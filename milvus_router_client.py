import asyncio
import numpy as np
import httpx
import logging
from typing import List, Dict, Tuple, Union, Optional
import time
from collections import defaultdict
import random
import os

# Configure logging
logger = logging.getLogger("async-milvus-router-client")

class AsyncMilvusRouterClient:
    """Asynchronous client for interacting with the Milvus Router API from tracker code"""
    
    def __init__(self, router_url, store_id, connection_timeout=10, 
                 embedding_dim=768, batch_size=50, max_retries=2,
                 enable_rate_limiting=None, enable_semaphore=None):
        """
        Initialize client that connects to the Milvus Router instead of directly to Milvus
        
        Args:
            router_url: Base URL of the router API
            store_id: ID of the store this client handles
            connection_timeout: Timeout for HTTP requests in seconds
            embedding_dim: Dimension of the feature embeddings
            batch_size: Size of batches for insert operations
            max_retries: Maximum number of retry attempts for operations
        """
        self.router_url = router_url.rstrip('/')  # Remove trailing slash if present
        self.store_id = store_id
        self.connection_timeout = connection_timeout
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.max_retries = max_retries

        self.enable_rate_limiting = (
            enable_rate_limiting if enable_rate_limiting is not None 
            else os.environ.get("MILVUS_ENABLE_RATE_LIMITING", "true").lower() in ("true", "1", "yes")
        )
        self.enable_semaphore = (
            enable_semaphore if enable_semaphore is not None
            else os.environ.get("MILVUS_ENABLE_SEMAPHORE", "true").lower() in ("true", "1", "yes")
        )

        # Configure rate limiting and semaphore based on settings
        if self.enable_semaphore:
            logger.info(f"semaphore enabled")
            self._max_concurrent = int(os.environ.get("MILVUS_MAX_CONCURRENT", "25"))
            self._request_semaphore = asyncio.Semaphore(self._max_concurrent)
            self._search_semaphore = asyncio.Semaphore(8)      # Dedicated for searches
            self._batch_semaphore = asyncio.Semaphore(5)       # Dedicated for batch ops
            self._feature_semaphore = asyncio.Semaphore(6)     # Dedicated for feature retrieval
            logger.debug(f"Semaphore enabled with max concurrent requests: {self._max_concurrent}")
        else:
            self._request_semaphore = None
            self._search_semaphore = None     # Dedicated for searches
            self._batch_semaphore = None       # Dedicated for batch ops
            self._feature_semaphore = None     # Dedicated for feature retrieval
            logger.debug("Semaphore disabled - unlimited concurrent requests")

        if self.enable_rate_limiting:
            self._min_interval = float(os.environ.get("MILVUS_MIN_INTERVAL", "0.01"))
            self._last_request_time = 0
            self._min_request_interval = self._min_interval
            logger.debug(f"Rate limiting enabled with min interval: {self._min_interval}s")
        else:
            self._min_interval = 0
            self._last_request_time = 0
            self._min_request_interval = 0
            logger.info("Rate limiting disabled - no minimum interval between requests")
        
        # Create the HTTP client during initialization
        self._http_client = None
        
        logger.debug(f"Initialized AsyncMilvusRouterClient for store_id={self.store_id}, router={router_url}")
        logger.debug(f"Rate limiting: {'enabled' if self.enable_rate_limiting else 'disabled'}, "
                   f"Semaphore: {'enabled' if self.enable_semaphore else 'disabled'}")

    async def _rate_limited_request(self, operation):
        """Apply rate limiting and semaphore controls if enabled"""
        # Handle semaphore if enabled
        if self.enable_semaphore:
            async with self._request_semaphore:
                return await self._execute_with_rate_limit(operation)
        else:
            return await self._execute_with_rate_limit(operation)

    async def _execute_with_rate_limit(self, operation):
        """Apply rate limiting if enabled"""
        if self.enable_rate_limiting:
            now = time.time()
            time_since_last = now - self._last_request_time
            
            if time_since_last < self._min_request_interval:
                sleep_time = self._min_request_interval - time_since_last
                #asyncio.sleep(sleep_time)
            
            self._last_request_time = time.time()
        
        return await operation()
    
    async def _direct_request(self, operation):
        """Execute operation without any rate limiting or semaphore controls"""
        return await operation()


    async def _high_priority_request(self, operation, operation_type="general"):
        """Use dedicated semaphores for critical operations"""
        if operation_type == "search":
            semaphore = self._search_semaphore
        elif operation_type == "batch": 
            semaphore = self._batch_semaphore
        elif operation_type == "features":
            semaphore = self._feature_semaphore
        else:
            semaphore = self._request_semaphore
            
        async with semaphore:
            # Skip rate limiting for critical ops to maximize throughput
            return await operation()
    
    async def _get_client(self):
        """Get or create the HTTP client for async operations"""
        if self._http_client is None:
            timeout_config = httpx.Timeout(
                connect=10.0,   # Connection timeout
                read=30.0,      # Read timeout (important for large queries)
                write=15.0,     # Write timeout
                pool=20.0       # Pool timeout
            )
            self._http_client = httpx.AsyncClient(
                timeout=timeout_config,
                limits=httpx.Limits(
                    max_keepalive_connections=150,  # Keep connections alive
                    max_connections=200,
                    keepalive_expiry=45.0
                ),
                headers={"Content-Type": "application/json"}
            )
        return self._http_client
    
    async def __aenter__(self):
        """Support for async context manager protocol"""
        await self._get_client()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when used as a context manager"""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
    
    async def close(self):
        """Close the client and release resources"""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
    
    async def check_connection_health(self):
        """Check if the connection to the router is healthy"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.router_url}/topology")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection health check failed: {e}")
            return False
    
    async def insert_embedding(self, track_id, embedding, store_id, camera_id, timestamp):
        """
        Insert a single embedding with retry logic for timeouts
        
        Returns:
            bool: True if successful, False otherwise
        """
        async def _insert_operation():
            # Process track_id if it's a tuple
            if isinstance(track_id, tuple) and len(track_id) > 0:
                track_id = track_id[0]

            if track_id is None:
                logger.error("Cannot insert embedding with None track_id")
                return False
            
            # Ensure embedding is in the correct format
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            else:
                embedding_list = embedding
                
            if len(embedding) != self.embedding_dim:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
                return False
            
            # Prepare request data
            data = {
                "track_id": track_id,
                "feature_vector": embedding,
                "store_id": store_id,
                "camera_id": camera_id,
                "timestamp": timestamp
            }

            try:
                client = await self._get_client()
                                
                # Send request to router with extended timeout
                response = await client.post(
                    f"{self.router_url}/insert",
                    json=data
                )
                                
                if response.status_code == 200:
                    logger.debug(f"[Milvus] Inserted feature for track_id {track_id}")
                    return True
                else:
                    logger.error(f"Insert failed with status {response.status_code}: {response.text}")
                    return False
                    
                
            except Exception as e:
                # Log other exceptions in detail
                logger.error(f"Error inserting feature_vector for track_id {track_id}: {str(e)}", exc_info=True)
                # For other exceptions, don't retry
                return False
        if self.enable_rate_limiting or self.enable_semaphore:
            return await self._rate_limited_request(_insert_operation)
        else:
            return await self._direct_request(_insert_operation)
        


    def _calculate_retry_delay(self, attempt):
        """Calculate retry delay with exponential backoff and jitter"""
        base_delay = 0.5
        max_delay = 10.0
        # Exponential backoff with jitter
        delay = min(max_delay, base_delay * (2 ** attempt))
        # Add jitter (±20%)
        jitter = delay * 0.2 * (random.random() * 2 - 1)
        return delay + jitter

    async def insert_embeddings_batch(
        self,
        track_ids,
        embeddings,
        store_ids,
        camera_ids,
        timestamps,
        batch_size=None
        ):
        """
        Insert multiple embeddings in batches
        
        Returns:
            int: Number of successfully inserted embeddings
        """
        async def _batch_insert_operation():
            if not track_ids or len(track_ids) == 0:
                logger.warning("No embeddings provided for batch insert")
                return 0
            
            # Validate input lists have same length
            if not (len(track_ids) == len(embeddings) == len(store_ids) == len(camera_ids) == len(timestamps)):
                logger.error("All input lists must have the same length for batch insert")
                return 0
            
            # Convert numpy arrays to lists
            processed_embeddings = []
            for emb in embeddings:
                if isinstance(emb, np.ndarray):
                    processed_embeddings.append(emb.tolist())
                else:
                    processed_embeddings.append(emb)

            effective_batch_size = min(batch_size or self.batch_size, 50)  # FIXED: 30 items per batch for efficiency
            total_records = len(track_ids)
            inserted_count = 0
        
            client = await self._get_client()

            # Process in batches
            for i in range(0, total_records, effective_batch_size):
                end_idx = min(i + effective_batch_size, total_records)

                # Prepare batch data
                batch_data = {
                    "track_ids": track_ids[i:end_idx],
                    "feature_vectors": processed_embeddings[i:end_idx],
                    "store_ids": store_ids[i:end_idx],
                    "camera_ids": camera_ids[i:end_idx],
                    "timestamps": timestamps[i:end_idx]
                }
                batch_success = False
                for attempt in range(2):
                    try:

                        # Send batch to router
                        response = await client.post(
                            f"{self.router_url}/batch_insert",
                            json=batch_data,
                            timeout = 10.0
                        )
            
                        if response.status_code == 200:
                            batch_result = response.json()
                            inserted_count += batch_result.get("total_inserted", 0)
                            batch_success = True
                            break
                        elif response.status_code >= 500 and attempt == 0:
                            logger.error(f"Batch insert failed with status {response.status_code}: {response.text}")
                            #asyncio.sleep(0.1)
                            continue
                        else:
                            logger.error(f"Batch insert failed with status {response.status_code}: {response.text}")
                            break

                    except Exception as e:
                        if attempt == 0:
                            #asyncio.sleep(0.1)
                            continue
                        else:
                            logger.warning(f"Batch insert failed: {e}")
                            break
                if not batch_success:
                    logger.debug(f"Failed to insert batch {i//effective_batch_size + 1}")
            
                # if end_idx < total_records:
                    #asyncio.sleep(0.1)
                    
            return inserted_count
        
        # Use rate limited or direct request based on configuration
        if self.enable_rate_limiting or self.enable_semaphore:
            return await self._rate_limited_request(_batch_insert_operation)
        else:
            return await self._direct_request(_batch_insert_operation)

    
    async def search_embedding(
        self,
        query_embedding,
        top_k=5,
        store_filter=None,
        camera_filter=None,
        min_similarity=0.0,
        max_retries=None,
        retry_delay=1.0,
        use_partition=True
        ):
        """
        Search for similar embeddings
        
        Returns:
            List of tuples: [(track_id, distance), ...]ct
        """
        async def _search_operation():
            store_id = store_filter if store_filter is not None else self.store_id
            effective_max_retries = max_retries or self.max_retries
            # Convert numpy array to list

            if isinstance(query_embedding, np.ndarray):
                effective_query_embedding = query_embedding.tolist()
            else:
                effective_query_embedding = query_embedding
                
            # Prepare request data
            data = {
                "feature_vector": effective_query_embedding,
                "store_id": store_id,
                "top_k": top_k
            }
            
            if camera_filter is not None:
                data["camera_filter"] = camera_filter
                
            if min_similarity > 0:
                data["min_similarity"] = min_similarity
            
            client = await self._get_client()
                
            # Execute with retry logic
            for attempt in range(effective_max_retries):
                try:
                    response = await client.post(
                        f"{self.router_url}/track/search",
                        json=data,
                        timeout = 3.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result.get("matches", [])
                    elif response.status_code < 500:
                        logger.debug(f"Search client error {response.status_code} - not retrying")
                        return []
                    else:
                        logger.error(f"Search attempt {attempt+1}/{effective_max_retries} failed with status {response.status_code}: {response.text}")
                        if attempt < effective_max_retries - 1:
                            #asyncio.sleep(0.1)
                            logger.info("will sleep here 0.1")
                            continue

                except asyncio.TimeoutError:
                    if attempt < effective_max_retries - 1:
                        #asyncio.sleep(0.05)
                        logger.info("will sleep here 0.05")
                        continue

                except Exception as e:
                    logger.error(f"Search attempt {attempt+1}/{effective_max_retries} failed: {e}")
                    # Retry with exponential backoff (use asyncio.sleep instead of time.sleep)
                    if attempt < effective_max_retries - 1:
                        #asyncio.sleep(0.05)
                        logger.info("will sleep here 0.05")
                        continue
                    
            logger.error("All search attempts failed")
            return []

        # Use rate limited or direct request based on configuration
        if self.enable_rate_limiting or self.enable_semaphore:
            return await self._rate_limited_request(_search_operation)
        else:
            return await self._direct_request(_search_operation)
    
    async def get_features_by_track_id(self, track_id, store_id=None):
        """
        Retrieve all feature embeddings for a specific track_id
        
        Returns:
            List of feature embeddings (numpy arrays)
        """
        async def _get_features_operation():
            logger.debug("INSIDE MILVUS ROUTER CLIENT CODE")
            effective_store_id = store_id if store_id is not None else self.store_id
            real_track_id = track_id

            if isinstance(real_track_id, tuple) and len(real_track_id) > 0:
                real_track_id = real_track_id[0]
            
            # If track_id is None, return empty list
            if real_track_id is None:
                logger.debug(f"Cannot retrieve features for None track_id")
                return []
                
            try:
                client = await self._get_client()
                response = await client.get(
                    f"{self.router_url}/track/features/{track_id}/{effective_store_id}",
                    timeout=15.0 
                )

                logger.debug(f"Get Features Response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    # Convert list features to numpy arrays
                    features = [np.array(f, dtype=np.float32) for f in result.get("features", [])]
                    return features
                else:
                    logger.error(f"Get features failed for track_id={track_id}: status={response.status_code}, response={response.text}")
                    logger.error(f"Get features failed with status {response.status_code}: {response.text}")
                    return []
            except httpx.TimeoutException as e:
                logger.error(f"Timeout retrieving features for track_id={track_id}: {str(e)}")
                return []
            except httpx.RequestError as e:
                logger.error(f"Request error retrieving features for track_id={track_id}: {str(e)}")
                return []
            except Exception as e:
                logger.error(f"Error retrieving features for track_id={real_track_id}: {e}")
                return []
        # Use rate limited or direct request based on configuration
        if self.enable_rate_limiting or self.enable_semaphore:
            return await self._rate_limited_request(_get_features_operation)
        else:
            return await self._direct_request(_get_features_operation)
    
    async def get_all_track_features(self, store_id, limit=100):
        """
        Returns a dictionary where keys are track_ids and values are lists of embeddings
        
        Returns:
            Dict mapping track_ids to lists of feature embeddings
        """
        async def _get_all_features_operation():
            effective_store_id = store_id if store_id is not None else self.store_id
            safe_limit = min(limit, 16384)
            try:
                client = await self._get_client()
                response = await client.get(
                    f"{self.router_url}/track/allfeatures/{effective_store_id}?limit={limit}"
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Convert to dict of numpy arrays
                    feature_map = {}
                    for track_id_str, embeddings in result.items():
                        track_id = int(track_id_str)
                        feature_map[track_id] = [np.array(emb, dtype=np.float32) for emb in embeddings]
                    
                    return feature_map
                else:
                    logger.error(f"Get all features failed with status {response.status_code}: {response.text}")
                    return {}
                    
            except Exception as e:
                logger.error(f"Failed to query embeddings: {e}")
                return {}
        # Use rate limited or direct request based on configuration
        if self.enable_rate_limiting or self.enable_semaphore:
            return await self._rate_limited_request(_get_all_features_operation)
        else:
            return await self._direct_request(_get_all_features_operation)

    async def _get_all_features_without_rate_limit(self, store_id, limit=100):
        """Internal method to get all features without rate limiting"""
        safe_limit = min(limit, 16384)
        
        client = await self._get_client()
        response = await client.get(
            f"{self.router_url}/track/allfeatures/{store_id}?limit={safe_limit}"
        )
        
        if response.status_code == 200:
            result = response.json()
            feature_map = {}
            for track_id_str, embeddings in result.items():
                track_id = int(track_id_str)
                feature_map[track_id] = [np.array(emb, dtype=np.float32) for emb in embeddings]
            return feature_map
        else:
            return {}
    
    async def get_all_track_ids(self, store_id):
        """
        Fetch all unique track_ids in the collection for a given store_id
        
        Returns:
            List of unique track IDs
        """
        async def _get_ids_operation():
            effective_store_id = store_id if store_id is not None else self.store_id
            
            try:
                # Get all track features and extract just the keys
                feature_map = await self._get_all_features_without_rate_limit(effective_store_id)
                track_ids = list(feature_map.keys())
                return track_ids
                    
            except Exception as e:
                logger.error(f"Error fetching track_ids for store_id {effective_store_id}: {e}")
                return []
        return await self._rate_limited_request(_get_ids_operation)
    
    async def delete_track(self, track_id, store_id):
        """
        Delete all embeddings associated with a given track_id and store_id
        
        Returns:
            bool: True if successful, False otherwise
        """
        async def _delete_operation():
            effective_store_id = store_id if store_id is not None else self.store_id
            
            try:
                # Prepare request data
                data = {
                    "track_id": track_id,
                    "store_id": effective_store_id
                }
                
                # Send delete request to router
                client = await self._get_client()
                response = await client.post(
                    f"{self.router_url}/delete",
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("deleted", False)
                else:
                    logger.error(f"Delete failed with status {response.status_code}: {response.text}")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to delete track_id={track_id}, store_id={effective_store_id}: {e}")
                return False
        # Use rate limited or direct request based on configuration
        if self.enable_rate_limiting or self.enable_semaphore:
            return await self._rate_limited_request(_delete_operation)
        else:
            return await self._direct_request(_delete_operation)
    
    async def find_next_best_match(self, feature, assigned_ids, max_candidates=5, distance_threshold=0.7):
        """
        Find the next best match from Milvus, excluding already assigned IDs
        
        Args:
            feature: Feature vector of the detection
            assigned_ids: Set of track IDs already assigned in the current frame
            max_candidates: Maximum number of candidates to consider
            distance_threshold: Maximum distance threshold for a valid match
                
        Returns:
            Tuple[int or None, float]: (track_id, distance) or (None, inf) if no good match found
        """
        async def _find_match_operation():
            try:
                # Perform general search
                results = await self.search_embedding(
                    query_embedding=feature,
                    store_filter=self.store_id,
                    top_k=max_candidates * 2  # Get more results for filtering
                )

                # Filter out assigned IDs and keep only those below threshold
                unassigned_matches = []
                for track_id, distance in results:
                    if track_id not in assigned_ids and distance < distance_threshold:
                        unassigned_matches.append((track_id, distance))
                
                # Sort by distance and return best match
                unassigned_matches.sort(key=lambda x: x[1])
                
                if unassigned_matches:
                    best_track_id, best_distance = unassigned_matches[0]
                    logger.debug(f"Found unassigned match: track_id={best_track_id}, distance={best_distance}")
                    return best_track_id, best_distance
                
                logger.debug(f"No suitable unassigned match found below threshold {distance_threshold}")
                return None, float('inf')
            except Exception as e:
                logger.error(f"Error finding next best match: {e}")
                return None, float('inf')
        # Use rate limited or direct request based on configuration
        if self.enable_rate_limiting or self.enable_semaphore:
            return await self._rate_limited_request(_find_match_operation)
        else:
            return await self._direct_request(_find_match_operation)
            
    # Batch processing methods
    
    async def search_embeddings_batch(self, embeddings_list, top_k=5, store_id=None):
        """HIGH-VOLUME FIX: Optimized batch search for 10s of cameras at 5-10 FPS"""
        
        async def _batch_search_operation():
            effective_store_id = store_id if store_id is not None else self.store_id
            logger.debug(f"effective store id is {effective_store_id}")
            if not embeddings_list:
                return []
            
            max_batch_size = 25 if (self.enable_rate_limiting or self.enable_semaphore) else 50
            # HIGH-VOLUME FIX: Larger threshold before fallback - batch is more efficient
            if len(embeddings_list) > max_batch_size:  # FIXED: Allow larger batches for efficiency
                logger.info(f"Very large batch ({len(embeddings_list)} items), splitting into chunks")
                # Split into manageable chunks
                chunk_size = max_batch_size // 2
                all_results = []
                for i in range(0, len(embeddings_list), chunk_size):
                    chunk = embeddings_list[i:i + chunk_size]
                    chunk_results = await self.search_embeddings_batch(chunk, top_k, effective_store_id)
                    all_results.extend(chunk_results)
                    # Small delay between chunks
                    if i + chunk_size < len(embeddings_list):
                        #asyncio.sleep(0.02)  # 20ms between chunks
                        logger.info("will sleep here 0.02")
                return all_results

            processed_embeddings = []
            for embedding in embeddings_list:
                if isinstance(embedding, np.ndarray):
                    processed_embeddings.append(embedding.tolist())
                else:
                    processed_embeddings.append(embedding)

            batch_data = {
                "feature_vectors": processed_embeddings,
                "store_id": effective_store_id,
                "top_k": top_k
            }
            
            try:
                client = await self._get_client()
                max_attempts = 2 if (self.enable_rate_limiting or self.enable_semaphore) else 3
                # HIGH-VOLUME FIX: Try batch twice with short timeout, then fallback
                for attempt in range(max_attempts):
                    try:
                        timeout = 5.0 if (self.enable_rate_limiting or self.enable_semaphore) else 8.0
                        response = await client.post(
                            f"{self.router_url}/batch_search",
                            json=batch_data,
                            timeout = timeout  # 5 second timeout for batches
                        )
                        
                        if response.status_code == 200:
                            # logger.info("200 for batch search. No timeouts")
                            result = response.json()
                            return result.get("results", [])
                        elif response.status_code >= 500 and attempt == 0:
                            # Server error - quick retry once
                            sleep_time = 0.1 if self.enable_rate_limiting else 0.05
                            #asyncio.sleep(sleep_time)
                            logger.info("5XX for batch search")
                            logger.info(f"will sleep here {sleep_time}s")
                            continue
                        else:
                            logger.info("4XX for batch search")
                            break
                            
                    except asyncio.TimeoutError:
                        if attempt == 0:
                            sleep_time = 0.1 if self.enable_rate_limiting else 0.05
                            #asyncio.sleep(sleep_time)
                            logger.info(f"will sleep here {sleep_time}s")
                            continue
                        else:
                            break
                            
                # Fallback to smart individual/micro-batch processing
                logger.info(f"Batch search failed, using fallback for {len(embeddings_list)} items")
                return await self._fallback_individual_searches(embeddings_list, top_k, effective_store_id)
                    
            except Exception as e:
                logger.info(f"Batch search error: {e}, using fallback")
                return await self._fallback_individual_searches(embeddings_list, top_k, effective_store_id)
        
        # Use rate limited or direct request based on configuration
        if self.enable_rate_limiting or self.enable_semaphore:
            return await self._rate_limited_request(_batch_search_operation)
        else:
            return await self._direct_request(_batch_search_operation)

    # Add this helper method to handle fallbacks
    async def _fallback_individual_searches(self, embeddings_list, top_k, store_id):
        """Fallback: perform individual searches when batch fails"""
        results = []
        effective_store_id = store_id if store_id is not None else self.store_id
        micro_batch_size = 3 if (self.enable_rate_limiting or self.enable_semaphore) else 6
        for i in range(0, len(embeddings_list), micro_batch_size):
            micro_batch = embeddings_list[i:i + micro_batch_size]
            if len(micro_batch) == 1:
                try:
                    individual_result = await self._individual_search_without_rate_limit(
                    micro_batch[0], top_k, effective_store_id
                    )
                    results.append(individual_result)
                except Exception as e:
                    logger.debug(f"Individual search failed: {e}")
                    results.append([])
            else:
                try:
                    processed_embeddings = []
                    for embedding in micro_batch:
                        if isinstance(embedding, np.ndarray):
                            processed_embeddings.append(embedding.tolist())
                        else:
                            processed_embeddings.append(embedding)
                    batch_data = {
                        "feature_vectors": processed_embeddings,
                        "store_id": effective_store_id,
                        "top_k": top_k
                    }   
                    client = await self._get_client()
                    response = await client.post(f"{self.router_url}/batch_search", json=batch_data, timeout=2.0)
                    if response.status_code == 200:
                        batch_results = response.json().get("results", [])
                        results.extend(batch_results)
                    else:
                        # Fall back to individual for this micro-batch
                        for embedding in micro_batch:
                            try:
                                individual_result = await self._individual_search_without_rate_limit(
                                    embedding, top_k, effective_store_id
                                )
                                results.append(individual_result)
                            except:
                                results.append([])
                except:
                    # Fall back to individual for this micro-batch
                    for embedding in micro_batch:
                        try:
                            individual_result = await self._individual_search_without_rate_limit(
                                embedding, top_k, effective_store_id
                            )
                            results.append(individual_result)
                        except:
                            results.append([])
            if i + micro_batch_size < len(embeddings_list):
                #asyncio.sleep(0.05)
                logger.info(f"will sleep here 0.05s")
        return results


    async def _individual_search_without_rate_limit(self, embedding, top_k, store_id):
        """Individual search without rate limiting (for internal use only)"""
        effective_store_id = store_id if store_id is not None else self.store_id
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        data = {
            "feature_vector": embedding,
            "store_id": effective_store_id,
            "top_k": top_k
        }
        
        client = await self._get_client()
        response = await client.post(f"{self.router_url}/track/search", json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("results", [])
        else:
            return []

    
    async def insert_embeddings_concurrent(self, track_ids, embeddings, store_ids, camera_ids, timestamps):
        """
        Insert multiple embeddings with concurrent processing for better performance
        
        Returns:
            int: Number of successful insertions
        """
        if not track_ids or len(track_ids) == 0:
            return 0
            
        # Validate input lists have same length
        if not (len(track_ids) == len(embeddings) == len(store_ids) == len(camera_ids) == len(timestamps)):
            logger.error("All input lists must have the same length")
            return 0
            
        # Create tasks for all insertions (with reasonable concurrency limits)
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
        
        async def insert_with_limit(idx):
            async with semaphore:
                return await self.insert_embedding(
                    track_id=track_ids[idx],
                    embedding=embeddings[idx],
                    store_id=store_ids[idx],
                    camera_id=camera_ids[idx],
                    timestamp=timestamps[idx]
                )
        
        # Create all insertion tasks
        tasks = [insert_with_limit(i) for i in range(len(track_ids))]
        
        # Execute all insertions concurrently and count successes
        results = await asyncio.gather(*tasks)
        success_count = sum(1 for result in results if result)
        
        return success_count

    def get_performance_info(self):
        """Get information about current performance settings"""
        return {
            "rate_limiting_enabled": self.enable_rate_limiting,
            "semaphore_enabled": self.enable_semaphore,
            "max_concurrent": getattr(self, '_max_concurrent', None),
            "min_interval": self._min_interval if self.enable_rate_limiting else 0,
            "configuration_source": {
                "rate_limiting": "environment" if "MILVUS_ENABLE_RATE_LIMITING" in os.environ else "default",
                "semaphore": "environment" if "MILVUS_ENABLE_SEMAPHORE" in os.environ else "default"
            }
        }

# Example usage with async context:
#
# async def main():
#     async with AsyncMilvusRouterClient(router_url="http://localhost:8000", store_id=1) as client:
#         features = await client.get_features_by_track_id(123)
#         print(f"Found {len(features)} features")
#
# asyncio.run(main())