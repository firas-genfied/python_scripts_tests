import asyncio
import numpy as np
import httpx
import logging
from typing import List, Dict, Tuple, Union, Optional
import time
from collections import defaultdict

# Configure logging
logger = logging.getLogger("async-milvus-router-client")

class AsyncMilvusRouterClient:
    """Asynchronous client for interacting with the Milvus Router API from tracker code"""
    
    def __init__(self, router_url, store_id, connection_timeout=10, 
                 embedding_dim=768, batch_size=100, max_retries=3):
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
        
        # Create the HTTP client during initialization
        self._http_client = None
        
        logger.info(f"Initialized AsyncMilvusRouterClient for store_id={store_id}, router={router_url}")
    
    async def _get_client(self):
        """Get or create the HTTP client for async operations"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self.connection_timeout,
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
        Insert a single embedding
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure embedding is in the correct format
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
                
            if len(embedding) != self.embedding_dim:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
                return False
                
            # Prepare request data
            data = {
                "track_id": track_id,
                "embedding": embedding,
                "store_id": store_id,
                "camera_id": camera_id,
                "timestamp": timestamp
            }
            
            # Send request to router
            client = await self._get_client()
            response = await client.post(
                f"{self.router_url}/insert", 
                json=data
            )
            
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Insert failed with status {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error inserting embedding: {e}")
            return False
            
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
        
        batch_size = batch_size or self.batch_size
        total_records = len(track_ids)
        inserted_count = 0
        
        try:
            client = await self._get_client()
            
            # Process in batches
            for i in range(0, total_records, batch_size):
                end_idx = min(i + batch_size, total_records)
                
                # Prepare batch data
                batch_data = {
                    "track_ids": track_ids[i:end_idx],
                    "embeddings": processed_embeddings[i:end_idx],
                    "store_ids": store_ids[i:end_idx],
                    "camera_ids": camera_ids[i:end_idx],
                    "timestamps": timestamps[i:end_idx]
                }
                
                # Send batch to router
                response = await client.post(
                    f"{self.router_url}/batch_insert",
                    json=batch_data
                )
                
                if response.status_code == 200:
                    batch_result = response.json()
                    inserted_count += batch_result.get("total_inserted", 0)
                else:
                    logger.error(f"Batch insert failed with status {response.status_code}: {response.text}")
            
            logger.info(f"Successfully inserted {inserted_count} embeddings in batches")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error in batch insert: {e}")
            return inserted_count
    
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
            List of tuples: [(track_id, distance), ...]
        """
        store_id = store_filter if store_filter is not None else self.store_id
        max_retries = max_retries or self.max_retries
        
        # Convert numpy array to list
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
            
        # Prepare request data
        data = {
            "embedding": query_embedding,
            "store_id": store_id,
            "top_k": top_k
        }
        
        if camera_filter is not None:
            data["camera_filter"] = camera_filter
            
        if min_similarity > 0:
            data["min_similarity"] = min_similarity
        
        client = await self._get_client()
            
        # Execute with retry logic
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    f"{self.router_url}/track/search",
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("results", [])
                else:
                    logger.error(f"Search attempt {attempt+1}/{max_retries} failed with status {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Search attempt {attempt+1}/{max_retries} failed: {e}")
                
            # Retry with exponential backoff (use asyncio.sleep instead of time.sleep)
            if attempt < max_retries - 1:
                backoff = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {backoff:.2f} seconds...")
                await asyncio.sleep(backoff)
                
        logger.error("All search attempts failed")
        return []
    
    async def get_features_by_track_id(self, track_id, store_id=None):
        """
        Retrieve all feature embeddings for a specific track_id
        
        Returns:
            List of feature embeddings (numpy arrays)
        """
        store_id = store_id if store_id is not None else self.store_id
        
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.router_url}/track/features/{track_id}/{store_id}"
            )
            
            if response.status_code == 200:
                result = response.json()
                # Convert list features to numpy arrays
                features = [np.array(f, dtype=np.float32) for f in result.get("features", [])]
                return features
            else:
                logger.error(f"Get features failed with status {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving features for track_id={track_id}: {e}")
            return []
    
    async def get_all_track_features(self, store_id, limit=100):
        """
        Returns a dictionary where keys are track_ids and values are lists of embeddings
        
        Returns:
            Dict mapping track_ids to lists of feature embeddings
        """
        store_id = store_id if store_id is not None else self.store_id
        safe_limit = min(limit, 16384)
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.router_url}/track/allfeatures/{store_id}?limit={limit}"
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
    
    async def get_all_track_ids(self, store_id):
        """
        Fetch all unique track_ids in the collection for a given store_id
        
        Returns:
            List of unique track IDs
        """
        store_id = store_id if store_id is not None else self.store_id
        
        try:
            # Get all track features and extract just the keys
            feature_map = await self.get_all_track_features(store_id)
            track_ids = list(feature_map.keys())
            return track_ids
                
        except Exception as e:
            logger.error(f"Error fetching track_ids for store_id {store_id}: {e}")
            return []
    
    async def delete_track(self, track_id, store_id):
        """
        Delete all embeddings associated with a given track_id and store_id
        
        Returns:
            bool: True if successful, False otherwise
        """
        store_id = store_id if store_id is not None else self.store_id
        
        try:
            # Prepare request data
            data = {
                "track_id": track_id,
                "store_id": store_id
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
            logger.error(f"Failed to delete track_id={track_id}, store_id={store_id}: {e}")
            return False
    
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
                logger.info(f"Found unassigned match: track_id={best_track_id}, distance={best_distance}")
                return best_track_id, best_distance
            
            logger.info(f"No suitable unassigned match found below threshold {distance_threshold}")
            return None, float('inf')
        except Exception as e:
            logger.error(f"Error finding next best match: {e}")
            return None, float('inf')
            
    # Batch processing methods
    
    async def search_embeddings_batch(self, embeddings_list, top_k=5, store_id=None):
        """
        Search for multiple embeddings in parallel
        
        Args:
            embeddings_list: List of embedding vectors to search for
            top_k: Maximum number of results to return for each query
            store_id: Store ID to filter by (defaults to client's store_id)
            
        Returns:
            List of search results (one per input embedding)
        """
        store_id = store_id if store_id is not None else self.store_id
        
        # Create tasks for all searches
        tasks = []
        for embedding in embeddings_list:
            task = self.search_embedding(
                query_embedding=embedding, 
                top_k=top_k, 
                store_filter=store_id
            )
            tasks.append(task)
            
        # Run all searches in parallel
        results = await asyncio.gather(*tasks)
        return results
    
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

# Example usage with async context:
#
# async def main():
#     async with AsyncMilvusRouterClient(router_url="http://localhost:8000", store_id=1) as client:
#         features = await client.get_features_by_track_id(123)
#         print(f"Found {len(features)} features")
#
# asyncio.run(main())