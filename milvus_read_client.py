import os
import time
import asyncio
import logging
from dotenv import load_dotenv
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
from typing import List, Tuple, Union, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time
# Import the helper
from config import get_milvus_host_port
import asyncio
from collections import deque

# Load .env variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("milvus_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("milvus-client")

import asyncio
from collections import deque

class AsyncTrackManager:
    """Manages asynchronous operations with Milvus for person tracking"""
    
    def __init__(self, milvus_client):
        self.milvus_client = milvus_client
        self.operation_queue = asyncio.Queue()
        self.batch_size = 50  # Max operations per batch
        self.batch_timeout = 0.1  # Max seconds to wait for a full batch
        self.processing_task = None
        self.is_running = False
    
    async def start(self):
        """Start the background processing task"""
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_operations())
        return self.processing_task
    
    async def stop(self):
        """Stop the background processing task"""
        self.is_running = False
        if self.processing_task:
            await self.operation_queue.put(None)  # Signal to stop
            await self.processing_task
    
    async def queue_insert(self, track_id, feature, store_id, camera_id, timestamp):
        """Queue a feature for insertion"""
        await self.operation_queue.put({
            "op": "insert",
            "track_id": track_id,
            "feature": feature,
            "store_id": store_id,
            "camera_id": camera_id,
            "timestamp": timestamp
        })
    
    async def queue_delete(self, track_id, store_id):
        """Queue a track for deletion"""
        await self.operation_queue.put({
            "op": "delete",
            "track_id": track_id,
            "store_id": store_id
        })
    
    async def get_features(self, track_id, store_id):
        """Directly fetch features (still async but not queued)"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.milvus_client.get_features_by_track_id(track_id, store_id)
        )
    
    async def search_embedding(self, feature, store_id, top_k=5):
        """Directly search for similar embeddings (still async but not queued)"""
        return await self.milvus_client.search_embedding_async(
            query_embedding=feature,
            top_k=top_k,
            store_filter=store_id
        )
    
    async def _process_operations(self):
        """Background task to process queued operations in batches"""
        while self.is_running:
            try:
                # Collect a batch of operations
                batch = []
                try:
                    # Always get at least one operation
                    operation = await self.operation_queue.get()
                    if operation is None:  # Stop signal
                        break
                    batch.append(operation)
                    
                    # Try to get more operations up to batch_size or timeout
                    while len(batch) < self.batch_size:
                        try:
                            operation = await asyncio.wait_for(
                                self.operation_queue.get(), 
                                timeout=self.batch_timeout
                            )
                            if operation is None:  # Stop signal
                                break
                            batch.append(operation)
                        except asyncio.TimeoutError:
                            break  # Timeout collecting batch, process what we have
                
                except Exception as e:
                    logger.error(f"Error collecting operations batch: {e}")
                    await asyncio.sleep(1)  # Avoid tight loop on error
                    continue
                
                if not batch:
                    continue
                
                # Group operations by type
                inserts = []
                deletes = []
                
                for op in batch:
                    if op["op"] == "insert":
                        inserts.append(op)
                    elif op["op"] == "delete":
                        deletes.append(op)
                
                # Process inserts in a batch
                if inserts:
                    try:
                        track_ids = [op["track_id"] for op in inserts]
                        features = [op["feature"] for op in inserts]
                        store_ids = [op["store_id"] for op in inserts]
                        camera_ids = [op["camera_id"] for op in inserts]
                        timestamps = [op["timestamp"] for op in inserts]
                        
                        await asyncio.to_thread(
                            self.milvus_client.insert_embeddings_batch,
                            track_ids, features, store_ids, camera_ids, timestamps
                        )
                        logger.info(f"Batch inserted {len(inserts)} features")
                    except Exception as e:
                        logger.error(f"Error batch inserting features: {e}")
                
                # Process deletes sequentially (could be batched in future)
                for delete_op in deletes:
                    try:
                        await asyncio.to_thread(
                            self.milvus_client.delete_track,
                            delete_op["track_id"],
                            delete_op["store_id"]
                        )
                        logger.info(f"Deleted track {delete_op['track_id']}")
                    except Exception as e:
                        logger.error(f"Error deleting track {delete_op['track_id']}: {e}")
                
                # Mark tasks as done
                for _ in range(len(batch)):
                    self.operation_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Error in operation processor: {e}")
                await asyncio.sleep(1)  # Avoid tight loop on error
                
class MilvusReIDClient:
    # Class-level connection pool
    _connection_pool = {}
    
    def __init__(self, store_id: int, collection_name: str = "person_embeddings", 
                embedding_dim: int = 768, connection_timeout: int = 10):
        """
        Initialize MilvusReIDClient with store-specific connection
        
        Args:
            store_id: ID of the store to connect to
            collection_name: Name of the Milvus collection
            embedding_dim: Dimension of feature vectors
            connection_timeout: Connection timeout in seconds
        """
        self.store_id = store_id
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.connection_timeout = connection_timeout
        
        # Get host and port for this store_id
        host, port = get_milvus_host_port(store_id)
        connection_key = f"{host}:{port}"
        self.connection_alias = connection_key
        
        # Establish connection (with retry logic)
        self._connect_with_retry(host, port)
        
        # Setup collection
        self.collection = self._get_or_create_collection()
        
    def _connect_with_retry(self, host: str, port: str, max_retries: int = 3):
        """Establish connection to Milvus with retry logic"""
        connection_key = f"{host}:{port}"
        
        # Check if already connected
        if connection_key in MilvusReIDClient._connection_pool:
            logger.info(f"Using existing connection to Milvus at {host}:{port}")
            return
            
        # Attempt connection with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"[store_id={self.store_id}] Connecting to Milvus at {host}:{port} (attempt {attempt+1}/{max_retries})...")
                connections.connect(connection_key, host=host, port=port, timeout=self.connection_timeout)
                MilvusReIDClient._connection_pool[connection_key] = True
                logger.info(f"Successfully connected to Milvus at {host}:{port}")
                return
            except Exception as e:
                logger.error(f"Connection attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # If we get here, all retries failed
        raise ConnectionError(f"Failed to connect to Milvus at {host}:{port} after {max_retries} attempts")
        
    def _get_or_create_collection(self) -> Collection:
        """Get or create the collection with optimal schema and indexing"""
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' exists. Loading...")
                collection = Collection(self.collection_name)
            else:
                logger.info(f"Collection '{self.collection_name}' not found. Creating...")
                fields = [
                    FieldSchema(name="track_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                    FieldSchema(name="store_id", dtype=DataType.INT64),
                    FieldSchema(name="camera_id", dtype=DataType.INT64),
                    FieldSchema(name="timestamp", dtype=DataType.INT64),
                ]
                schema = CollectionSchema(fields, description="Person ReID Embeddings")
                collection = Collection(name=self.collection_name, schema=schema)
                
                # Create partitions for better performance
                # Each partition will handle a range of store IDs
                partition_size = 1
                max_stores = 10  # Adjust based on expected number of stores
                
                for start in range(1, max_stores + 1, partition_size):
                    end = min(start + partition_size - 1, max_stores)
                    partition_name = f"stores_{start}_to_{end}"
                    collection.create_partition(partition_name)
                    logger.info(f"Created partition '{partition_name}'")
                
                # Create index for vector similarity search
                index_params = {
                    "index_type": "HNSW",  # Better for high recall and faster queries than IVF_FLAT
                    "metric_type": "COSINE",  
                    "params": {
                        "M": 16,            # Number of bi-directional links
                        "efConstruction": 200  # Higher for better index quality
                    }
                }
                
                logger.info(f"Creating index on embedding field...")
                collection.create_index("embedding", index_params)
                
                # Create additional scalar field indices for faster filtering
                collection.create_index("store_id", {"index_type": "FLAT"})
                collection.create_index("camera_id", {"index_type": "FLAT"})
                collection.create_index("track_id", {"index_type": "FLAT"})
                
                logger.info(f"Collection '{self.collection_name}' successfully created with indices")
            
            # Calculate which partition to load based on store_id
            partition_size = 1
            max_stores = 10
            partition_start = ((self.store_id - 1) // partition_size) * partition_size + 1
            partition_end = min(partition_start + partition_size - 1, max_stores)
            partition_name = f"stores_{partition_start}_to_{partition_end}"
            
            logger.info(f"Loading partition '{partition_name}' for store_id={self.store_id}...")
            try:
                collection.load(partition_names=[partition_name])
            except Exception as e:
                logger.warning(f"Error loading partition '{partition_name}', falling back to full collection: {e}")
                collection.load()
                
            return collection
            
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise

    def check_connection_health(self):
        """Check if the connection to Milvus is healthy and reconnect if needed"""
        try:
            # Simple collection existence check to test connection
            is_connected = self.collection_name in utility.list_collections()
            return is_connected
        except Exception as e:
            logger.error(f"Connection health check failed: {e}")
            # Attempt to reconnect
            host, port = get_milvus_host_port(self.store_id)
            try:
                connections.disconnect(self.connection_alias)
            except:
                pass
            
            connections.connect(self.connection_alias, host=host, port=port)
            return self.collection_name in utility.list_collections()
    
    def insert_embedding(
        self,
        track_id: int,
        embedding: Union[List[float], np.ndarray],
        store_id: int,
        camera_id: int,
        timestamp: int
        ) -> bool:
        """
        Insert a single embedding into the collection
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure embedding is in the correct format and dimension
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
                
            if len(embedding) != self.embedding_dim:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
                return False
                
            data = [[track_id], [embedding], [store_id], [camera_id], [timestamp]]
            logger.info(f"Inserting embedding for track_id={track_id}, store_id={store_id}")
            self.collection.insert(data)
            self.collection.flush()
            return True
        except Exception as e:
            logger.error(f"Error inserting embedding: {e}")
            return False
    
    def insert_embeddings_batch(
        self,
        track_ids: List[int],
        embeddings: List[Union[List[float], np.ndarray]],
        store_ids: List[int],
        camera_ids: List[int],
        timestamps: List[int],
        batch_size: int = 1000
        ) -> int:
        """
        Insert multiple embeddings in batches for better performance
        
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
            
        # Convert numpy arrays to lists if needed
        processed_embeddings = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                processed_embeddings.append(emb.tolist())
            else:
                processed_embeddings.append(emb)
        
        total_records = len(track_ids)
        inserted_count = 0
        
        try:
            # Process in batches
            for i in range(0, total_records, batch_size):
                end_idx = min(i + batch_size, total_records)
                batch_track_ids = track_ids[i:end_idx]
                batch_embeddings = processed_embeddings[i:end_idx]
                batch_store_ids = store_ids[i:end_idx]
                batch_camera_ids = camera_ids[i:end_idx]
                batch_timestamps = timestamps[i:end_idx]
                
                batch_size = len(batch_track_ids)
                logger.info(f"Inserting batch of {batch_size} embeddings ({i+1}-{end_idx} of {total_records})...")
                
                data = [batch_track_ids, batch_embeddings, batch_store_ids, batch_camera_ids, batch_timestamps]
                self.collection.insert(data)
                inserted_count += batch_size
                
            # Final flush to ensure all data is committed
            self.collection.flush()
            logger.info(f"Successfully inserted {inserted_count} embeddings in batches")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error in batch insert (completed {inserted_count}/{total_records}): {e}")
            # Try to flush what we have
            try:
                self.collection.flush()
            except:
                pass
            return inserted_count
    
    def search_embedding(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
        store_filter: Optional[int] = None,
        camera_filter: Optional[int] = None,
        min_similarity: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_partition: bool = True
        ) -> List[Tuple[int, float, Dict]]:
        """
        Search for similar embeddings with advanced filtering
        
        Args:
            query_embedding: Vector to search for
            top_k: Maximum number of results to return
            store_filter: Optional filter by store_id
            camera_filter: Optional filter by camera_id
            min_similarity: Minimum similarity threshold (0-1, higher is more similar)
            max_retries: Maximum number of retry attempts
            use_partition: Whether to use store-specific partitions
            
        Returns:
            List of tuples: (track_id, similarity_score, metadata_dict)
        """
        # Prepare query embedding
        if isinstance(query_embedding, list):
            query_embedding = np.array([query_embedding])
        elif isinstance(query_embedding, np.ndarray):
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
        
        # Setup search parameters for optimal performance
        search_params = {
            "metric_type": "COSINE", 
            "params": {
                "ef": 64  # Higher for better recall at the cost of speed
            }
        }

        partition_names = None
        if use_partition and store_filter is not None:
            partition_name = f"stores_{store_filter}_to_{store_filter}"
            partition_names = [partition_name]
            logger.info(f"Using partition: {partition_name}")
        
        # Build expression for filtering
        expr_parts = []
        if store_filter is not None:
            expr_parts.append(f"store_id == {store_filter}")
        if camera_filter is not None:
            expr_parts.append(f"camera_id == {camera_filter}")
            
        expr = " && ".join(expr_parts) if expr_parts else ""
        
        # Execute search with retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"Searching for top {top_k} similar embeddings{' with filter: ' + expr if expr else ''}"
                        f"{' in partition: ' + str(partition_names[0]) if partition_names else ''}...")
                
                results = self.collection.search(
                    data=query_embedding,
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    expr=expr,
                    output_fields=["track_id", "store_id", "camera_id", "timestamp"],
                    partition_names=partition_names
                )
                
                # Process results and convert cosine distance to similarity
                matches = []
                for hit in results[0]:
                    # Convert distance to similarity (cosine distance of 0 = similarity of 1)
                    similarity = 1.0 - hit.distance
                    
                    # Only include results above minimum similarity threshold
                    if similarity >= min_similarity:
                        # Extract metadata
                        metadata = {
                            "store_id": hit.entity.get("store_id"),
                            "camera_id": hit.entity.get("camera_id"),
                            "timestamp": hit.entity.get("timestamp")
                        }
                        
                        matches.append((hit.entity.get("track_id"), similarity, metadata))
                
                return matches
                
            except Exception as e:
                logger.error(f"Search attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    backoff = (2 ** attempt) + np.random.uniform(0, 1)
                    logger.info(f"Retrying in {backoff:.2f} seconds...")
                    time.sleep(backoff)
                else:
                    logger.error("All search attempts failed")
                    raise
    
    async def search_embedding_async(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
        store_filter: Optional[int] = None,
        camera_filter: Optional[int] = None,
        min_similarity: float = 0.0,
        use_partition: bool = True
        ) -> List[Tuple[int, float, Dict]]:
        """
        Asynchronous version of search_embedding
        """
        loop = asyncio.get_running_loop()
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: self.search_embedding(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    store_filter=store_filter,
                    camera_filter=camera_filter,
                    min_similarity=min_similarity,
                    use_partition = use_partition
                )
            )
        
        return result
    
    def calculate_cosine_similarity(self, feature1, feature2):
        """Calculate cosine similarity between two feature vectors (optimized version)"""
        # Use numpy's optimized operations
        return 1.0 - np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2) + 1e-8)
    

    def match_features_batch(self, query_features: List[np.ndarray], candidates: dict, threshold: float = 0.4):
        """
        Efficient batch matching of multiple query features against candidate tracks
        Returns: List of (track_id, best_score) tuples that match above threshold
        """
        matches = []
        
        # Pre-normalize all candidate features for efficiency
        normalized_candidates = {}
        for track_id, features in candidates.items():
            normalized_candidates[track_id] = [
                feat / (np.linalg.norm(feat) + 1e-8) for feat in features
            ]
        
        # Normalize query features
        normalized_queries = [
            query / (np.linalg.norm(query) + 1e-8) for query in query_features
        ]
        
        # Batch matching
        for i, query in enumerate(normalized_queries):
            best_match_id = None
            best_match_score = 0.0
            
            for track_id, features in normalized_candidates.items():
                # Compute dot products (cosine similarity since vectors are normalized)
                similarities = [np.dot(query, feat) for feat in features]
                if similarities:
                    max_similarity = max(similarities)
                    if max_similarity > best_match_score:
                        best_match_score = max_similarity
                        best_match_id = track_id
            
            if best_match_id and best_match_score >= threshold:
                matches.append((best_match_id, best_match_score))
        
        return matches

    def get_features_by_track_id(self, track_id: int, store_id: int = None) -> List[np.ndarray]:
        """
        Retrieve all feature embeddings for a specific track_id
        
        Args:
            track_id: The unique identifier for the tracked person
            store_id: Optional store ID filter (defaults to the client's store_id if None)
            
        Returns:
            List of feature embeddings (numpy arrays) for this track_id
        """
        # Use provided store_id or fall back to client's store_id
        store_filter = store_id if store_id is not None else self.store_id
        
        # Build query expression
        expr = f"track_id == {track_id}"
        if store_filter is not None:
            expr += f" && store_id == {store_filter}"
            
        logger.info(f"Fetching features for track_id={track_id} with filter: {expr}")
        
        try:
            # Execute query
            results = self.collection.query(
                expr=expr,
                output_fields=["embedding", "timestamp", "camera_id"],
                limit=1000  # Adjust based on expected maximum features per person
            )
            
            if not results:
                logger.warning(f"No features found for track_id={track_id}")
                return []
                
            # Extract and convert embeddings to numpy arrays
            features = []
            for row in results:
                embedding = np.array(row["embedding"], dtype=np.float32)
                features.append(embedding)
                
            logger.info(f"Retrieved {len(features)} features for track_id={track_id}")
            return features
            
        except Exception as e:
            logger.error(f"Error retrieving features for track_id={track_id}: {e}")
            return []

    def get_all_track_features(self, store_id: int, limit: int = 100000) -> Dict[int, List[np.ndarray]]:
        """
        Returns a dictionary where keys are track_ids and values are lists of embeddings
        
        Args:
            store_id: Filter results by store_id
            limit: Maximum number of embeddings to retrieve
            
        Returns:
            Dict mapping track_ids to lists of feature embeddings
        """
        expr = f"store_id == {store_id}"
        logger.info(f"Querying all embeddings for store_id={store_id} (limit={limit})...")

        try:
            # Execute query with pagination if needed
            if limit <= 10000:
                results = self.collection.query(
                    expr=expr,
                    output_fields=["track_id", "embedding"],
                    limit=limit
                )
            else:
                # For large result sets, use pagination
                results = []
                offset = 0
                page_size = 10000
                
                while offset < limit:
                    page = self.collection.query(
                        expr=expr,
                        output_fields=["track_id", "embedding"],
                        limit=page_size,
                        offset=offset
                    )
                    
                    if not page:
                        break  # No more results
                        
                    results.extend(page)
                    offset += page_size
                    
                    if len(results) >= limit:
                        results = results[:limit]
                        break
                        
            # Process results
            feature_map = defaultdict(list)
            for row in results:
                track_id = row["track_id"]
                embedding = np.array(row["embedding"], dtype=np.float32)
                feature_map[track_id].append(embedding)

            logger.info(f"Retrieved features for {len(feature_map)} unique track_ids")
            return dict(feature_map)
            
        except Exception as e:
            logger.error(f"Failed to query embeddings: {e}")
            return {}
    
    def get_all_track_ids(self, store_id: int) -> List[int]:
        """
        Fetch all unique track_ids in the collection for a given store_id
        
        Args:
            store_id: Store ID to filter by
            
        Returns:
            List of unique track IDs
        """
        try:
            expr = f"store_id == {store_id}"
            output_fields = ["track_id"]
            
            # Use a more efficient query that only retrieves distinct track_ids
            results = self.collection.query(
                expr=expr,
                output_fields=output_fields,
                limit=100000  # Adjust if needed
            )

            # Extract unique track_ids
            track_ids = list({r["track_id"] for r in results})
            logger.info(f"Found {len(track_ids)} unique track_ids for store_id={store_id}")
            return track_ids
            
        except Exception as e:
            logger.error(f"Error fetching track_ids for store_id {store_id}: {e}")
            return []
    
    def delete_track(self, track_id: int, store_id: int) -> bool:
        """
        Delete all embeddings associated with a given track_id and store_id
        
        Args:
            track_id: Track ID to delete
            store_id: Store ID for validation
            
        Returns:
            bool: True if successful, False otherwise
        """
        expr = f"track_id == {track_id} and store_id == {store_id}"
        logger.info(f"Deleting records for track_id={track_id}, store_id={store_id}...")
        
        try:
            # Check if records exist before attempting deletion
            count_query = self.collection.query(
                expr=expr,
                output_fields=["count(*)"],
                limit=1
            )
            
            if not count_query or count_query[0].get("count(*)", 0) == 0:
                logger.warning(f"No records found for track_id={track_id}, store_id={store_id}")
                return False
                
            result = self.collection.delete(expr)
            self.collection.flush()
            
            logger.info(f"Successfully deleted records for track_id={track_id}, store_id={store_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete track_id={track_id}, store_id={store_id}: {e}")
            return