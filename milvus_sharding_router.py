\# """
# Corrected Milvus Router - Compatible with optimized collection schema

# Fixed issues:
# - Correct composite key calculation
# - Proper field order matching schema
# - Correct data types (INT32 for store_id, camera_id)
# - Better error handling
# """

# import os
# import json
# import logging
# import asyncio
# from typing import Dict, List, Optional
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from pymilvus import Collection, connections, utility
# from collections import defaultdict
# import uuid

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# )
# logger = logging.getLogger("corrected-milvus-router")

# # Create FastAPI app
# app = FastAPI(title="Corrected Milvus Router", version="2.0.1")

# # Keep the exact same API models as your original router
# class EmbeddingRequest(BaseModel):
#     track_id: int
#     embedding: List[float]
#     store_id: int
#     camera_id: int
#     timestamp: int

# class BatchEmbeddingRequest(BaseModel):
#     track_ids: List[int]
#     embeddings: List[List[float]]
#     store_ids: List[int]
#     camera_ids: List[int]
#     timestamps: List[int]

# class SearchRequest(BaseModel):
#     embedding: List[float]
#     store_id: int
#     top_k: int = 5
#     camera_filter: Optional[int] = None
#     min_similarity: float = 0.0

# class BatchSearchRequest(BaseModel):
#     embeddings: List[List[float]]
#     store_id: int
#     top_k: int = 5
#     min_similarity: float = 0.0

# class DeleteRequest(BaseModel):
#     track_id: int
#     store_id: int

# class ShardConfig(BaseModel):
#     id: str
#     host: str
#     port: int
#     replica_of: Optional[str] = None
#     status: str = "active"

# class TopologyResponse(BaseModel):
#     shards: Dict[str, ShardConfig]
#     store_mappings: Dict[str, str]

# class CorrectedMilvusRouter:
#     """Corrected router for optimized collection schema"""
    
#     def __init__(self, collection_name="person_reid_features", embedding_dim=768):
#         self.collection_name = collection_name
#         self.embedding_dim = embedding_dim
#         self.shard_configs = {}
#         self.store_to_shard = {}
        
#         # Load simple configuration
#         self.milvus_host = os.environ.get("MILVUS_HOST", "74.235.98.141")
#         self.milvus_port = int(os.environ.get("MILVUS_PORT", "19530"))
        
#         logger.debug(f"Corrected router configured for {self.milvus_host}:{self.milvus_port}")
#         logger.debug(f"Collection: {self.collection_name}")
    
#     def _get_connection(self):
#         """Get a simple connection - creates new one each time"""
#         alias = f"conn_{uuid.uuid4().hex[:8]}"
#         connections.connect(
#             alias,
#             host=self.milvus_host,
#             port=self.milvus_port,
#             timeout=30
#         )
#         return alias
    
#     def _cleanup_connection(self, alias):
#         """Clean up connection"""
#         try:
#             connections.disconnect(alias)
#         except:
#             pass
    
#     async def _get_connection_for_store(self, store_id: int):
#         """Mimic original API - just return connection and dummy shard_id"""
#         alias = self._get_connection()
#         return alias, "shard1"
    
#     def _create_store_camera_composite(self, store_id: int, camera_id: int) -> int:
#         """Create composite key - FIXED to match the test script formula"""
#         # Use the same formula as in the test scripts: store_id * 10000 + camera_id
#         return store_id * 10000 + camera_id
    
#     def _validate_embedding(self, embedding: List[float]) -> List[float]:
#         """Validate and normalize embedding"""
#         if len(embedding) != self.embedding_dim:
#             raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
        
#         # Ensure it's a list of floats
#         return [float(x) for x in embedding]
    
#     def _prepare_insert_data(self, embeddings, track_ids, store_ids, camera_ids, timestamps):
#         """Prepare data in the EXACT order required by the schema"""
        
#         # Validate all embeddings
#         validated_embeddings = []
#         for emb in embeddings:
#             if isinstance(emb, np.ndarray):
#                 validated_embeddings.append(emb.astype(np.float32).tolist())
#             else:
#                 validated_embeddings.append(self._validate_embedding(emb))
        
#         # Ensure proper data types for schema compatibility
#         validated_track_ids = [int(x) for x in track_ids]
#         validated_store_ids = [int(x) for x in store_ids]  # INT32 in schema
#         validated_camera_ids = [int(x) for x in camera_ids]  # INT32 in schema
#         validated_timestamps = [int(x) for x in timestamps]  # INT64 in schema
        
#         # Create composite keys
#         store_camera_composites = [
#             self._create_store_camera_composite(store_id, camera_id)
#             for store_id, camera_id in zip(validated_store_ids, validated_camera_ids)
#         ]
        
#         # CRITICAL: Data must be in EXACT schema field order
#         # Schema order: id (auto), feature_vector, track_id, store_id, camera_id, timestamp, store_camera_composite
#         data = [
#             validated_embeddings,        # feature_vector (FLOAT_VECTOR, dim=768)
#             validated_track_ids,         # track_id (INT64)
#             validated_store_ids,         # store_id (INT32)
#             validated_camera_ids,        # camera_id (INT32)
#             validated_timestamps,        # timestamp (INT64)
#             store_camera_composites      # store_camera_composite (INT64)
#         ]
        
#         return data
    
#     async def insert_embedding(self, request: EmbeddingRequest):
#         """Insert a single embedding - CORRECTED"""
#         alias = None
#         try:
#             alias, shard_id = await self._get_connection_for_store(request.store_id)
#             collection = Collection(self.collection_name, using=alias)
            
#             # Prepare data using corrected helper
#             data = self._prepare_insert_data(
#                 embeddings=[request.embedding],
#                 track_ids=[request.track_id],
#                 store_ids=[request.store_id],
#                 camera_ids=[request.camera_id],
#                 timestamps=[request.timestamp]
#             )
            
#             logger.debug(f"Inserting single record: track_id={request.track_id}, store_id={request.store_id}")
            
#             # Insert without flush (as requested)
#             result = collection.insert(data)
            
#             return {"success": True, "shard_id": shard_id}
            
#         except Exception as e:
#             logger.error(f"Insert error: {e}")
#             logger.error(f"Request data: track_id={request.track_id}, store_id={request.store_id}, embedding_dim={len(request.embedding)}")
#             raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")
#         finally:
#             if alias:
#                 self._cleanup_connection(alias)
    
#     async def insert_embeddings_batch(self, request: BatchEmbeddingRequest):
#         """Insert multiple embeddings - CORRECTED"""
#         alias = None
#         try:
#             # Validate request data first
#             if not all(len(lst) == len(request.track_ids) for lst in [
#                 request.embeddings, request.store_ids, request.camera_ids, request.timestamps
#             ]):
#                 raise ValueError("All input lists must have the same length")
            
#             # Group by store_id like original
#             store_groups = defaultdict(list)
#             for i in range(len(request.track_ids)):
#                 store_id = request.store_ids[i]
#                 store_groups[store_id].append(i)
            
#             results = {}
#             total_inserted = 0
            
#             for store_id, indices in store_groups.items():
#                 alias, shard_id = await self._get_connection_for_store(store_id)
                
#                 try:
#                     collection = Collection(self.collection_name, using=alias)
                    
#                     # Extract data for this store group
#                     batch_embeddings = [request.embeddings[i] for i in indices]
#                     batch_track_ids = [request.track_ids[i] for i in indices]
#                     batch_store_ids = [request.store_ids[i] for i in indices]
#                     batch_camera_ids = [request.camera_ids[i] for i in indices]
#                     batch_timestamps = [request.timestamps[i] for i in indices]
                    
#                     # Prepare data using corrected helper
#                     data = self._prepare_insert_data(
#                         embeddings=batch_embeddings,
#                         track_ids=batch_track_ids,
#                         store_ids=batch_store_ids,
#                         camera_ids=batch_camera_ids,
#                         timestamps=batch_timestamps
#                     )
                    
#                     logger.debug(f"Inserting batch for store {store_id}: {len(indices)} records")
#                     logger.debug(f"First embedding dimension: {len(batch_embeddings[0]) if batch_embeddings else 0}")
                    
#                     # Insert without flush (as requested)
#                     collection.insert(data)
                    
#                     results[store_id] = {
#                         "shard_id": shard_id,
#                         "insert_count": len(indices)
#                     }
#                     total_inserted += len(indices)
                    
#                 finally:
#                     if alias:
#                         self._cleanup_connection(alias)
#                         alias = None  # Reset for next iteration
            
#             return {
#                 "success": True,
#                 "results": results,
#                 "total_inserted": total_inserted
#             }
            
#         except Exception as e:
#             logger.error(f"Batch insert error: {e}")
#             logger.error(f"Error type: {type(e).__name__}")
#             # Add more debugging info
#             logger.error(f"Request validation - track_ids: {len(request.track_ids) if hasattr(request, 'track_ids') else 'missing'}")
#             logger.error(f"Request validation - embeddings: {len(request.embeddings) if hasattr(request, 'embeddings') else 'missing'}")
#             logger.error(f"Request validation - store_ids: {len(request.store_ids) if hasattr(request, 'store_ids') else 'missing'}")
#             raise HTTPException(status_code=500, detail=f"Batch insert failed: {str(e)}")
#         finally:
#             if alias:
#                 self._cleanup_connection(alias)
    
#     async def search_embedding(self, request: SearchRequest):
#         """Search for similar embeddings - CORRECTED"""
#         alias = None
#         try:
#             alias, shard_id = await self._get_connection_for_store(request.store_id)
#             collection = Collection(self.collection_name, using=alias)
            
#             # Validate embedding
#             validated_embedding = self._validate_embedding(request.embedding)
            
#             # Optimized search parameters 
#             search_params = {
#                 "metric_type": "COSINE",
#                 "params": {"ef": min(request.top_k * 4, 64)}  # Conservative ef
#             }
            
#             # Build filter expression for new schema
#             expr_parts = [f"store_id == {request.store_id}"]
#             if request.camera_filter is not None:
#                 expr_parts.append(f"camera_id == {request.camera_filter}")
#             expr = " and ".join(expr_parts)
            
#             query_embedding = np.array([validated_embedding])
            
#             # Search using correct field name
#             results = collection.search(
#                 data=query_embedding,
#                 anns_field="feature_vector",  # Correct field name
#                 param=search_params,
#                 limit=request.top_k,
#                 expr=expr,
#                 output_fields=["track_id", "store_id", "camera_id", "timestamp"],
#                 consistency_level="Session"
#             )
            
#             # Process results 
#             matches = []
#             for hit in results[0]:
#                 similarity = 1.0 - hit.distance
#                 if similarity >= request.min_similarity:
#                     matches.append({
#                         "track_id": hit.entity.get("track_id"),
#                         "similarity": similarity,
#                         "metadata": {
#                             "store_id": hit.entity.get("store_id"),
#                             "camera_id": hit.entity.get("camera_id"),
#                             "timestamp": hit.entity.get("timestamp")
#                         }
#                     })
            
#             return {
#                 "success": True,
#                 "shard_id": shard_id,
#                 "matches": matches,
#                 "total_matches": len(matches)
#             }
            
#         except Exception as e:
#             logger.error(f"Search error: {e}")
#             raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
#         finally:
#             if alias:
#                 self._cleanup_connection(alias)
    
#     async def search_embeddings_batch(self, request: BatchSearchRequest):
#         """Batch search - CORRECTED"""
#         alias = None
#         try:
#             alias, shard_id = await self._get_connection_for_store(request.store_id)
#             collection = Collection(self.collection_name, using=alias)
            
#             # Validate all embeddings
#             validated_embeddings = [self._validate_embedding(emb) for emb in request.embeddings]
            
#             # Optimized search parameters
#             search_params = {
#                 "metric_type": "COSINE", 
#                 "params": {"ef": min(request.top_k * 4, 64)}
#             }
            
#             expr = f"store_id == {request.store_id}"
#             query_embeddings = np.array(validated_embeddings)
            
#             # Search using correct field name
#             results = collection.search(
#                 data=query_embeddings,
#                 anns_field="feature_vector",  # Correct field name
#                 param=search_params,
#                 limit=request.top_k,
#                 expr=expr,
#                 output_fields=["track_id", "store_id", "camera_id", "timestamp"],
#                 consistency_level="Session"
#             )
            
#             # Process results for tracker compatibility
#             batch_results = []
#             for query_results in results:
#                 query_matches = []
#                 for hit in query_results:
#                     similarity = 1.0 - hit.distance
#                     if similarity >= request.min_similarity:
#                         # Return (track_id, distance) format for tracker
#                         query_matches.append((hit.entity.get("track_id"), hit.distance))
#                 batch_results.append(query_matches)
            
#             return {
#                 "success": True,
#                 "shard_id": shard_id,
#                 "results": batch_results
#             }
            
#         except Exception as e:
#             logger.error(f"Batch search error: {e}")
#             raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")
#         finally:
#             if alias:
#                 self._cleanup_connection(alias)
    
#     async def delete_track(self, request: DeleteRequest):
#         """Delete a track - CORRECTED"""
#         alias = None
#         try:
#             alias, shard_id = await self._get_connection_for_store(request.store_id)
#             collection = Collection(self.collection_name, using=alias)
            
#             expr = f"track_id == {request.track_id} and store_id == {request.store_id}"
#             collection.delete(expr)
#             # No flush as requested
            
#             return {
#                 "success": True,
#                 "shard_id": shard_id,
#                 "deleted": True
#             }
            
#         except Exception as e:
#             logger.error(f"Delete error: {e}")
#             raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
#         finally:
#             if alias:
#                 self._cleanup_connection(alias)
    
#     async def get_features_by_track_id(self, track_id: int, store_id: int):
#         """Get features for a track - CORRECTED"""
#         alias = None
#         try:
#             alias, shard_id = await self._get_connection_for_store(store_id)
#             collection = Collection(self.collection_name, using=alias)
            
#             expr = f"track_id == {track_id} && store_id == {store_id}"
            
#             # Query using correct field names
#             results = collection.query(
#                 expr=expr,
#                 output_fields=["feature_vector", "timestamp", "camera_id"],
#                 limit=1000,
#                 consistency_level="Session"
#             )
            
#             # Convert to original API format
#             features = []
#             for row in results:
#                 features.append({
#                     "embedding": row["feature_vector"],  # Map to original API name
#                     "timestamp": row.get("timestamp"),
#                     "camera_id": row.get("camera_id")
#                 })
            
#             return {
#                 "success": True,
#                 "shard_id": shard_id,
#                 "track_id": track_id,
#                 "store_id": store_id,
#                 "features": features,
#                 "feature_count": len(features)
#             }
            
#         except Exception as e:
#             logger.error(f"Get features error: {e}")
#             raise HTTPException(status_code=500, detail=f"Failed to get features: {str(e)}")
#         finally:
#             if alias:
#                 self._cleanup_connection(alias)
    
#     async def get_all_track_features(self, store_id: int, limit: int = 100000):
#         """Get all track features - CORRECTED"""
#         alias = None
#         try:
#             alias, shard_id = await self._get_connection_for_store(store_id)
#             collection = Collection(self.collection_name, using=alias)
            
#             expr = f"store_id == {store_id}"
            
#             # Query using correct field names
#             results = collection.query(
#                 expr=expr,
#                 output_fields=["track_id", "feature_vector"],
#                 limit=min(limit, 16384),  # Milvus limit
#                 consistency_level="Session"
#             )
            
#             # Group by track_id
#             feature_map = defaultdict(list)
#             for row in results:
#                 track_id = row["track_id"]
#                 embedding = np.array(row["feature_vector"], dtype=np.float32)
#                 feature_map[track_id].append(embedding)
            
#             return {
#                 "success": True,
#                 "shard_id": shard_id,
#                 "track_count": len(feature_map),
#                 "features": dict(feature_map)
#             }
            
#         except Exception as e:
#             logger.error(f"Get all features error: {e}")
#             raise HTTPException(status_code=500, detail=f"Failed to get track features: {str(e)}")
#         finally:
#             if alias:
#                 self._cleanup_connection(alias)
    
#     async def get_topology(self):
#         """Get topology - same API as original"""
#         return TopologyResponse(
#             shards={
#                 "shard1": ShardConfig(
#                     id="shard1",
#                     host=self.milvus_host,
#                     port=self.milvus_port,
#                     status="active"
#                 )
#             },
#             store_mappings=self.store_to_shard
#         )

# # Initialize corrected router
# router = CorrectedMilvusRouter()

# # Same startup and endpoints as original...
# @app.on_event("startup")
# async def startup():
#     """Initialize router on application startup"""
#     config_path = os.path.join(os.path.dirname(__file__), "config.json")
#     if os.path.exists(config_path):
#         try:
#             with open(config_path, "r") as f:
#                 config = json.load(f)
#                 logger.debug(f"Loaded configuration from {config_path}")
            
#             # Load shard configs
#             if "shards" in config:
#                 for shard_id, shard_data in config["shards"].items():
#                     shard_config = ShardConfig(**shard_data)
#                     router.shard_configs[shard_id] = shard_config
#                     router.milvus_host = shard_config.host
#                     router.milvus_port = shard_config.port
#                     logger.debug(f"Using shard {shard_id}: {shard_config.host}:{shard_config.port}")
#                     break
                
#             # Load store mappings
#             if "store_mappings" in config:
#                 for store_id, shard_id in config["store_mappings"].items():
#                     router.store_to_shard[store_id] = shard_id
#                 logger.debug(f"Mapped {len(config['store_mappings'])} stores to shards")
                
#             logger.debug(f"Corrected router initialized successfully")
#         except Exception as e:
#             logger.error(f"Error loading configuration: {e}")
#     else:
#         logger.debug(f"No config file found, using defaults")

# # All the same API endpoints...
# @app.get("/health")
# async def health_check():
#     return JSONResponse(status_code=200, content={"status": "ok"})

# @app.post("/insert")
# async def insert_embedding(request: EmbeddingRequest):
#     return await router.insert_embedding(request)

# @app.post("/batch_insert")
# async def insert_embeddings_batch(request: BatchEmbeddingRequest):
#     return await router.insert_embeddings_batch(request)

# @app.post("/search")
# async def search_embedding(request: SearchRequest):
#     return await router.search_embedding(request)

# @app.post("/batch_search")
# async def search_embeddings_batch(request: BatchSearchRequest):
#     return await router.search_embeddings_batch(request)

# @app.post("/delete")
# async def delete_track(request: DeleteRequest):
#     return await router.delete_track(request)

# @app.get("/topology")
# async def get_topology():
#     return await router.get_topology()

# @app.get("/features/{track_id}/{store_id}")
# async def get_features(track_id: int, store_id: int):
#     return await router.get_features_by_track_id(track_id, store_id)

# @app.get("/track/features/{track_id}/{store_id}")
# async def get_track_features_for_tracker(track_id: int, store_id: int):
#     """Get features for tracker"""
#     result = await router.get_features_by_track_id(track_id, store_id)
#     features = [f["embedding"] for f in result["features"]]
#     return {"track_id": track_id, "store_id": store_id, "features": features}

# @app.post("/track/search")
# async def search_for_tracker(request: SearchRequest):
#     """Search for tracker"""
#     result = await router.search_embedding(request)
#     tracker_results = [
#         (match["track_id"], 1.0 - match["similarity"])
#         for match in result["matches"]
#     ]
#     return {"results": tracker_results}

# @app.get("/track/allfeatures/{store_id}")
# async def get_all_track_features_for_tracker(store_id: int, limit: int = 100000):
#     """Get all features for tracker"""
#     result = await router.get_all_track_features(store_id, limit)
    
#     tracker_features = {}
#     for track_id, embeddings in result["features"].items():
#         track_id_int = int(track_id)
#         tracker_features[track_id_int] = [e.tolist() for e in embeddings]
    
#     return tracker_features

# @app.get("/test_connection/{store_id}")
# async def test_connection(store_id: int):
#     """Test connection"""
#     try:
#         alias, shard_id = await router._get_connection_for_store(store_id)
        
#         try:
#             collection_list = utility.list_collections(using=alias)
            
#             return {
#                 "success": True,
#                 "store_id": store_id,
#                 "shard_id": shard_id,
#                 "host": router.milvus_host,
#                 "port": router.milvus_port,
#                 "collections": collection_list
#             }
#         finally:
#             router._cleanup_connection(alias)
            
#     except Exception as e:
#         logger.error(f"Connection test failed: {e}")
#         return {
#             "success": False,
#             "store_id": store_id,
#             "error": str(e)
#         }

# @app.get("/performance/breakdown")
# async def get_performance_breakdown():
#     """Performance endpoint for compatibility"""
#     return {
#         "bottleneck_analysis": "healthy",
#         "timing_breakdown": {
#             "avg_total": 0.05,
#             "avg_connection_wait": 0.01,
#             "avg_database": 0.03,
#             "avg_processing": 0.01,
#             "slow_requests": 0
#         },
#         "connection_pool_health": {
#             "avg_wait_time": 0.01,
#             "max_wait_time": 0.02,
#             "waiting_requests": {},
#             "total_connections": 1
#         },
#         "diagnosis": {
#             "router_overloaded": False,
#             "database_slow": False,
#             "many_slow_requests": False
#         }
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Milvus Sharding Router - Routes requests to appropriate Milvus shards based on store_id.

This router sits between client applications and Milvus shards, providing:
1. Central connection management and connection pooling to all shards
2. Store-to-shard mapping based on a configuration service
3. Request routing to appropriate shards
4. Health monitoring and failover
5. A unified API for clients regardless of sharding topology
"""

import os
import time
import asyncio
import logging
import json
import random
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import redis
from pymilvus import Collection, connections, utility
import uvicorn
from collections import defaultdict, deque
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("milvus_router.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("milvus-router")

# Create FastAPI app
app = FastAPI(title="Milvus Sharding Router", version="1.0.0")

# Models for API requests
class EmbeddingRequest(BaseModel):
    track_id: int
    feature_vector: List[float]
    store_id: int
    camera_id: int
    timestamp: int

class BatchEmbeddingRequest(BaseModel):
    track_ids: List[int]
    feature_vectors: List[List[float]]
    store_ids: List[int]
    camera_ids: List[int]
    timestamps: List[int]

class SearchRequest(BaseModel):
    feature_vector: List[float]
    store_id: int
    top_k: int = 5
    camera_filter: Optional[int] = None
    min_similarity: float = 0.0

class BatchSearchRequest(BaseModel):
    feature_vectors: List[List[float]]
    store_id: int
    top_k: int = 5
    min_similarity: float = 0.0

class DeleteRequest(BaseModel):
    track_id: int
    store_id: int

class ShardConfig(BaseModel):
    id: str
    host: str
    port: int
    replica_of: Optional[str] = None
    status: str = "active"

class StoreShardMapping(BaseModel):
    store_id: int
    shard_id: str

class TopologyResponse(BaseModel):
    shards: Dict[str, ShardConfig]
    store_mappings: Dict[str, str]

class ConnectionPoolConfigRequest(BaseModel):
    max_connections_per_shard: int = Field(ge=1, le=512)  # Min 1, Max 512
    connection_timeout: Optional[int] = Field(default=None, ge=1, le=60)

class SimpleTimingMonitor:
   def __init__(self):
       self.request_timings = deque(maxlen=500)  # Store last 500 requests
       self.connection_pool_stats = deque(maxlen=100)
       
   def record_request_timing(self, request_id: str, timings: dict, metadata: dict):
       """Record complete request timing breakdown"""
       self.request_timings.append({
           'request_id': request_id,
           'timestamp': time.time(),
           'total_time': timings.get('total', 0),
           'connection_wait': timings.get('connection_wait', 0),
           'database_time': timings.get('database', 0),
           'processing_time': timings.get('processing', 0),
           'operation': metadata.get('operation', 'unknown'),
           'store_id': metadata.get('store_id'),
           'shard_id': metadata.get('shard_id')
       })
       
       # Log slow requests immediately
       total_time = timings.get('total', 0)
       if total_time > 3.0:
           logger.debug(f"SLOW REQUEST {request_id}: {total_time:.2f}s - "
                 f"conn:{timings.get('connection_wait', 0):.2f}s, "
                 f"db:{timings.get('database', 0):.2f}s, "
                 f"proc:{timings.get('processing', 0):.2f}s")
   
   def get_recent_stats(self, last_n=50):
       """Get stats for recent requests"""
       recent = list(self.request_timings)[-last_n:]
       if not recent:
           return {}
           
       return {
           'avg_total': sum(r['total_time'] for r in recent) / len(recent),
           'avg_connection_wait': sum(r['connection_wait'] for r in recent) / len(recent),
           'avg_database': sum(r['database_time'] for r in recent) / len(recent),
           'avg_processing': sum(r['processing_time'] for r in recent) / len(recent),
           'slow_requests': len([r for r in recent if r['total_time'] > 3.0])
       }

class ConnectionPool:
    """Manages and pools connections to Milvus shards"""
    
    def __init__(self, max_connections_per_shard=16, connection_timeout=10):
        self.connections = {}  # {shard_id: {connection_alias: last_used_time}}
        self.shard_info = {}  # {shard_id: ShardConfig}
        self.max_connections_per_shard = max_connections_per_shard
        self.connection_timeout = connection_timeout
        self.lock = asyncio.Lock()
        self.active_connections = {}
        self.connection_health = {}
        
        self.wait_times = deque(maxlen=50)  # Track connection wait times
        self.waiting_requests = defaultdict(int)  # Current waiting requests per shard
        
    # async def get_connection_alias(self, shard_id: str) -> str:
    #     """Get a connection alias for the specified shard"""
    #     async with self.lock:
    #         if shard_id not in self.shard_info:
    #             raise ValueError(f"Unknown shard ID: {shard_id}")
                
    #         # Check if we have any existing connections for this shard
    #         if shard_id not in self.connections:
    #             self.connections[shard_id] = {}
                
    #         # If we have fewer than max connections, create a new one
    #         if len(self.connections[shard_id]) < self.max_connections_per_shard:
    #             connection_alias = f"{shard_id}_{len(self.connections[shard_id])}"
    #             # Create connection if it doesn't exist
    #             if connection_alias not in self.connections[shard_id]:
    #                 config = self.shard_info[shard_id]
    #                 try:
    #                     connections.connect(
    #                         connection_alias,
    #                         host=config.host,
    #                         port=config.port,
    #                         timeout=self.connection_timeout
    #                     )
    #                     self.connections[shard_id][connection_alias] = time.time()
    #                     logger.debug(f"Created new connection to shard {shard_id}: {connection_alias}")
    #                 except Exception as e:
    #                     logger.error(f"Failed to connect to shard {shard_id}: {e}")
    #                     raise
    #             return connection_alias
                
    #         # Reuse the least recently used connection
    #         connection_alias = min(
    #             self.connections[shard_id].keys(),
    #             key=lambda k: self.connections[shard_id][k]
    #         )
    #         self.connections[shard_id][connection_alias] = time.time()
    #         logger.debug(f"Reusing existing connection for shard {shard_id}: {connection_alias}")
    #         return connection_alias


    async def get_connection_alias(self, shard_id: str) -> str:
        wait_start = time.time()
        self.waiting_requests[shard_id] += 1

        try:
            async with self.lock:
                connection_alias = await self._get_connection_impl(shard_id)
                wait_time = time.time() - wait_start
                self.wait_times.append(wait_time)

                # Log long waits
                if wait_time > 1.0:
                    logger.debug(f"LONG CONNECTION WAIT: {wait_time:.2f}s for shard {shard_id}, "
                          f"waiting: {self.waiting_requests[shard_id]}, "
                          f"active: {len(self.connections.get(shard_id, {}))}")

                return connection_alias
        finally:
            self.waiting_requests[shard_id] -= 1

    async def _get_connection_impl(self, shard_id: str) -> str:
        """Your existing get_connection_alias logic here"""
        if shard_id not in self.shard_info:
            raise ValueError(f"Unknown shard ID: {shard_id}")
            
        # Check if we have any existing connections for this shard
        if shard_id not in self.connections:
            self.connections[shard_id] = {}
            
        # If we have fewer than max connections, create a new one
        if len(self.connections[shard_id]) < self.max_connections_per_shard:
            connection_alias = f"{shard_id}_{len(self.connections[shard_id])}"
            # Create connection if it doesn't exist
            if connection_alias not in self.connections[shard_id]:
                config = self.shard_info[shard_id]
                try:
                    connections.connect(
                        connection_alias,
                        host=config.host,
                        port=config.port,
                        timeout=self.connection_timeout
                    )
                    self.connections[shard_id][connection_alias] = time.time()
                    logger.debug(f"Created new connection to shard {shard_id}: {connection_alias}")
                except Exception as e:
                    logger.error(f"Failed to connect to shard {shard_id}: {e}")
                    raise
            return connection_alias
        # Reuse the least recently used connection
        connection_alias = min(
            self.connections[shard_id].keys(),
            key=lambda k: self.connections[shard_id][k]
        )
        self.connections[shard_id][connection_alias] = time.time()
        logger.debug(f"Reusing existing connection for shard {shard_id}: {connection_alias}")
        return connection_alias

    async def update_pool_configuration(self, max_connections: int, timeout: Optional[int] = None):
        """Update connection pool configuration dynamically"""
        async with self.lock:
            old_max = self.max_connections_per_shard
            
            # Update configuration
            self.max_connections_per_shard = max_connections
            if timeout is not None:
                self.connection_timeout = timeout
            
            logger.debug(f"Pool configuration updated: max_connections {old_max} -> {max_connections}")
            
            # If we reduced the limit, close excess connections
            if max_connections < old_max:
                for shard_id in self.connections:
                    connections_to_close = []
                    current_connections = list(self.connections[shard_id].keys())
                    
                    if len(current_connections) > max_connections:
                        # Close the oldest connections (keep the most recently used)
                        sorted_connections = sorted(
                            current_connections,
                            key=lambda k: self.connections[shard_id][k]
                        )
                        connections_to_close = sorted_connections[:len(current_connections) - max_connections]
                    
                    for conn_alias in connections_to_close:
                        try:
                            connections.disconnect(conn_alias)
                            del self.connections[shard_id][conn_alias]
                            logger.debug(f"Closed excess connection {conn_alias} on shard {shard_id}")
                        except Exception as e:
                            logger.error(f"Error closing connection {conn_alias}: {e}")
            
            return {
                "old_max_connections": old_max,
                "new_max_connections": self.max_connections_per_shard,
                "current_connections": {
                    shard_id: len(conns) for shard_id, conns in self.connections.items()
                }
            }

    def get_pool_health(self):
        """Get connection pool health metrics"""
        recent_waits = list(self.wait_times)[-20:]
        return {
            'avg_wait_time': sum(recent_waits) / len(recent_waits) if recent_waits else 0,
            'max_wait_time': max(recent_waits) if recent_waits else 0,
            'waiting_requests': dict(self.waiting_requests),
            'total_connections': sum(len(conns) for conns in self.connections.values())
        }

    async def update_shard_config(self, shard_id: str, config: ShardConfig):
        """Update the configuration for a shard"""
        async with self.lock:
            self.shard_info[shard_id] = config
            
    async def reset_connections(self, shard_id: str = None):
        """Reset connections for a specific shard or all shards"""
        async with self.lock:
            if shard_id:
                if shard_id in self.connections:
                    for connection_alias in self.connections[shard_id]:
                        try:
                            connections.disconnect(connection_alias)
                        except:
                            pass
                    self.connections[shard_id] = {}
            else:
                for shard_id in self.connections:
                    for connection_alias in self.connections[shard_id]:
                        try:
                            connections.disconnect(connection_alias)
                        except:
                            pass
                self.connections = {}
                
    async def cleanup_stale_connections(self, max_idle_time=300):
        """Close connections that haven't been used recently"""
        async with self.lock:
            current_time = time.time()
            for shard_id in list(self.connections.keys()):
                for connection_alias in list(self.connections[shard_id].keys()):
                    if current_time - self.connections[shard_id][connection_alias] > max_idle_time:
                        try:
                            connections.disconnect(connection_alias)
                            del self.connections[shard_id][connection_alias]
                            logger.debug(f"Closed stale connection to shard {shard_id}: {connection_alias}")
                        except Exception as e:
                            logger.error(f"Error closing connection {connection_alias}: {e}")

class ShardingManager:
    """Manages store-to-shard mapping and shard topology"""
    
    def __init__(self, config_service_url=None, redis_url=None, refresh_interval=60):
        """
        Args:
        config_service_url: ShardingManager will make HTTP requests to this URL to fetch the latest store-to-shard mappings and shard configurations
        redis_url: Alternative to the HTTP-based config service, using Redis as a distributed configuration store
        refresh_interval: Controls how frequently the system checks for configuration updates
        """
        self.config_service_url = config_service_url
        self.redis_url = redis_url
        self.refresh_interval = refresh_interval
        self.last_refresh = 0
        self.store_to_shard = {}  # {store_id: shard_id}
        self.shard_configs = {}  # {shard_id: ShardConfig}
        self.replicas = {}  # {primary_shard_id: [replica_shard_ids]}
        self.lock = asyncio.Lock()
        
        # Use either Redis or direct HTTP for config management
        if redis_url:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        else:
            self.redis_client = None
        
        self.http_client = None
        if config_service_url:
            import httpx
            self.http_client = httpx.AsyncClient(
                timeout=10.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                headers={"User-Agent": "MilvusRouter/1.0"}
            )
    
    async def __del__(self):
        """Cleanup resources when object is destroyed"""
        if self.http_client:
            await self.http_client.aclose()
        
    async def refresh_topology(self, force=False):
        """Refresh shard topology from configuration service or Redis"""
        current_time = time.time()
        if not force and current_time - self.last_refresh < self.refresh_interval:
            return
            
        async with self.lock:
            try:
                if self.redis_client:
                    await self._refresh_from_redis()
                elif self.config_service_url:
                    await self._refresh_from_config_service()
                else:
                    logger.warning("No configuration source specified, using default mapping")
                    
                self.last_refresh = current_time
                logger.debug(f"Topology refreshed with {len(self.store_to_shard)} store mappings and {len(self.shard_configs)} shards")
                
                # Update replica mapping
                self.replicas = {}
                for shard_id, config in self.shard_configs.items():
                    if config.replica_of:
                        if config.replica_of not in self.replicas:
                            self.replicas[config.replica_of] = []
                        self.replicas[config.replica_of].append(shard_id)
                
            except Exception as e:
                logger.error(f"Failed to refresh topology: {e}")
                
    async def _refresh_from_redis(self):
        """Refresh topology from Redis"""
        # Get shard configs
        shard_configs_json = self.redis_client.get("milvus:shard_configs")
        if shard_configs_json:
            self.shard_configs = {
                k: ShardConfig(**v) 
                for k, v in json.loads(shard_configs_json).items()
            }
        
        # Get store mappings
        store_mappings_json = self.redis_client.get("milvus:store_mappings")
        if store_mappings_json:
            self.store_to_shard = json.loads(store_mappings_json)
            
    async def _refresh_from_config_service(self):
        """Refresh topology from HTTP config service"""
        if not self.config_service_url:
            logger.warning("No config service URL provided, skipping HTTP refresh")
            return
            
        try:
            # If we don't have an HTTP client yet, create one
            if not self.http_client:
                import httpx
                self.http_client = httpx.AsyncClient(
                    timeout=10.0,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                )
                
            # Fetch the topology from the config service
            response = await self.http_client.get(f"{self.config_service_url}/topology")
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch topology: {response.status_code} {response.text}")
                return
                
            data = response.json()
            
            # Update shard configurations
            if "shards" in data:
                self.shard_configs = {
                    k: ShardConfig(**v) if isinstance(v, dict) else v
                    for k, v in data["shards"].items()
                }
                
            # Update store-to-shard mappings
            if "store_mappings" in data:
                self.store_to_shard = data["store_mappings"]
                
            logger.debug(f"Successfully refreshed topology from config service: "
                       f"{len(self.shard_configs)} shards, {len(self.store_to_shard)} store mappings")
                       
        except Exception as e:
            logger.error(f"Error refreshing from config service: {str(e)}")
                
    async def get_shard_for_store(self, store_id: int) -> str:
        """Get the shard ID for a specific store"""
        # Refresh if needed
        await self.refresh_topology()
        
        store_id_str = str(store_id)
        
        # Check if we have a mapping for this store
        if store_id_str in self.store_to_shard:
            return self.store_to_shard[store_id_str]
            
        # If no explicit mapping, use consistent hashing
        if not self.shard_configs:
            raise ValueError("No shard configuration available")
            
        # Get only primary (non-replica) shards
        primary_shards = [
            shard_id for shard_id, config in self.shard_configs.items()
            if not config.replica_of and config.status == "active"
        ]
        
        if not primary_shards:
            raise ValueError("No active primary shards available")
            
        # Use consistent hashing to assign to a shard
        shard_idx = hash(store_id_str) % len(primary_shards)
        assigned_shard = primary_shards[shard_idx]
        
        # Cache this mapping
        self.store_to_shard[store_id_str] = assigned_shard
        
        logger.debug(f"Assigned store {store_id} to shard {assigned_shard} using consistent hashing")
        return assigned_shard
        
    async def get_shard_config(self, shard_id: str) -> ShardConfig:
        """Get configuration for a specific shard"""
        await self.refresh_topology()
        
        if shard_id not in self.shard_configs:
            raise ValueError(f"Unknown shard ID: {shard_id}")
            
        return self.shard_configs[shard_id]
        
    async def get_topology(self) -> TopologyResponse:
        """Get the complete topology"""
        await self.refresh_topology()
        
        return TopologyResponse(
            shards=self.shard_configs,
            store_mappings=self.store_to_shard
        )
        
    async def get_replica_shards(self, primary_shard_id: str) -> List[str]:
        """Get replica shard IDs for a primary shard"""
        await self.refresh_topology()
        
        return self.replicas.get(primary_shard_id, [])

class MilvusCollectionManager:
    """Manages Milvus collection metadata and operations"""
    
    def __init__(self, collection_name="person_reid_features", embedding_dim=768):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection_loaded = {}  # {shard_id: bool}
        
    async def ensure_collection_loaded(self, connection_alias: str, shard_id: str):
        """Ensure the collection is loaded for a specific shard connection"""
        if shard_id in self.collection_loaded and self.collection_loaded[shard_id]:
            return
            
        try:
            collection = Collection(self.collection_name, using=connection_alias)
            # if not collection.is_loaded:
            collection.load()
            self.collection_loaded[shard_id] = True
            logger.debug(f"Loaded collection {self.collection_name} on shard {shard_id}")
        except Exception as e:
            logger.error(f"Failed to load collection on shard {shard_id}: {e}")
            raise
            
    async def execute_operation(self, connection_alias: str, shard_id: str, operation: str, **kwargs):
        """Execute an operation on a specific shard"""
        await self.ensure_collection_loaded(connection_alias, shard_id)
        
        collection = Collection(self.collection_name, using=connection_alias)
        
        if operation == "insert":
            partition_name = kwargs.pop('partition_name', None)
            if partition_name:
                partitions = collection.partitions
                partition_names = [p.name for p in partitions]
                if partition_name not in partition_names:
                    try:
                        collection.create_partition(partition_name)
                        logger.debug(f"Created partition {partition_name} in collection {self.collection_name}")
                    except Exception as e:
                        logger.warning(f"Error creating partition {partition_name}: {e}")

                return collection.insert(partition_name=partition_name, **kwargs)
            else:
                # Insert to default partition
                return collection.insert(**kwargs)

        elif operation == "search":
            return collection.search(**kwargs)
        elif operation == "query":
            return collection.query(**kwargs)
        elif operation == "delete":
            return collection.delete(**kwargs)
        elif operation == "flush":
            return collection.flush()
        else:
            raise ValueError(f"Unknown operation: {operation}")

class RouterHealthMonitor:
    """Monitors shard health and manages failover"""
    
    def __init__(self, sharding_manager: ShardingManager, connection_pool: ConnectionPool, 
                check_interval=30):
        self.sharding_manager = sharding_manager
        self.connection_pool = connection_pool
        self.check_interval = check_interval
        self.shard_health = {}  # {shard_id: bool}
        self.health_check_task = None
        
    async def start_monitoring(self):
        """Start the health monitoring background task"""
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
    async def stop_monitoring(self):
        """Stop the health monitoring background task"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            
    async def _health_check_loop(self):
        """Background task to periodically check shard health"""
        while True:
            try:
                # Get all shards
                topology = await self.sharding_manager.get_topology()
                
                for shard_id, config in topology.shards.items():
                    if config.status != "active":
                        continue
                        
                    # Check health of this shard
                    is_healthy = await self._check_shard_health(shard_id, config)
                    
                    # Update health status
                    self.shard_health[shard_id] = is_healthy
                    
                    if not is_healthy:
                        logger.warning(f"Shard {shard_id} is unhealthy")
                        
                        # If primary is down, consider using replicas
                        if not config.replica_of:
                            replicas = await self.sharding_manager.get_replica_shards(shard_id)
                            logger.debug(f"Primary shard {shard_id} is down, will try replicas: {replicas}")
                    
                # Sleep until next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.check_interval)
                
    async def _check_shard_health(self, shard_id: str, config: ShardConfig) -> bool:
        """Check health of a specific shard using a temporary connection"""
        temp_alias = f"temp_health_{shard_id}"
        try:
            # Create a temporary connection with a short timeout
            connections.connect(
                temp_alias,
                host=config.host,
                port=config.port,
                timeout=5
            )
            logger.debug(f"Temp health connection '{temp_alias}' created for shard {shard_id}")

            # Perform a simple operation to verify health; here, list the collections
            collections_list = utility.list_collections(using=temp_alias)
            logger.debug(f"Temp health check for shard {shard_id} succeeded. Collections: {collections_list}")

            # Immediately disconnect the temporary connection
            connections.disconnect(temp_alias)
            logger.debug(f"Temp health connection '{temp_alias}' disconnected for shard {shard_id}")
            return True
        except Exception as e:
            logger.error(f"Health check failed for shard {shard_id} with alias {temp_alias}: {e}")
            try:
                connections.disconnect(temp_alias)
            except Exception:
                pass
            return False

            
    async def is_shard_healthy(self, shard_id: str) -> bool:
        """Check if a shard is currently healthy"""
        if shard_id in self.shard_health:
            return self.shard_health[shard_id]
            
        # If we haven't checked yet, assume it's healthy
        return True
        
    async def get_healthy_shard(self, primary_shard_id: str) -> str:
        """Get a healthy shard (primary or replica) for the given primary"""
        # Check if primary is healthy
        if await self.is_shard_healthy(primary_shard_id):
            return primary_shard_id
            
        # Try replicas
        replicas = await self.sharding_manager.get_replica_shards(primary_shard_id)
        for replica_id in replicas:
            if await self.is_shard_healthy(replica_id):
                logger.debug(f"Using healthy replica {replica_id} instead of primary {primary_shard_id}")
                return replica_id
                
        # No healthy shards found, return primary and hope for the best
        logger.warning(f"No healthy shards found for {primary_shard_id}, using primary anyway")
        return primary_shard_id

class ShardedMilvusRouter:
    """Main router class that coordinates routing, connection management, and operations"""
    
    def __init__(self, 
                config_service_url=None, 
                redis_url=None,
                collection_name="person_reid_features",
                embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.sharding_manager = ShardingManager(config_service_url, redis_url)
        self.connection_pool = ConnectionPool()
        self.collection_manager = MilvusCollectionManager(collection_name, embedding_dim)
        self.health_monitor = RouterHealthMonitor(self.sharding_manager, self.connection_pool)

        #DEBUG
        self.timing_monitor = SimpleTimingMonitor()
        
    async def start(self):
        """Start the router services"""
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
        # Initial topology refresh
        await self.sharding_manager.refresh_topology(force=True)
        
        # Start background tasks
        asyncio.create_task(self._background_maintenance())
        
    async def stop(self):
        """Stop the router services"""
        await self.health_monitor.stop_monitoring()
        await self.connection_pool.reset_connections()
        
    async def _background_maintenance(self):
        """Background task for maintenance operations"""
        while True:
            try:
                # Cleanup connections
                await self.connection_pool.cleanup_stale_connections()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Error in background maintenance: {e}")
                await asyncio.sleep(60)

    async def configure_connection_pool(self, max_connections: int, timeout: Optional[int] = None):
        """Configure connection pool settings"""
        try:
            result = await self.connection_pool.update_pool_configuration(max_connections, timeout)
            logger.debug(f"Connection pool reconfigured successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to configure connection pool: {e}")
            raise

    async def _get_connection_for_store(self, store_id: int):
        """Get connection details for a specific store"""
        # Determine which shard this store maps to
        primary_shard_id = await self.sharding_manager.get_shard_for_store(store_id)
        
        # Get a healthy shard (primary or replica)
        shard_id = await self.health_monitor.get_healthy_shard(primary_shard_id)
        
        # Get shard configuration
        shard_config = await self.sharding_manager.get_shard_config(shard_id)
        
        # Update connection pool with latest config
        await self.connection_pool.update_shard_config(shard_id, shard_config)
        
        # Get a connection from the pool
        connection_alias = await self.connection_pool.get_connection_alias(shard_id)
        
        return connection_alias, shard_id

    def _prepare_insert_data(
        self,
        embeddings,
        track_ids,
        store_ids,
        camera_ids,
        timestamps
    ):
        """Prepare data in the EXACT order required by the schema"""

        # Validate all embeddings
        validated_embeddings = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                validated_embeddings.append(emb.astype(np.float32).tolist())
            else:
                validated_embeddings.append(self._validate_embedding(emb))

        # Ensure proper data types for schema compatibility
        validated_track_ids  = [int(x) for x in track_ids]
        validated_store_ids  = [int(x) for x in store_ids]    # INT32 in schema
        validated_camera_ids = [int(x) for x in camera_ids]   # INT32 in schema
        validated_timestamps = [int(x) for x in timestamps]   # INT64 in schema

        # Create composite keys
        store_camera_composites = [
            self._create_store_camera_composite(store_id, camera_id)
            for store_id, camera_id in zip(validated_store_ids, validated_camera_ids)
        ]

        # CRITICAL: Data must be in EXACT schema field order
        # Schema order: id (auto), feature_vector, track_id, store_id, camera_id, timestamp, store_camera_composite
        data = [
            validated_embeddings,        # feature_vector (FLOAT_VECTOR, dim=768)
            validated_track_ids,         # track_id (INT64)
            validated_store_ids,         # store_id (INT32)
            validated_camera_ids,        # camera_id (INT32)
            validated_timestamps,        # timestamp (INT64)
            store_camera_composites      # store_camera_composite (INT64)
        ]

        return data

    def _create_store_camera_composite(self, store_id: int, camera_id: int) -> int:
        """Create composite key - FIXED to match the test script formula"""
        # Use the same formula as in the test scripts: store_id * 10000 + camera_id
        return store_id * 10000 + camera_id
        
    async def insert_embedding(self, request: EmbeddingRequest):
        """Insert a single embedding"""
        try:
            connection_alias, shard_id = await self._get_connection_for_store(request.store_id)
            
            # Prepare data for insertion
            if isinstance(request.feature_vector, np.ndarray):
                request.feature_vector = request.feature_vector.tolist()
            
            if len(request.feature_vector) != self.embedding_dim:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(request.feature_vector)}")
                raise HTTPException(status_code=400, detail="Embedding dimension mismatch")
                
            data = self._prepare_insert_data(
                embeddings   = [request.feature_vector],
                track_ids    = [request.track_id],
                store_ids    = [request.store_id],
                camera_ids   = [request.camera_id],
                timestamps   = [request.timestamp]
            )
            partition_name = f"stores_{request.store_id}_to_{request.store_id}"
            # Execute insert operation
            result = await self.collection_manager.execute_operation(
                connection_alias=connection_alias,
                shard_id=shard_id,
                operation="insert",
                data=data,
                partition_name = partition_name
            )
            
            # await self.collection_manager.execute_operation(
            #     connection_alias=connection_alias,
            #     shard_id=shard_id,
            #     operation="flush"
            # )
            
            return {"success": True, "shard_id": shard_id}
            
        except Exception as e:
            logger.error(f"Insert error: {e}")
            raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")
            
    async def insert_embeddings_batch(self, request: BatchEmbeddingRequest):
        """Insert multiple embeddings in a batch"""
        request_id = str(uuid.uuid4())[:8]
        total_start = time.time()
        try:
            # Group embeddings by store_id for efficiency
            store_groups = defaultdict(list)
            results = {}
            total_connection_time = 0
            total_db_time = 0
            all_shard_ids = set()
            for i in range(len(request.track_ids)):
                store_id = request.store_ids[i]
                store_groups[store_id].append(i)
                
            # Process each store's data separately
            for store_id, indices in store_groups.items():
                connection_start = time.time()
                connection_alias, shard_id = await self._get_connection_for_store(store_id)
                connection_time = time.time() - connection_start
                total_connection_time += connection_time

                try:
                    
                    # Prepare data for this store
                    db_start = time.time()
                    track_ids = [request.track_ids[i] for i in indices]
                    embeddings = []
                    for i in indices:
                        emb = request.feature_vectors[i]
                        if isinstance(emb, np.ndarray):
                            embeddings.append(emb.tolist())
                        else:
                            embeddings.append(emb)
                    # embeddings = [request.embeddings[i] for i in indices]
                    store_ids = [request.store_ids[i] for i in indices]
                    camera_ids = [request.camera_ids[i] for i in indices]
                    timestamps = [request.timestamps[i] for i in indices]

                    partition_name = f"stores_{store_id}_to_{store_id}"
                    data = self._prepare_insert_data(
                        embeddings  = embeddings,
                        track_ids   = track_ids,
                        store_ids   = store_ids,
                        camera_ids  = camera_ids,
                        timestamps  = timestamps
                    )
                    
                    # Execute insert operation
                    store_result = await self.collection_manager.execute_operation(
                        connection_alias=connection_alias,
                        shard_id=shard_id,
                        operation="insert",
                        data=data,
                        partition_name=partition_name
                    )
                    db_time = time.time() - db_start
                    total_db_time += db_time
                    all_shard_ids.add(shard_id)
                    
                    results[store_id] = {
                    "shard_id": shard_id,
                    "insert_count": len(indices)
                    }
                except Exception as store_error:
                    logger.error(f"Error inserting data for store {store_id}: {store_error}")
                    results[store_id] = {
                        "shard_id": shard_id,
                        "insert_count": 0,
                        "success": False,
                        "error": str(store_error)
                    }
            # flush_start = time.time()
            # for shard_id in all_shard_ids:
            #     try:
            #         connection_alias = await self.connection_pool.get_connection_alias(shard_id)
            #         await self.collection_manager.execute_operation(
            #             connection_alias=connection_alias,
            #             shard_id=shard_id,
            #             operation="flush"
            #         )
            #         logger.debug(f"Flushed shard {shard_id}")
            #     except Exception as flush_error:
            #         logger.error(f"Error flushing shard {shard_id}: {flush_error}")
            # flush_time = time.time() - flush_start
            # total_db_time += flush_time 
            total_time = time.time() - total_start
            self.timing_monitor.record_request_timing(
                request_id,
                {
                    'total': total_time,
                    'connection_wait': total_connection_time,
                    'database': total_db_time,  # Includes only insert
                    'processing': total_time - total_connection_time - total_db_time
                },
                {
                    'operation': 'insert_batch',
                    'batch_size': len(request.track_ids),
                    'store_count': len(store_groups),
                    'shard_count': len(all_shard_ids)
                }
            )
            return {
                "success": True,
                "results": results,
                "total_inserted": len(request.track_ids)
            }
            
        except Exception as e:
            logger.error(f"Batch insert error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch insert failed: {str(e)}")
            
    async def search_embedding(self, request: SearchRequest):
        """Search for similar embeddings"""
        request_id = str(uuid.uuid4())[:8]
        total_start = time.time()

        connection_start = time.time()
        connection_alias, shard_id = await self._get_connection_for_store(request.store_id)
        connection_time = time.time() - connection_start
        db_start = time.time()
        try:
            # Prepare search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 32}
            }
            
            # Prepare filter expression
            expr_parts = [f"store_id == {request.store_id}"]
            if request.camera_filter is not None:
                expr_parts.append(f"camera_id == {request.camera_filter}")
            expr = " && ".join(expr_parts)
            
            # Prepare query vector
            query_embedding = np.array([request.feature_vector])
            
            # Execute search operation
            results = await self.collection_manager.execute_operation(
                connection_alias=connection_alias,
                shard_id=shard_id,
                operation="search",
                data=query_embedding,
                anns_field="feature_vector",
                param=search_params,
                limit=request.top_k,
                expr=expr,
                output_fields=["track_id", "store_id", "camera_id", "timestamp"]
            )
            db_time = time.time() - db_start

            processing_start = time.time()
            # Process results
            matches = []
            for hit in results[0]:
                # Convert distance to similarity
                similarity = 1.0 - hit.distance
                
                # Only include results above minimum similarity threshold
                if similarity >= request.min_similarity:
                    # Extract metadata
                    metadata = {
                        "store_id": hit.entity.get("store_id"),
                        "camera_id": hit.entity.get("camera_id"),
                        "timestamp": hit.entity.get("timestamp")
                    }
                    
                    matches.append({
                        "track_id": hit.entity.get("track_id"),
                        "similarity": similarity,
                        "metadata": metadata
                    })
            processing_time = time.time() - processing_start
            total_time = time.time() - total_start
            self.timing_monitor.record_request_timing(
               request_id,
               {
                   'total': total_time,
                   'connection_wait': connection_time,
                   'database': db_time,
                   'processing': processing_time
               },
               {
                   'operation': 'search_embedding',
                   'store_id': request.store_id,
                   'shard_id': shard_id
               }
            )
            return {
                "success": True,
                "shard_id": shard_id,
                "matches": matches,
                "total_matches": len(matches)
            }
            
        except Exception as e:
            total_time = time.time() - total_start
            logger.error(f"ERROR {request_id}: {total_time:.2f}s - {str(e)}")
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Add this method to the ShardedMilvusRouter class
    async def search_embeddings_batch(self, request: BatchSearchRequest):
        """Search for similar embeddings in batch"""
        request_id = str(uuid.uuid4())[:8]
        total_start = time.time()
        try:
            connection_start = time.time()
            connection_alias, shard_id = await self._get_connection_for_store(request.store_id)
            connection_time = time.time() - connection_start

            db_start = time.time()
            # Prepare search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 32}
            }
            
            # Prepare filter expression
            expr = f"store_id == {request.store_id}"
            
            # Convert embeddings to numpy array
            query_embeddings = np.array(request.feature_vectors)
            
            # Execute batch search operation
            results = await self.collection_manager.execute_operation(
                connection_alias=connection_alias,
                shard_id=shard_id,
                operation="search",
                data=query_embeddings,
                anns_field="feature_vector",
                param=search_params,
                limit=request.top_k,
                expr=expr,
                output_fields=["track_id", "store_id", "camera_id", "timestamp"]
            )
            db_time = time.time() - db_start
            processing_start = time.time()

            # Process results for each query
            batch_results = []
            for query_results in results:
                query_matches = []
                for hit in query_results:
                    # Convert distance to similarity
                    similarity = 1.0 - hit.distance
                    
                    # Only include results above minimum similarity threshold
                    if similarity >= request.min_similarity:
                        # For tracker compatibility, return (track_id, distance) tuples
                        query_matches.append((hit.entity.get("track_id"), hit.distance))
                
                batch_results.append(query_matches)
            processing_time = time.time() - processing_start
            # Record timing breakdown
            total_time = time.time() - total_start
            self.timing_monitor.record_request_timing(
                request_id,
                {
                    'total': total_time,
                    'connection_wait': connection_time,
                    'database': db_time,
                    'processing': processing_time
                },
                {
                    'operation': 'search_batch',
                    'store_id': request.store_id,
                    'shard_id': shard_id,
                    'batch_size': len(request.feature_vectors),
                    'total_results': sum(len(matches) for matches in batch_results)
                }
            )

            return {
                "success": True,
                "shard_id": shard_id,
                "results": batch_results
            }
            
        except Exception as e:
            total_time = time.time() - total_start
            logger.error(f"ERROR {request_id}: {total_time:.2f}s - {str(e)}")
            logger.error(f"Batch search error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")

    async def search_embedding_for_tracker(self, query_embedding, store_id, top_k=10):
        """
        Search for similar embeddings specifically optimized for tracker usage.
        Returns track_ids and distances in a format compatible with tracker code.
        
        Args:
            query_embedding: Vector to search for
            store_id: Store ID to filter by
            top_k: Maximum number of results to return
            
        Returns:
            List of tuples: [(track_id, distance), ...]
        """
        try:
            # Use existing search method but transform result format
            search_result = await self.search_embedding(SearchRequest(
                embedding=query_embedding,
                store_id=store_id,
                top_k=top_k,
                min_similarity=0.0  # Return all results and let tracker filter
            ))
            
            # Transform matches to the format expected by tracker
            tracker_results = [
                (match["track_id"], 1.0 - match["similarity"])  # Convert similarity back to distance
                for match in search_result["matches"]
            ]
            
            return tracker_results
        except Exception as e:
            logger.error(f"Search error for tracker: {e}")
            return []  # Return empty list on error so tracker can continue

    async def find_next_best_match(self, feature, store_id, assigned_ids, max_candidates=5, distance_threshold=0.7):
        """
        Find the next best match from Milvus, excluding already assigned IDs
        
        Args:
            feature: Feature vector of the detection
            store_id: Store ID to filter by
            assigned_ids: Set of track IDs already assigned in the current frame
            max_candidates: Maximum number of candidates to consider
            distance_threshold: Maximum distance threshold for a valid match
                
        Returns:
            Tuple[int or None, float]: (track_id, distance) or (None, inf) if no good match found
        """
        try:
            # Perform general search
            results = await self.search_embedding_for_tracker(
                query_embedding=feature,
                store_id=store_id,
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
         
    async def delete_track(self, request: DeleteRequest):
        """Delete a track from the database"""
        try:
            connection_alias, shard_id = await self._get_connection_for_store(request.store_id)
            
            # Prepare deletion expression and log details
            expr = f"track_id == {request.track_id} and store_id == {request.store_id}"
            logger.debug(f"Deletion expression for track {request.track_id}: {expr}")
            logger.debug(f"Using connection alias: {connection_alias} on shard: {shard_id}")
            
            # Execute delete operation and log the operation result
            result = await self.collection_manager.execute_operation(
                connection_alias=connection_alias,
                shard_id=shard_id,
                operation="delete",
                expr=expr
            )
            logger.debug(f"Result from delete operation: {result}")
            
            # Flush to ensure deletion is committed
            # flush_result = await self.collection_manager.execute_operation(
            #     connection_alias=connection_alias,
            #     shard_id=shard_id,
            #     operation="flush"
            # )
            # logger.debug(f"Result from flush operation: {flush_result}")
            
            return {
                "success": True,
                "shard_id": shard_id,
                "deleted": True
            }
            
        except Exception as e:
            logger.error(f"Delete error for track_id={request.track_id}, store_id={request.store_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

    async def get_topology(self):
        """Get the current sharding topology"""
        try:
            return await self.sharding_manager.get_topology()
        except Exception as e:
            logger.error(f"Get topology error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get topology: {str(e)}")
            
    async def get_features_by_track_id(self, track_id: int, store_id: int):
        """Get all features for a specific track_id"""
        try:
            connection_alias, shard_id = await self._get_connection_for_store(store_id)
            
            # Prepare query expression
            expr = f"track_id == {track_id} && store_id == {store_id}"
            
            # Execute query operation
            results = await self.collection_manager.execute_operation(
                connection_alias=connection_alias,
                shard_id=shard_id,
                operation="query",
                expr=expr,
                output_fields=["feature_vector", "timestamp", "camera_id"],
                limit=1000
            )
            
            # Process features
            features = []
            for row in results:
                feature = {
                    "feature_vector": row["feature_vector"],
                    "timestamp": row.get("timestamp"),
                    "camera_id": row.get("camera_id")
                }
                features.append(feature)
                
            return {
                "success": True,
                "shard_id": shard_id,
                "track_id": track_id,
                "store_id": store_id,
                "features": features,
                "feature_count": len(features)
            }
            
        except Exception as e:
            logger.error(f"Get features error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get features: {str(e)}")

    async def get_features_by_track_id_for_tracker(self, track_id, store_id):
        """
        Gets features for a specific track_id formatted for the tracker
        
        Args:
            track_id: ID of the track to get features for
            store_id: Store ID filter
            
        Returns:
            List[numpy.ndarray]: List of feature embeddings as numpy arrays
        """
        try:
            # Use existing method but transform result
            features_result = await self.get_features_by_track_id(track_id, store_id)
            
            # Convert to list of numpy arrays as expected by tracker
            embeddings = [np.array(f["embedding"], dtype=np.float32) for f in features_result["features"]]
            return embeddings
        except Exception as e:
            logger.error(f"Error getting features for track_id {track_id}: {e}")
            return []
        
    async def get_all_track_features(self, store_id: int, limit: int = 100000):
        """
        Returns a dictionary where keys are track_ids and values are lists of embeddings
        
        Args:
            store_id: Filter results by store_id
            limit: Maximum number of embeddings to retrieve
            
        Returns:
            Dict mapping track_ids to lists of feature embeddings
        """
        request_id = str(uuid.uuid4())[:8]
        total_start = time.time()
        try:
            connection_start = time.time()
            connection_alias, shard_id = await self._get_connection_for_store(store_id)
            connection_time = time.time() - connection_start


            # Build query expression
            expr = f"store_id == {store_id}"
            logger.debug(f"Querying all embeddings for store_id={store_id} (limit={limit})...")
            
            db_start = time.time()

            # Execute query with pagination
            if limit <= 10000:
                results = await self.collection_manager.execute_operation(
                    connection_alias=connection_alias,
                    shard_id=shard_id,
                    operation="query",
                    expr=expr,
                    output_fields=["track_id", "feature_vector"],
                    limit=limit
                )
            else:
                # For large result sets, use pagination
                results = []
                offset = 0
                # page_size = 10000
                
                while offset < limit:
                    page_size = min(10000, 16384 - offset)
                    page = await self.collection_manager.execute_operation(
                        connection_alias=connection_alias,
                        shard_id=shard_id,
                        operation="query",
                        expr=expr,
                        output_fields=["track_id", "feature_vector"],
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

            db_time = time.time() - db_start
       
            # Phase 3: Result processing
            processing_start = time.time()     

            # Process results
            feature_map = defaultdict(list)
            for row in results:
                track_id = row["track_id"]
                embedding = np.array(row["feature_vector"], dtype=np.float32)
                feature_map[track_id].append(embedding)
                
            processing_time = time.time() - processing_start
            logger.debug(f"Retrieved features for {len(feature_map)} unique track_ids from shard {shard_id}")
            
            # Record timing breakdown
            total_time = time.time() - total_start
            self.timing_monitor.record_request_timing(
                request_id,
                {
                    'total': total_time,
                    'connection_wait': connection_time,
                    'database': db_time,
                    'processing': processing_time
                },
                {
                    'operation': 'get_all_track_features',
                    'store_id': store_id,
                    'shard_id': shard_id,
                    'limit': limit,
                    'total_records': len(results),
                    'unique_tracks': len(feature_map),
                    'used_pagination': limit > 10000
                }
            )

            return {
                "success": True,
                "shard_id": shard_id,
                "track_count": len(feature_map),
                "features": dict(feature_map)
            }
            
        except Exception as e:
            total_time = time.time() - total_start
            logger.error(f"ERROR {request_id}: {total_time:.2f}s - {str(e)}")
            logger.error(f"Failed to query embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get track features: {str(e)}")

    async def get_all_track_features_for_tracker(self, store_id, limit=100000):
        """
        Returns a dictionary where keys are track_ids and values are lists of embeddings
        formatted specifically for the tracker code
        
        Args:
            store_id: Filter results by store_id
            limit: Maximum number of embeddings to retrieve
            
        Returns:
            Dict[int, List[numpy.ndarray]]: Dictionary mapping track_ids to lists of feature embeddings
        """
        try:
            # Get all features using previously implemented method
            features_result = await self.get_all_track_features(store_id, limit)
            
            # Convert to tracker-compatible format
            feature_map = {}
            for track_id, features_data in features_result["features"].items():
                try:
                    track_id_int = int(track_id)
                except Exception as e:
                    logger.error(f"Cannot convert track ID '{track_id}' to integer: {e}")
                embeddings = []
                for f in features_data:
                    logger.debug(f"Track {track_id_int}, feature entry type: {type(f)}, value: {f}")
                    if isinstance(f, dict) and "embedding" in f:
                        try:
                            emb_array = np.array(f["embedding"], dtype=np.float32)
                            embeddings.append(emb_array)
                        except Exception as conv_ex:
                            logger.error(f"Error converting embedding for track {track_id_int} from dict: {conv_ex}")
                    elif isinstance(f, (list, np.ndarray)):
                        try:
                            # If already a list/array, convert directly
                            emb_array = np.array(f, dtype=np.float32)
                            embeddings.append(emb_array)
                        except Exception as conv_ex:
                            logger.error(f"Error converting embedding for track {track_id_int} from list/array: {conv_ex}")
                    else:
                        logger.error(f"Unexpected feature format for track {track_id_int}: {f}")
                feature_map[track_id_int] = embeddings
                # Convert each embedding to numpy array as expected by tracker
                # embeddings = [np.array(f["embedding"], dtype=np.float32) for f in features_data]
                # feature_map[track_id_int] = embeddings
                
            return feature_map
        except Exception as e:
            logger.error(f"Error getting all track features for tracker: {e}")
            return {}

    async def get_all_track_ids(self, store_id: int):
        """
        Fetch all unique track_ids in the collection for a given store_id
        
        Args:
            store_id: Store ID to filter by
            
        Returns:
            List of unique track IDs
        """
        try:
            connection_alias, shard_id = await self._get_connection_for_store(store_id)
            
            # Build query expression
            expr = f"store_id == {store_id}"
            
            # Execute query to get all track_ids
            results = await self.collection_manager.execute_operation(
                connection_alias=connection_alias,
                shard_id=shard_id,
                operation="query",
                expr=expr,
                output_fields=["track_id"],
                limit=100000  # Adjust if needed
            )
            
            # Extract unique track_ids
            track_ids = list({r["track_id"] for r in results})
            logger.debug(f"Found {len(track_ids)} unique track_ids for store_id={store_id} on shard {shard_id}")
            
            return {
                "success": True,
                "shard_id": shard_id,
                "store_id": store_id,
                "track_ids": track_ids,
                "count": len(track_ids)
            }
            
        except Exception as e:
            logger.error(f"Error fetching track_ids for store_id {store_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get track IDs: {str(e)}")

# Initialize router
router = ShardedMilvusRouter(
    config_service_url=os.environ.get("CONFIG_SERVICE_URL"),
    redis_url=os.environ.get("REDIS_URL")
)

@app.on_event("startup")
async def startup():
    """Initialize router on application startup"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.debug(f"Loaded configuration from {config_path}")
            # Initialize shards from config
            if "shards" in config:
                for shard_id, shard_data in config["shards"].items():
                    shard_config = ShardConfig(**shard_data)
                    router.sharding_manager.shard_configs[shard_id] = shard_config
                    logger.debug(f"Added shard {shard_id}: {shard_config.host}:{shard_config.port}")
                
            # Initialize store-to-shard mappings
            if "store_mappings" in config:
                for store_id, shard_id in config["store_mappings"].items():
                    router.sharding_manager.store_to_shard[store_id] = shard_id
                logger.debug(f"Mapped {len(config['store_mappings'])} stores to shards")
                
             # Log the configuration
            logger.debug(f"Router initialized with {len(router.sharding_manager.shard_configs)} shards and {len(router.sharding_manager.store_to_shard)} store mappings")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}", exc_info=True)
    else:
        logger.debug(f"No config file found at {config_path}, using environment variables or defaults")
    await router.start()

@app.on_event("shutdown")
async def shutdown():
    """Cleanup router on application shutdown"""
    await router.stop()

@app.get("/health")
async def health_check():
    """
    A simple liveness probe for your router.
    """
    return JSONResponse(status_code=200, content={"status": "ok"})

# Add this endpoint with your other endpoints
@app.post("/configure/connection_pool")
async def configure_connection_pool(request: ConnectionPoolConfigRequest):
    """Configure connection pool settings dynamically"""
    try:
        result = await router.configure_connection_pool(
            max_connections=request.max_connections_per_shard,
            timeout=request.connection_timeout
        )
        
        return {
            "success": True,
            "message": f"Connection pool configured to {request.max_connections_per_shard} connections per shard",
            "configuration": result
        }
        
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to configure connection pool: {str(e)}"
        )

# Also add a GET endpoint to check current configuration
@app.get("/configure/connection_pool")
async def get_connection_pool_config():
    """Get current connection pool configuration"""
    return {
        "max_connections_per_shard": router.connection_pool.max_connections_per_shard,
        "connection_timeout": router.connection_pool.connection_timeout,
        "current_connections": {
            shard_id: len(conns) 
            for shard_id, conns in router.connection_pool.connections.items()
        },
        "pool_health": router.connection_pool.get_pool_health()
    }

# API Endpoints
@app.post("/insert")
async def insert_embedding(request: EmbeddingRequest):
    """Insert a single embedding"""
    return await router.insert_embedding(request)

@app.post("/batch_insert")
async def insert_embeddings_batch(request: BatchEmbeddingRequest):
    """Insert multiple embeddings in a batch"""
    return await router.insert_embeddings_batch(request)

@app.post("/search")
async def search_embedding(request: SearchRequest):
    """Search for similar embeddings"""
    return await router.search_embedding(request)

# Add this API endpoint at the bottom with other endpoints
@app.post("/batch_search")
async def search_embeddings_batch(request: BatchSearchRequest):
    """Search for multiple embeddings in a single batch"""
    return await router.search_embeddings_batch(request)

@app.post("/delete")
async def delete_track(request: DeleteRequest):
    """Delete a track from the database"""
    return await router.delete_track(request)

@app.get("/topology")
async def get_topology():
    """Get the current sharding topology"""
    return await router.get_topology()

# Add this to your milvus_sharding_router.py

@app.get("/test_connection/{store_id}")
async def test_connection(store_id: int):
    """Test connection to the Milvus shard for a specific store"""
    try:
        # Get connection details
        logger.debug(f"[{store_id}] start health check")
        t0 = time.time()
        connection_alias, shard_id = await router._get_connection_for_store(store_id)
        t1 = time.time()
        logger.debug(f"[{store_id}] got alias in {t1 - t0:.3f}s")
        shard_config = router.connection_pool.shard_info[shard_id]
        
        # Try to list collections as a test
        t2 = time.time()
        collection_list = utility.list_collections(using=connection_alias)
        t3 = time.time()
        logger.debug(f"[{store_id}] list_collections took {t3 - t2:.3f}s")
        total = t3 - t0
        logger.debug(f"[{store_id}] total endpoint latency: {total:.3f}s")
        return {
            "success": True,
            "store_id": store_id,
            "shard_id": shard_id,
            "host": shard_config.host,
            "port": shard_config.port,
            "collections": collection_list
        }
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {
            "success": False,
            "store_id": store_id,
            "error": str(e)
        }

@app.get("/features/{track_id}/{store_id}")
async def get_features(track_id: int, store_id: int):
    """Get all features for a specific track_id"""
    # return await router.get_features_by_track_id(track)
    return await router.get_features_by_track_id(track_id, store_id)

@app.get("/track/features/{track_id}/{store_id}")
async def get_track_features_for_tracker(track_id: int, store_id: int):
    """Get features for a specific track_id in tracker-compatible format"""
    features = await router.get_features_by_track_id_for_tracker(track_id, store_id)
    return {"track_id": track_id, "store_id": store_id, "features": [f.tolist() for f in features]}

@app.post("/track/search")
async def search_for_tracker(request: SearchRequest):
    """Search for similar embeddings in tracker-compatible format"""
    results = await router.search_embedding_for_tracker(
        query_embedding=request.feature_vector,
        store_id=request.store_id,
        top_k=request.top_k
    )
    return {"results": results}

# Add monitoring endpoint
@app.get("/performance/breakdown")
async def get_performance_breakdown():
   """Get detailed performance breakdown"""
   timing_stats = router.timing_monitor.get_recent_stats()
   pool_health = router.connection_pool.get_pool_health()
   
   # Determine bottleneck
   bottleneck = "unknown"
   if timing_stats.get('avg_connection_wait', 0) > 2.0:
       bottleneck = "connection_pool"
   elif timing_stats.get('avg_database', 0) > 3.0:
       bottleneck = "database"
   elif timing_stats.get('avg_processing', 0) > 1.0:
       bottleneck = "router_processing"
   else:
       bottleneck = "healthy"
   
   return {
       "bottleneck_analysis": bottleneck,
       "timing_breakdown": timing_stats,
       "connection_pool_health": pool_health,
       "diagnosis": {
           "router_overloaded": timing_stats.get('avg_connection_wait', 0) > 2.0,
           "database_slow": timing_stats.get('avg_database', 0) > 3.0,
           "many_slow_requests": timing_stats.get('slow_requests', 0) > 5
       }
   }

@app.get("/track/allfeatures/{store_id}")
async def get_all_track_features_for_tracker(store_id: int, limit: int = 100000):
    """Get all track features in tracker-compatible format"""
    features = await router.get_all_track_features_for_tracker(store_id, limit)
    # Convert numpy arrays to lists for JSON serialization
    result = {}
    for track_id, embeddings in features.items():
        result[track_id] = [e.tolist() for e in embeddings]
    return result
