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
from pydantic import BaseModel, Field
import httpx
import redis
from pymilvus import Collection, connections, utility
import uvicorn
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
    embedding: List[float]
    store_id: int
    camera_id: int
    timestamp: int

class BatchEmbeddingRequest(BaseModel):
    track_ids: List[int]
    embeddings: List[List[float]]
    store_ids: List[int]
    camera_ids: List[int]
    timestamps: List[int]

class SearchRequest(BaseModel):
    embedding: List[float]
    store_id: int
    top_k: int = 5
    camera_filter: Optional[int] = None
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

class ConnectionPool:
    """Manages and pools connections to Milvus shards"""
    
    def __init__(self, max_connections_per_shard=5, connection_timeout=10):
        self.connections = {}  # {shard_id: {connection_alias: last_used_time}}
        self.shard_info = {}  # {shard_id: ShardConfig}
        self.max_connections_per_shard = max_connections_per_shard
        self.connection_timeout = connection_timeout
        self.lock = asyncio.Lock()
        
    async def get_connection_alias(self, shard_id: str) -> str:
        """Get a connection alias for the specified shard"""
        async with self.lock:
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
                        logger.info(f"Created new connection to shard {shard_id}: {connection_alias}")
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
                            logger.info(f"Closed stale connection to shard {shard_id}: {connection_alias}")
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
                logger.info(f"Topology refreshed with {len(self.store_to_shard)} store mappings and {len(self.shard_configs)} shards")
                
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
                
            logger.info(f"Successfully refreshed topology from config service: "
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
        
        logger.info(f"Assigned store {store_id} to shard {assigned_shard} using consistent hashing")
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
    
    def __init__(self, collection_name="person_embeddings", embedding_dim=768):
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
            logger.info(f"Loaded collection {self.collection_name} on shard {shard_id}")
        except Exception as e:
            logger.error(f"Failed to load collection on shard {shard_id}: {e}")
            raise
            
    async def execute_operation(self, connection_alias: str, shard_id: str, operation: str, **kwargs):
        """Execute an operation on a specific shard"""
        await self.ensure_collection_loaded(connection_alias, shard_id)
        
        collection = Collection(self.collection_name, using=connection_alias)
        
        if operation == "insert":
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
                            logger.info(f"Primary shard {shard_id} is down, will try replicas: {replicas}")
                    
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
                logger.info(f"Using healthy replica {replica_id} instead of primary {primary_shard_id}")
                return replica_id
                
        # No healthy shards found, return primary and hope for the best
        logger.warning(f"No healthy shards found for {primary_shard_id}, using primary anyway")
        return primary_shard_id

class ShardedMilvusRouter:
    """Main router class that coordinates routing, connection management, and operations"""
    
    def __init__(self, 
                config_service_url=None, 
                redis_url=None,
                collection_name="person_embeddings",
                embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.sharding_manager = ShardingManager(config_service_url, redis_url)
        self.connection_pool = ConnectionPool()
        self.collection_manager = MilvusCollectionManager(collection_name, embedding_dim)
        self.health_monitor = RouterHealthMonitor(self.sharding_manager, self.connection_pool)
        
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
        
    async def insert_embedding(self, request: EmbeddingRequest):
        """Insert a single embedding"""
        try:
            connection_alias, shard_id = await self._get_connection_for_store(request.store_id)
            
            # Prepare data for insertion
            if isinstance(request.embedding, np.ndarray):
                request.embedding = request.embedding.tolist()
            
            if len(request.embedding) != self.embedding_dim:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(request.embedding)}")
                raise HTTPException(status_code=400, detail="Embedding dimension mismatch")
                
            data = [
                [request.track_id], 
                [request.embedding], 
                [request.store_id], 
                [request.camera_id], 
                [request.timestamp]
            ]
            
            # Execute insert operation
            result = await self.collection_manager.execute_operation(
                connection_alias=connection_alias,
                shard_id=shard_id,
                operation="insert",
                data=data
            )
            
            await self.collection_manager.execute_operation(
                connection_alias=connection_alias,
                shard_id=shard_id,
                operation="flush"
            )
            
            return {"success": True, "shard_id": shard_id}
            
        except Exception as e:
            logger.error(f"Insert error: {e}")
            raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")
            
    async def insert_embeddings_batch(self, request: BatchEmbeddingRequest):
        """Insert multiple embeddings in a batch"""
        try:
            # Group embeddings by store_id for efficiency
            store_groups = defaultdict(list)
            
            for i in range(len(request.track_ids)):
                store_id = request.store_ids[i]
                store_groups[store_id].append(i)
                
            results = {}
            
            # Process each store's data separately
            for store_id, indices in store_groups.items():
                connection_alias, shard_id = await self._get_connection_for_store(store_id)
                
                # Prepare data for this store
                track_ids = [request.track_ids[i] for i in indices]
                embeddings = []
                for i in indices:
                    emb = request.embeddings[i]
                    if isinstance(emb, np.ndarray):
                        embeddings.append(emb.tolist())
                    else:
                        embeddings.append(emb)
                # embeddings = [request.embeddings[i] for i in indices]
                store_ids = [request.store_ids[i] for i in indices]
                camera_ids = [request.camera_ids[i] for i in indices]
                timestamps = [request.timestamps[i] for i in indices]
                
                data = [track_ids, embeddings, store_ids, camera_ids, timestamps]
                
                # Execute insert operation
                store_result = await self.collection_manager.execute_operation(
                    connection_alias=connection_alias,
                    shard_id=shard_id,
                    operation="insert",
                    data=data
                )
                
                # Flush to ensure data is committed
                await self.collection_manager.execute_operation(
                    connection_alias=connection_alias,
                    shard_id=shard_id,
                    operation="flush"
                )
                
                results[store_id] = {
                    "shard_id": shard_id,
                    "insert_count": len(indices)
                }
                
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
        try:
            connection_alias, shard_id = await self._get_connection_for_store(request.store_id)
            
            # Prepare search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            # Prepare filter expression
            expr_parts = [f"store_id == {request.store_id}"]
            if request.camera_filter is not None:
                expr_parts.append(f"camera_id == {request.camera_filter}")
            expr = " && ".join(expr_parts)
            
            # Prepare query vector
            query_embedding = np.array([request.embedding])
            
            # Execute search operation
            results = await self.collection_manager.execute_operation(
                connection_alias=connection_alias,
                shard_id=shard_id,
                operation="search",
                data=query_embedding,
                anns_field="embedding",
                param=search_params,
                limit=request.top_k,
                expr=expr,
                output_fields=["track_id", "store_id", "camera_id", "timestamp"]
            )
            
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
            
            return {
                "success": True,
                "shard_id": shard_id,
                "matches": matches,
                "total_matches": len(matches)
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

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
                logger.info(f"Found unassigned match: track_id={best_track_id}, distance={best_distance}")
                return best_track_id, best_distance
            
            logger.info(f"No suitable unassigned match found below threshold {distance_threshold}")
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
            flush_result = await self.collection_manager.execute_operation(
                connection_alias=connection_alias,
                shard_id=shard_id,
                operation="flush"
            )
            logger.debug(f"Result from flush operation: {flush_result}")
            
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
                output_fields=["embedding", "timestamp", "camera_id"],
                limit=1000
            )
            
            # Process features
            features = []
            for row in results:
                feature = {
                    "embedding": row["embedding"],
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
        try:
            connection_alias, shard_id = await self._get_connection_for_store(store_id)
            
            # Build query expression
            expr = f"store_id == {store_id}"
            logger.info(f"Querying all embeddings for store_id={store_id} (limit={limit})...")
            
            # Execute query with pagination
            if limit <= 10000:
                results = await self.collection_manager.execute_operation(
                    connection_alias=connection_alias,
                    shard_id=shard_id,
                    operation="query",
                    expr=expr,
                    output_fields=["track_id", "embedding"],
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
                
            logger.info(f"Retrieved features for {len(feature_map)} unique track_ids from shard {shard_id}")
            
            return {
                "success": True,
                "shard_id": shard_id,
                "track_count": len(feature_map),
                "features": dict(feature_map)
            }
            
        except Exception as e:
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
            logger.info(f"Found {len(track_ids)} unique track_ids for store_id={store_id} on shard {shard_id}")
            
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
    await router.start()

@app.on_event("shutdown")
async def shutdown():
    """Cleanup router on application shutdown"""
    await router.stop()

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

@app.post("/delete")
async def delete_track(request: DeleteRequest):
    """Delete a track from the database"""
    return await router.delete_track(request)

@app.get("/topology")
async def get_topology():
    """Get the current sharding topology"""
    return await router.get_topology()

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
        query_embedding=request.embedding,
        store_id=request.store_id,
        top_k=request.top_k
    )
    return {"results": results}

@app.get("/track/allfeatures/{store_id}")
async def get_all_track_features_for_tracker(store_id: int, limit: int = 100000):
    """Get all track features in tracker-compatible format"""
    features = await router.get_all_track_features_for_tracker(store_id, limit)
    # Convert numpy arrays to lists for JSON serialization
    result = {}
    for track_id, embeddings in features.items():
        result[track_id] = [e.tolist() for e in embeddings]
    return result
