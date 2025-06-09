import asyncio
import time
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class StoreCacheManager:
    """
    Manages a shared cache of track features for all cameras in a store.
    Thread-safe and optimized for concurrent access.
    """
    
    def __init__(self, store_id: int, milvus_client, cache_config: dict = None):
        self.store_id = store_id
        self.milvus_client = milvus_client
        
        # Cache configuration
        self.config = cache_config or {
            'time_interval': 3.0,      # Refresh every 3 seconds
            'stale_threshold': 15.0,   # Force refresh after 15 seconds
            'max_cache_size': 5000,    # Maximum tracks to cache
            'features_per_track': 10,  # Recent features per track
        }
        
        # Cache storage
        self._track_features = {}  # {track_id: [features]}
        self._feature_arrays = {}  # {track_id: numpy array}
        
        # Cache state
        self._last_refresh_time = 0
        self._cache_valid = False
        self._refresh_lock = asyncio.Lock()
        self._read_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'refresh_count': 0,
            'total_queries': 0,
            'cache_hits': 0,
            'last_refresh_duration': 0
        }
    
    async def ensure_cache_fresh(self):
        """
        Ensure cache is fresh. Multiple cameras can call this concurrently.
        Only one refresh will happen due to the lock.
        """
        current_time = time.time()
        time_since_refresh = current_time - self._last_refresh_time
        
        # Quick check without lock
        if (self._cache_valid and 
            time_since_refresh < self.config['time_interval']):
            return
        
        # Need to refresh - acquire lock
        async with self._refresh_lock:
            # Double-check after acquiring lock (another thread might have refreshed)
            time_since_refresh = time.time() - self._last_refresh_time
            if (self._cache_valid and 
                time_since_refresh < self.config['time_interval']):
                return
            
            # Perform refresh
            await self._refresh_cache()
    
    async def _refresh_cache(self):
        """
        Actually refresh the cache from Milvus.
        Called with lock held.
        """
        start_time = time.time()
        
        try:
            logger.info(f"[StoreCache-{self.store_id}] Refreshing cache...")
            
            # Fetch all track features for this store
            all_features = await self.milvus_client.get_all_track_features(
                self.store_id,
                limit=self.config['max_cache_size']
            )
            
            # Build new cache structures
            new_track_features = {}
            new_feature_arrays = {}
            
            for track_id, features in all_features.items():
                if features:
                    # Keep only recent features
                    recent_features = features[:self.config['features_per_track']]
                    new_track_features[track_id] = recent_features
                    
                    # Pre-compute numpy array for fast search
                    if len(recent_features) > 0:
                        new_feature_arrays[track_id] = np.array(recent_features)
            
            # Atomic swap of cache data
            self._track_features = new_track_features
            self._feature_arrays = new_feature_arrays
            self._last_refresh_time = time.time()
            self._cache_valid = True
            
            # Update stats
            duration = time.time() - start_time
            self.stats['refresh_count'] += 1
            self.stats['last_refresh_duration'] = duration
            
            logger.info(f"[StoreCache-{self.store_id}] Refreshed {len(new_track_features)} tracks "
                       f"with {sum(len(f) for f in new_track_features.values())} features "
                       f"in {duration*1000:.1f}ms")
            
        except Exception as e:
            logger.error(f"[StoreCache-{self.store_id}] Failed to refresh cache: {e}")
            # Keep existing cache on failure
    
    async def search_in_cache(self, feature: np.ndarray, exclude_ids: set = None, 
                            threshold: float = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar features in the cache.
        Returns list of (track_id, distance) tuples.
        """
        exclude_ids = exclude_ids or set()
        results = []
        
        # Ensure cache is fresh
        await self.ensure_cache_fresh()
        
        # Quick read with minimal locking
        async with self._read_lock:
            feature_arrays_snapshot = self._feature_arrays.copy()
        
        # Perform search outside the lock
        if feature_arrays_snapshot:
            feature_array = np.array(feature).reshape(1, -1)
            
            for track_id, cached_features in feature_arrays_snapshot.items():
                if track_id in exclude_ids:
                    continue
                
                # Vectorized distance calculation
                distances = np.linalg.norm(cached_features - feature_array, axis=1)
                min_distance = np.min(distances)
                
                if threshold is None or min_distance < threshold:
                    results.append((track_id, min_distance))
            
            # Sort by distance and limit results
            results.sort(key=lambda x: x[1])
            results = results[:top_k]
        
        # Update stats
        self.stats['total_queries'] += 1
        if results:
            self.stats['cache_hits'] += 1
        
        return results
    
    async def batch_search_in_cache(self, features: List[np.ndarray], exclude_ids: set = None,
                                  threshold: float = None, top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """
        Batch search for multiple features at once.
        Returns list of results for each query feature.
        """
        exclude_ids = exclude_ids or set()
        all_results = []
        
        # Ensure cache is fresh (only one refresh for entire batch)
        await self.ensure_cache_fresh()
        
        # Get snapshot of feature arrays
        async with self._read_lock:
            feature_arrays_snapshot = self._feature_arrays.copy()
        
        # Process each query feature
        for query_feature in features:
            results = []
            
            if feature_arrays_snapshot:
                feature_array = np.array(query_feature).reshape(1, -1)
                
                for track_id, cached_features in feature_arrays_snapshot.items():
                    if track_id in exclude_ids:
                        continue
                    
                    # Vectorized distance calculation
                    distances = np.linalg.norm(cached_features - feature_array, axis=1)
                    min_distance = np.min(distances)
                    
                    if threshold is None or min_distance < threshold:
                        results.append((track_id, min_distance))
                
                # Sort by distance and limit
                results.sort(key=lambda x: x[1])
                results = results[:top_k]
            
            all_results.append(results)
        
        # Update stats
        self.stats['total_queries'] += len(features)
        self.stats['cache_hits'] += sum(1 for r in all_results if r)
        
        return all_results
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        hit_rate = (self.stats['cache_hits'] / max(1, self.stats['total_queries'])) * 100
        cache_age = time.time() - self._last_refresh_time
        
        return {
            'store_id': self.store_id,
            'cache_size': len(self._track_features),
            'total_features': sum(len(f) for f in self._track_features.values()),
            'cache_age_seconds': cache_age,
            'refresh_count': self.stats['refresh_count'],
            'hit_rate_percent': hit_rate,
            'last_refresh_ms': self.stats['last_refresh_duration'] * 1000
        }
