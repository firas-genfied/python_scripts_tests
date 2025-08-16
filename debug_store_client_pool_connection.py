#!/usr/bin/env python3
"""
Store Client Pool Simulation - Replicates Production Behavior

This script implements the same client management pattern as your production code:
- Per-store client pools (not per-frame clients)
- Connection reuse across all operations
- Simulates 2 cameras processing frames
- Performs batch insertions without flush

Key differences from original test:
- Clients created once per store and reused
- No per-frame client creation overhead
- Mimics production CameraProcessor + KafkaProcessor pattern
"""

import asyncio
import time
import numpy as np
import random
import logging
import signal
import sys
import os
from typing import List, Dict, Optional
from collections import defaultdict
# from simple_milvus_router_client import SimpleMilvusRouterClient
from milvus_router_client import AsyncMilvusRouterClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"store_client_pool_test_{int(time.time())}.log"),
        logging.StreamHandler()
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("store-client-pool")

class StoreClientManager:
    """Manages Milvus router clients per store (like production KafkaProcessor)"""
    
    def __init__(self, router_url: str):
        self.router_url = router_url
        self.store_clients = {}  # store_id -> SimpleMilvusRouterClient
        self.store_last_access = {}  # Track usage for cleanup
        self.max_stores = 10  # Limit like production
        
        logger.debug(f"Initialized StoreClientManager with router: {router_url}")
    
    async def get_client_for_store(self, store_id: int) -> AsyncMilvusRouterClient:
        """Get or create a client for the given store (reuses existing clients)"""
        
        # Update last access time
        self.store_last_access[store_id] = time.time()
        
        # Return existing client if available
        if store_id in self.store_clients:
            logger.debug(f"Reusing existing client for store {store_id}")
            return self.store_clients[store_id]
        
        # Check if we need to evict old stores (LRU)
        if len(self.store_clients) >= self.max_stores:
            lru_store_id = min(self.store_last_access, key=self.store_last_access.get)
            if lru_store_id != store_id:
                logger.debug(f"Evicting LRU store client: {lru_store_id}")
                await self._close_store_client(lru_store_id)
        
        # Create new client for this store
        logger.debug(f"Creating new client for store {store_id}")
        # client = SimpleMilvusRouterClient(self.router_url, store_id)
        client = AsyncMilvusRouterClient(self.router_url, store_id)
        self.store_clients[store_id] = client
        
        return client
    
    async def _close_store_client(self, store_id: int):
        """Close and remove a store client"""
        if store_id in self.store_clients:
            try:
                client = self.store_clients[store_id]
                await client.close()  # If your client has a close method
                logger.debug(f"Closed client for store {store_id}")
            except Exception as e:
                logger.warning(f"Error closing client for store {store_id}: {e}")
            finally:
                del self.store_clients[store_id]
                if store_id in self.store_last_access:
                    del self.store_last_access[store_id]
    
    async def cleanup_all(self):
        """Close all store clients"""
        for store_id in list(self.store_clients.keys()):
            await self._close_store_client(store_id)
        logger.debug("All store clients closed")

class CameraProcessor:
    """Simulates a single camera processor (like production CameraProcessor)"""
    
    def __init__(self, camera_id: int, store_id: int, client_manager: StoreClientManager):
        self.camera_id = camera_id
        self.store_id = store_id
        self.client_manager = client_manager
        self.frame_count = 0
        
        # Operation patterns (same as production)
        self.batch_search_size = 10
        self.feature_retrievals_per_frame = 1
        self.batch_insert_size = 16
        
        logger.debug(f"Created CameraProcessor for camera {camera_id} in store {store_id}")
    
    def generate_realistic_embedding(self) -> List[float]:
        """Generate realistic person embeddings"""
        base_pattern = np.random.normal(0, 0.3, 768)
        base_pattern[:100] = np.random.normal(0.5, 0.2, 100)
        base_pattern[100:200] = np.random.normal(-0.3, 0.2, 100)
        
        norm = np.linalg.norm(base_pattern)
        if norm > 0:
            base_pattern = base_pattern / norm
            
        return base_pattern.astype(np.float32).tolist()
    
    async def process_frame(self, frame_id: int) -> Dict:
        """
        Process a single frame using the store's shared client
        This is the key difference - reusing the same client, not creating new ones
        """
        self.frame_count += 1
        frame_start = time.time()
        
        # Get the shared client for this store (reuses existing connection)
        client = await self.client_manager.get_client_for_store(self.store_id)
        
        logger.debug(f"🎬 Camera {self.camera_id} Frame {frame_id} - Processing with reused client")
        
        frame_stats = {
            'camera_id': self.camera_id,
            'frame_id': frame_id,
            'operations': [],
            'total_duration': 0,
            'client_reused': True  # Track that we reused the client
        }
        
        try:
            # === STEP 1: Batch Search ===
            search_start = time.time()
            search_embeddings = [self.generate_realistic_embedding() for _ in range(self.batch_search_size)]
            
            try:
                batch_search_results = await client.search_embeddings_batch(
                    embeddings_list=search_embeddings,
                    top_k=20,
                    store_id=self.store_id
                )
                search_duration = time.time() - search_start
                frame_stats['operations'].append(('batch_search', search_duration, True))
                logger.info(f"✅ Batch search: {search_duration:.3f}s for {self.batch_search_size} embeddings")
                
            except Exception as e:
                search_duration = time.time() - search_start
                frame_stats['operations'].append(('batch_search', search_duration, False))
                logger.error(f"❌ Batch search failed: {e}")
            
            # === STEP 2: Feature Retrievals ===
            for i in range(self.feature_retrievals_per_frame):
                retrieval_start = time.time()
                track_id = random.randint(1, 50)
                
                try:
                    features = await client.get_features_by_track_id(
                        track_id=track_id,
                        store_id=self.store_id
                    )
                    retrieval_duration = time.time() - retrieval_start
                    frame_stats['operations'].append(('get_features', retrieval_duration, True))
                    logger.info(f"✅ Feature retrieval {i+1}: {retrieval_duration:.3f}s for track {track_id}")
                    
                except Exception as e:
                    retrieval_duration = time.time() - retrieval_start
                    frame_stats['operations'].append(('get_features', retrieval_duration, False))
                    logger.error(f"❌ Feature retrieval {i+1} failed: {e}")
            
            # === STEP 3: Batch Insert (NO FLUSH) ===
            insert_start = time.time()
            
            track_ids = [random.randint(100, 1000) for _ in range(self.batch_insert_size)]
            embeddings = [self.generate_realistic_embedding() for _ in range(self.batch_insert_size)]
            store_ids = [self.store_id] * self.batch_insert_size
            camera_ids = [self.camera_id] * self.batch_insert_size
            timestamps = [int(time.time() * 1000) + i for i in range(self.batch_insert_size)]
            
            try:
                insert_result = await client.insert_embeddings_batch(
                    track_ids=track_ids,
                    embeddings=embeddings,
                    store_ids=store_ids,
                    camera_ids=camera_ids,
                    timestamps=timestamps
                )
                insert_duration = time.time() - insert_start
                frame_stats['operations'].append(('batch_insert_no_flush', insert_duration, True))
                logger.info(f"✅ Batch insert: {insert_duration:.3f}s for {self.batch_insert_size} embeddings")
                
            except Exception as e:
                insert_duration = time.time() - insert_start
                frame_stats['operations'].append(('batch_insert_no_flush', insert_duration, False))
                logger.error(f"❌ Batch insert failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Frame processing failed: {e}")
        
        frame_stats['total_duration'] = time.time() - frame_start
        
        # Log frame summary
        if frame_stats['total_duration'] > 10:
            logger.warning(f"🚨 Slow frame: Camera {self.camera_id} Frame {frame_id} - {frame_stats['total_duration']:.2f}s")
        else:
            logger.debug(f"✅ Fast frame: Camera {self.camera_id} Frame {frame_id} - {frame_stats['total_duration']:.2f}s")
        
        return frame_stats

class ProductionLikeSimulator:
    """Simulates the production system's client management pattern"""
    
    def __init__(self, connection_pool_size=None, num_cameras=None):
        self.router_url = os.environ.get("MILVUS_ROUTER_URL", "http://localhost:8000")
        self.store_id = int(os.environ.get("STORE_ID", "5"))
        
        # Production-like settings
        self.num_cameras = num_cameras or int(os.environ.get("NUM_CAMERAS", "5"))
        self.processing_fps = 10
        self.frame_interval = 1.0 / self.processing_fps
        
        # Initialize store client manager (like production KafkaProcessor)
        self.client_manager = StoreClientManager(self.router_url)
        self.connection_pool_size = connection_pool_size or int(os.environ.get("CONNECTION_POOL_SIZE", "64"))
        
        # Create camera processors (like production)
        self.camera_processors = {}
        for camera_id in range(1, self.num_cameras + 1):
            self.camera_processors[camera_id] = CameraProcessor(
                camera_id, self.store_id, self.client_manager
            )
        
        # Control and stats
        self.running = True
        self.stats = {
            'frames_processed': 0,
            'total_operations': 0,
            'client_reuses': 0,
            'start_time': time.time()
        }
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.debug(f"🚀 Production-like Simulator initialized")
        logger.debug(f"Router: {self.router_url}")
        logger.debug(f"Store ID: {self.store_id}")
        logger.debug(f"Cameras: {self.num_cameras}")
        logger.debug(f"✅ Using per-store client pools (like production)")

    async def configure_router_pool_size(self):
            """Configure the router's connection pool size before testing"""
            try:
                client = await self.client_manager.get_client_for_store(self.store_id)
                http_client = await client._get_client()
                
                # Call router configuration endpoint (we'll need to add this to router)
                response = await http_client.post(
                    f"{self.router_url}/configure/connection_pool",
                    json={"max_connections_per_shard": self.connection_pool_size}
                )
                
                if response.status_code == 200:
                    logger.debug(f"✅ Router pool size configured to {self.connection_pool_size}")
                    return True
                else:
                    logger.warning(f"⚠️ Failed to configure pool size: {response.status_code}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error configuring pool size: {e}")
                return False
    
    async def collect_pool_metrics(self):
        """Collect detailed pool metrics from router"""
        try:
            client = await self.client_manager.get_client_for_store(self.store_id)
            http_client = await client._get_client()
            
            # Get performance breakdown from router
            response = await http_client.get(f"{self.router_url}/performance/breakdown")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get pool metrics: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting pool metrics: {e}")
            return None
    
    async def print_enhanced_statistics(self):
            """Print statistics including pool performance"""
            # Existing stats calculation...
            runtime = time.time() - self.stats['start_time']
            frames_per_sec = self.stats['frames_processed'] / runtime if runtime > 0 else 0
            
            # NEW: Get pool metrics from router
            pool_metrics = await self.collect_pool_metrics()
            
            logger.info(f"📊 === CONNECTION POOL TEST RESULTS ===")
            logger.info(f"🔧 Pool Configuration: {self.connection_pool_size} connections per shard")
            logger.info(f"📈 Processing Rate: {frames_per_sec:.2f} frames/sec")
            logger.info(f"")
            
            if pool_metrics:
                timing = pool_metrics.get('timing_breakdown', {})
                pool_health = pool_metrics.get('connection_pool_health', {})
                
                logger.info(f"⏱️ Router Performance Breakdown:")
                logger.info(f"  Avg Connection Wait: {timing.get('avg_connection_wait', 0):.3f}s")
                logger.info(f"  Avg Database Time:   {timing.get('avg_database', 0):.3f}s")
                logger.info(f"  Avg Processing Time: {timing.get('avg_processing', 0):.3f}s")
                logger.info(f"")
                logger.info(f"🔗 Connection Pool Health:")
                logger.info(f"  Total Connections: {pool_health.get('total_connections', 0)}")
                logger.info(f"  Avg Wait Time: {pool_health.get('avg_wait_time', 0):.3f}s")
                logger.info(f"  Max Wait Time: {pool_health.get('max_wait_time', 0):.3f}s")
                
                # Identify bottleneck
                bottleneck = pool_metrics.get('bottleneck_analysis', 'unknown')
                logger.info(f"🎯 Primary Bottleneck: {bottleneck}")
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        logger.debug(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def simulate_camera_stream(self, camera_id: int):
        """Simulate continuous frame processing for one camera"""
        frame_id = 0
        processor = self.camera_processors[camera_id]
        
        logger.debug(f"📹 Camera {camera_id} stream started (target: {self.processing_fps} FPS)")
        
        while self.running:
            try:
                frame_id += 1
                
                # Process frame using the shared store client
                frame_stats = await processor.process_frame(frame_id)
                
                # Update global statistics
                self.stats['frames_processed'] += 1
                self.stats['total_operations'] += len(frame_stats['operations'])
                if frame_stats.get('client_reused', False):
                    self.stats['client_reuses'] += 1
                
                # Maintain target FPS
                elapsed = frame_stats['total_duration']
                if elapsed < self.frame_interval:
                    sleep_time = self.frame_interval - elapsed
                    logger.info(f"sleeping for {sleep_time} seconds")
                    await asyncio.sleep(sleep_time)
                # else:
                #     pass
                    # logger.warning(f"📹 Camera {camera_id} falling behind! Frame took {elapsed:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ Camera {camera_id} stream error: {e}")
                await asyncio.sleep(1)

    async def print_enhanced_statistics_for_ete_load(self):
        """Print statistics including pool performance"""
        runtime = time.time() - self.stats['start_time']
        frames_per_sec = self.stats['frames_processed'] / runtime if runtime > 0 else 0
        ops_per_sec = self.stats['total_operations'] / runtime if runtime > 0 else 0
        
        # NEW: Get pool metrics from router
        pool_metrics = await self.collect_pool_metrics()
        
        logger.info(f"📊 === CAMERA SCALING TEST RESULTS ({self.num_cameras} cameras) ===")
        logger.info(f"Runtime: {runtime:.1f} seconds")
        logger.info(f"Processing rate: {frames_per_sec:.2f} frames/sec (target: {self.num_cameras * self.processing_fps})")
        logger.info(f"Total operations: {self.stats['total_operations']} ({ops_per_sec:.2f}/sec)")
        logger.info(f"")
        
        if pool_metrics:
            timing = pool_metrics.get('timing_breakdown', {})
            pool_health = pool_metrics.get('connection_pool_health', {})
            
            logger.info(f"⏱️ Router Performance Breakdown:")
            logger.info(f"  Avg Connection Wait: {timing.get('avg_connection_wait', 0):.3f}s")
            logger.info(f"  Avg Database Time:   {timing.get('avg_database', 0):.3f}s")
            logger.info(f"  Avg Processing Time: {timing.get('avg_processing', 0):.3f}s")
            logger.info(f"")
            logger.info(f"🔗 Connection Pool Health:")
            logger.info(f"  Total Connections: {pool_health.get('total_connections', 0)}")
            logger.info(f"  Avg Wait Time: {pool_health.get('avg_wait_time', 0):.3f}s")
            logger.info(f"  Max Wait Time: {pool_health.get('max_wait_time', 0):.3f}s")
            
            # Identify bottleneck
            bottleneck = pool_metrics.get('bottleneck_analysis', 'unknown')
            logger.info(f"🎯 Primary Bottleneck: {bottleneck}")
        else:
            logger.warning("⚠️ Could not retrieve router metrics")
        
        logger.info(f"================================================")
    
    async def run_simulation(self):
        """Run the complete simulation"""
        logger.info(f"�� Starting Production-Like Client Pool Simulation")
        logger.info(f"")
        logger.info(f"�� KEY PRODUCTION PATTERNS:")
        logger.info(f"  ✅ Per-store client pools (not per-frame)")
        logger.info(f"  ✅ Connection reuse across all operations")
        logger.info(f"  ✅ CameraProcessor pattern with shared clients")
        logger.info(f"  ✅ No manual flush operations")
        logger.info(f"")
        logger.info(f"🔬 EXPECTED RESULTS:")
        logger.info(f"  - No connection overhead per frame")
        logger.info(f"  - Consistent performance across frames")
        logger.info(f"  - High client reuse rate (should be 100%)")
        logger.info(f"")
        logger.info(f"Press Ctrl+C to stop gracefully")
        logger.info(f"=" * 80)
        # Start camera streams (concurrent like production)
        camera_tasks = []
        for camera_id in range(1, self.num_cameras + 1):
            task = asyncio.create_task(self.simulate_camera_stream(camera_id))
            camera_tasks.append(task)
        
        # Statistics reporting task
        async def stats_reporter():
            while self.running:
                await asyncio.sleep(30)  # Report every 30 seconds
                if self.running:
                    await self.print_enhanced_statistics_for_ete_load()
        
        stats_task = asyncio.create_task(stats_reporter())
        
        try:
            # Run until interrupted
            await asyncio.gather(*camera_tasks, stats_task)
        except asyncio.CancelledError:
            logger.debug("Simulation cancelled")
        finally:
            # Final statistics and cleanup
            logger.debug(f"🏁 Simulation stopped")
            await self.print_enhanced_statistics_for_ete_load()
            
            # Cleanup store clients
            await self.client_manager.cleanup_all()
            
            # Cancel tasks
            for task in camera_tasks + [stats_task]:
                if not task.done():
                    task.cancel()

async def test_multiple_pool_sizes():
    """Test different connection pool sizes"""
    pool_sizes = [8, 16, 32, 64, 128]
    test_duration = 120  # 2 minutes per test
    
    results = {}
    
    for pool_size in pool_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TESTING POOL SIZE: {pool_size}")
        logger.info(f"{'='*60}")
        
        # Create simulator with specific pool size
        simulator = ProductionLikeSimulator(connection_pool_size=pool_size)
        
        # Configure router
        await simulator.configure_router_pool_size()
        
        # Run test for fixed duration
        start_time = time.time()
        simulator.running = True
        
        try:
            # Start camera tasks
            camera_tasks = []
            for camera_id in range(1, simulator.num_cameras + 1):
                task = asyncio.create_task(simulator.simulate_camera_stream(camera_id))
                camera_tasks.append(task)
            
            # Run for test duration
            await asyncio.sleep(test_duration)
            simulator.running = False
            
            # Wait for tasks to finish
            for task in camera_tasks:
                task.cancel()
            
            # Collect final metrics
            await simulator.print_enhanced_statistics()
            pool_metrics = await simulator.collect_pool_metrics()
            
            # Store results
            results[pool_size] = {
                'frames_processed': simulator.stats['frames_processed'],
                'runtime': time.time() - start_time,
                'pool_metrics': pool_metrics
            }
            
        except Exception as e:
            logger.error(f"Test failed for pool size {pool_size}: {e}")
        finally:
            # Cleanup
            await simulator.client_manager.cleanup_all()
    
    # Print comparison
    print_pool_size_comparison(results)

def print_pool_size_comparison(results):
    """Print comparison table of all pool sizes tested"""
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 CONNECTION POOL SIZE COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"{'Size':<6} {'FPS':<8} {'ConnWait':<10} {'DBTime':<8} {'Bottleneck':<15}")
    logger.info(f"{'-'*50}")
    
    for pool_size, data in results.items():
        runtime = data['runtime']
        fps = data['frames_processed'] / runtime if runtime > 0 else 0
        
        pool_metrics = data.get('pool_metrics', {})
        timing = pool_metrics.get('timing_breakdown', {})
        
        conn_wait = timing.get('avg_connection_wait', 0)
        db_time = timing.get('avg_database', 0)
        bottleneck = pool_metrics.get('bottleneck_analysis', 'unknown')
        
        logger.debug(f"{pool_size:<6} {fps:<8.2f} {conn_wait:<10.3f} {db_time:<8.3f} {bottleneck:<15}")

async def main():
    """Main entry point"""
    # Validate environment
    if not os.environ.get("MILVUS_ROUTER_URL"):
        logger.error("Please set MILVUS_ROUTER_URL environment variable")
        logger.error("Example: export MILVUS_ROUTER_URL='http://localhost:8000'")
        sys.exit(1)

    if os.environ.get("TEST_POOL_SIZES", "false").lower() == "true":
        await test_multiple_pool_sizes()
    
    else:
        logger.info("Production-Like Client Pool Simulation")
        logger.info("=" * 60)
        logger.info("🎯 PURPOSE:")
        logger.info("  Test the production pattern of per-store client pools")
        logger.info("  instead of creating new clients for each frame")
        logger.info("")
        logger.info("🔄 CLIENT MANAGEMENT:")
        logger.info("  - One SimpleMilvusRouterClient per store")
        logger.info("  - Clients reused across all frames and operations")
        logger.info("  - LRU eviction for inactive stores")
        logger.info("  - Proper cleanup on shutdown")
        logger.info("")
        logger.info("📈 PERFORMANCE BENEFITS:")
        logger.info("  - No connection overhead per frame")
        logger.info("  - Reduced latency and resource usage")
        logger.info("  - Matches production architecture")
        logger.info("=" * 60)
    
    simulator = ProductionLikeSimulator()
    await simulator.configure_router_pool_size()
    await simulator.run_simulation()

if __name__ == "__main__":
    asyncio.run(main())
run_simulation