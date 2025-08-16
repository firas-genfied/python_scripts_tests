#!/usr/bin/env python3
"""
Production Scenario Replication Test

This script replicates the exact conditions that cause 68+ second frame processing:
1. Multiple cameras processing simultaneously 
2. High connection pool contention
3. Concurrent batch searches + individual feature retrievals + inserts
4. Continuous perpetual operation (like production)

Run until manually stopped with Ctrl+C
"""

import asyncio
import time
import numpy as np
import random
import logging
import signal
import sys
from typing import List, Dict
from collections import defaultdict
from simple_milvus_router_client import SimpleMilvusRouterClient
import os

# Configure logging to see the pain
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"production_replication_{int(time.time())}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("production-replication")

class ProductionLoadSimulator:
    """Simulates the exact production load pattern that causes 68+ second delays"""
    
    def __init__(self):
        self.router_url = os.environ.get("MILVUS_ROUTER_URL", "http://localhost:8000")
        self.store_id = int(os.environ.get("STORE_ID", "5"))
        
        # Production-like configuration
        self.num_cameras = 2  # From your system_config.json
        self.processing_fps = 5  # From system_config.json  
        self.frame_interval = 1.0 / self.processing_fps  # 0.2 seconds between frames
        
        # Tracker-like operation patterns
        self.detections_per_frame = 3  # Average detections per frame
        self.batch_search_size = 5  # Unmatched detections requiring search
        self.feature_retrievals_per_frame = 3  # New track IDs needing features
        self.batch_insert_size = 8  # Features to insert per frame
        
        # Control flags
        self.running = True
        self.stats = {
            'frames_processed': 0,
            'total_operations': 0,
            'slow_frames': 0,
            'connection_errors': 0,
            'timeouts': 0,
            'start_time': time.time()
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Production Load Simulator Initialized")
        logger.info(f"Router: {self.router_url}")
        logger.info(f"Store ID: {self.store_id}")
        logger.info(f"Cameras: {self.num_cameras}")
        logger.info(f"Processing FPS: {self.processing_fps}")
        logger.info(f"Expected operations per second: ~{self.num_cameras * self.processing_fps * 4}")
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def generate_realistic_embedding(self) -> List[float]:
        """Generate embeddings that look like real person embeddings (not random noise)"""
        # Real person embeddings have certain patterns - not pure random
        base_pattern = np.random.normal(0, 0.3, 768)  # Lower variance than pure random
        
        # Add some structure that real embeddings have
        base_pattern[:100] = np.random.normal(0.5, 0.2, 100)  # Some features tend positive
        base_pattern[100:200] = np.random.normal(-0.3, 0.2, 100)  # Some tend negative
        
        # Normalize like real embeddings
        norm = np.linalg.norm(base_pattern)
        if norm > 0:
            base_pattern = base_pattern / norm
            
        return base_pattern.astype(np.float32).tolist()
    
    async def simulate_tracker_frame_processing(self, camera_id: int, frame_id: int) -> Dict:
        """
        Simulate the exact sequence of operations your tracker performs per frame
        This is what causes the 68+ second delays in production
        """
        frame_start_time = time.time()
        frame_stats = {
            'camera_id': camera_id,
            'frame_id': frame_id,
            'operations': [],
            'total_duration': 0,
            'milvus_operations': 0,
            'connection_errors': 0,
            'timeouts': 0
        }
        
        logger.info(f"🎬 Camera {camera_id} Frame {frame_id} - Starting tracker processing...")
        
        try:
            # Create client (each camera/frame uses its own client like your tracker)
            async with SimpleMilvusRouterClient(self.router_url, self.store_id) as client:
                
                # === STEP 1: Batch Search for Unmatched Detections ===
                # This is the 68-second operation from your logs
                search_start = time.time()
                search_embeddings = [self.generate_realistic_embedding() for _ in range(self.batch_search_size)]
                
                try:
                    batch_search_results = await client.search_embeddings_batch(
                        embeddings_list=search_embeddings,
                        top_k=20,  # Same as your tracker
                        store_id=self.store_id
                    )
                    search_duration = time.time() - search_start
                    frame_stats['operations'].append(('batch_search', search_duration, True))
                    frame_stats['milvus_operations'] += 1
                    
                    if search_duration > 10:
                        logger.warning(f"🚨 SLOW BATCH SEARCH: {search_duration:.1f}s (expected <1s)")
                    
                except asyncio.TimeoutError:
                    search_duration = time.time() - search_start
                    frame_stats['operations'].append(('batch_search', search_duration, False))
                    frame_stats['timeouts'] += 1
                    logger.error(f"❌ Batch search TIMEOUT after {search_duration:.1f}s")
                except Exception as e:
                    search_duration = time.time() - search_start
                    frame_stats['operations'].append(('batch_search', search_duration, False))
                    frame_stats['connection_errors'] += 1
                    logger.error(f"❌ Batch search ERROR: {e}")
                
                # === STEP 2: Individual Feature Retrievals ===
                # These are the 15+ second operations from your logs
                for i in range(self.feature_retrievals_per_frame):
                    retrieval_start = time.time()
                    
                    # Use realistic track IDs (your tracker queries existing track IDs)
                    track_id = random.randint(1, 50)  # Realistic range
                    
                    try:
                        features = await client.get_features_by_track_id(
                            track_id=track_id,
                            store_id=self.store_id
                        )
                        retrieval_duration = time.time() - retrieval_start
                        frame_stats['operations'].append(('get_features', retrieval_duration, True))
                        frame_stats['milvus_operations'] += 1
                        
                        if retrieval_duration > 5:
                            logger.warning(f"🚨 SLOW FEATURE RETRIEVAL: {retrieval_duration:.1f}s for track {track_id}")
                            
                    except asyncio.TimeoutError:
                        retrieval_duration = time.time() - retrieval_start
                        frame_stats['operations'].append(('get_features', retrieval_duration, False))
                        frame_stats['timeouts'] += 1
                        logger.error(f"❌ Feature retrieval TIMEOUT for track {track_id} after {retrieval_duration:.1f}s")
                    except Exception as e:
                        retrieval_duration = time.time() - retrieval_start
                        frame_stats['operations'].append(('get_features', retrieval_duration, False))
                        frame_stats['connection_errors'] += 1
                        logger.error(f"❌ Feature retrieval ERROR for track {track_id}: {e}")
                
                # === STEP 3: Batch Insert New Features ===
                insert_start = time.time()
                
                # Generate batch insert data (like your tracker does)
                track_ids = [random.randint(100, 1000) for _ in range(self.batch_insert_size)]
                embeddings = [self.generate_realistic_embedding() for _ in range(self.batch_insert_size)]
                store_ids = [self.store_id] * self.batch_insert_size
                camera_ids = [camera_id] * self.batch_insert_size
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
                    frame_stats['operations'].append(('batch_insert', insert_duration, True))
                    frame_stats['milvus_operations'] += 1
                    
                    if insert_duration > 5:
                        logger.warning(f"🚨 SLOW BATCH INSERT: {insert_duration:.1f}s")
                        
                except asyncio.TimeoutError:
                    insert_duration = time.time() - insert_start
                    frame_stats['operations'].append(('batch_insert', insert_duration, False))
                    frame_stats['timeouts'] += 1
                    logger.error(f"❌ Batch insert TIMEOUT after {insert_duration:.1f}s")
                except Exception as e:
                    insert_duration = time.time() - insert_start
                    frame_stats['operations'].append(('batch_insert', insert_duration, False))
                    frame_stats['connection_errors'] += 1
                    logger.error(f"❌ Batch insert ERROR: {e}")
        
        except Exception as e:
            logger.error(f"❌ Frame processing FAILED for camera {camera_id}: {e}")
            frame_stats['connection_errors'] += 1
        
        # Calculate frame statistics
        frame_stats['total_duration'] = time.time() - frame_start_time
        
        # Log frame summary (like your tracker does)
        milvus_time = sum(op[1] for op in frame_stats['operations'])
        non_milvus_time = frame_stats['total_duration'] - milvus_time
        
        if frame_stats['total_duration'] > 30:
            logger.warning(f"🔥 === SLOW FRAME DETECTED ===")
            logger.warning(f"Camera {camera_id} Frame {frame_id}: {frame_stats['total_duration']:.1f}s total")
            logger.warning(f"Milvus time: {milvus_time:.1f}s ({frame_stats['milvus_operations']} ops)")
            logger.warning(f"Non-Milvus time: {non_milvus_time:.1f}s")
            logger.warning(f"Operations: {frame_stats['operations']}")
            logger.warning(f"Errors: {frame_stats['connection_errors']} connections, {frame_stats['timeouts']} timeouts")
            logger.warning(f"================================")
            
            if frame_stats['total_duration'] > 60:
                logger.error(f"🎯 REPRODUCED THE 68+ SECOND ISSUE!")
        
        return frame_stats
    
    async def simulate_camera_stream(self, camera_id: int):
        """Simulate continuous frame processing for one camera"""
        frame_id = 0
        
        logger.info(f"📹 Camera {camera_id} stream started (target: {self.processing_fps} FPS)")
        
        while self.running:
            try:
                frame_id += 1
                
                # Process frame (this is where the 68+ second delays happen)
                frame_stats = await self.simulate_tracker_frame_processing(camera_id, frame_id)
                
                # Update global statistics
                self.stats['frames_processed'] += 1
                self.stats['total_operations'] += frame_stats['milvus_operations']
                self.stats['connection_errors'] += frame_stats['connection_errors']
                self.stats['timeouts'] += frame_stats['timeouts']
                
                if frame_stats['total_duration'] > 30:
                    self.stats['slow_frames'] += 1
                
                # Maintain target FPS (if frame processing was fast enough)
                elapsed = frame_stats['total_duration']
                if elapsed < self.frame_interval:
                    sleep_time = self.frame_interval - elapsed
                    await asyncio.sleep(sleep_time)
                else:
                    # Frame took longer than target interval - we're falling behind!
                    logger.warning(f"📹 Camera {camera_id} falling behind! Frame took {elapsed:.1f}s (target: {self.frame_interval:.1f}s)")
                
            except Exception as e:
                logger.error(f"❌ Camera {camera_id} stream error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
    
    def print_statistics(self):
        """Print current performance statistics"""
        runtime = time.time() - self.stats['start_time']
        frames_per_sec = self.stats['frames_processed'] / runtime if runtime > 0 else 0
        ops_per_sec = self.stats['total_operations'] / runtime if runtime > 0 else 0
        
        logger.info(f"📊 === PERFORMANCE STATISTICS ===")
        logger.info(f"Runtime: {runtime:.1f} seconds")
        logger.info(f"Frames processed: {self.stats['frames_processed']}")
        logger.info(f"Processing rate: {frames_per_sec:.2f} frames/sec (target: {self.num_cameras * self.processing_fps})")
        logger.info(f"Total Milvus operations: {self.stats['total_operations']}")
        logger.info(f"Operations per second: {ops_per_sec:.2f}")
        logger.info(f"Slow frames (>30s): {self.stats['slow_frames']}")
        logger.info(f"Connection errors: {self.stats['connection_errors']}")
        logger.info(f"Timeouts: {self.stats['timeouts']}")
        
        if self.stats['slow_frames'] > 0:
            logger.warning(f"🎯 ISSUE REPRODUCED: {self.stats['slow_frames']} slow frames detected!")
        else:
            logger.info("✅ No slow frames detected yet")
        
        logger.info(f"================================")
    
    async def run_production_simulation(self):
        """Run the complete production simulation"""
        logger.info(f"🚀 Starting Production Load Simulation")
        logger.info(f"This will replicate the conditions causing 68+ second frame processing")
        logger.info(f"Press Ctrl+C to stop gracefully")
        logger.info(f"=" * 80)
        
        # Start camera streams (concurrent like in production)
        camera_tasks = []
        for camera_id in range(1, self.num_cameras + 1):
            task = asyncio.create_task(self.simulate_camera_stream(camera_id))
            camera_tasks.append(task)
        
        # Statistics reporting task
        async def stats_reporter():
            while self.running:
                await asyncio.sleep(30)  # Report every 30 seconds
                if self.running:
                    self.print_statistics()
        
        stats_task = asyncio.create_task(stats_reporter())
        
        try:
            # Run until interrupted
            await asyncio.gather(*camera_tasks, stats_task)
        except asyncio.CancelledError:
            logger.info("Simulation cancelled")
        finally:
            # Final statistics
            logger.info(f"🏁 Simulation stopped")
            self.print_statistics()
            
            # Cancel all tasks
            for task in camera_tasks + [stats_task]:
                if not task.done():
                    task.cancel()

async def main():
    """Main entry point"""
    # Validate environment
    if not os.environ.get("MILVUS_ROUTER_URL"):
        logger.error("Please set MILVUS_ROUTER_URL environment variable")
        logger.error("Example: export MILVUS_ROUTER_URL='http://localhost:8000'")
        sys.exit(1)
    
    simulator = ProductionLoadSimulator()
    
    logger.info("Production Scenario Replication Test")
    logger.info("=" * 50)
    logger.info("This test simulates the EXACT conditions that cause")
    logger.info("68+ second frame processing times in your tracker:")
    logger.info("")
    logger.info("1. Multiple cameras processing simultaneously")
    logger.info("2. High connection pool contention") 
    logger.info("3. Concurrent batch searches + feature retrievals + inserts")
    logger.info("4. Continuous perpetual operation")
    logger.info("")
    logger.info("Expected result: Reproduces the 68+ second delays")
    logger.info("=" * 50)
    
    await simulator.run_production_simulation()

if __name__ == "__main__":
    asyncio.run(main())
