#!/usr/bin/env python3
"""
Stress Testing Script for Retail Analytics System
Tests: Camera simulation → Frame buffering → Feature extraction → Store-wise batch insertion

This script simulates multiple cameras across multiple stores, generates realistic feature vectors,
and stress tests the batch insertion pipeline to the Milvus router.
"""

import asyncio
import time
import logging
import argparse
import json
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import uuid
import aiohttp
from datetime import datetime
import threading
import statistics

# Import existing components from the codebase
from robust_frame_buffer import RobustFrameBuffer
from milvus_router_client import AsyncMilvusRouterClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("stress_test")

@dataclass
class StressTestConfig:
    """Configuration for stress testing parameters"""
    num_stores: int = 1
    cameras_per_store: int = 10
    fps_per_camera: float = 5.0
    test_duration_seconds: int = 120  # 2 minutes
    batch_size_threshold: int = 50
    time_threshold_seconds: float = 2.0
    router_url: str = "http://localhost:8000"
    max_concurrent_requests: int = 300
    
    # Frame buffer configuration
    buffer_max_total_size: int = 300
    buffer_max_size_per_camera: int = buffer_max_total_size // (num_stores * cameras_per_store) 
    
    # Feature vector configuration
    embedding_dim: int = 768
    features_per_frame: int = 2  # 2 detected people per frame

@dataclass
class PerformanceMetrics:
    """Tracks performance metrics during stress testing"""
    total_frames_generated: int = 0
    total_features_generated: int = 0
    total_batch_requests_sent: int = 0
    total_features_inserted: int = 0
    total_insertion_errors: int = 0
    
    # Timing metrics
    batch_insertion_times: List[float] = field(default_factory=list)
    buffer_utilization_samples: List[float] = field(default_factory=list)
    concurrent_requests_samples: List[int] = field(default_factory=list)
    
    # Per-store metrics
    store_batch_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    store_feature_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Per-camera metrics (new for per-camera batching)
    camera_batch_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    camera_feature_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    camera_error_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    def add_batch_timing(self, duration: float):
        """Add batch insertion timing"""
        self.batch_insertion_times.append(duration)
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics"""
        if not self.batch_insertion_times:
            return {"avg": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}
        
        times = sorted(self.batch_insertion_times)
        return {
            "avg": statistics.mean(times),
            "min": min(times),
            "max": max(times),
            "p95": times[int(len(times) * 0.95)] if len(times) > 20 else max(times),
            "p99": times[int(len(times) * 0.99)] if len(times) > 100 else max(times)
        }

class CameraSimulator:
    """Simulates a single camera generating frames"""
    
    def __init__(self, camera_id: int, store_id: int, fps: float):
        self.camera_id = camera_id
        self.store_id = store_id
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.frame_count = 0
        self.running = False
        
    def generate_frame(self) -> np.ndarray:
        """Generate a dummy frame (we don't actually process it)"""
        # Create a small dummy frame to minimize memory usage
        return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    def create_frame_metadata(self) -> Dict:
        """Create metadata for the frame"""
        return {
            "store_id": self.store_id,
            "camera_id": self.camera_id,
            "frame_id": f"{self.camera_id}_{self.frame_count}",
            "timestamp": datetime.utcnow().isoformat(),
            "queued_time": time.time()
        }
    
    async def start_simulation(self, frame_buffer: RobustFrameBuffer, duration_seconds: int):
        """Start generating frames for the specified duration"""
        self.running = True
        start_time = time.time()
        
        logger.info(f"Starting camera {self.camera_id} simulation at {self.fps} FPS for {duration_seconds}s")
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                # Generate frame and metadata
                frame = self.generate_frame()
                metadata = self.create_frame_metadata()
                
                # Add to buffer
                await frame_buffer.add_frame(frame, metadata)
                
                self.frame_count += 1
                
                # Sleep to maintain FPS
                await asyncio.sleep(self.frame_interval)
                
            except Exception as e:
                logger.error(f"Error in camera {self.camera_id} simulation: {e}")
                break
        
        self.running = False
        logger.info(f"Camera {self.camera_id} simulation completed. Generated {self.frame_count} frames")

class FeatureVectorGenerator:
    """Generates realistic feature vectors for stress testing"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
    
    def generate_realistic_embedding(self) -> List[float]:
        """Generate realistic person embeddings"""
        base_pattern = np.random.normal(0, 0.3, self.embedding_dim)
        base_pattern[:100] = np.random.normal(0.5, 0.2, 100)
        base_pattern[100:200] = np.random.normal(-0.3, 0.2, 100)
        norm = np.linalg.norm(base_pattern)
        if norm > 0:
            base_pattern = base_pattern / norm
        return base_pattern.astype(np.float32).tolist()
    
    def generate_features_for_frame(self, metadata: Dict, num_features: int = 2) -> List[Dict]:
        """Generate feature vectors for a frame (simulating detected people)"""
        features = []
        current_time = int(time.time())
        
        for i in range(num_features):
            # Generate a unique track_id (simulating person tracking)
            track_id = hash(f"{metadata['camera_id']}_{metadata['frame_id']}_{i}") % 1000000
            
            feature_data = {
                "track_id": abs(track_id),
                "feature_vector": self.generate_realistic_embedding(),
                "store_id": metadata["store_id"],
                "camera_id": metadata["camera_id"],
                "timestamp": current_time
            }
            features.append(feature_data)
        
        return features

class CameraBatchAccumulator:
    """Manages batch accumulation and insertion for a single camera (like production)"""
    
    def __init__(self, camera_id: int, store_id: int, config: StressTestConfig, 
                 router_client: AsyncMilvusRouterClient, metrics: PerformanceMetrics):
        self.camera_id = camera_id
        self.store_id = store_id
        self.config = config
        self.router_client = router_client
        self.metrics = metrics
        
        # Per-camera feature batch (like production tracker)
        self.feature_batch = {
            'track_ids': [],
            'embeddings': [],
            'store_ids': [],
            'camera_ids': [],
            'timestamps': []
        }
        self.last_batch_time = time.time()
        self.batch_size = config.batch_size_threshold  # Smaller per-camera batches
        self.time_threshold = config.time_threshold_seconds
        
        # Individual camera stats
        self.camera_batches_sent = 0
        self.camera_features_sent = 0
        self.camera_errors = 0
    
    def add_features(self, features: List[Dict]):
        """Add features to this camera's batch"""
        for feature in features:
            self.feature_batch['track_ids'].append(feature["track_id"])
            self.feature_batch['embeddings'].append(feature["feature_vector"])
            self.feature_batch['store_ids'].append(feature["store_id"])
            self.feature_batch['camera_ids'].append(feature["camera_id"])
            self.feature_batch['timestamps'].append(feature["timestamp"])
            
            self.metrics.total_features_generated += 1
            self.metrics.store_feature_counts[self.store_id] += 1
    
    def should_flush(self) -> bool:
        """Check if this camera's batch should be flushed"""
        current_time = time.time()
        batch_size = len(self.feature_batch['track_ids'])
        time_since_last = current_time - self.last_batch_time
        
        return (batch_size >= self.batch_size or 
                (batch_size > 0 and time_since_last >= self.time_threshold))
    
    async def flush_if_ready(self, semaphore: asyncio.Semaphore) -> bool:
        """Flush this camera's batch if ready"""
        if not self.should_flush():
            return False
            
        await self._send_batch(semaphore)
        return True
    
    async def force_flush(self, semaphore: asyncio.Semaphore):
        """Force flush this camera's batch (for cleanup)"""
        if len(self.feature_batch['track_ids']) > 0:
            await self._send_batch(semaphore)
    
    async def _send_batch(self, semaphore: asyncio.Semaphore):
        """Send this camera's batch to the router"""
        if not self.feature_batch['track_ids']:
            return
            
        async with semaphore:
            batch_size = len(self.feature_batch['track_ids'])
            start_time = time.time()
            
            try:
                # Send batch insertion request
                result = await self.router_client.insert_embeddings_batch(
                    track_ids=self.feature_batch['track_ids'].copy(),
                    embeddings=self.feature_batch['embeddings'].copy(),
                    store_ids=self.feature_batch['store_ids'].copy(),
                    camera_ids=self.feature_batch['camera_ids'].copy(),
                    timestamps=self.feature_batch['timestamps'].copy()
                )
                
                duration = time.time() - start_time
                self.metrics.add_batch_timing(duration)
                
                if result and result > 0:
                    self.metrics.total_features_inserted += result
                    self.metrics.total_batch_requests_sent += 1
                    self.metrics.store_batch_counts[self.store_id] += 1
                    
                    # Update camera-specific stats
                    self.camera_batches_sent += 1
                    self.camera_features_sent += result
                    
                    logger.info(f"Camera {self.camera_id} batch sent: {batch_size} features in {duration:.3f}s")
                else:
                    self.metrics.total_insertion_errors += 1
                    self.camera_errors += 1
                    logger.error(f"Camera {self.camera_id} batch insertion failed")
                
            except Exception as e:
                self.metrics.total_insertion_errors += 1
                self.camera_errors += 1
                logger.error(f"Error sending batch for camera {self.camera_id}: {e}")
            finally:
                # Reset batch
                self.feature_batch = {
                    'track_ids': [],
                    'embeddings': [],
                    'store_ids': [],
                    'camera_ids': [],
                    'timestamps': []
                }
                self.last_batch_time = time.time()

class BatchAccumulator:
    """Coordinates multiple camera batch accumulators (per-camera batching strategy)"""
    
    def __init__(self, config: StressTestConfig, router_clients: Dict[int, AsyncMilvusRouterClient], metrics: PerformanceMetrics):
        self.config = config
        self.router_clients = router_clients
        self.metrics = metrics
        
        # Create per-camera batch accumulators
        self.camera_accumulators: Dict[int, CameraBatchAccumulator] = {}
        
        # Concurrency control (shared across all cameras)
        self.active_requests = 0
        self.request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Background task for time-based flushing
        self.flush_task = None
        self.running = False
    
    def register_camera(self, camera_id: int, store_id: int):
        """Register a camera for batch processing"""
        if camera_id not in self.camera_accumulators:
            router_client = self.router_clients[store_id]
            self.camera_accumulators[camera_id] = CameraBatchAccumulator(
                camera_id, store_id, self.config, router_client, self.metrics
            )
            logger.info(f"Registered camera {camera_id} for store {store_id}")
    
    def add_features(self, camera_id: int, features: List[Dict]):
        """Add features to the appropriate camera's batch"""
        if camera_id in self.camera_accumulators:
            self.camera_accumulators[camera_id].add_features(features)
        else:
            logger.error(f"Camera {camera_id} not registered for batching")
    
    
    async def check_and_send_batches(self):
        """Check all cameras and flush ready batches"""
        tasks = []
        
        for camera_id, accumulator in self.camera_accumulators.items():
            if accumulator.should_flush():
                # Create task for this camera's batch
                task = asyncio.create_task(
                    self._flush_camera_with_tracking(camera_id, accumulator)
                )
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _flush_camera_with_tracking(self, camera_id: int, accumulator: CameraBatchAccumulator):
        """Flush a camera's batch while tracking active requests"""
        self.active_requests += 1
        self.metrics.concurrent_requests_samples.append(self.active_requests)
        
        try:
            await accumulator.flush_if_ready(self.request_semaphore)
        finally:
            self.active_requests -= 1
    
    async def start_time_based_flushing(self):
        """Start background task for time-based batch flushing"""
        self.running = True
        
        while self.running:
            try:
                await self.check_and_send_batches()
                await asyncio.sleep(0.2)  # Check every 200ms for more responsive per-camera flushing
            except Exception as e:
                logger.error(f"Error in time-based flushing: {e}")
                await asyncio.sleep(1)
    
    async def flush_all_remaining(self):
        """Flush all remaining batches from all cameras"""
        self.running = False
        
        if self.flush_task and not self.flush_task.done():
            self.flush_task.cancel()
        
        # Send all remaining batches from each camera
        tasks = []
        for camera_id, accumulator in self.camera_accumulators.items():
            task = asyncio.create_task(accumulator.force_flush(self.request_semaphore))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_camera_stats(self) -> Dict[int, Dict]:
        """Get per-camera statistics"""
        stats = {}
        for camera_id, accumulator in self.camera_accumulators.items():
            stats[camera_id] = {
                'batches_sent': accumulator.camera_batches_sent,
                'features_sent': accumulator.camera_features_sent,
                'errors': accumulator.camera_errors,
                'current_batch_size': len(accumulator.feature_batch['track_ids']),
                'store_id': accumulator.store_id
            }
        return stats

class StressTestOrchestrator:
    """Main orchestrator for the stress test"""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.frame_buffer = None
        self.router_client = None
        self.feature_generator = FeatureVectorGenerator(config.embedding_dim)
        self.batch_accumulator = None
        self.cameras = []
        
        # Monitoring
        self.monitor_task = None
        self.router_clients = {}
    
    async def setup(self):
        """Setup all components for stress testing"""
        logger.info("Setting up stress test components...")
        
        # Initialize frame buffer
        self.frame_buffer = RobustFrameBuffer(
            max_size_per_camera=self.config.buffer_max_size_per_camera,
            max_total_size=self.config.buffer_max_total_size,
            timeout_seconds=5.0,
            drop_strategy='smart',
            auto_adjust=True
        )
        await self.frame_buffer.start_monitors()
        
        for store_id in range(1, self.config.num_stores + 1):
            self.router_clients[store_id] = AsyncMilvusRouterClient(
                router_url=self.config.router_url,
                store_id=store_id,  # Each client gets its own store_id
                connection_timeout=15,
                embedding_dim=self.config.embedding_dim,
                batch_size=self.config.batch_size_threshold
            )

        # # Initialize router client
        # self.router_client = AsyncMilvusRouterClient(
        #     router_url=self.config.router_url,
        #     store_id=1,  # Default store for client initialization
        #     connection_timeout=15,
        #     embedding_dim=self.config.embedding_dim,
        #     batch_size=self.config.batch_size_threshold
        # )
        
        # Initialize batch accumulator
        self.batch_accumulator = BatchAccumulator(self.config, self.router_clients, self.metrics)
        # self.batch_accumulator = BatchAccumulator(self.config, self.router_client, self.metrics)
        
        # Create camera simulators and register them
        for store_id in range(1, self.config.num_stores + 1):
            for cam_idx in range(self.config.cameras_per_store):
                camera_id = store_id * 100 + cam_idx  # Unique camera ID
                camera = CameraSimulator(camera_id, store_id, self.config.fps_per_camera)
                self.cameras.append(camera)
                
                # Register camera with batch accumulator
                self.batch_accumulator.register_camera(camera_id, store_id)
        
        logger.info(f"Setup complete: {len(self.cameras)} cameras across {self.config.num_stores} stores")
        logger.info(f"Using per-camera batching strategy with {self.config.batch_size_threshold // self.config.cameras_per_store} features per camera batch")
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        self.monitor_task = asyncio.create_task(self._monitor_performance())
    
    async def _monitor_performance(self):
        """Monitor performance metrics periodically"""
        while True:
            try:
                buffer_status = self.frame_buffer.get_buffer_status()
                utilization = buffer_status.get('utilization_percent', 0)
                self.metrics.buffer_utilization_samples.append(utilization)
                
                # Log current status
                logger.info(f"Buffer utilization: {utilization:.1f}%, "
                           f"Features generated: {self.metrics.total_features_generated}, "
                           f"Batches sent: {self.metrics.total_batch_requests_sent}, "
                           f"Active requests: {self.batch_accumulator.active_requests}")
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _process_frames(self):
        """Process frames from buffer and generate features"""
        processed_frames = 0
        
        while True:
            try:
                # Get batch of frames from buffer
                batch = await self.frame_buffer.get_next_batch(
                    max_batch_size=32,
                    strategy='fair'
                )
                
                if not batch:
                    await asyncio.sleep(0.1)
                    continue
                
                # Generate features for each frame
                all_features = []
                for frame, metadata in batch:
                    features = self.feature_generator.generate_features_for_frame(
                        metadata, self.config.features_per_frame
                    )
                    
                    # Add features to the appropriate camera's batch
                    if features:
                        camera_id = metadata['camera_id']
                        self.batch_accumulator.add_features(camera_id, features)
                    
                    processed_frames += 1
                
                self.metrics.total_frames_generated = processed_frames
                
            except Exception as e:
                logger.error(f"Error processing frames: {e}")
                await asyncio.sleep(1)
    
    async def run_stress_test(self):
        """Run the complete stress test"""
        logger.info(f"Starting stress test for {self.config.test_duration_seconds} seconds...")
        
        # Start all components
        tasks = []
        
        # Start camera simulations
        for camera in self.cameras:
            task = asyncio.create_task(
                camera.start_simulation(self.frame_buffer, self.config.test_duration_seconds)
            )
            tasks.append(task)
        
        # Start frame processing
        frame_processor_task = asyncio.create_task(self._process_frames())
        tasks.append(frame_processor_task)
        
        # Start batch accumulator
        batch_task = asyncio.create_task(self.batch_accumulator.start_time_based_flushing())
        tasks.append(batch_task)
        
        # Start monitoring
        await self.start_monitoring()
        
        try:
            # Wait for test duration
            await asyncio.sleep(self.config.test_duration_seconds)
            
            logger.info("Test duration completed, stopping components...")
            
            # Stop cameras (they should stop automatically)
            # Cancel frame processor
            frame_processor_task.cancel()
            
            # Flush remaining batches
            await self.batch_accumulator.flush_all_remaining()
            
            # Wait a bit for final batches to complete
            await asyncio.sleep(5)
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        finally:
            # Cancel all tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            if self.monitor_task:
                self.monitor_task.cancel()
            
            # Cleanup
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.frame_buffer:
            await self.frame_buffer.stop()
        
        for store_id, client in self.router_clients.items():
            await client.close()
    
    def print_results(self):
        """Print comprehensive test results"""
        print("\n" + "="*80)
        print("STRESS TEST RESULTS")
        print("="*80)
        
        # Basic metrics
        print(f"Test Duration: {self.config.test_duration_seconds} seconds")
        print(f"Stores: {self.config.num_stores}")
        print(f"Cameras per Store: {self.config.cameras_per_store}")
        print(f"Total Cameras: {len(self.cameras)}")
        print(f"FPS per Camera: {self.config.fps_per_camera}")
        print(f"Expected Frames: {len(self.cameras) * self.config.fps_per_camera * self.config.test_duration_seconds:.0f}")
        
        print("\n" + "-"*40 + " FRAME PROCESSING " + "-"*40)
        print(f"Frames Generated: {self.metrics.total_frames_generated}")
        print(f"Features Generated: {self.metrics.total_features_generated}")
        print(f"Features per Frame: {self.metrics.total_features_generated / max(1, self.metrics.total_frames_generated):.2f}")
        
        print("\n" + "-"*40 + " BATCH INSERTION " + "-"*40)
        print(f"Batch Requests Sent: {self.metrics.total_batch_requests_sent}")
        print(f"Features Inserted: {self.metrics.total_features_inserted}")
        print(f"Insertion Errors: {self.metrics.total_insertion_errors}")
        print(f"Success Rate: {(self.metrics.total_features_inserted / max(1, self.metrics.total_features_generated)) * 100:.2f}%")
        
        # Timing statistics
        timing_stats = self.metrics.get_timing_stats()
        print("\n" + "-"*40 + " TIMING STATS " + "-"*40)
        print(f"Avg Batch Insert Time: {timing_stats['avg']:.3f}s")
        print(f"Min Batch Insert Time: {timing_stats['min']:.3f}s")
        print(f"Max Batch Insert Time: {timing_stats['max']:.3f}s")
        print(f"P95 Batch Insert Time: {timing_stats['p95']:.3f}s")
        print(f"P99 Batch Insert Time: {timing_stats['p99']:.3f}s")
        
        # Throughput
        if self.metrics.batch_insertion_times:
            total_insert_time = sum(self.metrics.batch_insertion_times)
            avg_features_per_batch = self.metrics.total_features_inserted / max(1, self.metrics.total_batch_requests_sent)
            throughput = self.metrics.total_features_inserted / max(1, total_insert_time)
            print(f"Avg Features per Batch: {avg_features_per_batch:.1f}")
            print(f"Feature Insertion Throughput: {throughput:.1f} features/second")
        
        # Concurrency
        if self.metrics.concurrent_requests_samples:
            avg_concurrency = statistics.mean(self.metrics.concurrent_requests_samples)
            max_concurrency = max(self.metrics.concurrent_requests_samples)
            print(f"Avg Concurrent Requests: {avg_concurrency:.2f}")
            print(f"Max Concurrent Requests: {max_concurrency}")
        
        # Buffer utilization
        if self.metrics.buffer_utilization_samples:
            avg_buffer_util = statistics.mean(self.metrics.buffer_utilization_samples)
            max_buffer_util = max(self.metrics.buffer_utilization_samples)
            print(f"Avg Buffer Utilization: {avg_buffer_util:.1f}%")
            print(f"Max Buffer Utilization: {max_buffer_util:.1f}%")
        
        # Per-store breakdown
        print("\n" + "-"*40 + " PER-STORE STATS " + "-"*40)
        for store_id in range(1, self.config.num_stores + 1):
            features = self.metrics.store_feature_counts[store_id]
            batches = self.metrics.store_batch_counts[store_id]
            print(f"Store {store_id}: {features} features, {batches} batches")
        
        # Per-camera breakdown (new for per-camera batching)
        print("\n" + "-"*40 + " PER-CAMERA STATS " + "-"*40)
        camera_stats = self.batch_accumulator.get_camera_stats()
        for camera_id in sorted(camera_stats.keys()):
            stats = camera_stats[camera_id]
            print(f"Camera {camera_id} (Store {stats['store_id']}): "
                  f"{stats['features_sent']} features, "
                  f"{stats['batches_sent']} batches, "
                  f"{stats['errors']} errors, "
                  f"pending: {stats['current_batch_size']}")
        
        print("="*80)

async def main():
    parser = argparse.ArgumentParser(description='Stress test the retail analytics system')
    parser.add_argument('--stores', type=int, default=20, help='Number of stores to simulate')
    parser.add_argument('--cameras-per-store', type=int, default=5, help='Number of cameras per store')
    parser.add_argument('--fps', type=float, default=5.0, help='FPS per camera')
    parser.add_argument('--duration', type=int, default=120, help='Test duration in seconds')
    parser.add_argument('--batch-size', type=int, default=70, help='Batch size threshold for insertion')
    parser.add_argument('--batch-time', type=float, default=10.0, help='Time threshold for batch insertion')
    parser.add_argument('--router-url', type=str, default='http://172.210.56.49:8000', help='Router URL')
    parser.add_argument('--max-concurrent', type=int, default=300, help='Max concurrent requests')
    parser.add_argument('--buffer-size', type=int, default=300, help='Total buffer size')
    parser.add_argument('--batching-strategy', type=str, choices=['per-camera', 'per-store'], 
                        default='per-camera', help='Batching strategy: per-camera (like production) or per-store (aggregated)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = StressTestConfig(
        num_stores=args.stores,
        cameras_per_store=args.cameras_per_store,
        fps_per_camera=args.fps,
        test_duration_seconds=args.duration,
        batch_size_threshold=args.batch_size,
        time_threshold_seconds=args.batch_time,
        router_url=args.router_url,
        max_concurrent_requests=args.max_concurrent,
        buffer_max_total_size=args.buffer_size
    )
    
    logger.info(f"Using {args.batching_strategy} batching strategy")
    
    # Create and run stress test
    if args.batching_strategy == 'per-camera':
        orchestrator = StressTestOrchestrator(config)
    else:
        # Keep the old per-store implementation available as an option
        orchestrator = StoreWiseStressTestOrchestrator(config)  # You could implement this variant
    
    try:
        await orchestrator.setup()
        await orchestrator.run_stress_test()
    finally:
        orchestrator.print_results()

if __name__ == "__main__":
    asyncio.run(main())
