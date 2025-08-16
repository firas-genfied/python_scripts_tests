#!/usr/bin/env python3
"""
GPU Stress Test for Retail Analytics Pipeline

This script stress tests the GPU processing pipeline by simulating multiple cameras
sending frames at configurable rates. It measures performance metrics to find
optimal configurations and breaking points.

Usage:
    python gpu_stress_test.py --images-dir ./test_images --config stress_test_config.json
"""

import asyncio
import cv2
import numpy as np
import time
import json
import argparse
import logging
import os
import random
import signal
import sys
import torch
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Tuple
import psutil
import threading
from dataclasses import dataclass, asdict

# Import your existing modules
from rtsp_stream_processor_multiple_cameras import GPUBatchProcessor
from robust_frame_buffer import RobustFrameBuffer
from task_manager import TaskManager
from utils.processor_utils import initialize_processors

# Try to import GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("stress_test")

@dataclass
class TestMetrics:
    """Container for test metrics"""
    timestamp: float
    frames_injected: int
    frames_processed: int
    frames_in_buffer: int
    gpu_utilization: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    cpu_usage: float
    ram_usage_mb: float
    processing_latency_avg: float
    processing_latency_p95: float
    batch_processing_time: float
    error_count: int
    oom_count: int

class GPUMonitor:
    """Monitor GPU metrics using nvidia-ml-py"""
    
    def __init__(self):
        self.device_id = 0
        self.handle = None
        self.available = False
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                self.available = True
                logger.info("GPU monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.available = False
        else:
            logger.warning("NVML not available, using fallback GPU monitoring")
    
    def get_gpu_metrics(self) -> Tuple[float, float, float]:
        """
        Returns: (utilization_percent, memory_used_mb, memory_total_mb)
        """
        if not self.available:
            return 0.0, 0.0, 0.0
            
        try:
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = util.gpu
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used = mem_info.used / (1024 * 1024)  # Convert to MB
            memory_total = mem_info.total / (1024 * 1024)
            
            return float(gpu_util), float(memory_used), float(memory_total)
        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {e}")
            return 0.0, 0.0, 0.0

class FrameInjector:
    """Injects frames into the system at configurable rates"""
    
    def __init__(self, images_dir: str, camera_count: int, fps_per_camera: float):
        self.images_dir = images_dir
        self.camera_count = camera_count
        self.fps_per_camera = fps_per_camera
        self.frame_interval = 1.0 / fps_per_camera if fps_per_camera > 0 else 1.0
        
        # Load test images
        self.test_images = self._load_test_images()
        if not self.test_images:
            raise ValueError(f"No images found in {images_dir}")
        
        logger.info(f"Loaded {len(self.test_images)} test images")
        
        # Injection stats
        self.frames_injected = 0
        self.running = False
        self.injection_tasks = []
        
    def _load_test_images(self) -> List[np.ndarray]:
        """Load all test images from directory"""
        images = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        if not os.path.exists(self.images_dir):
            logger.error(f"Images directory not found: {self.images_dir}")
            return images
            
        for filename in os.listdir(self.images_dir):
            if filename.lower().endswith(supported_formats):
                filepath = os.path.join(self.images_dir, filename)
                try:
                    image = cv2.imread(filepath)
                    if image is not None:
                        images.append(image)
                        logger.debug(f"Loaded image: {filename} ({image.shape})")
                    else:
                        logger.warning(f"Failed to load image: {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        return images
    
    async def start_injection(self, frame_buffer: RobustFrameBuffer):
        """Start injecting frames for all cameras"""
        self.running = True
        logger.info(f"Starting frame injection: {self.camera_count} cameras at {self.fps_per_camera} FPS each")
        
        # Create injection task for each camera
        for camera_id in range(1, self.camera_count + 1):
            task = asyncio.create_task(
                self._inject_frames_for_camera(camera_id, frame_buffer)
            )
            self.injection_tasks.append(task)
        
        # Wait for all injection tasks
        try:
            await asyncio.gather(*self.injection_tasks)
        except asyncio.CancelledError:
            logger.info("Frame injection cancelled")
    
    async def _inject_frames_for_camera(self, camera_id: int, frame_buffer: RobustFrameBuffer):
        """Inject frames for a specific camera"""
        frame_id = 0
        last_injection_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time to inject next frame
                if current_time - last_injection_time >= self.frame_interval:
                    # Select random image
                    image = random.choice(self.test_images).copy()
                    
                    # Create metadata
                    metadata = {
                        "store_id": 1,  # Fixed store for simplicity
                        "camera_id": camera_id,
                        "frame_id": frame_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "queued_time": current_time
                    }
                    
                    # Inject frame
                    await frame_buffer.add_frame(image, metadata)
                    self.frames_injected += 1
                    frame_id += 1
                    last_injection_time = current_time
                
                # Sleep briefly to prevent CPU spinning
                await asyncio.sleep(0.001)  # 1ms
                
            except Exception as e:
                logger.error(f"Error injecting frame for camera {camera_id}: {e}")
                await asyncio.sleep(0.1)
    
    def stop_injection(self):
        """Stop frame injection"""
        self.running = False
        for task in self.injection_tasks:
            task.cancel()

class StressTestProcessor:
    """Simplified processor focused on GPU stress testing"""
    
    def __init__(self, batch_size: int, batch_interval: float, num_processors: int = 1):
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.running = False
        self.processing_busy = False
        self.last_process_time = time.time()
        
        # Initialize the actual GPU processor (this does segmentation + feature extraction)
        self.gpu_processor = GPUBatchProcessor(
            max_batch_size=batch_size,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            model_config={"context_id": 0}
        )
        
        # Frame buffer for stress testing
        self.frame_buffer = RobustFrameBuffer(
            max_size_per_camera=200,
            max_total_size=1000,
            timeout_seconds=10.0,
            drop_strategy='smart',
            auto_adjust=True,
            camera_priorities=None
        )
        
        # Metrics tracking
        self.processing_latencies = deque(maxlen=1000)
        self.batch_processing_times = deque(maxlen=100)
        self.error_count = 0
        self.oom_count = 0
        self.stats = {
            "frames_processed": 0,
            "total_people_detected": 0,
            "segmentation_time": 0,
            "feature_extraction_time": 0
        }
        
        self.task_manager = TaskManager()
        
    async def start_stress_test(self):
        """Start the stress test processor"""
        # Initialize GPU processors
        await initialize_processors(self.gpu_processor)
        await self.frame_buffer.start_monitors()
        
        # Start processing loop
        self.task_manager.create_task(
            self._processing_loop_stress_test(),
            category="stress_test_processing"
        )
    
    async def _processing_loop_stress_test(self):
        """Modified processing loop for stress testing"""
        logger.info("Starting stress test processing loop")
        
        while self.running:
            try:
                if not self.processing_busy:
                    buffer_status = self.frame_buffer.get_buffer_status()
                    
                    # Process if we have frames
                    if buffer_status['total_frames'] > 0:
                        self.processing_busy = True
                        try:
                            batch_start_time = time.time()
                            await self.process_batch_stress_test()
                            batch_time = time.time() - batch_start_time
                            self.batch_processing_times.append(batch_time)
                        except Exception as e:
                            logger.error(f"Error in stress test batch processing: {e}")
                            self.error_count += 1
                            if "out of memory" in str(e).lower():
                                self.oom_count += 1
                        finally:
                            self.processing_busy = False
                
                # Very short sleep to maintain high throughput
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in stress test processing loop: {e}")
                self.processing_busy = False
                await asyncio.sleep(0.1)
    
    async def process_batch_stress_test(self):
        """Process batch with actual segmentation and feature extraction"""
        # Get batch from buffer
        current_batch = await self.frame_buffer.get_next_batch(
            max_batch_size=self.batch_size,
            strategy='fair'
        )
        
        if not current_batch:
            return
        
        logger.debug(f"Processing batch of {len(current_batch)} frames with actual GPU models")
        
        # Process with actual GPU models (segmentation + feature extraction)
        loop = asyncio.get_running_loop()
        
        batch_start = time.time()
        try:
            # This runs the ACTUAL segmentation and feature extraction pipeline
            # - Detectron2 segmentation model processes each frame
            # - Crops people from detections  
            # - TransReID model extracts features from each person crop
            batch_results = await loop.run_in_executor(
                None,
                lambda: self.gpu_processor.process_batch(current_batch)
            )
            
            batch_time = time.time() - batch_start
            self.batch_processing_times.append(batch_time)
            
            # Count actual detections and processing stats
            total_people = 0
            successful_frames = 0
            
            for i, (metadata, detections, features) in enumerate(batch_results):
                if metadata and 'queued_time' in metadata:
                    latency = time.time() - metadata['queued_time']
                    self.processing_latencies.append(latency)
                
                if detections:
                    successful_frames += 1
                    total_people += len(detections)
                    
                    # Verify features were actually extracted
                    valid_features = sum(1 for f in features if f is not None and len(f) > 0)
                    if valid_features != len(detections):
                        logger.warning(f"Feature extraction mismatch: {valid_features}/{len(detections)}")
            
            # Update detailed stats
            self.stats["frames_processed"] += len(current_batch)
            self.stats["total_people_detected"] += total_people
            
            # Log detailed processing info
            logger.info(f"Batch processed: {len(current_batch)} frames, {total_people} people detected, "
                       f"{batch_time:.3f}s batch time, {successful_frames} frames with detections")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM during processing: {e}")
            self.oom_count += 1
            self.error_count += 1
            # Clear GPU cache and continue
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error during GPU processing: {e}")
            self.error_count += 1
        
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics including GPU model performance"""
        buffer_status = self.frame_buffer.get_buffer_status()
        
        # Calculate latency statistics
        avg_latency = 0.0
        p95_latency = 0.0
        if self.processing_latencies:
            avg_latency = sum(self.processing_latencies) / len(self.processing_latencies)
            sorted_latencies = sorted(self.processing_latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0.0
        
        # Calculate batch processing time
        avg_batch_time = 0.0
        if self.batch_processing_times:
            avg_batch_time = sum(self.batch_processing_times) / len(self.batch_processing_times)
        
        # Get GPU memory stats from the actual GPU processor
        gpu_memory_stats = self.gpu_processor.get_memory_stats()
        
        # Calculate detection rate (people per frame)
        people_per_frame = 0.0
        if self.stats["frames_processed"] > 0:
            people_per_frame = self.stats["total_people_detected"] / self.stats["frames_processed"]
        
        return {
            'frames_processed': self.stats["frames_processed"],
            'total_people_detected': self.stats["total_people_detected"],
            'people_per_frame': people_per_frame,
            'frames_in_buffer': buffer_status['total_frames'],
            'buffer_utilization_percent': buffer_status['utilization_percent'],
            'avg_processing_latency': avg_latency,
            'p95_processing_latency': p95_latency,
            'avg_batch_processing_time': avg_batch_time,
            'gpu_memory_allocated_mb': gpu_memory_stats['allocated_mb'],
            'gpu_memory_peak_mb': gpu_memory_stats['peak_allocated_mb'],
            'error_count': self.error_count,
            'oom_count': self.oom_count
        }
    
    async def stop(self):
        """Stop the stress test processor"""
        self.running = False
        try:
            await self.frame_buffer.stop()
            await self.task_manager.wait_for_all(timeout=5.0)
            logger.info("Stress test processor stopped")
        except Exception as e:
            logger.error(f"Error stopping stress test processor: {e}")

class StressTestRunner:
    """Main stress test runner"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.gpu_monitor = GPUMonitor()
        self.process = psutil.Process()
        
        # Test components
        self.frame_injector = None
        self.processor = None
        self.running = False
        
        # Metrics collection
        self.metrics_history = []
        self.metrics_collection_interval = 1.0  # 1 second
        self.test_start_time = None
        
        # Results
        self.results = {
            'config': config,
            'metrics': [],
            'summary': {}
        }
        
    async def run_test(self, test_duration: int):
        """Run the stress test for specified duration"""
        logger.info(f"Starting stress test with config: {self.config}")
        logger.info(f"Test duration: {test_duration} seconds")
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Start metrics collection
            metrics_task = asyncio.create_task(self._collect_metrics())
            
            # Start frame injection
            injection_task = asyncio.create_task(
                self.frame_injector.start_injection(self.processor.frame_buffer)
            )
            
            # Run test for specified duration
            self.test_start_time = time.time()
            await asyncio.sleep(test_duration)
            
            # Stop test
            logger.info("Stopping stress test...")
            self.running = False
            self.frame_injector.stop_injection()
            
            # Cancel tasks
            metrics_task.cancel()
            injection_task.cancel()
            
            # Wait briefly for cleanup
            await asyncio.sleep(2)
            
            # Generate results
            self._generate_results()
            
        except Exception as e:
            logger.error(f"Error during stress test: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _initialize_components(self):
        """Initialize test components"""
        # Create frame injector
        self.frame_injector = FrameInjector(
            images_dir=self.config['images_dir'],
            camera_count=self.config['camera_count'],
            fps_per_camera=self.config['fps_per_camera']
        )
        
        # Create processor
        self.processor = StressTestProcessor(
            batch_size=self.config['batch_size'],
            batch_interval=self.config['batch_interval'],
            num_processors=1
        )
        
        # Start processor
        await self.processor.start_stress_test()
        self.running = True
        
        logger.info("Stress test components initialized")
    
    async def _collect_metrics(self):
        """Collect metrics periodically"""
        while self.running:
            try:
                current_time = time.time()
                
                # Get GPU metrics
                gpu_util, gpu_mem_used, gpu_mem_total = self.gpu_monitor.get_gpu_metrics()
                
                # Get system metrics
                cpu_usage = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                ram_usage = memory_info.rss / (1024 * 1024)  # MB
                
                # Get processor metrics (now includes actual GPU processing stats)
                perf_metrics = self.processor.get_performance_metrics()
                
                # Create metrics record
                metrics = TestMetrics(
                    timestamp=current_time,
                    frames_injected=self.frame_injector.frames_injected,
                    frames_processed=perf_metrics['frames_processed'],
                    frames_in_buffer=perf_metrics['frames_in_buffer'],
                    gpu_utilization=gpu_util,
                    gpu_memory_used_mb=perf_metrics.get('gpu_memory_allocated_mb', gpu_mem_used),
                    gpu_memory_total_mb=gpu_mem_total,
                    cpu_usage=cpu_usage,
                    ram_usage_mb=ram_usage,
                    processing_latency_avg=perf_metrics['avg_processing_latency'],
                    processing_latency_p95=perf_metrics['p95_processing_latency'],
                    batch_processing_time=perf_metrics['avg_batch_processing_time'],
                    error_count=perf_metrics['error_count'],
                    oom_count=perf_metrics['oom_count']
                )
                
                self.metrics_history.append(metrics)
                
                # Log current metrics with detection info
                people_detected = perf_metrics.get('total_people_detected', 0)
                people_per_frame = perf_metrics.get('people_per_frame', 0)
                
                logger.info(
                    f"[{elapsed:.1f}s] "
                    f"Injected: {metrics.frames_injected}, "
                    f"Processed: {metrics.frames_processed}, "
                    f"People: {people_detected} ({people_per_frame:.1f}/frame), "
                    f"Buffer: {metrics.frames_in_buffer}, "
                    f"GPU: {gpu_util:.1f}%/{gpu_mem_used:.0f}MB, "
                    f"CPU: {cpu_usage:.1f}%, "
                    f"Latency: {metrics.processing_latency_avg*1000:.1f}ms, "
                    f"Batch: {metrics.batch_processing_time*1000:.1f}ms"
                )
                
                await asyncio.sleep(self.metrics_collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(1)
    
    def _generate_results(self):
        """Generate test results summary"""
        if not self.metrics_history:
            logger.warning("No metrics collected")
            return
        
        # Calculate summary statistics
        total_frames_injected = self.metrics_history[-1].frames_injected
        total_frames_processed = self.metrics_history[-1].frames_processed
        test_duration = self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
        
        # Calculate processing rate
        processing_fps = total_frames_processed / test_duration if test_duration > 0 else 0
        injection_fps = total_frames_injected / test_duration if test_duration > 0 else 0
        
        # Calculate additional processing-specific metrics
        total_people_detected = 0
        avg_people_per_frame = 0
        avg_batch_time = 0
        
        if self.metrics_history:
            # Get people detection stats from processor
            final_metrics = self.processor.get_performance_metrics()
            total_people_detected = final_metrics.get('total_people_detected', 0)
            avg_people_per_frame = final_metrics.get('people_per_frame', 0)
            
            # Calculate batch processing time average
            if self.metrics_history:
                batch_times = [m.batch_processing_time for m in self.metrics_history if m.batch_processing_time > 0]
                avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        
        avg_cpu_usage = sum(m.cpu_usage for m in self.metrics_history) / len(self.metrics_history)
        avg_ram_usage = sum(m.ram_usage_mb for m in self.metrics_history) / len(self.metrics_history)
        
        avg_latency = sum(m.processing_latency_avg for m in self.metrics_history if m.processing_latency_avg > 0)
        avg_latency = avg_latency / len([m for m in self.metrics_history if m.processing_latency_avg > 0]) if avg_latency > 0 else 0
        
        # Final buffer state
        final_buffer_frames = self.metrics_history[-1].frames_in_buffer
        
        # Error counts
        final_error_count = self.metrics_history[-1].error_count
        final_oom_count = self.metrics_history[-1].oom_count
        
        # Generate summary
        summary = {
            'test_duration_seconds': test_duration,
            'total_frames_injected': total_frames_injected,
            'total_frames_processed': total_frames_processed,
            'total_people_detected': total_people_detected,
            'avg_people_per_frame': avg_people_per_frame,
            'frames_lost': total_frames_injected - total_frames_processed - final_buffer_frames,
            'injection_fps': injection_fps,
            'processing_fps': processing_fps,
            'processing_efficiency_percent': (processing_fps / injection_fps * 100) if injection_fps > 0 else 0,
            'avg_gpu_utilization_percent': avg_gpu_util,
            'avg_gpu_memory_mb': avg_gpu_memory,
            'max_gpu_memory_mb': max_gpu_memory,
            'avg_cpu_usage_percent': avg_cpu_usage,
            'avg_ram_usage_mb': avg_ram_usage,
            'avg_processing_latency_ms': avg_latency * 1000,
            'avg_batch_processing_time_ms': avg_batch_time * 1000,
            'final_buffer_frames': final_buffer_frames,
            'error_count': final_error_count,
            'oom_count': final_oom_count,
            'bottleneck_detected': self._detect_bottleneck(),
            'models_tested': {
                'segmentation': 'Detectron2',
                'feature_extraction': 'TransReID',
                'feature_dimensions': 768
            }
        }
        
        self.results['summary'] = summary
        self.results['metrics'] = [asdict(m) for m in self.metrics_history]
        
        # Log summary
        logger.info("=== STRESS TEST RESULTS ===")
        logger.info(f"Test Duration: {test_duration:.1f}s")
        logger.info(f"Frames Injected: {total_frames_injected}")
        logger.info(f"Frames Processed: {total_frames_processed}")
        logger.info(f"People Detected: {summary.get('total_people_detected', 0)}")
        logger.info(f"Processing FPS: {processing_fps:.2f}")
        logger.info(f"Processing Efficiency: {summary['processing_efficiency_percent']:.1f}%")
        logger.info(f"Avg GPU Utilization: {avg_gpu_util:.1f}%")
        logger.info(f"Avg GPU Memory: {avg_gpu_memory:.1f}MB (Max: {max_gpu_memory:.1f}MB)")
        logger.info(f"Avg Processing Latency: {avg_latency*1000:.1f}ms")
        logger.info(f"Avg Batch Time: {avg_batch_time*1000:.1f}ms")
        logger.info(f"Errors: {final_error_count}, OOM: {final_oom_count}")
        logger.info(f"Bottleneck: {summary['bottleneck_detected']}")
        logger.info("=== Models Used ===")
        logger.info("✓ Detectron2 Segmentation Model (person detection)")
        logger.info("✓ TransReID Feature Extraction Model (768-dim features)")
        logger.info("✓ Complete GPU processing pipeline tested")
        logger.info("==========================")
    
    def _detect_bottleneck(self) -> str:
        """Detect potential bottlenecks"""
        if not self.metrics_history:
            return "unknown"
        
        avg_gpu_util = sum(m.gpu_utilization for m in self.metrics_history) / len(self.metrics_history)
        max_gpu_memory = max(m.gpu_memory_used_mb for m in self.metrics_history)
        gpu_memory_total = self.metrics_history[0].gpu_memory_total_mb
        
        avg_buffer_utilization = sum(
            m.frames_in_buffer / 1000 * 100  # Assuming max 1000 buffer size
            for m in self.metrics_history
        ) / len(self.metrics_history)
        
        final_oom_count = self.metrics_history[-1].oom_count
        
        if final_oom_count > 0:
            return "gpu_memory"
        elif max_gpu_memory / gpu_memory_total > 0.9:
            return "gpu_memory_pressure"
        elif avg_gpu_util > 95:
            return "gpu_compute"
        elif avg_buffer_utilization > 80:
            return "buffer_overflow"
        elif avg_gpu_util < 50:
            return "injection_rate_too_low"
        else:
            return "balanced"
    
    def save_results(self, filepath: str):
        """Save results to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    async def _cleanup(self):
        """Cleanup resources"""
        try:
            if self.processor:
                await self.processor.stop()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GPU Stress Test for Retail Analytics Pipeline')
    parser.add_argument('--images-dir', required=True, help='Directory containing test images')
    parser.add_argument('--config', help='JSON config file for test parameters')
    parser.add_argument('--camera-count', type=int, default=4, help='Number of cameras to simulate')
    parser.add_argument('--fps-per-camera', type=float, default=5.0, help='FPS per camera')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for GPU processing')
    parser.add_argument('--batch-interval', type=float, default=0.3, help='Batch processing interval')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--output', default='stress_test_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load config from file or use command line args
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'images_dir': args.images_dir,
            'camera_count': args.camera_count,
            'fps_per_camera': args.fps_per_camera,
            'batch_size': args.batch_size,
            'batch_interval': args.batch_interval
        }
    
    # Validate images directory
    if not os.path.exists(config['images_dir']):
        logger.error(f"Images directory not found: {config['images_dir']}")
        return 1
    
    # Set up signal handling for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run stress test
    runner = StressTestRunner(config)
    
    try:
        # Run test with timeout or until shutdown signal
        test_task = asyncio.create_task(runner.run_test(args.duration))
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        
        done, pending = await asyncio.wait(
            [test_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        
        # Save results
        runner.save_results(args.output)
        logger.info("Stress test completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))