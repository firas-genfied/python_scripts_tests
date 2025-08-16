class BufferDropMonitor:
    """Monitor frame buffer drops by wrapping add_frame method"""
    def __init__(self, frame_buffer, metrics):
        self.frame_buffer = frame_buffer
        self.metrics = metrics
        self.original_add_frame = frame_buffer.add_frame
        
        # Wrap the add_frame method to detect drops
        frame_buffer.add_frame = self._monitored_add_frame
    
    async def _monitored_add_frame(self, frame, metadata):
        """Wrapped add_frame that detects and logs drops"""
        camera_id = metadata.get('camera_id', 'unknown')
        
        # Get buffer status before attempting to add
        buffer_status = self.frame_buffer.get_buffer_status()
        
        try:
            # Try to add frame
            result = await self.original_add_frame(frame, metadata)
            
            # Check if frame was actually added by comparing buffer status
            new_buffer_status = self.frame_buffer.get_buffer_status()
            
            # If total frames didn't increase, it means frame was dropped
            if new_buffer_status['total_frames'] <= buffer_status['total_frames']:
                # Determine drop reason based on buffer state
                if buffer_status['utilization_percent'] >= 95:
                    reason = f"Buffer full ({buffer_status['utilization_percent']:.1f}%)"
                elif buffer_status['per_camera'].get(camera_id, 0) >= self.frame_buffer.max_size_per_camera * 0.9:
                    reason = f"Camera {camera_id} buffer full"
                else:
                    reason = "Buffer drop (unknown reason)"
                
                self.metrics.record_frame_dropped(camera_id, reason, buffer_status)
            
            return result
            
        except Exception as e:
            # Frame addition failed
            self.metrics.record_frame_dropped(camera_id, f"Add failed: {str(e)}", buffer_status)
            raise



#!/usr/bin/env python3

import asyncio
import cv2
import numpy as np
import os
import time
import logging
import json
import argparse
from pathlib import Path
from collections import defaultdict, deque
import psutil
import torch
from PIL import Image
# Import your existing components
from robust_frame_buffer import RobustFrameBuffer
from rtsp_stream_processor_multiple_cameras import GPUBatchProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

from utils.detection_utils import (
    crop_without_resize, 
    is_entering_store_percent, 
    has_left_store_percent, 
    filter_duplicate_detections, 
    compute_iou
)
from utils.log_utils import log_total_memory
from utils.processor_utils import (
    initialize_processors, 
    set_memory_limit
)

class TimedGPUBatchProcessor(GPUBatchProcessor):
    """GPU processor with detailed timing breakdown"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_timing_breakdown = {}
    
    def process_batch(self, image_batch):
        """Process batch with detailed timing breakdown"""
        batch_start = time.time()
        timing = {}
        
        logger.info(f"Processing batch of {len(image_batch)} images on GPU")
        batch_results = []
        total_people = 0
        
        try:
            # Step 1: Preprocessing (CPU)
            preprocessing_start = time.time()
            batch_inputs = []
            original_sizes = []
            for image, metadata in image_batch:
                original_sizes.append((image.shape[0], image.shape[1]))
                height, width = image.shape[:2]
                transformed_image = self.aug.get_transform(image).apply_image(image)
                transformed_image = torch.as_tensor(transformed_image.astype("float32").transpose(2, 0, 1))
                batch_inputs.append({
                    "image": transformed_image.to(self.device),
                    "height": height,
                    "width": width,
                })
            timing['preprocessing'] = time.time() - preprocessing_start
            
            # Step 2: GPU Segmentation
            gpu_seg_start = time.time()
            with torch.cuda.stream(self.stream):
                with torch.no_grad():
                    batch_outputs = self.seg_model(batch_inputs)
            self.stream.synchronize()
            timing['gpu_seg_time'] = time.time() - gpu_seg_start
            
            # Step 3: Postprocessing (CPU)
            postprocess_start = time.time()
            all_crops = []
            crop_info = []
            
            for i, (outputs, (image, metadata)) in enumerate(zip(batch_outputs, image_batch)):
                instances = outputs["instances"]
                person_indices = (instances.pred_classes == 0).nonzero().flatten()
                if len(person_indices) == 0:
                    batch_results.append((metadata, [], []))
                    continue
                
                # Get person boxes, scores, and masks
                person_boxes = instances.pred_boxes.tensor[person_indices].cpu().numpy()
                person_scores = instances.scores[person_indices].cpu().numpy()
                person_masks = instances.pred_masks[person_indices].cpu().numpy()
                
                # Filter and process (existing logic)
                filtered_boxes, filtered_scores, filtered_masks = filter_duplicate_detections(
                    person_boxes, person_scores, person_masks, image, iou_threshold=0.9
                )
                
                height, width = image.shape[:2]
                line_y = int(height * 0.10)
                green_box = [0, line_y, width, height]
                
                valid_detections = []
                converted_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                for j, bbox in enumerate(filtered_boxes):
                    if self.is_blacklisted(bbox):
                        continue
                    score = filtered_scores[j]
                    mask = filtered_masks[j]   
                    overlap_area, bbox_area = self.calculate_overlap(bbox, green_box)
                    
                    if overlap_area / bbox_area > 0.7:
                        tlwh_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        detection = (tlwh_bbox, score, "person", mask)
                        valid_detections.append(detection)
                        
                        # Prepare crops for feature extraction
                        x, y, w, h = map(int, tlwh_bbox)
                        crop_masked = crop_without_resize(converted_frame, [x, y, x+w, y+h], mask)
                        if crop_masked.size > 0 and w > 0 and h > 0:
                            crop_pil = Image.fromarray(crop_masked)
                            transformed_crop = self.transform(crop_pil).unsqueeze(0)
                            all_crops.append(transformed_crop)
                            crop_info.append((i, len(valid_detections)-1))
                
                batch_results.append((metadata, valid_detections, [None] * len(valid_detections)))
                total_people += len(valid_detections)
            
            timing['postprocessing'] = time.time() - postprocess_start
            
            # Step 4: GPU Feature Extraction
            gpu_feat_start = time.time()
            if all_crops:
                chunk_size = min(self.max_batch_size * 2, len(all_crops), 32)
                for chunk_start in range(0, len(all_crops), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(all_crops))
                    chunk_crops = all_crops[chunk_start:chunk_end]
                    chunk_info = crop_info[chunk_start:chunk_end]
                    
                    batch_tensor = torch.cat(chunk_crops, dim=0).to(self.device)
                    if self.use_half_precision and self.device.type == "cuda":
                        batch_tensor = batch_tensor.half()
                        
                    with torch.no_grad():
                        features = self.extract_features(self.model, batch_tensor).cpu().float().numpy()
                    
                    # Assign features back
                    for feat_idx, (batch_idx, det_idx) in enumerate(chunk_info):
                        if batch_results[batch_idx][2][det_idx] is None:
                            batch_results[batch_idx][2][det_idx] = features[feat_idx].reshape(-1)
            timing['gpu_feat_time'] = time.time() - gpu_feat_start
            
            # Calculate CPU time (everything except GPU operations)
            total_time = time.time() - batch_start
            gpu_time = timing['gpu_seg_time'] + timing['gpu_feat_time']
            timing['cpu_time'] = total_time - gpu_time
            timing['total_time'] = total_time
            
            # Store timing for external access
            self.last_timing_breakdown = timing
            
            # Log detailed timing
            logger.info(f"=== TIMING BREAKDOWN for {len(image_batch)} images ===")
            logger.info(f"Preprocessing (CPU): {timing['preprocessing']*1000:.1f}ms ({timing['preprocessing']/total_time*100:.1f}%)")
            logger.info(f"Segmentation (GPU): {timing['gpu_seg_time']*1000:.1f}ms ({timing['gpu_seg_time']/total_time*100:.1f}%)")
            logger.info(f"Postprocessing (CPU): {timing['postprocessing']*1000:.1f}ms ({timing['postprocessing']/total_time*100:.1f}%)")
            logger.info(f"Feature extraction (GPU): {timing['gpu_feat_time']*1000:.1f}ms ({timing['gpu_feat_time']/total_time*100:.1f}%)")
            logger.info(f"Total CPU time: {timing['cpu_time']*1000:.1f}ms ({timing['cpu_time']/total_time*100:.1f}%)")
            logger.info(f"Total GPU time: {gpu_time*1000:.1f}ms ({gpu_time/total_time*100:.1f}%)")
            logger.info(f"TOTAL: {total_time*1000:.1f}ms")
            
            # Calculate efficiency metrics
            per_image_total = total_time * 1000 / len(image_batch)
            per_image_gpu = gpu_time * 1000 / len(image_batch)
            gpu_utilization = gpu_time / total_time * 100
            
            logger.info(f"Per image: Total={per_image_total:.1f}ms, GPU={per_image_gpu:.1f}ms")
            logger.info(f"GPU utilization: {gpu_utilization:.1f}%")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in timed batch processing: {e}")
            self.last_timing_breakdown = {'error': str(e)}
            return [(metadata, [], []) for metadata, _ in image_batch]

class StressTestConfig:
    """Configuration for stress test parameters"""
    def __init__(self, config_file=None):
        # Default values
        self.num_cameras = 5
        self.fps_per_camera = 5.0
        self.batch_size = 8
        self.test_duration_seconds = 60
        self.images_dir = "test_images"
        self.log_interval_seconds = 5
        
        # Performance thresholds
        self.max_gpu_memory_mb = 8000  # Alert if GPU memory exceeds this
        self.max_processing_latency_ms = 2000  # Alert if processing takes too long
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

class StressTestMetrics:
    """Collects and tracks performance metrics"""
    def __init__(self):
        self.start_time = None
        self.frames_sent = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.processing_times = deque(maxlen=1000)
        self.gpu_memory_usage = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)
        self.ram_usage = deque(maxlen=1000)
        
        # Per-second tracking
        self.last_log_time = 0
        self.last_frames_sent = 0
        self.last_frames_processed = 0
        self.last_frames_dropped = 0
        
        # Drop tracking
        self.drop_events = deque(maxlen=100)  # Store recent drop events
        self.last_drop_log_time = 0
    
    def record_frame_dropped(self, camera_id, reason, buffer_status=None):
        """Record that a frame was dropped by the buffer"""
        self.frames_dropped += 1
        
        # Store drop event details
        drop_event = {
            'timestamp': time.time(),
            'camera_id': camera_id,
            'reason': reason,
            'buffer_status': buffer_status
        }
        self.drop_events.append(drop_event)
        
        # Log drop event immediately (but rate-limit to avoid spam)
        current_time = time.time()
        if current_time - self.last_drop_log_time >= 1.0:  # Log at most once per second
            self.log_drop_event(drop_event)
            self.last_drop_log_time = current_time
    
    def log_drop_event(self, drop_event):
        """Log a frame drop event"""
        elapsed = drop_event['timestamp'] - (self.start_time or drop_event['timestamp'])
        logger.warning(f"[DROP@{elapsed:.1f}s] Camera {drop_event['camera_id']}: {drop_event['reason']}")
        
        if drop_event['buffer_status']:
            status = drop_event['buffer_status']
            logger.warning(f"  Buffer: {status.get('total_frames', 0)} frames, "
                          f"{status.get('utilization_percent', 0):.1f}% full")
    
    def record_frame_sent(self):
        """Record that a frame was sent to buffer"""
        self.frames_sent += 1
    
    def record_frame_processed(self, processing_time_ms):
        """Record that a frame was processed"""
        self.frames_processed += 1
        self.processing_times.append(processing_time_ms)
    
    def record_system_metrics(self, gpu_processor):
        """Record system performance metrics"""
        # GPU metrics
        if gpu_processor:
            gpu_stats = gpu_processor.get_memory_stats()
            self.gpu_memory_usage.append(gpu_stats['allocated_mb'])
        
        # System metrics
        process = psutil.Process()
        self.cpu_usage.append(process.cpu_percent())
        self.ram_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
    
    def get_current_stats(self, final_calculation=False):
        """Get current performance statistics"""
        if not self.start_time:
            return {}
        
        elapsed = time.time() - self.start_time
        
        # Calculate rates
        current_time = time.time()
        time_since_last = current_time - self.last_log_time
        
        if final_calculation:
            # For final stats, calculate over entire test duration
            if elapsed > 0:
                send_fps = self.frames_sent / elapsed
                process_fps = self.frames_processed / elapsed
                drop_fps = self.frames_dropped / elapsed
            else:
                send_fps = process_fps = drop_fps = 0
        else:
            # For periodic stats, calculate over recent interval
            if time_since_last > 0:
                send_fps = (self.frames_sent - self.last_frames_sent) / time_since_last
                process_fps = (self.frames_processed - self.last_frames_processed) / time_since_last
                drop_fps = (self.frames_dropped - self.last_frames_dropped) / time_since_last
            else:
                send_fps = process_fps = drop_fps = 0
            
            # Update for next calculation
            self.last_log_time = current_time
            self.last_frames_sent = self.frames_sent
            self.last_frames_processed = self.frames_processed
            self.last_frames_dropped = self.frames_dropped
        
        stats = {
            'elapsed_seconds': elapsed,
            'total_frames_sent': self.frames_sent,
            'total_frames_processed': self.frames_processed,
            'total_frames_dropped': self.frames_dropped,
            'send_fps': send_fps,
            'process_fps': process_fps,
            'drop_fps': drop_fps,
            'processing_efficiency': (self.frames_processed / max(1, self.frames_sent)) * 100,
            'drop_rate': (self.frames_dropped / max(1, self.frames_sent)) * 100
        }
        
        # Add aggregate metrics
        if self.processing_times:
            stats['avg_processing_time_ms'] = sum(self.processing_times) / len(self.processing_times)
            stats['max_processing_time_ms'] = max(self.processing_times)
        
        if self.gpu_memory_usage:
            stats['current_gpu_memory_mb'] = self.gpu_memory_usage[-1]
            stats['peak_gpu_memory_mb'] = max(self.gpu_memory_usage)
        
        if self.cpu_usage:
            stats['current_cpu_percent'] = self.cpu_usage[-1]
            stats['avg_cpu_percent'] = sum(self.cpu_usage) / len(self.cpu_usage)
        
        if self.ram_usage:
            stats['current_ram_mb'] = self.ram_usage[-1]
            stats['peak_ram_mb'] = max(self.ram_usage)
        
        return stats

class BufferDropMonitor:
    """Monitor frame buffer drops by wrapping add_frame method"""
    def __init__(self, frame_buffer, metrics):
        self.frame_buffer = frame_buffer
        self.metrics = metrics
        self.original_add_frame = frame_buffer.add_frame
        
        # Wrap the add_frame method to detect drops
        frame_buffer.add_frame = self._monitored_add_frame
    
    async def _monitored_add_frame(self, frame, metadata):
        """Wrapped add_frame that detects and logs drops"""
        camera_id = metadata.get('camera_id', 'unknown')
        
        # Get buffer status before attempting to add
        buffer_status = self.frame_buffer.get_buffer_status()
        
        try:
            # Try to add frame
            result = await self.original_add_frame(frame, metadata)
            
            # Check if frame was actually added by comparing buffer status
            new_buffer_status = self.frame_buffer.get_buffer_status()
            
            # If total frames didn't increase, it means frame was dropped
            if new_buffer_status['total_frames'] <= buffer_status['total_frames']:
                # Determine drop reason based on buffer state
                if buffer_status['utilization_percent'] >= 95:
                    reason = f"Buffer full ({buffer_status['utilization_percent']:.1f}%)"
                elif buffer_status['per_camera'].get(camera_id, 0) >= self.frame_buffer.max_size_per_camera * 0.9:
                    reason = f"Camera {camera_id} buffer full"
                else:
                    reason = "Buffer drop (unknown reason)"
                
                self.metrics.record_frame_dropped(camera_id, reason, buffer_status)
            
            return result
            
        except Exception as e:
            # Frame addition failed
            self.metrics.record_frame_dropped(camera_id, f"Add failed: {str(e)}", buffer_status)
            raise

class ImageLoader:
    """Loads and cycles through test images"""
    def __init__(self, images_dir):
        self.images_dir = Path(images_dir)
        self.images = []
        self.current_index = 0
        self.load_images()
    
    def load_images(self):
        """Load all images from directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for img_path in self.images_dir.glob('*'):
            if img_path.suffix.lower() in image_extensions:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        self.images.append(img)
                        logger.info(f"Loaded image: {img_path.name} ({img.shape})")
                except Exception as e:
                    logger.error(f"Failed to load {img_path}: {e}")
        
        if not self.images:
            raise ValueError(f"No images found in {self.images_dir}")
        
        logger.info(f"Loaded {len(self.images)} test images")
    
    def get_next_image(self):
        """Get next image in cycle"""
        img = self.images[self.current_index].copy()
        self.current_index = (self.current_index + 1) % len(self.images)
        return img

class StressTester:
    """Main stress testing class"""
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.metrics = StressTestMetrics()
        self.image_loader = ImageLoader(config.images_dir)
        self.running = False
        
        # Initialize components
        self.frame_buffer = RobustFrameBuffer(
            max_size_per_camera=100,
            max_total_size=500,
            timeout_seconds=10.0
        )
        
        # Add drop monitoring
        self.drop_monitor = BufferDropMonitor(self.frame_buffer, self.metrics)
        
        self.gpu_processor = TimedGPUBatchProcessor(
            max_batch_size=config.batch_size,
            device=torch.device("cuda:0"),
            model_config={"context_id": 0}
        )
        
        logger.info(f"Initialized stress tester: {config.num_cameras} cameras, "
                   f"{config.fps_per_camera} FPS each, batch size {config.batch_size}")
    
    async def generate_frames(self, camera_id):
        """Generate frames for a specific camera"""
        frame_interval = 1.0 / self.config.fps_per_camera
        frame_count = 0
        
        while self.running:
            try:
                # Get next test image
                image = self.image_loader.get_next_image()
                
                # Create metadata
                metadata = {
                    'store_id': 1,
                    'camera_id': camera_id,
                    'frame_id': frame_count,
                    'timestamp': time.time(),
                    'queued_time': time.time()
                }
                
                # Add to buffer
                await self.frame_buffer.add_frame(image, metadata)
                self.metrics.record_frame_sent()
                
                frame_count += 1
                
                # Wait for next frame
                await asyncio.sleep(frame_interval)
                
            except Exception as e:
                logger.error(f"Error generating frame for camera {camera_id}: {e}")
                await asyncio.sleep(0.1)
    
    async def process_frames(self):
        """Process frames from buffer using GPU processor"""
        while self.running:
            try:
                # Get batch from buffer
                batch = await self.frame_buffer.get_next_batch(
                    max_batch_size=self.config.batch_size,
                    strategy='fair'
                )
                
                if not batch:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process batch
                start_time = time.time()
                results = self.gpu_processor.process_batch(batch)
                processing_time_ms = (time.time() - start_time) * 1000
                logger.info(f"processing_time_ms for {len(batch)} images is {processing_time_ms}")
                # Record metrics for each frame in batch
                for _ in results:
                    self.metrics.record_frame_processed(processing_time_ms / len(results))
                
            except Exception as e:
                logger.error(f"Error processing frames: {e}")
                await asyncio.sleep(0.1)
    
    async def monitor_metrics(self):
        """Monitor and log performance metrics"""
        while self.running:
            try:
                # Record system metrics
                self.metrics.record_system_metrics(self.gpu_processor)
                
                # Log stats periodically
                current_time = time.time()
                if current_time - self.metrics.last_log_time >= self.config.log_interval_seconds:
                    stats = self.metrics.get_current_stats()
                    self.log_performance_stats(stats)
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error monitoring metrics: {e}")
                await asyncio.sleep(1.0)
    
    def log_performance_stats(self, stats):
        """Log current performance statistics"""
        logger.info(f"=== PERFORMANCE STATS (t={stats['elapsed_seconds']:.1f}s) ===")
        logger.info(f"Throughput: Send={stats['send_fps']:.1f} FPS, Process={stats['process_fps']:.1f} FPS, Drop={stats['drop_fps']:.1f} FPS")
        logger.info(f"Efficiency: {stats['processing_efficiency']:.1f}% ({stats['total_frames_processed']}/{stats['total_frames_sent']})")
        logger.info(f"Drop Rate: {stats['drop_rate']:.1f}% ({stats['total_frames_dropped']}/{stats['total_frames_sent']})")
        
        if 'avg_processing_time_ms' in stats:
            logger.info(f"Processing: Avg={stats['avg_processing_time_ms']:.1f}ms, Max={stats['max_processing_time_ms']:.1f}ms")
        
        if 'current_gpu_memory_mb' in stats:
            logger.info(f"GPU Memory: Current={stats['current_gpu_memory_mb']:.1f}MB, Peak={stats['peak_gpu_memory_mb']:.1f}MB")
        
        if 'current_cpu_percent' in stats:
            logger.info(f"System: CPU={stats['current_cpu_percent']:.1f}%, RAM={stats['current_ram_mb']:.1f}MB")
        
        # Check for performance issues
        self.check_performance_alerts(stats)
    
    def check_performance_alerts(self, stats):
        """Check for performance issues and log alerts"""
        if stats['processing_efficiency'] < 80:
            logger.warning(f"LOW EFFICIENCY: Only {stats['processing_efficiency']:.1f}% of frames processed!")
        
        if stats['drop_rate'] > 20:
            logger.warning(f"HIGH DROP RATE: {stats['drop_rate']:.1f}% of frames dropped!")
        
        if 'peak_gpu_memory_mb' in stats and stats['peak_gpu_memory_mb'] > self.config.max_gpu_memory_mb:
            logger.warning(f"HIGH GPU MEMORY: {stats['peak_gpu_memory_mb']:.1f}MB > {self.config.max_gpu_memory_mb}MB")
        
        if 'max_processing_time_ms' in stats and stats['max_processing_time_ms'] > self.config.max_processing_latency_ms:
            logger.warning(f"HIGH LATENCY: {stats['max_processing_time_ms']:.1f}ms > {self.config.max_processing_latency_ms}ms")
    
    def test_pure_gpu_batching(self):
        """Test pure GPU performance to isolate batching benefits"""
        logger.info("=== TESTING PURE GPU BATCHING ===")
        
        # Create dummy data that bypasses all CPU processing
        dummy_crops = [torch.randn(3, 224, 224) for _ in range(20)]
        
        # Test different batch sizes
        for batch_size in [1, 5, 10, 15, 20]:
            crops = dummy_crops[:batch_size]
            batch_tensor = torch.cat(crops, dim=0).to(self.gpu_processor.device)
            
            # Test feature extraction only (pure GPU)
            start = time.time()
            with torch.no_grad():
                features = self.gpu_processor.extract_features(self.gpu_processor.model, batch_tensor)
            elapsed = time.time() - start
            
            per_image = elapsed * 1000 / batch_size
            logger.info(f"Pure GPU batch {batch_size}: {elapsed*1000:.1f}ms total, {per_image:.1f}ms per image")

    async def run_test(self):
        """Run the complete stress test"""
        logger.info("Starting GPU stress test...")

        # self.test_pure_gpu_batching()
        
        # Initialize
        await self.frame_buffer.start_monitors()
        self.running = True
        self.metrics.start_time = time.time()
        self.metrics.last_log_time = time.time()
        
        # Start tasks
        tasks = []
        
        # Frame generators (one per camera)
        for camera_id in range(self.config.num_cameras):
            task = asyncio.create_task(self.generate_frames(camera_id))
            tasks.append(task)
        
        # Frame processor
        processor_task = asyncio.create_task(self.process_frames())
        tasks.append(processor_task)
        
        # Metrics monitor
        monitor_task = asyncio.create_task(self.monitor_metrics())
        tasks.append(monitor_task)
        
        try:
            # Run for specified duration
            await asyncio.sleep(self.config.test_duration_seconds)
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        
        finally:
            # Cleanup
            logger.info("Stopping stress test...")
            self.running = False
            
            # Cancel tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Stop frame buffer
            await self.frame_buffer.stop()
            
            # Final stats
            final_stats = self.metrics.get_current_stats(final_calculation=True)
            self.log_final_results(final_stats)
    
    def log_final_results(self, stats):
        """Log final test results"""
        logger.info("\n" + "="*60)
        logger.info("STRESS TEST FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"Test Duration: {stats['elapsed_seconds']:.1f} seconds")
        logger.info(f"Configuration: {self.config.num_cameras} cameras @ {self.config.fps_per_camera} FPS, batch size {self.config.batch_size}")
        
        expected_fps = self.config.num_cameras * self.config.fps_per_camera
        logger.info(f"Expected FPS: {expected_fps:.1f}")
        logger.info(f"Actual Process FPS: {stats['process_fps']:.1f}")
        logger.info(f"Efficiency: {stats['processing_efficiency']:.1f}%")
        logger.info(f"Frames Dropped: {stats['total_frames_dropped']} ({stats['drop_rate']:.1f}%)")
        
        # Calculate throughput ratio
        throughput_ratio = stats['process_fps'] / expected_fps if expected_fps > 0 else 0
        logger.info(f"Throughput Ratio: {throughput_ratio:.1%} ({stats['process_fps']:.1f}/{expected_fps:.1f})")
        
        if 'avg_processing_time_ms' in stats:
            logger.info(f"Processing Time: Avg={stats['avg_processing_time_ms']:.1f}ms, Max={stats['max_processing_time_ms']:.1f}ms")
            theoretical_max_fps = 1000 / stats['avg_processing_time_ms']
            logger.info(f"Theoretical Max FPS: {theoretical_max_fps:.1f} (based on avg processing time)")
        
        if 'peak_gpu_memory_mb' in stats:
            logger.info(f"Peak GPU Memory: {stats['peak_gpu_memory_mb']:.1f}MB")
        
        if 'peak_ram_mb' in stats:
            logger.info(f"Peak RAM Usage: {stats['peak_ram_mb']:.1f}MB")
        
        # More accurate performance assessment
        if throughput_ratio >= 0.95:
            logger.info("RESULT: EXCELLENT - GPU handles full load efficiently")
        elif throughput_ratio >= 0.80:
            logger.info("RESULT: GOOD - GPU handles most of the load")
        elif throughput_ratio >= 0.60:
            logger.info("RESULT: MODERATE - GPU is bottlenecked, some frames dropped")
        elif throughput_ratio >= 0.30:
            logger.info("RESULT: POOR - GPU heavily bottlenecked, many frames dropped")
        else:
            logger.info("RESULT: OVERLOADED - GPU cannot keep up, reduce load significantly")

def main():
    parser = argparse.ArgumentParser(description='GPU Stress Test for Video Processing')
    parser.add_argument('--config', help='Configuration file (JSON)')
    parser.add_argument('--cameras', type=int, default=15, help='Number of cameras to simulate')
    parser.add_argument('--fps', type=float, default=5.0, help='FPS per camera')
    parser.add_argument('--batch-size', type=int, default=20, help='Processing batch size')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--images-dir', default='test_images', help='Directory with test images')
    
    args = parser.parse_args()
    
    # Create configuration
    config = StressTestConfig(args.config)
    
    # Override with command line args
    if args.cameras:
        config.num_cameras = args.cameras
    if args.fps:
        config.fps_per_camera = args.fps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.duration:
        config.test_duration_seconds = args.duration
    if args.images_dir:
        config.images_dir = args.images_dir
    
    # Run test
    tester = StressTester(config)
    asyncio.run(tester.run_test())

if __name__ == "__main__":
    main()