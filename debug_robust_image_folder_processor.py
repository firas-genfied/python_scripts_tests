import asyncio
import cv2
import numpy as np
import uuid
import os
import logging
from datetime import datetime, timedelta
import time
import torch
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import argparse
from PIL import Image
import aiohttp
import glob
from pathlib import Path

# Import your custom modules
from processor_segment_with_transreid import Segmentation_DeepSort, confirm_human
from processor_segment_with_transreid import setup_predictor

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
from sender import send_detection_data

# Initialize TransReID model
from TransReID.config import cfg
from TransReID.model import make_model
from TransReID.datasets.transforms import build_transforms
from TransReID.processor import extract_features
from typing import List, Dict, Tuple, Union, Optional

# Create Detection objects for DeepSORT
from yolov4_deepsort.deep_sort.detection import Detection

# Get classification
from status_checker import improved_human_status

from robust_frame_buffer import RobustFrameBuffer, FrameBufferStats
from task_manager import TaskManager
from milvus_router_client import AsyncMilvusRouterClient
from store_cache_manager import StoreCacheManager

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|error_concealment;1"

# Import needed to match original code
class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class GPUBatchProcessor:
    """Handles batch processing of images using GPU for shared operations"""
    def __init__(self, max_batch_size=16, device=None, model_config=None):
        
        self.max_batch_size = max_batch_size
        self.dynamic_batch_size = max_batch_size 
        self.min_batch_size = max_batch_size // 2
        self.oom_strikes = 0
        self.memory_high_water_mark = 0
        self.last_oom_batch_size = None

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model_config = model_config or {}
        self.context_id = self.model_config.get("context_id", 0)
        self.stream = torch.cuda.Stream(device=self.device)
        
        # Track GPU memory stats
        self.memory_stats = {
            "total_allocated": 0,
            "peak_allocated": 0,
            "total_reserved": 0,
            "peak_reserved": 0
        }
        
        # Initialize the segmentation model (Detectron2)
        self.seg_predictor = setup_predictor()
        self.seg_model = self.seg_predictor.model
        self.seg_model.eval()
        self.aug = self.seg_predictor.aug
        logger.info("Segmentation model initialized")
        
        self.model = make_model(cfg, num_class=1041, camera_num=0, view_num=0).to(self.device)
        self.model.load_param(cfg.TEST.WEIGHT)
        self.model.eval()
        self.transform = build_transforms(cfg, is_train=False)
        self.extract_features = extract_features
        logger.info("TransReID model initialized")

        # Initialize stats tracking
        self.stats = {
            "total_processing_time": 0,
            "total_batches_processed": 0,
            "total_frames_processed": 0,
            "total_people_detected": 0,
            "last_log_time": time.time()
        }
        self.false_positive_blacklist = []

        # If using half-precision (FP16)
        self.use_half_precision = self.model_config.get("half_precision", False)
        if self.use_half_precision and self.device.type == "cuda":
            logger.info("Using half precision (FP16)")
            self.model = self.model.half()
        
        # Update GPU memory stats
        self._update_memory_stats()

    def _update_memory_stats(self):
        """Update GPU memory statistics with context awareness"""
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
                
                device_idx = self.device.index
                current_allocated = torch.cuda.memory_allocated(device_idx)
                current_reserved = torch.cuda.memory_reserved(device_idx)
                
                self.memory_stats["context_id"] = self.context_id
                self.memory_stats["total_allocated"] = current_allocated
                self.memory_stats["total_reserved"] = current_reserved
                
                if current_allocated > self.memory_stats["peak_allocated"]:
                    self.memory_stats["peak_allocated"] = current_allocated
                
                if current_reserved > self.memory_stats["peak_reserved"]:
                    self.memory_stats["peak_reserved"] = current_reserved
                
            except Exception as e:
                logger.error(f"Error updating memory stats for context {self.context_id}: {e}")

    def get_memory_stats(self):
        """Get current GPU memory statistics in MB"""
        self._update_memory_stats()
        return {
            "allocated_mb": self.memory_stats["total_allocated"] / (1024 * 1024),
            "peak_allocated_mb": self.memory_stats["peak_allocated"] / (1024 * 1024),
            "reserved_mb": self.memory_stats["total_reserved"] / (1024 * 1024),
            "peak_reserved_mb": self.memory_stats["peak_reserved"] / (1024 * 1024)
        }

    def is_blacklisted(self, bbox, tolerance=10):
        """
        Checks if the given bbox is similar to one in the blacklist.
        bbox format: [x1, y1, x2, y2]
        """
        for bb in self.false_positive_blacklist:
            if (abs(bbox[0] - bb[0]) <= tolerance and
                abs(bbox[1] - bb[1]) <= tolerance and
                abs(bbox[2] - bb[2]) <= tolerance and
                abs(bbox[3] - bb[3]) <= tolerance):
                return True
        return False

    def process_batch(self, image_batch):
        """
        Process a batch of images to extract detections and features
        
        Args:
            image_batch: List of (image, metadata) tuples
        
        Returns:
            List of (metadata, detections, features) tuples
        """
        
        self.start_time = time.time()

        # Check available memory before processing
        if torch.cuda.is_available():
            free_memory = torch.cuda.mem_get_info(self.device.index)[0] / 1024**3  # GB
            if free_memory < 2.0:
                logger.info(f"Low GPU memory: {free_memory:.1f}GB free")
                return self._process_batch_chunked(image_batch, chunk_size=4)
            if self.last_oom_batch_size and len(image_batch) >= self.last_oom_batch_size:
                return self._process_batch_chunked(image_batch, self.last_oom_batch_size - 2)

            try:
                return self._process_batch_impl(image_batch)
            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM with batch size {len(image_batch)}, splitting batch")
                torch.cuda.empty_cache()
                self.oom_strikes += 1

                if self.oom_strikes > 2:
                    self.dynamic_batch_size = max(self.min_batch_size, self.dynamic_batch_size // 2)
                    logger.info(f"Reducing dynamic batch size to {self.dynamic_batch_size}")
                    self.oom_strikes = 0
                
                return self._process_batch_chunked(image_batch, chunk_size=self.min_batch_size)

    def _process_batch_chunked(self, image_batch, chunk_size=4, max_retries=2, retry_delay=0.1):
        """
        Process a batch in smaller chunks to avoid OOM.
        On OOM, halves the chunk and retries up to `max_retries`.
        """
        all_results = []
        idx = 0
        current_chunk_size = chunk_size

        while idx < len(image_batch):
            chunk = image_batch[idx : idx + current_chunk_size]
            camera_ids = [meta["camera_id"] for _, meta in chunk]
            store_ids = [meta["store_id"] for _, meta in chunk]
            unique_cameras = list(set(camera_ids))
            unique_stores = list(set(store_ids))

            for attempt in range(max_retries):
                try:
                    results = self._process_batch_impl(chunk)
                    all_results.extend(results)
                    idx += len(chunk)
                    current_chunk_size = chunk_size
                    break

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    logger.error(
                        f"OOM on {self.device} | Cameras={unique_cameras} "
                        f"| Stores={unique_stores} | ChunkSize={len(chunk)} "
                        f"| Attempt={attempt+1}/{max_retries}"
                    )
                    if attempt < max_retries - 1 and len(chunk) > 1:
                        current_chunk_size = max(1, len(chunk) // 2)
                        chunk = image_batch[idx : idx + current_chunk_size]
                        camera_ids = [meta["camera_id"] for _, meta in chunk]
                        store_ids = [meta["store_id"] for _, meta in chunk]
                        unique_cameras = list(set(camera_ids))
                        unique_stores = list(set(store_ids))
                        logger.info(
                            f"Reducing chunk to {len(chunk)} for retry | "
                            f"Cameras={unique_cameras} | Stores={unique_stores}"
                        )
                        continue
                    logger.error(
                        f"Skipping chunk after {max_retries} OOMs | "
                        f"FinalChunkSize={len(chunk)}"
                    )
                    for _, meta in chunk:
                        all_results.append((meta, [], []))
                    idx += len(chunk)
                    current_chunk_size = chunk_size
                    break

            else:
                logger.error(
                    f"Non-OOM exception or retries exhausted | "
                    f"Cameras={unique_cameras} | Stores={unique_stores}"
                )
                for _, meta in chunk:
                    all_results.append((meta, [], []))
                idx += len(chunk)
                current_chunk_size = chunk_size

            if idx < len(image_batch):
                time.sleep(retry_delay)

        return all_results

    def _process_batch_impl(self, image_batch):
        batch_results = []
        total_people = 0

        batch_metadata = [metadata for _, metadata in image_batch]
        camera_ids = [meta["camera_id"] for meta in batch_metadata]
        store_ids = [meta["store_id"] for meta in batch_metadata]

        try:
            with torch.cuda.stream(self.stream):
                batch_inputs = []
                original_sizes = []
                transforms = []
                for image, metadata in image_batch:
                    original_sizes.append((image.shape[0], image.shape[1]))
                    original_height, original_width = image.shape[:2]
                    transform = self.aug.get_transform(image)
                    transformed_image = transform.apply_image(image)
                    transforms.append(transform)
                    logger.info(f"Original image size: {image.shape}, "
                           f"Transformed size: {transformed_image.shape}")
                    transformed_image = torch.as_tensor(transformed_image.astype("float32").transpose(2, 0, 1))
                    batch_inputs.append({
                        "image": transformed_image.to(self.device),
                        "height": original_height,
                        "width": original_width,
                    })
                
                with torch.no_grad():
                    batch_outputs = self.seg_model(batch_inputs)
                
                for i, (outputs, (image, metadata)) in enumerate(zip(batch_outputs, image_batch)):
                    instances = outputs["instances"]
                    person_indices = (instances.pred_classes == 0).nonzero().flatten()
                    if len(person_indices) == 0:
                        batch_results.append((metadata, [], []))
                        continue
                    
                    person_boxes = instances.pred_boxes.tensor[person_indices].cpu().numpy()
                    person_scores = instances.scores[person_indices].cpu().numpy()
                    person_masks = instances.pred_masks[person_indices].cpu().numpy()

                    logger.info(f"Image {i}: Using original bboxes without scaling/transformation")

                    filtered_boxes, filtered_scores, filtered_masks = filter_duplicate_detections(
                        person_boxes, person_scores, person_masks, image, iou_threshold=0.9
                    )
                    
                    height, width = image.shape[:2]
                    shrink_percentage_top = 0.10
                    line_y = int(height * shrink_percentage_top)
                    green_box = [0, line_y, width, height]
                    
                    valid_detections = []
                    
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
                        
                    batch_results.append((metadata, valid_detections, [None] * len(valid_detections)))
                    total_people += len(valid_detections)
                
                all_crops = []
                crop_info = []
                for batch_idx, (metadata, detections, _) in enumerate(batch_results):
                    if not detections:
                        continue
                    
                    image = image_batch[batch_idx][0]
                    converted_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    for det_idx, detection in enumerate(detections):
                        tlwh_bbox, score, class_name, mask = detection
                        x, y, w, h = map(int, tlwh_bbox)
                        crop_masked = crop_without_resize(converted_frame, [x, y, x+w, y+h], mask)
                        if crop_masked.size == 0 or w <= 0 or h <= 0:
                            continue
                        crop_pil = Image.fromarray(crop_masked)
                        transformed_crop = self.transform(crop_pil).unsqueeze(0)
                        all_crops.append(transformed_crop)
                        crop_info.append((batch_idx, det_idx))
                
                if all_crops:
                    chunk_size = min(self.max_batch_size * 2, len(all_crops), 32)
                    for chunk_start in range(0, len(all_crops), chunk_size):
                        torch.cuda.empty_cache()

                        chunk_end = min(chunk_start + chunk_size, len(all_crops))
                        chunk_crops = all_crops[chunk_start:chunk_end]
                        chunk_info = crop_info[chunk_start:chunk_end]

                        use_size = chunk_size
                        if torch.cuda.is_available():
                            free_mem = torch.cuda.mem_get_info(self.device.index)[0] / 1024**3
                            if free_mem < 2.0: 
                                logger.warning(f"Low memory before feature extraction: {free_mem:.1f}GB")
                                use_size = max(8, chunk_size // 2)
                        
                        for sub_start in range(0, len(chunk_crops), use_size):
                            sub_end = min(sub_start + use_size, len(chunk_crops))
                            sub_crops = chunk_crops[sub_start:sub_end]
                            sub_info = chunk_info[sub_start:sub_end]

                            batch_tensor = torch.cat(sub_crops, dim=0).to(self.device)
                            if self.use_half_precision and self.device.type == "cuda":
                                batch_tensor = batch_tensor.half()
                            with torch.no_grad():
                                sub_features = self.extract_features(self.model, batch_tensor).cpu().float().numpy()
                            
                            for feat_idx, (batch_idx, det_idx) in enumerate(sub_info):
                                if batch_results[batch_idx][2][det_idx] is None:
                                    batch_results[batch_idx][2][det_idx] = sub_features[feat_idx].reshape(-1)
                
                for batch_idx, (metadata, detections, features) in enumerate(batch_results):
                    for feat_idx in range(len(features)):
                        if features[feat_idx] is None:
                            features[feat_idx] = np.zeros((768,))

                processing_time = time.time() - self.start_time
                self.stats["total_processing_time"] += processing_time
                self.stats["total_batches_processed"] += 1
                self.stats["total_frames_processed"] += len(image_batch)
                self.stats["total_people_detected"] += total_people
                
                current_time = time.time()
                if current_time - self.stats["last_log_time"] > 60:
                    self._log_performance_stats()
                    self.stats["last_log_time"] = current_time
                
                self._update_memory_stats()
            
            self.stream.synchronize()
            return batch_results
        
        except torch.cuda.OutOfMemoryError:
            unique_cameras = list(set(camera_ids))
            unique_stores = list(set(store_ids))
            logger.error(
                f"CUDA out of memory error on {self.device} - "
                f"Cameras: {unique_cameras}, Stores: {unique_stores}, "
                f"Batch size: {len(image_batch)}, Total people detected: {total_people}"
            )
            torch.cuda.empty_cache()
            return [(metadata, [], []) for metadata, _ in image_batch]
            
        except Exception as e:
            unique_cameras = list(set(camera_ids))
            unique_stores = list(set(store_ids))
            logger.error(
                f"Error processing batch on {self.device} - "
                f"Cameras: {unique_cameras}, Stores: {unique_stores}, "
                f"Batch size: {len(image_batch)}, Error: {e}", 
                exc_info=True
            )
            return [(metadata, [], []) for metadata, _ in image_batch]

    def _log_performance_stats(self):
        """Log performance statistics"""
        try:
            avg_batch_time = self.stats["total_processing_time"] / max(1, self.stats["total_batches_processed"])
            avg_frame_time = self.stats["total_processing_time"] / max(1, self.stats["total_frames_processed"])
            avg_people_per_frame = self.stats["total_people_detected"] / max(1, self.stats["total_frames_processed"])
            
            mem_stats = self.get_memory_stats()
            
            logger.info(f"GPU {self.device} stats: "
                       f"avg_batch_time={avg_batch_time:.3f}s, "
                       f"avg_frame_time={avg_frame_time:.3f}s, "
                       f"avg_people={avg_people_per_frame:.1f}, "
                       f"memory={mem_stats['allocated_mb']:.1f}MB/{mem_stats['peak_allocated_mb']:.1f}MB")
        except Exception as e:
            logger.error(f"Error logging performance stats: {e}")
    
    def calculate_overlap(self, bbox, green_box):
        """
        Calculate the overlap area between a detection bounding box and the green box.
        """
        x1_inter = max(bbox[0], green_box[0])
        y1_inter = max(bbox[1], green_box[1])
        x2_inter = min(bbox[2], green_box[2])
        y2_inter = min(bbox[3], green_box[3])

        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        overlap_area = inter_width * inter_height

        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        return overlap_area, bbox_area

class CameraProcessor:
    """Handles per-camera tracking and processing"""
    def __init__(self, camera_id, store_id, milvus_client, store_cache=True):
        logger.info("Inside CameraProcessor Constructor")
        logger.info(f"Received milvus client for store {milvus_client.store_id}")
        self.camera_id = camera_id
        self.store_id = store_id
        self.milvus_client = milvus_client
        self.store_cache = store_cache 
        self.processor = Segmentation_DeepSort(info_flag=True, camera_id=self.camera_id, store_id=self.store_id, milvus_client=self.milvus_client, store_cache=self.store_cache)
        self.tracker = self.processor.tracker
        self.person_status = {}
        self.recent_entries = {"entries": [], "classified": {}}
        self.previous_positions = {}
        self.counted_track_ids = set()
        self.singles = 0
        self.couples = 0
        self.groups = 0
        self.false_positive_blacklist = []
        self.frame_count = 0
        
        # Add error tracking and recovery
        self.consecutive_errors = 0
        self.last_successful_frame = None
        self.max_consecutive_errors = 5
        
        logger.info(f"Initialized CameraProcessor for camera {camera_id} in store {store_id}")
    
    async def process_frame(self, frame, frame_id, detections, features, timestamp):
        """
        Process a single frame for this camera using pre-computed detections and features
        """
        self.frame_count += 1
        logger.info(f"Camera {self.camera_id}: Processing frame {frame_id} with {len(detections)} detections")
        
        height, width = frame.shape[:2]
        
        shrink_percentage_top = 0.10
        line_y = int(height * shrink_percentage_top)
        
        x1_box, y1_box = 0, line_y
        x2_box, y2_box = width, height
        
        roi_x1, roi_y1, roi_x2, roi_y2 = (768, 270, 1152, 800)

        detection_objs = [
            Detection(det[0], det[1], det[2], feat)
            for det, feat in zip(detections, features)
        ]
        
        logger.info(f"Camera {self.camera_id}: Created {len(detection_objs)} detection objects")
        
        # Add error handling around tracker operations with enhanced sync
        try:
            self.tracker.predict()
            
            # ENHANCED FIX: Ensure metric samples are in sync before update
            await self._sync_metric_samples_with_tracks()
            
            await self.tracker.update(detection_objs)
        except KeyError as e:
            logger.error(f"Camera {self.camera_id}: KeyError in tracker update: {e}")
            logger.error(f"Camera {self.camera_id}: Current tracks: {[t.track_id for t in self.tracker.tracks]}")
            
            # Get the missing track ID from the error
            missing_track_id = None
            try:
                missing_track_id = int(str(e).strip("'\""))
            except:
                missing_track_id = str(e).strip("'\"")
            
            logger.error(f"Camera {self.camera_id}: Missing track ID in metric samples: {missing_track_id}")
            
            # Specific fix for this KeyError
            success = await self._fix_keyerror_and_retry(missing_track_id, detection_objs, frame_id)
            if not success:
                logger.error(f"Camera {self.camera_id}: Failed to fix KeyError, returning empty result")
                return {
                    "camera_id": int(self.camera_id),
                    "image_url": "",
                    "frame_id": frame_id,
                    "is_organised": True,
                    "no_of_people": 0,
                    "date_time": timestamp,
                    "persons": [],
                }
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Unexpected error in tracker update: {e}", exc_info=True)
            return {
                "camera_id": int(self.camera_id),
                "image_url": "",
                "frame_id": frame_id,
                "is_organised": True,
                "no_of_people": 0,
                "date_time": timestamp,
                "persons": [],
            }
        
        active_tracks = []
        entity_coordinates = []
        
    async def _sync_metric_samples_with_tracks(self):
        """
        DEFENSIVE SYNC: Ensure metric samples are synchronized with current tracks
        """
        try:
            if not hasattr(self.tracker, 'metric') or not hasattr(self.tracker.metric, 'samples'):
                logger.warning(f"Camera {self.camera_id}: Tracker metric not properly initialized")
                return
                
            # Get current track IDs (only valid, non-deleted tracks)
            current_track_ids = set()
            for track in self.tracker.tracks:
                if hasattr(track, 'track_id') and hasattr(track, 'state'):
                    # Only include confirmed and tentative tracks, exclude deleted ones
                    if track.state in [1, 2]:  # TrackState.Tentative=1, Confirmed=2
                        current_track_ids.add(track.track_id)
            
            logger.debug(f"Camera {self.camera_id}: Current valid track IDs: {current_track_ids}")
            
            # LAZY CREATION: Ensure all current tracks have metric samples
            missing_samples = 0
            for track_id in current_track_ids:
                if track_id not in self.tracker.metric.samples:
                    logger.debug(f"Camera {self.camera_id}: Creating empty metric samples for track {track_id}")
                    self.tracker.metric.samples[track_id] = []
                    missing_samples += 1
            
            if missing_samples > 0:
                logger.info(f"Camera {self.camera_id}: Created {missing_samples} missing metric samples")
            
            # DEFENSIVE CLEANUP: Only remove samples for tracks that are definitely gone
            # Be conservative - only remove if track has been deleted for a while
            metric_track_ids = set(self.tracker.metric.samples.keys())
            potentially_orphaned = metric_track_ids - current_track_ids
            
            # Check if these tracks are actually deleted (not just temporarily missing)
            confirmed_orphaned = set()
            for track_id in potentially_orphaned:
                # Only consider it orphaned if NO track with this ID exists at all
                track_exists = any(t.track_id == track_id for t in self.tracker.tracks)
                if not track_exists:
                    confirmed_orphaned.add(track_id)
            
            # Remove only confirmed orphaned samples
            for track_id in confirmed_orphaned:
                logger.debug(f"Camera {self.camera_id}: Removing orphaned metric samples for track {track_id}")
                del self.tracker.metric.samples[track_id]
            
            if confirmed_orphaned:
                logger.info(f"Camera {self.camera_id}: Removed {len(confirmed_orphaned)} orphaned metric samples")
            
            logger.debug(f"Camera {self.camera_id}: Metric sync complete - "
                        f"Tracks: {len(current_track_ids)}, "
                        f"Samples: {len(self.tracker.metric.samples)}")
            
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Error syncing metric samples: {e}", exc_info=True)
        """
        Fix KeyError by handling missing track ID in metric samples and retry tracker update
        
        Args:
            missing_track_id: The track ID that's missing from metric samples
            detection_objs: Detection objects to process
            frame_id: Current frame ID for logging
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            logger.info(f"Camera {self.camera_id}: Attempting to fix KeyError for track {missing_track_id}")
            
            # Step 1: Get current state
            current_track_ids = set()
            valid_tracks = []
            
            for track in self.tracker.tracks:
                if hasattr(track, 'track_id') and hasattr(track, 'state'):
                    # Only keep confirmed and tentative tracks (state 1 and 2)
                    if track.state in [1, 2]:  # Tentative=1, Confirmed=2
                        current_track_ids.add(track.track_id)
                        valid_tracks.append(track)
                    else:
                        logger.info(f"Camera {self.camera_id}: Removing deleted track {track.track_id}")
            
            # Update tracker tracks list to only include valid tracks
            self.tracker.tracks = valid_tracks
            logger.info(f"Camera {self.camera_id}: Valid tracks after cleanup: {list(current_track_ids)}")
            
            # Step 2: Clean up metric samples
            if hasattr(self.tracker, 'metric') and hasattr(self.tracker.metric, 'samples'):
                samples_keys = set(self.tracker.metric.samples.keys())
                logger.info(f"Camera {self.camera_id}: Metric samples before cleanup: {samples_keys}")
                
                # Remove samples for tracks that don't exist anymore
                orphaned_samples = samples_keys - current_track_ids
                for orphaned_id in orphaned_samples:
                    logger.info(f"Camera {self.camera_id}: Removing orphaned sample for track {orphaned_id}")
                    del self.tracker.metric.samples[orphaned_id]
                
                # If the missing track ID is in current tracks but not in samples, 
                # we need to either remove the track or create samples
                if missing_track_id in current_track_ids and missing_track_id not in self.tracker.metric.samples:
                    logger.warning(f"Camera {self.camera_id}: Track {missing_track_id} exists but has no metric samples")
                    
                    # Option 1: Remove the problematic track
                    self.tracker.tracks = [t for t in self.tracker.tracks if t.track_id != missing_track_id]
                    current_track_ids.discard(missing_track_id)
                    logger.info(f"Camera {self.camera_id}: Removed problematic track {missing_track_id}")
                    
                    # Option 2: Alternative - create empty samples (commented out)
                    # self.tracker.metric.samples[missing_track_id] = []
                    # logger.info(f"Camera {self.camera_id}: Created empty samples for track {missing_track_id}")
                
                logger.info(f"Camera {self.camera_id}: Metric samples after cleanup: {set(self.tracker.metric.samples.keys())}")
            
            # Step 3: Clean up related state
            if hasattr(self, 'person_status') and missing_track_id in self.person_status:
                del self.person_status[missing_track_id]
                logger.info(f"Camera {self.camera_id}: Cleaned up person_status for track {missing_track_id}")
            
            # Step 4: Try tracker update again
            logger.info(f"Camera {self.camera_id}: Retrying tracker update after cleanup")
            self.tracker.predict()
            await self.tracker.update(detection_objs)
            
            logger.info(f"Camera {self.camera_id}: Successfully recovered from KeyError for track {missing_track_id}")
            self.consecutive_errors = 0  # Reset error counter
            self.last_successful_frame = frame_id
            return True
            
        except KeyError as retry_error:
            logger.error(f"Camera {self.camera_id}: KeyError persists after fix attempt: {retry_error}")
            # Try more aggressive cleanup
            return await self._aggressive_tracker_reset(frame_id)
            
        except Exception as fix_error:
            logger.error(f"Camera {self.camera_id}: Error during KeyError fix: {fix_error}", exc_info=True)
            return await self._aggressive_tracker_reset(frame_id)
    
    async def _aggressive_tracker_reset(self, frame_id):
        """
        Perform aggressive tracker reset as last resort
        
        Args:
            frame_id: Current frame ID for logging
            
        Returns:
            bool: True if reset successful, False otherwise
        """
        try:
            logger.warning(f"Camera {self.camera_id}: Performing aggressive tracker reset at frame {frame_id}")
            
            # Clear all tracker state
            self.tracker.tracks = []
            
            # Clear metric samples
            if hasattr(self.tracker, 'metric') and hasattr(self.tracker.metric, 'samples'):
                self.tracker.metric.samples.clear()
                logger.info(f"Camera {self.camera_id}: Cleared all metric samples")
            
            # Reset tracker ID counter
            if hasattr(self.tracker, '_next_id'):
                self.tracker._next_id = 1
            
            # Clear camera processor state
            self.person_status.clear()
            self.recent_entries = {"entries": [], "classified": {}}
            self.previous_positions.clear()
            self.counted_track_ids.clear()
            
            # Call tracker cleanup if available
            if hasattr(self.tracker, 'cleanup'):
                await self.tracker.cleanup()
            
            self.consecutive_errors = 0
            self.last_successful_frame = frame_id
            
            logger.info(f"Camera {self.camera_id}: Aggressive tracker reset completed at frame {frame_id}")
            return True
            
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Failed to perform aggressive reset: {e}", exc_info=True)
            self.consecutive_errors += 1
            return False
        
        logger.info(f"Camera {self.camera_id}: Tracker has {len(self.tracker.tracks)} total tracks")
        
        confirmed_tracks = 0
        tentative_tracks = 0
        deleted_tracks = 0
        
        for track in self.tracker.tracks:
            if track.state == TrackState.Confirmed:
                confirmed_tracks += 1
            elif track.state == TrackState.Tentative:
                tentative_tracks += 1
            elif track.state == TrackState.Deleted:
                deleted_tracks += 1
                
        logger.info(f"Camera {self.camera_id}: Track states - Confirmed: {confirmed_tracks}, Tentative: {tentative_tracks}, Deleted: {deleted_tracks}")
            if track.state == TrackState.Tentative:
                bbox = track.to_tlbr()
                logger.info(f"BBOX IS {bbox}")
                logger.info(f"frame size is {frame.shape}")
                if not confirm_human(frame, bbox):
                    self.false_positive_blacklist.append(bbox)
                    track.mark_missed()
                else:
                    if not getattr(track, "has_entered", False):
                        has_entered, fraction = is_entering_store_percent(bbox, line_y)
                        if has_entered:
                            track.has_entered = True
                            logger.info(f"Track {track.track_id} has entered the store")
            
            if track.time_since_update > 1:
                bbox = track.to_tlbr()
                has_left, fraction = has_left_store_percent(bbox, line_y)
                has_entered, fraction = is_entering_store_percent(bbox, line_y)
                if has_left and track.mean[5] < 0:
                    self.tracker.mark_track_as_left(track)
                    logger.info(f"Track {track.track_id} has left the store")
                if has_entered and track.mean[5] > 0:
                    logger.info(f"Track {track.track_id} has entered the store (bbox y1: {bbox[1]} is above line {line_y}).")
            
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr()
            bbox_center_x = int(round((bbox[0] + bbox[2]) / 2))
            bbox_center_y = int(round((bbox[1] + bbox[3]) / 2))
            bbox_y_80 = int(round(bbox[1] + 0.8 * (bbox[3] - bbox[1])))
            
            current_timestamp = datetime.utcnow().timestamp()
            classification, is_final = improved_human_status(
                track, current_timestamp, self.recent_entries, 
                self.previous_positions, roi_y1
            )
            
            if classification:
                if "_" in classification:
                    classification_type, group_id_str = classification.split("_", 1)
                    group_id = int(group_id_str)
                else:
                    classification_type = classification
                    group_id = None
            else:
                classification_type = None
                group_id = None
            
            if is_final and track.track_id not in self.counted_track_ids:
                if classification_type == "alone":
                    self.singles += 1
                elif classification_type == "couple":
                    self.couples += 1
                elif classification_type == "group":
                    self.groups += 1
                self.counted_track_ids.add(track.track_id)
            
            if track.track_id not in self.person_status:
                self.person_status[track.track_id] = {"status": "", "group_id": ""}
            
            if is_final:
                logger.info(f"Finalizing status of {track.track_id} to {classification_type}")
                self.person_status[track.track_id]["status"] = classification_type if classification_type else ""
                self.person_status[track.track_id]["group_id"] = str(group_id) if group_id is not None else ""
            else:
                self.person_status[track.track_id]["status"] = ""
                self.person_status[track.track_id]["group_id"] = ""
            
            active_tracks.append(track.track_id)
            if bbox_center_x is not None and bbox_center_y is not None:
                status = self.person_status.get(track.track_id, {}).get("status", "undetermined")
                group_id_str = self.person_status.get(track.track_id, {}).get("group_id", "")
                
                entity_coordinates.append({
                    "person_id": str(track.track_id),
                    "type": status,
                    "group_id": group_id_str,
                    "coords": {
                        "x": str(bbox_center_x),
                        "y": str(bbox_y_80),
                    },
                    "bbox": {
                        "x1": int(bbox[0]),
                        "y1": int(bbox[1]),
                        "x2": int(bbox[2]),
                        "y2": int(bbox[3])
                    }
                })
        
        result = {
            "camera_id": int(self.camera_id),  # Ensure integer
            "image_url": "",
            "frame_id": frame_id,
            "is_organised": True,
            "no_of_people": int(len(active_tracks)),
            "date_time": timestamp,
            "persons": entity_coordinates,
        }
        
        return result

    async def _fix_keyerror_and_retry(self, missing_track_id, detection_objs, frame_id):
        """
        Fix KeyError by handling missing track ID in metric samples and retry tracker update
        
        Args:
            missing_track_id: The track ID that's missing from metric samples
            detection_objs: Detection objects to process
            frame_id: Current frame ID for logging
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            logger.info(f"Camera {self.camera_id}: Attempting to fix KeyError for track {missing_track_id}")
            
            # Step 1: Get current state
            current_track_ids = set()
            valid_tracks = []
            
            for track in self.tracker.tracks:
                if hasattr(track, 'track_id') and hasattr(track, 'state'):
                    # Only keep confirmed and tentative tracks (state 1 and 2)
                    if track.state in [1, 2]:  # Tentative=1, Confirmed=2
                        current_track_ids.add(track.track_id)
                        valid_tracks.append(track)
                    else:
                        logger.info(f"Camera {self.camera_id}: Removing deleted track {track.track_id}")
            
            # Update tracker tracks list to only include valid tracks
            self.tracker.tracks = valid_tracks
            logger.info(f"Camera {self.camera_id}: Valid tracks after cleanup: {list(current_track_ids)}")
            
            # Step 2: Clean up metric samples
            if hasattr(self.tracker, 'metric') and hasattr(self.tracker.metric, 'samples'):
                samples_keys = set(self.tracker.metric.samples.keys())
                logger.info(f"Camera {self.camera_id}: Metric samples before cleanup: {samples_keys}")
                
                # Remove samples for tracks that don't exist anymore
                orphaned_samples = samples_keys - current_track_ids
                for orphaned_id in orphaned_samples:
                    logger.info(f"Camera {self.camera_id}: Removing orphaned sample for track {orphaned_id}")
                    del self.tracker.metric.samples[orphaned_id]
                
                # If the missing track ID is in current tracks but not in samples, 
                # we need to either remove the track or create samples
                if missing_track_id in current_track_ids and missing_track_id not in self.tracker.metric.samples:
                    logger.warning(f"Camera {self.camera_id}: Track {missing_track_id} exists but has no metric samples")
                    
                    # Option 1: Remove the problematic track
                    self.tracker.tracks = [t for t in self.tracker.tracks if t.track_id != missing_track_id]
                    current_track_ids.discard(missing_track_id)
                    logger.info(f"Camera {self.camera_id}: Removed problematic track {missing_track_id}")
                
                logger.info(f"Camera {self.camera_id}: Metric samples after cleanup: {set(self.tracker.metric.samples.keys())}")
            
            # Step 3: Clean up related state
            if hasattr(self, 'person_status') and missing_track_id in self.person_status:
                del self.person_status[missing_track_id]
                logger.info(f"Camera {self.camera_id}: Cleaned up person_status for track {missing_track_id}")
            
            # Step 4: Try tracker update again
            logger.info(f"Camera {self.camera_id}: Retrying tracker update after cleanup")
            self.tracker.predict()
            await self.tracker.update(detection_objs)
            
            logger.info(f"Camera {self.camera_id}: Successfully recovered from KeyError for track {missing_track_id}")
            self.consecutive_errors = 0  # Reset error counter
            self.last_successful_frame = frame_id
            return True
            
        except KeyError as retry_error:
            logger.error(f"Camera {self.camera_id}: KeyError persists after fix attempt: {retry_error}")
            # Try more aggressive cleanup
            return await self._aggressive_tracker_reset(frame_id)
            
        except Exception as fix_error:
            logger.error(f"Camera {self.camera_id}: Error during KeyError fix: {fix_error}", exc_info=True)
            return await self._aggressive_tracker_reset(frame_id)
    
    async def _aggressive_tracker_reset(self, frame_id):
        """
        Perform aggressive tracker reset as last resort
        
        Args:
            frame_id: Current frame ID for logging
            
        Returns:
            bool: True if reset successful, False otherwise
        """
        try:
            logger.warning(f"Camera {self.camera_id}: Performing aggressive tracker reset at frame {frame_id}")
            
            # Clear all tracker state
            self.tracker.tracks = []
            
            # Clear metric samples
            if hasattr(self.tracker, 'metric') and hasattr(self.tracker.metric, 'samples'):
                self.tracker.metric.samples.clear()
                logger.info(f"Camera {self.camera_id}: Cleared all metric samples")
            
            # Reset tracker ID counter
            if hasattr(self.tracker, '_next_id'):
                self.tracker._next_id = 1
            
            # Clear camera processor state
            self.person_status.clear()
            self.recent_entries = {"entries": [], "classified": {}}
            self.previous_positions.clear()
            self.counted_track_ids.clear()
            
            # Call tracker cleanup if available
            if hasattr(self.tracker, 'cleanup'):
                await self.tracker.cleanup()
            
            self.consecutive_errors = 0
            self.last_successful_frame = frame_id
            
            logger.info(f"Camera {self.camera_id}: Aggressive tracker reset completed at frame {frame_id}")
            return True
            
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Failed to perform aggressive reset: {e}", exc_info=True)
            self.consecutive_errors += 1
            return False

    async def cleanup(self):
        """Clean up resources including flushing tracker features"""
        try:
            if hasattr(self.tracker, 'cleanup'):
                await self.tracker.cleanup()
                logger.info(f"Cleaned up tracker for camera {self.camera_id}")
        except Exception as e:
            logger.error(f"Error cleaning up tracker: {e}")

class ImageFolderProcessor:
    """Processes images from a folder instead of Kafka streams"""
    def __init__(self, input_folder, output_folder, json_output_path, num_processors=2, batch_size=8, 
                 camera_id=1, store_id=1, clients_per_store=4):
        set_memory_limit(fraction=0.9)
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.json_output_path = Path(json_output_path)
        self.camera_id = camera_id
        self.store_id = store_id
        
        # Create output directories
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.json_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU processors
        if torch.cuda.device_count() > 1:
            self.gpu_processors = [
                GPUBatchProcessor(
                    max_batch_size=batch_size,
                    device=torch.device(f"cuda:{i}"),
                    model_config={"context_id": i}
                )
                for i in range(min(num_processors, torch.cuda.device_count()))
            ]
        else:
            self.gpu_processors = [GPUBatchProcessor(
                max_batch_size=batch_size, 
                device=torch.device("cuda:0"),
                model_config={"context_id": 0}
            )]

        log_total_memory(self.gpu_processors)
        self.current_processor_index = 0
        
        # Initialize Milvus client
        router_url = os.environ.get("MILVUS_ROUTER_URL", "http://localhost:8000")
        self.milvus_client = AsyncMilvusRouterClient(
            router_url=router_url,
            store_id=store_id,
            embedding_dim=768,
            connection_timeout=15,
            batch_size=50
        )
        
        # Ensure compatibility with your collection schema
        self.collection_name = "person_reid_features"
        logger.info(f"Using collection: {self.collection_name}")
        
        # Initialize camera processor
        self.camera_processor = None
        
        # Results storage
        self.all_results = []
        
        # Performance tracking
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "total_people_detected": 0,
            "processing_start_time": None,
            "processing_end_time": None
        }
        
        logger.info(f"Initialized ImageFolderProcessor with {len(self.gpu_processors)} GPU processors")
        logger.info(f"Input folder: {self.input_folder}")
        logger.info(f"Output folder: {self.output_folder}")
        logger.info(f"JSON output: {self.json_output_path}")

    def get_image_files(self):
        """Get all image files from the input folder in numerical order"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.input_folder.glob(f"*{ext}"))
            image_files.extend(self.input_folder.glob(f"*{ext.upper()}"))
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        # Sort numerically by extracting frame numbers
        def extract_frame_number(path):
            """Extract frame number from filename for proper sorting"""
            filename = path.stem.lower()
            
            # Try different patterns: frame123, img123, 123, etc.
            import re
            patterns = [
                r'frame(\d+)',     # frame123
                r'img(\d+)',       # img123  
                r'image(\d+)',     # image123
                r'(\d+)',          # just numbers
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    return int(match.group(1))
            
            # If no number found, return 0
            logger.warning(f"Could not extract frame number from {filename}, using 0")
            return 0
        
        # Sort by frame number
        image_files.sort(key=extract_frame_number)
        
        logger.info(f"Found {len(image_files)} image files in {self.input_folder}")
        
        # Log first few files to verify ordering
        if image_files:
            first_five = image_files[:5]
            first_five_with_numbers = [(f, extract_frame_number(f)) for f in first_five]
            logger.info(f"First five images (with frame numbers): {first_five_with_numbers}")
            
            # Verify sorting is correct
            frame_numbers = [extract_frame_number(f) for f in image_files[:10]]
            logger.info(f"First 10 frame numbers: {frame_numbers}")
        
        return image_files

    def _select_processor_for_batch(self):
        """Select the least busy processor for the next batch"""
        if len(self.gpu_processors) == 1:
            return self.gpu_processors[0]

        if all(hasattr(p, 'get_memory_stats') for p in self.gpu_processors):
            proc_idx = min(range(len(self.gpu_processors)), 
                        key=lambda i: self.gpu_processors[i].get_memory_stats()['allocated_mb'])
        else:
            proc_idx = self.current_processor_index
            self.current_processor_index = (self.current_processor_index + 1) % len(self.gpu_processors)
        
        return self.gpu_processors[proc_idx]

    def draw_annotations(self, image, result, timestamp):
        """Draw bounding boxes, coordinates, and timestamp on the image"""
        # Create a copy for annotation
        annotated_image = image.copy()
        height, width = image.shape[:2]
        
        # Draw timestamp in top-right corner
        timestamp_text = f"Time: {timestamp}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)  # Green
        
        # Get text size to position it properly
        (text_width, text_height), baseline = cv2.getTextSize(timestamp_text, font, font_scale, thickness)
        
        # Draw background rectangle for timestamp
        cv2.rectangle(annotated_image, 
                     (width - text_width - 10, 5), 
                     (width - 5, text_height + baseline + 10), 
                     (0, 0, 0), -1)
        
        # Draw timestamp text
        cv2.putText(annotated_image, timestamp_text, 
                   (width - text_width - 8, text_height + 8), 
                   font, font_scale, color, thickness)
        
        # Draw green box (store area)
        shrink_percentage_top = 0.10
        line_y = int(height * shrink_percentage_top)
        cv2.rectangle(annotated_image, (0, line_y), (width, height), (0, 255, 0), 2)
        cv2.putText(annotated_image, "Store Area", (10, line_y + 25), 
                   font, 0.6, (0, 255, 0), 2)
        
        # Draw person annotations
        for person in result.get("persons", []):
            person_id = person["person_id"]
            person_type = person["type"]
            group_id = person["group_id"]
            
            # Get coordinates
            center_x = int(person["coords"]["x"])
            center_y = int(person["coords"]["y"])
            
            # Get bounding box
            bbox = person.get("bbox", {})
            if bbox:
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw center point
                cv2.circle(annotated_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Draw coordinate text
                coord_text = f"({center_x}, {center_y})"
                cv2.putText(annotated_image, coord_text, 
                           (center_x + 10, center_y), 
                           font, 0.5, (0, 0, 255), 1)
                
                # Draw person info
                info_text = f"ID:{person_id}"
                if person_type:
                    info_text += f" Type:{person_type}"
                if group_id:
                    info_text += f" Group:{group_id}"
                
                cv2.putText(annotated_image, info_text, 
                           (x1, y1 - 10), 
                           font, 0.5, (255, 0, 0), 1)
        
        return annotated_image

    async def initialize_milvus_client(self):
        """Initialize Milvus client with retry logic"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing Milvus client (attempt {attempt + 1}/{max_retries})...")
                is_healthy = await self.milvus_client.check_connection_health()
                if not is_healthy:
                    logger.error(f"Milvus connection health check failed (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Failed to connect to Milvus after {max_retries} attempts")
                        return False
                        
                logger.info(f"Successfully connected to Milvus for store {self.store_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error initializing Milvus client (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Failed to initialize Milvus client after {max_retries} attempts")
                    return False
        
        return False

    async def initialize_camera_processor(self):
        """Initialize camera processor with store cache"""
        try:
            # Create store cache manager
            cache_config = {
                'time_interval': 5.0,
                'stale_threshold': 30.0,
                'max_cache_size': 1000,
                'features_per_track': 50,
            }
            
            store_cache = StoreCacheManager(
                store_id=self.store_id,
                milvus_client=self.milvus_client,
                cache_config=cache_config
            )
            
            self.camera_processor = CameraProcessor(
                self.camera_id, 
                self.store_id,
                milvus_client=self.milvus_client,
                store_cache=store_cache
            )
            logger.info(f"Initialized camera processor for camera {self.camera_id} in store {self.store_id}")
            return True
        except Exception as e:
            logger.error(f"Error initializing camera processor: {e}")
            return False

    async def process_batch(self, image_batch):
        """Process a batch of images"""
        try:
            # Select GPU processor
            processor = self._select_processor_for_batch()
            
            # Process batch on GPU
            loop = asyncio.get_running_loop()
            batch_results = await loop.run_in_executor(
                None,
                lambda: processor.process_batch(image_batch)
            )
            
            # Process each result with camera processor
            processed_results = []
            for i, (metadata, detections, features) in enumerate(batch_results):
                if not detections:
                    continue
                
                original_frame = image_batch[i][0]
                frame_id = metadata['frame_id']
                timestamp = metadata['timestamp']
                
                # Process with camera processor with robust error handling
                try:
                    result = await self.camera_processor.process_frame(
                        original_frame,
                        frame_id,
                        detections,
                        features,
                        timestamp
                    )
                    
                    processed_results.append({
                        'result': result,
                        'metadata': metadata,
                        'original_frame': original_frame
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_id}: {e}", exc_info=True)
                    # Create empty result to continue processing
                    empty_result = {
                        "camera_id": int(self.camera_id),
                        "image_url": "",
                        "frame_id": frame_id,
                        "is_organised": True,
                        "no_of_people": 0,
                        "date_time": timestamp,
                        "persons": [],
                    }
                    
                    processed_results.append({
                        'result': empty_result,
                        'metadata': metadata,
                        'original_frame': original_frame
                    })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            return []

    async def save_results_to_json(self):
        """Save all results to JSON file with request body payload format"""
        try:
            # Prepare the data in the exact format that would be sent to the API
            api_payload_format = []
            
            for result in self.all_results:
                # Convert to the exact API request body format
                api_result = {
                    "camera_id": int(result.get("camera_id", 0)),
                    "image_url": str(result.get("image_url", "")),
                    "is_organised": bool(result.get("is_organised", True)),
                    "no_of_people": int(result.get("no_of_people", 0)),
                    "date_time": str(result.get("date_time", "")),
                    "persons": []
                }
                
                # Process persons array to match API format
                for person in result.get("persons", []):
                    api_person = {
                        "person_id": str(person.get("person_id", "")),
                        "coords": {
                            "x": str(person.get("coords", {}).get("x", "0")),
                            "y": str(person.get("coords", {}).get("y", "0"))
                        },
                        "type": str(person.get("type", "customer"))
                    }
                    
                    # Add group_id only if it exists and is not empty
                    group_id = person.get("group_id", "")
                    if group_id and str(group_id).strip():
                        api_person["group_id"] = str(group_id)
                    
                    api_result["persons"].append(api_person)
                
                # Only include results with people detected
                if api_result["no_of_people"] > 0:
                    api_payload_format.append(api_result)
            
            # Create comprehensive output structure
            output_data = {
                "metadata": {
                    "processing_stats": self.stats,
                    "camera_id": self.camera_id,
                    "store_id": self.store_id,
                    "total_images_processed": self.stats["processed_images"],
                    "total_results_with_people": len(api_payload_format),
                    "total_people_detected": self.stats["total_people_detected"],
                    "database_sending_disabled": True,
                    "format": "API_REQUEST_BODY_FORMAT"
                },
                "api_request_payload": api_payload_format,
                "raw_results": self.all_results  # Keep the original results as well
            }
            
            # Save to JSON file
            with open(self.json_output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(api_payload_format)} API-formatted results to {self.json_output_path}")
            logger.info(f"Total results with people: {len(api_payload_format)}")
            logger.info(f"Raw results saved: {len(self.all_results)}")
            
            # Also save just the API payload format for easy testing
            api_payload_file = self.json_output_path.parent / f"api_payload_{self.json_output_path.stem}.json"
            with open(api_payload_file, 'w') as f:
                json.dump(api_payload_format, f, indent=2, default=str)
            
            logger.info(f"API payload format saved to: {api_payload_file}")
            
            # Log sample of the API payload for verification
            if api_payload_format:
                logger.info("Sample API payload (first result):")
                logger.info(json.dumps(api_payload_format[0], indent=2))
                
        except Exception as e:
            logger.error(f"Error saving results to JSON: {e}", exc_info=True)

    async def process_images(self, batch_size=8):
        """Main method to process all images in the folder"""
        # Initialize components
        logger.info("Initializing Milvus client...")
        if not await self.initialize_milvus_client():
            logger.error("Failed to initialize Milvus client")
            return
            
        logger.info("Initializing camera processor...")
        if not await self.initialize_camera_processor():
            logger.error("Failed to initialize camera processor")
            return
        
        # Get all image files
        image_files = self.get_image_files()
        if not image_files:
            logger.error("No image files found in input folder")
            return
        
        # Log the actual order we'll process them in
        if len(image_files) >= 5:
            logger.info(f"First five images in processing order: {image_files[:5]}")
        if len(image_files) >= 10:
            logger.info(f"First ten images in processing order: {image_files[:10]}")
        
        self.stats["total_images"] = len(image_files)
        self.stats["processing_start_time"] = datetime.utcnow().isoformat()
        
        logger.info(f"Starting processing of {len(image_files)} images...")
        
        # Process images in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size} "
                       f"({len(batch_files)} images)")
            
            # Load images and create metadata
            image_batch = []
            for j, image_path in enumerate(batch_files):
                try:
                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        logger.warning(f"Failed to load image: {image_path}")
                        continue
                    
                    # Create metadata
                    metadata = {
                        "store_id": self.store_id,
                        "camera_id": self.camera_id,
                        "frame_id": f"{image_path.stem}",
                        "timestamp": datetime.utcnow().isoformat(),
                        "image_path": str(image_path),
                        "queued_time": time.time()
                    }
                    
                    image_batch.append((image, metadata))
                    
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}")
                    continue
            
            if not image_batch:
                logger.warning(f"No valid images in batch {i//batch_size + 1}")
                continue
            
            # Process batch
            processed_results = await self.process_batch(image_batch)
            
            # Save results and annotated images
            for result_data in processed_results:
                try:
                    result = result_data["result"]
                    metadata = result_data["metadata"]
                    original_frame = result_data["original_frame"]
                    
                    # Skip if no people detected
                    if result["no_of_people"] == 0:
                        logger.info(f"No people detected in {metadata['frame_id']}")
                        # Still add to all_results for comprehensive logging
                        self.all_results.append(result)
                        continue
                    
                    # Add to results collection (all results, including those with 0 people)
                    self.all_results.append(result)
                    
                    # Create annotated image
                    annotated_image = self.draw_annotations(
                        original_frame, 
                        result, 
                        metadata["timestamp"]
                    )
                    
                    # Save annotated image
                    output_filename = f"processed_{metadata['frame_id']}.jpg"
                    output_path = self.output_folder / output_filename
                    cv2.imwrite(str(output_path), annotated_image)
                    
                    # Update stats
                    self.stats["processed_images"] += 1
                    self.stats["total_people_detected"] += result["no_of_people"]
                    
                    logger.info(f"Processed {metadata['frame_id']}: {result['no_of_people']} people detected, saved to {output_filename}")
                    
                    # Log the result in API payload format for verification
                    if result["no_of_people"] > 0:
                        logger.debug(f"Result data for {metadata['frame_id']}: {json.dumps(result, indent=2, default=str)}")
                    
                except Exception as e:
                    logger.error(f"Error saving result: {e}", exc_info=True)
            
            # Send to database (COMMENTED OUT - NOT SENDING TO DATABASE)
            # if processed_results:
            #     results_to_send = [r["result"] for r in processed_results if r["result"]["no_of_people"] > 0]
            #     if results_to_send:
            #         try:
            #             success = await send_detection_data(results_to_send)
            #             if success:
            #                 logger.info(f"Successfully sent {len(results_to_send)} results to database")
            #             else:
            #                 logger.warning(f"Failed to send {len(results_to_send)} results to database")
            #         except Exception as e:
            #             logger.error(f"Error sending data to database: {e}")
            
            logger.info("Database sending is disabled - results saved to JSON only")
        
        # Finalize processing
        self.stats["processing_end_time"] = datetime.utcnow().isoformat()
        
        # Calculate processing time
        if self.stats["processing_start_time"] and self.stats["processing_end_time"]:
            start_time = datetime.fromisoformat(self.stats["processing_start_time"])
            end_time = datetime.fromisoformat(self.stats["processing_end_time"])
            processing_duration = (end_time - start_time).total_seconds()
            self.stats["processing_duration_seconds"] = processing_duration
            self.stats["average_time_per_image"] = processing_duration / max(1, self.stats["processed_images"])
        
        # Save results to JSON
        await self.save_results_to_json()
        
        # Print final statistics
        logger.info("=" * 50)
        logger.info("PROCESSING COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Total images found: {self.stats['total_images']}")
        logger.info(f"Images processed: {self.stats['processed_images']}")
        logger.info(f"Total people detected: {self.stats['total_people_detected']}")
        logger.info(f"Processing duration: {self.stats.get('processing_duration_seconds', 0):.2f} seconds")
        logger.info(f"Average time per image: {self.stats.get('average_time_per_image', 0):.2f} seconds")
        logger.info(f"Results saved to: {self.json_output_path}")
        logger.info(f"Annotated images saved to: {self.output_folder}")
        
        # Cleanup
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.camera_processor:
                await self.camera_processor.cleanup()
            if self.milvus_client:
                await self.milvus_client.close()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main entry point for the image folder processor"""
    parser = argparse.ArgumentParser(description='Image Folder Processor for Retail Analytics')
    parser.add_argument('--input-folder', default='/home/azureuser/workstation/genfied/ObjectTracking/images/cam1',
                       help='Path to input folder containing images')
    parser.add_argument('--output-folder', default='/home/azureuser/workstation/genfied/ObjectTracking/images/processed_cam1',
                       help='Path to output folder for processed images')
    parser.add_argument('--json-output', default='/home/azureuser/workstation/genfied/ObjectTracking/results_cam1_video/results.json',
                       help='Path to JSON output file')
    parser.add_argument('--batch-size', type=int, default=8, 
                       help='Number of images to process in each batch')
    parser.add_argument('--camera-id', type=int, default=1, 
                       help='Camera ID to use for processing')
    parser.add_argument('--store-id', type=int, default=1, 
                       help='Store ID to use for processing')
    parser.add_argument('--num-processors', type=int, default=2, 
                       help='Number of GPU processors to use')
    
    args = parser.parse_args()
    
    logger.info("Starting Image Folder Processor service")
    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Output folder: {args.output_folder}")
    logger.info(f"JSON output: {args.json_output}")
    logger.info(f"Camera ID: {args.camera_id}")
    logger.info(f"Store ID: {args.store_id}")
    logger.info(f"Batch size: {args.batch_size}")
    
    processor = None
    try:
        processor = ImageFolderProcessor(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            json_output_path=args.json_output,
            num_processors=args.num_processors,
            batch_size=args.batch_size,
            camera_id=args.camera_id,
            store_id=args.store_id
        )
        
        await processor.process_images(batch_size=args.batch_size)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        if processor:
            await processor.cleanup()
        logger.info("Image Folder Processor service stopped")

if __name__ == "__main__":
    asyncio.run(main())
