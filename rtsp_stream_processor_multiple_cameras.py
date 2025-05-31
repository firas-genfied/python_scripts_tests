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
# Import your custom modules
from processor_segment_with_transreid import Segmentation_DeepSort, confirm_human, compute_iou
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
from typing import List, Dict, Tuple, Union, Optional  # Add List to existing import

# Create Detection objects for DeepSORT
from yolov4_deepsort.deep_sort.detection import Detection

# Get classification
from status_checker import improved_human_status

from robust_frame_buffer import RobustFrameBuffer, FrameBufferStats

from task_manager import TaskManager

from milvus_router_client import AsyncMilvusRouterClient

from store_cache_manager import StoreCacheManager
# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|error_concealment;1"

MILVUS_CLIENTS = {}

from kafka.admin import KafkaAdminClient
import re
from kafka import KafkaConsumer


# Import needed to match original code
class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class GPUBatchProcessor:
    """Handles batch processing of images using GPU for shared operations"""
    def __init__(self, max_batch_size=8, device=None, model_config=None):
        
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
        self.model_config = model_config or {}
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
                # Record current memory before operations
                torch.cuda.synchronize(self.device)
                
                device_idx = self.device.index
                current_allocated = torch.cuda.memory_allocated(device_idx)
                current_reserved = torch.cuda.memory_reserved(device_idx)
                
                # Associate with this context
                self.memory_stats["context_id"] = self.context_id
                self.memory_stats["total_allocated"] = current_allocated
                self.memory_stats["total_reserved"] = current_reserved
                
                # Update peaks
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
            if free_memory < 4.0:  # Less than 4GB free
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
    
    def _process_batch_chunked(self, image_batch, chunk_size=4):
        """Process batch in smaller chunks to avoid OOM"""
        all_results = []
        current_chunk_size = chunk_size

        for i in range(0, len(image_batch), chunk_size):
            retry_count = 0
            max_retries = 2
            chunk = image_batch[i:i + chunk_size]
            for attempt in range(2): 
                try:
                    chunk_results = self._process_batch_impl(chunk)
                    all_results.extend(chunk_results)
                    break
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"OOM even with chunk size {chunk_size}")
                    torch.cuda.empty_cache()
                    if attempt == 0:
                        if len(chunk) > 1:
                            chunk = chunk[:len(chunk)//2]
                            logger.info(f"Reducing chunk size to {len(chunk)} for retry")
                        else:
                            # Single image still failing, return empty result
                            logger.error("Single image OOM, skipping")
                            all_results.append((chunk[0][1], [], []))  # metadata, empty detections, empty features
                            break
                    else:
                        logger.error(f"Skipping chunk after 2 OOM attempts")
                        for img, meta in chunk:
                            all_results.append((meta, [], []))
                        break

                except Exception as e:
                    logger.error(f"Non-OOM error in chunk processing: {e}")
                    # Return empty results for this chunk
                    for img, meta in chunk:
                        all_results.append((meta, [], []))
                    break
            if i + chunk_size < len(image_batch):
                await asyncio.sleep(0.1)
        return all_results

    def _process_batch_impl(self, image_batch):
        # logger.info(f"Processing batch of {len(image_batch)} images on GPU")
        batch_results = []
        timestamps = []
        total_people = 0
        
        try:
            # Step 1: Run segmentation on each image
            with torch.cuda.stream(self.stream):
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
                with torch.no_grad():
                    batch_outputs = self.seg_model(batch_inputs)
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
                    
                    # Filter duplicate detections
                    # frame_copy = image.copy()  # For visualization of suppressed boxes
                    filtered_boxes, filtered_scores, filtered_masks = filter_duplicate_detections(
                        person_boxes, person_scores, person_masks, image, iou_threshold=0.9
                    )
                    
                    # Get frame dimensions for green box
                    height, width = image.shape[:2]
                    shrink_percentage_top = 0.10
                    line_y = int(height * shrink_percentage_top)
                    green_box = [0, line_y, width, height]
                    
                    # Collect valid detections
                    valid_detections = []
                    detection_crops = []
                    detection_indices = []
                    
                    for j, bbox in enumerate(filtered_boxes):
                        if self.is_blacklisted(bbox):
                            continue
                        score = filtered_scores[j]
                        mask = filtered_masks[j]   
                        overlap_area, bbox_area = self.calculate_overlap(bbox, green_box)
                        
                        if overlap_area / bbox_area > 0.7:
                            # Convert bbox for DeepSORT
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
                            sub_end    = min(sub_start + use_size, len(chunk_crops))
                            sub_crops  = chunk_crops[sub_start:sub_end]
                            sub_info   = chunk_info[sub_start:sub_end]

                        batch_tensor = torch.cat(chunk_crops, dim=0).to(self.device)
                        if self.use_half_precision and self.device.type == "cuda":
                            batch_tensor = batch_tensor.half()
                        with torch.no_grad():
                            features = self.extract_features(self.model, batch_tensor).cpu().float().numpy()
                        # Assign features back to detections
                        for feat_idx, (batch_idx, det_idx) in enumerate(chunk_info):
                            if batch_results[batch_idx][2][det_idx] is None:
                                batch_results[batch_idx][2][det_idx] = features[feat_idx].reshape(-1)
                
                # Fill any remaining None features with zeros
                for batch_idx, (metadata, detections, features) in enumerate(batch_results):
                    for feat_idx in range(len(features)):
                        if features[feat_idx] is None:
                            features[feat_idx] = np.zeros((768,))

                processing_time = time.time() - self.start_time
                self.stats["total_processing_time"] += processing_time
                self.stats["total_batches_processed"] += 1
                self.stats["total_frames_processed"] += len(image_batch)
                self.stats["total_people_detected"] += total_people
                
                # Log stats periodically
                current_time = time.time()
                if current_time - self.stats["last_log_time"] > 60:  # Log every minute
                    self._log_performance_stats()
                    self.stats["last_log_time"] = current_time
                
                # Update GPU memory stats
                self._update_memory_stats()
            
            self.stream.synchronize()

            return batch_results
        
        except torch.cuda.OutOfMemoryError:
            # Handle OOM error
            logger.error(f"CUDA out of memory error on {self.device}")
            torch.cuda.empty_cache()
            
            # Return empty results
            return [(metadata, [], []) for metadata, _ in image_batch]
            
        except Exception as e:
            logger.error(f"Error processing batch on {self.device}: {e}", exc_info=True)
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
        Args:
            bbox: [x1, y1, x2, y2] coordinates of the detection bounding box.
            green_box: [x1, y1, x2, y2] coordinates of the green box.
        Returns:
            overlap_area: Area of the intersection between the two boxes.
            bbox_area: Area of the detection bounding box.
        """
        # Calculate intersection coordinates
        x1_inter = max(bbox[0], green_box[0])
        y1_inter = max(bbox[1], green_box[1])
        x2_inter = min(bbox[2], green_box[2])
        y2_inter = min(bbox[3], green_box[3])

        # Calculate intersection area
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        overlap_area = inter_width * inter_height

        # Calculate detection bounding box area
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        return overlap_area, bbox_area
    
    def calculate_overlap_vectorized(self, bboxes, green_box):
        """Vectorized overlap calculation for multiple bboxes"""
        bboxes = np.array(bboxes)
        
        # Calculate intersections vectorized
        x1_inter = np.maximum(bboxes[:, 0], green_box[0])
        y1_inter = np.maximum(bboxes[:, 1], green_box[1])
        x2_inter = np.minimum(bboxes[:, 2], green_box[2])
        y2_inter = np.minimum(bboxes[:, 3], green_box[3])
        
        inter_widths = np.maximum(0, x2_inter - x1_inter)
        inter_heights = np.maximum(0, y2_inter - y1_inter)
        overlap_areas = inter_widths * inter_heights
        
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        overlap_ratios = overlap_areas / bbox_areas
        
        return overlap_ratios > 0.7
        

class CameraProcessor:
    """Handles per-camera tracking and processing"""
    def __init__(self, camera_id, store_id, milvus_client, store_cache=None):
        logger.info("Inside CameraProcessor Constructor")
        logger.info(f"Received milvus client for store {milvus_client.store_id}")
        self.camera_id = camera_id
        self.store_id = store_id
        # Initialize tracker and status tracking
        self.milvus_client = milvus_client
        self.store_cache = store_cache 
        self.processor = Segmentation_DeepSort(info_flag=True, camera_id = self.camera_id, store_id = self.store_id,  milvus_client = self.milvus_client, store_cache=self.store_cache)
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
        logger.info(f"Initialized CameraProcessor for camera {camera_id} in store {store_id}")
    
    async def process_frame(self, frame, frame_id, detections, features, timestamp):
        """
        Process a single frame for this camera using pre-computed detections and features
        
        Args:
            frame: The image frame
            frame_id: Frame identifier
            detections: List of detections from GPU processor
            features: List of features from GPU processor
            timestamp: Frame timestamp
        
        Returns:
            Dictionary containing tracking results
        """
        self.frame_count += 1
        # frame_copy = frame.copy()
        height, width = frame.shape[:2]
        
        # Define the line_y value (entrance line)
        shrink_percentage_top = 0.10
        line_y = int(height * shrink_percentage_top)
        
        # Draw green box
        x1_box, y1_box = 0, line_y
        x2_box, y2_box = width, height
        # cv2.rectangle(frame_copy, (x1_box, y1_box), (x2_box, y2_box), (0, 255, 0), 2)
        
        # Define ROI for entrance
        roi_x1, roi_y1, roi_x2, roi_y2 = (768, 270, 1152, 800)
        # cv2.rectangle(frame_copy, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
        

        detection_objs = [
            Detection(det[0], det[1], det[2], feat)
            for det, feat in zip(detections, features)
        ]
        
        # Update tracker
        self.tracker.predict()
        await self.tracker.update(detection_objs)
        
        # Process tracking results
        active_tracks = []
        entity_coordinates = []
        
        for track in self.tracker.tracks:
            # Check for false positives in tentative tracks
            if track.state == TrackState.Tentative:
                bbox = track.to_tlbr()
                if not confirm_human(frame, bbox):
                    self.false_positive_blacklist.append(bbox)
                    track.mark_missed()
                else:
                    if not getattr(track, "has_entered", False):
                        has_entered, fraction = is_entering_store_percent(bbox, line_y)
                        if has_entered:
                            track.has_entered = True
                            logger.info(f"Track {track.track_id} has entered the store")
            
            # Check for tracks leaving the store
            if track.time_since_update > 1:
                bbox = track.to_tlbr()
                has_left, fraction = has_left_store_percent(bbox, line_y)
                has_entered, fraction = is_entering_store_percent(bbox, line_y)
                if has_left and track.mean[5] < 0:
                    self.tracker.mark_track_as_left(track)
                    logger.info(f"Track {track.track_id} has left the store")
                if has_entered and track.mean[5] > 0:
                    logger.info(f"Track {track.track_id} has entered the store (bbox y1: {bbox[1]} is above line {line_y}).")
 
            
            # Skip if track is not confirmed or was missed
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            # Get bounding box and center points
            bbox = track.to_tlbr()
            bbox_center_x = int(round((bbox[0] + bbox[2]) / 2))
            bbox_center_y = int(round((bbox[1] + bbox[3]) / 2))
            bbox_y_80 = int(round(bbox[1] + 0.8 * (bbox[3] - bbox[1])))
            
            current_timestamp = datetime.utcnow().timestamp()
            classification, is_final = improved_human_status(
                track, current_timestamp, self.recent_entries, 
                self.previous_positions, roi_y1
            )
            
            # Parse classification
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
            
            # Update counters if classification is finalized
            if is_final and track.track_id not in self.counted_track_ids:
                if classification_type == "alone":
                    self.singles += 1
                elif classification_type == "couple":
                    self.couples += 1
                elif classification_type == "group":
                    self.groups += 1
                self.counted_track_ids.add(track.track_id)
            
            # Update person status
            if track.track_id not in self.person_status:
                self.person_status[track.track_id] = {"status": "", "group_id": ""}
            
            if is_final:
                logger.info(f"Finalizing status of {track.track_id} to {classification_type}")
                self.person_status[track.track_id]["status"] = classification_type if classification_type else ""
                self.person_status[track.track_id]["group_id"] = str(group_id) if group_id is not None else ""
            else:
                # For non-finalized tracks, leave the status empty
                self.person_status[track.track_id]["status"] = ""
                self.person_status[track.track_id]["group_id"] = ""
            
            # Draw bounding box and text on frame_copy
            # (Visualization code omitted for brevity)
            
            # Save track information
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
                    }

                    
                })
        
        # Create result dictionary
        result = {
            "camera_id": str(self.camera_id),
            "image_url": "",
            "frame_id": frame_id,
            "is_organised": True,
            "no_of_people": int(len(active_tracks)),
            "date_time": timestamp,
            "persons": entity_coordinates,
        }
        
        # return result, frame_copy
        return result

    async def cleanup(self):
        """Clean up resources including flushing tracker features"""
        try:
            if hasattr(self.tracker, 'cleanup'):
                await self.tracker.cleanup()
                logger.info(f"Cleaned up tracker for camera {self.camera_id}")
        except Exception as e:
            logger.error(f"Error cleaning up tracker: {e}")

class KafkaProcessor:
    """Manages processing for multiple Kafka topics/partitions"""
    def __init__(self, num_processors = 2, batch_size=8, batch_interval=0.5, processing_fps=5,
        gpu_processors=None, frame_buffer_config=None, thread_pool_size=8, send_fps = 5):
        # self.gpu_processor = GPUBatchProcessor(max_batch_size=batch_size)
        set_memory_limit(fraction=0.9)
        self.num_processors = num_processors
        self.active_processing_tasks = set()
        if gpu_processors is None:
            if torch.cuda.device_count() > 1:
                self.gpu_processors = [
                    GPUBatchProcessor(
                        max_batch_size=batch_size,
                        device=torch.device(f"cuda:{i}"),  # Different GPUs!
                        model_config={"context_id": i}
                    )
                    for i in range(min(num_processors, torch.cuda.device_count()))
                ]
            else:
                # Single GPU = Single Processor
                self.gpu_processors = [GPUBatchProcessor(
                    max_batch_size=batch_size, 
                    device=torch.device("cuda:0"),  # All use the same device
                    model_config={"context_id": 0}  # Give each a unique context ID
                )]
        else:
            self.gpu_processors = gpu_processors

        log_total_memory(self.gpu_processors)
        self.current_processor_index = 0
        # self.gpu_processor = gpu_processors[0]
        self.camera_processors = {}
        self.batch_interval = batch_interval  # Time to collect frames before batch processing (seconds)
        self.processing_fps = processing_fps  # How many frames to process per second
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        buffer_config = frame_buffer_config or {}

        self.frame_buffer = RobustFrameBuffer(
            max_size_per_camera=buffer_config.get('max_size_per_camera', 60),
            max_total_size=buffer_config.get('max_total_size', 300),
            timeout_seconds=buffer_config.get('timeout_seconds', 5.0),
            drop_strategy=buffer_config.get('drop_strategy', 'smart'),
            auto_adjust=buffer_config.get('auto_adjust', True),
            camera_priorities=buffer_config.get('camera_priorities', None)
        )
        # Track if the processing loop is busy
        self.processing_busy = False
        self.last_process_time = time.time()
        # Dictionary to store camera stream info
        self.running = True

        # Performance tracking
        self.stats = {
            "incoming_frames":    0,
            "frames_processed": 0,
            "frames_dropped": 0,
            "processing_times": deque(maxlen=1000),
            "last_stats_time": time.time(),
            "stats_interval": 60  # Log stats every minute
        }
        self.task_manager = TaskManager()
        self.send_interval = 1.0 / send_fps
        self.last_send_time = time.time()

        self.kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVER")
        logger.info(f"server is {self.kafka_bootstrap_servers}")
        self.kafka_consumer_group = os.getenv("KAFKA_CONSUMER_GROUP")
        self.KAFKA_TOPIC_PATTERN  = re.compile(os.getenv("KAFKA_TOPIC_PATTERN"))
        logger.info(f"Kafka bootstrap={self.kafka_bootstrap_servers},"
                    f"group={self.kafka_consumer_group}, "
                    f"pattern={self.KAFKA_TOPIC_PATTERN}")
        
        logger.info(f"Initialized KafkaProcessor with {len(self.gpu_processors)} GPU processors")
        
        # Log GPU device details
        for i, proc in enumerate(self.gpu_processors):
            logger.info(f"GPU processor {i}: device={proc.device}")
        
        self.milvus_clients = {}
        self.max_stores_per_pod = int(os.environ.get("MAX_STORES_PER_POD", "10"))
        self.store_last_access = {}  # Track when each store was last used

        self.store_cache_managers = {}
        self.cache_stats_interval = 60  # Log cache stats every minute
        self.last_cache_stats_time = time.time()

    def get_or_create_store_cache(self, store_id: int) -> StoreCacheManager:
        """Get or create a shared cache manager for a store."""
        if len(self.store_cache_managers) >= self.max_stores_per_pod:
            if self.store_last_access:
                lru_store_id = min(self.store_last_access, key=self.store_last_access.get)
                if lru_store_id != store_id:
                    logger.info(f"Evicting cache for store {lru_store_id} (LRU)")
                    del self.store_cache_managers[lru_store_id]
                    del self.store_last_access[lru_store_id]
        
        self.store_last_access[store_id] = time.time()

        if store_id not in self.store_cache_managers:
            # Get the milvus client for this store
            if store_id not in self.milvus_clients:
                router_url = os.environ.get("MILVUS_ROUTER_URL", "http://localhost:8000")
                self.milvus_clients[store_id] = AsyncMilvusRouterClient(
                    router_url=router_url,
                    store_id=store_id,
                    embedding_dim=768,
                    connection_timeout=10,
                    batch_size=100
                )
            
            # Create cache manager
            cache_config = {
                'time_interval': 5.0,      # Refresh every 5 seconds
                'stale_threshold': 30.0,   # Force refresh after 30 seconds
                'max_cache_size': max(1000, 10000 // max(1, len(self.store_cache_managers))),
                'features_per_track': 50 if len(self.store_cache_managers) < 5 else 25,
            }
            
            self.store_cache_managers[store_id] = StoreCacheManager(
                store_id=store_id,
                milvus_client=self.milvus_clients[store_id],
                cache_config=cache_config
            )
            
            logger.info(f"Created shared cache manager for store {store_id}")
        
        return self.store_cache_managers[store_id]
      
    def _select_processor_for_batch(self):
        """Select the least busy processor for the next batch"""

        # Fast path for single processor
        if len(self.gpu_processors) == 1:
            return self.gpu_processors[0]

        # If memory usage info is available, use it for balancing
        if all(hasattr(p, 'get_memory_stats') for p in self.gpu_processors):
            # Choose the processor with the lowest current memory usage
            proc_idx = min(range(len(self.gpu_processors)), 
                        key=lambda i: self.gpu_processors[i].get_memory_stats()['allocated_mb'])
        else:
            # Fall back to round-robin
            proc_idx = self.current_processor_index
            self.current_processor_index = (self.current_processor_index + 1) % len(self.gpu_processors)
        
        return self.gpu_processors[proc_idx]

    def get_camera_processor(self, camera_id, store_id):
        """Get or create a camera processor for the given camera"""
        key = f"{store_id}_{camera_id}"
        if key not in self.camera_processors:
            milvus_client = self.milvus_clients.get(store_id)
            if not milvus_client:
                logger.info("Store ID not handled by any milvus client")
                router_url = os.environ.get("MILVUS_ROUTER_URL", "http://localhost:8000")
                milvus_client = AsyncMilvusRouterClient(
                    router_url=router_url,
                    store_id=store_id,
                    embedding_dim=768,  # match your model's feature dimension
                    connection_timeout=10,
                    batch_size=100
                )
                self.milvus_clients[store_id] = milvus_client

            store_cache = self.get_or_create_store_cache(store_id)
            logger.info(f"Milvus Client found for store {self.milvus_clients[store_id].store_id}")
            logger.info(f"passing client for store id {milvus_client.store_id} to processor ")
    
            self.camera_processors[key] = CameraProcessor(
                camera_id, 
                store_id,
                milvus_client=milvus_client,
                store_cache=store_cache
            )
            logger.info(f"Created camera processor for camera {camera_id} in store {store_id} with shared cache")
        return self.camera_processors[key]

    async def _log_cache_statistics(self):
        """Log statistics for all store caches."""
        if not self.store_cache_managers:
            return
        
        logger.info("=== Store Cache Statistics ===")
        # Group by active/inactive
        active_stores = []
        inactive_stores = []
            
        total_memory_mb = 0
        total_tracks = 0
        current_time = time.time()
        for store_id, cache_manager in self.store_cache_managers.items():
            stats = cache_manager.get_stats()
            last_access = self.store_last_access.get(store_id, 0)
            idle_time = current_time - last_access
            if idle_time < 300:  # Active if used in last 5 minutes
                active_stores.append((store_id, stats, idle_time))
            else:
                inactive_stores.append((store_id, stats, idle_time))


            # Estimate memory usage (rough calculation)
            memory_mb = (stats['total_features'] * 768 * 4) / (1024 * 1024)
            total_memory_mb += memory_mb
            total_tracks += stats['cache_size']
            
            logger.info(f"Store {store_id}: "
                       f"tracks={stats['cache_size']}, "
                       f"features={stats['total_features']}, "
                       f"age={stats['cache_age_seconds']:.1f}s, "
                       f"hit_rate={stats['hit_rate_percent']:.1f}%, "
                       f"memory≈{memory_mb:.1f}MB")
        
        logger.info(f"Total: {len(self.store_cache_managers)} stores, "
                   f"{total_tracks} tracks, "
                   f"≈{total_memory_mb:.1f}MB cache memory")
        
            # Log active stores
        for store_id, stats, idle_time in active_stores:
            logger.info(f"Store {store_id} [ACTIVE]: "
                    f"idle={idle_time:.0f}s, "
                    f"tracks={stats['cache_size']}, "
                    f"hit_rate={stats['hit_rate_percent']:.1f}%")

    
    async def initialize_milvus_clients(self, store_ids):
        """Initialize AsyncMilvusRouterClient instances for all stores"""
        # Get router URL from environment or use default
        router_url = os.environ.get("MILVUS_ROUTER_URL", "http://localhost:8000")
        
        logger.info(f"Initializing Milvus clients using router URL: {router_url}")
        
        # Initialize a client for each store
        for store_id in store_ids:
            try:
                # Create new AsyncMilvusRouterClient
                client = AsyncMilvusRouterClient(
                    router_url=router_url,
                    store_id=store_id,
                    embedding_dim=768,  # Match your model's embedding dimension
                    connection_timeout=10,
                    batch_size=100
                )
                
                # Check connection health
                is_healthy = await client.check_connection_health()
                if is_healthy:
                    logger.info(f"Successfully connected to Milvus router for store {store_id}")
                    self.milvus_clients[store_id] = client
                else:
                    logger.error(f"Failed to connect to Milvus router for store {store_id}")
                    
            except Exception as e:
                logger.error(f"Error initializing Milvus client for store {store_id}: {e}")
        
        logger.info(f"Initialized {len(self.milvus_clients)} Milvus clients {self.milvus_clients}")
    
    def _task_done_callback(self, task):
        """Callback function when a processing task completes."""
        # Remove the task from our tracking set
        self.active_processing_tasks.discard(task)
        
        # Check if it raised any exceptions
        if not task.cancelled() and task.exception() is not None:
            logger.error(f"Processing task failed with error: {task.exception()}")
    
    async def _process_camera_batch(self, camera_id, frames_data_list):
        """Process all frames for a specific camera"""
        camera_results = []
        
        # Get store_id from first frame (assuming all frames for a camera are from same store)
        store_id = frames_data_list[0]['store_id']
        processor = self.get_camera_processor(camera_id, store_id)
        
        for frame_data in frames_data_list:
            try:
                # Extract all needed data
                original_frame = frame_data['original_frame']
                metadata = frame_data['metadata']
                detections = frame_data['detections']
                features = frame_data['features']
                
                # Process each frame for this camera
                result = await processor.process_frame(
                    original_frame,
                    metadata['frame_id'],
                    detections,
                    features,
                    metadata['timestamp']
                )
                camera_results.append({
                    'result': result,
                    'metadata': metadata
                })
            except Exception as e:
                logger.error(f"Error processing frame for camera {camera_id}: {e}", exc_info=True)
                # Continue with other frames even if one fails
    
        return camera_results
        
    async def process_batch(self):
        """Process all frames in the current batch"""
        try:
            # Get batch from the robust buffer using fair distribution
            batch_start_time = time.time()
            timing_stats = {
                "buffer_read": 0,
                "gpu_processing": 0,
                "camera_processing": 0,
                "result_preparation": 0,
                "sending_data": 0,
                "total": 0
            }
            buffer_start = time.time()
            loop = asyncio.get_running_loop()
            current_batch = await self.frame_buffer.get_next_batch(
                max_batch_size=self.gpu_processors[0].max_batch_size,
                strategy='fair'  # Ensures all cameras get processing time
            )
            timing_stats["buffer_read"] = time.time() - buffer_start
            logger.info(f"[TIMING] Buffer read: {timing_stats['buffer_read']:.3f}s, frames: {len(current_batch) if current_batch else 0}")
            if not current_batch:
                logger.info("No frames in buffer to process")
                self.processing_busy = False
                return
            
            gpu_start = time.time() 
            batch_start_time = time.time()
            processor = self._select_processor_for_batch()
            batch_results = await loop.run_in_executor(
                None,  # Use default executor
                lambda: processor.process_batch(current_batch)
            )
            timing_stats["gpu_processing"] = time.time() - gpu_start
            logger.info(f"[TIMING] GPU processing: {timing_stats['gpu_processing']:.3f}s for {len(current_batch)} frames")
            # Group by camera WITH original frames
            camera_start = time.time()
            camera_groups = defaultdict(list)
            tasks = []
            for i, (metadata, detections, features) in enumerate(batch_results):
                if not detections:  # Skip empty results
                    continue
                store_id = metadata['store_id']
                camera_id = metadata['camera_id']
                frame_id = metadata['frame_id']
                timestamp = metadata['timestamp']
                original_frame = current_batch[i][0]
                camera_groups[camera_id].append({
                    'original_frame': original_frame,
                    'metadata': metadata,
                    'detections': detections,
                    'features': features,
                    'store_id': store_id
                })
            
            camera_tasks = []
            for camera_id, frames_data in camera_groups.items():
                cam_start = time.time()
                task = self._process_camera_batch(camera_id, frames_data)
                camera_tasks.append(task)
                # Find the original frame
            all_results = await asyncio.gather(*camera_tasks)
            timing_stats["camera_processing"] = time.time() - camera_start
            logger.info(f"[TIMING] Camera processing: {timing_stats['camera_processing']:.3f}s for {len(camera_groups)} cameras")
            # Wait for all processing to complete
            send_tasks = []
            prep_start = time.time()
            results_to_send = []  # Create a list to store all results

            for camera_results in all_results:

                if isinstance(camera_results, Exception):
                    logger.error(f"Camera processing failed: {camera_results}")
                    continue

                for result_data in camera_results:
                    try:
                        result = result_data["result"]
                        metadata = result_data["metadata"]

                        # Skip if no people detected
                        if result["no_of_people"] == 0:
                            logger.info(f"Skipping result with no people detected: camera_id={result['camera_id']}, frame_id={metadata['frame_id']}")
                            continue
                        
                        # Ensure all required fields are present
                        if "image_url" not in result or not result["image_url"]:
                            result["image_url"] = ""
                        
                        if "camera_id" not in result or not result["camera_id"]:
                            result["camera_id"] = ""
                        
                        if "is_organised" not in result:
                            result["is_organised"] = True
                        
                        # Make sure date_time is properly set
                        if "date_time" not in result or not result["date_time"]:
                            result["date_time"] = metadata["timestamp"]
                        
                        if "persons" not in result:
                            result["persons"] = []
                        
                        results_to_send.append(result)

                        # Calculate processing latency
                        queued_time = metadata.get('queued_time', time.time())
                        total_latency = time.time() - queued_time
                        
                        # Track processing time for stats
                        self.stats["processing_times"].append(total_latency)
                        self.stats["frames_processed"] += 1
                        logger.info(f"Processed frame {metadata['frame_id']} from camera {result['camera_id']} "
                                f"with {result['no_of_people']} people (latency: {total_latency*1000:.1f}ms)")
                        
                    except Exception as e:
                        logger.error(f"Error processing result: {e}", exc_info=True)

                #     current_time = time.time()
                #     if current_time - self.last_send_time >= self.send_interval:
                #         payload_str = json.dumps(results_to_send, indent=2)
                #         logger.info("About to send detection payload:\n%s", payload_str)
                #         send_task = asyncio.create_task(send_detection_data(results_to_send))
                #         send_tasks.append(send_task)
                #         self.last_send_time = current_time
                #     else:
                #         logger.info("Skipping send to maintain configured send rate")
                # except Exception as task_error:
                #     logger.error(f"Error processing task: {task_error}", exc_info=True)

            # After collecting all results
            timing_stats["result_preparation"] = time.time() - prep_start
            logger.info(f"[TIMING] Result preparation: {timing_stats['result_preparation']:.3f}s, {len(results_to_send)} results")
            if results_to_send:
                send_start = time.time()
                payload_str = json.dumps(results_to_send, indent=2)
                logger.info("About to send detection payload:\n%s", payload_str)
                success = await send_detection_data(results_to_send)
                timing_stats["sending_data"] = time.time() - send_start
                logger.info(f"[TIMING] Send detection data: {timing_stats['sending_data']:.3f}s, success: {success}")
                if not success:
                    logger.warning("Failed to send batch of results after multiple attempts")

            # Make sure to send any remaining results after the loop
            if results_to_send:
                current_time = time.time()
                if current_time - self.last_send_time >= self.send_interval:
                    payload_str = json.dumps(results_to_send, indent=2)
                    logger.info("About to send detection payload:\n%s", payload_str)
                    send_task = asyncio.create_task(send_detection_data(results_to_send))
                    send_tasks.append(send_task)
                    self.last_send_time = current_time
                else:
                    logger.info(f"Skipping send to maintain configured send rate. Will send {len(results_to_send)} results later.")
                    # # If we need to respect the interval, schedule the send for later
                    # wait_time = self.send_interval - (current_time - self.last_send_time)
                    # logger.info(f"Waiting {wait_time:.2f}s before sending final batch of {len(results_to_send)} results")
                    # await asyncio.sleep(wait_time)
                    # payload_str = json.dumps(results_to_send, indent=2)
                    # logger.info("About to send detection payload:\n%s", payload_str)
                    # send_task = asyncio.create_task(send_detection_data(results_to_send))
                    # send_tasks.append(send_task)
                    # self.last_send_time = time.time()
                    # results_to_send = []  # Clear the list after sending
                    # # Wait for all send tasks to complete
            else:
                logger.info("No people detected in any frames, skipping API call")
            if send_tasks:
                await asyncio.gather(*send_tasks)

            timing_stats["total"] = time.time() - batch_start_time
            logger.info(f"[TIMING] BATCH TOTAL: {timing_stats['total']:.3f}s | "
                   f"Buffer: {timing_stats['buffer_read']:.3f}s, "
                   f"GPU: {timing_stats['gpu_processing']:.3f}s, "
                   f"Camera: {timing_stats['camera_processing']:.3f}s, "
                   f"Prep: {timing_stats['result_preparation']:.3f}s, "
                   f"Send: {timing_stats['sending_data']:.3f}s")
            # Log processing stats periodically
            current_time = time.time()
            if current_time - self.stats["last_stats_time"] > self.stats["stats_interval"]:
                self._log_processing_stats()
                self.stats["last_stats_time"] = current_time
            
            # Calculate batch processing time
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch processing completed in {batch_time:.3f}s")
            
            # Schedule next batch immediately if frames are available
            buffer_status = self.frame_buffer.get_buffer_status()
            if buffer_status['total_frames'] > 0:
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
                # asyncio.create_task(self.process_batch())
                self.task_manager.create_task(self.process_batch(), category="batch_processing")

            else:
                self.processing_busy = False
        except Exception as e:  # Add this exception handler
            logger.error(f"Error in process_batch: {e}")
            self.processing_busy = False  # Make sure to 
                
    def _log_processing_stats(self):
        """Log processing statistics"""
        now = time.time()
        if now - self.stats["last_stats_time"] < self.stats["stats_interval"]:
            return
        if not self.stats["processing_times"]:
            return
            
        avg_latency = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
        p95_latency = sorted(self.stats["processing_times"])[int(len(self.stats["processing_times"]) * 0.95)]
        
        # Get buffer stats
        buffer_status = self.frame_buffer.get_buffer_status()
        
        # Get GPU memory stats from each processor
        gpu_mem_stats = []
        for i, proc in enumerate(self.gpu_processors):
            mem_stats = proc.get_memory_stats()
            gpu_mem_stats.append(f"GPU{i}:{mem_stats['allocated_mb']:.1f}MB")
        
        logger.info(f"Processing stats: frames={self.stats['frames_processed']}, "
                   f"avg_latency={avg_latency*1000:.1f}ms, p95_latency={p95_latency*1000:.1f}ms, "
                   f"buffer={buffer_status['total_frames']}/{buffer_status['max_total_size']} "
                   f"({buffer_status['utilization_percent']:.1f}%), "
                   f"GPU memory=[{', '.join(gpu_mem_stats)}]")
        

        interval = self.stats["stats_interval"]
        proc_fps = self.stats["frames_processed"] / interval
        in_fps   = self.stats["incoming_frames"]  / interval
        logger.info(
            f"FPS: in={in_fps:.1f}/s, proc={proc_fps:.1f}/s, "
            f"avg_latency={avg_latency*1000:.1f}ms, "
            f"buffer={buffer_status['utilization_percent']:.1f}%"
        )
        self.stats["processing_times"].clear()       
        self.stats["frames_processed"] = 0
        self.stats["incoming_frames"]  = 0
        self.stats["last_stats_time"] = now
        
    def save_annotated_frame(self, frame, camera_id, frame_number):
        """Save annotated frame for debugging (optional)"""
        output_dir = f"output/{camera_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Only save every 10th frame to reduce storage usage
        if frame_number % 10 == 0:
            filename = f"{output_dir}/frame_{frame_number:06d}.jpg"
            cv2.imwrite(filename, frame)
    
    async def run(self):
        """Main method to run the processor"""
        logger.info("Starting Kafka Stream Processor")
        allowed_stores_str = os.getenv("STORE_IDS", "[]")
        try:
            store_ids = json.loads(allowed_stores_str)
            logger.info(f"Using store IDs from ConfigMap: {store_ids}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse STORE_IDS '{allowed_stores_str}' as JSON: {e}")
            # Fallback to comma-separated format
            store_ids = [s.strip() for s in allowed_stores_str.split(",") if s.strip()]
            logger.info(f"Parsed store IDs as comma-separated list: {store_ids}")
        
        # If store_ids is empty, try to discover from Kafka
        if not store_ids:
            logger.info("No store IDs configured, attempting discovery from Kafka")
            try:
                admin = KafkaAdminClient(bootstrap_servers=self.kafka_bootstrap_servers)
                all_topics = admin.list_topics()
                
                pattern = re.compile(os.getenv("KAFKA_TOPIC_PATTERN", "^store-([0-9]+)$"))
                discovered_store_ids = {
                    m.group(1)
                    for t in all_topics
                    if (m := pattern.match(t))
                }
                logger.info(f"Discovered store IDs from Kafka: {discovered_store_ids}")
                store_ids = list(discovered_store_ids)
            except Exception as e:
                logger.error(f"Failed to discover store IDs from Kafka: {e}")
                store_ids = []

        await self.initialize_milvus_clients(store_ids)
        await initialize_processors(self.gpu_processors)    
        await self.frame_buffer.start_monitors()

        # Start the Kafka consumer loop instead of Kafka readers
        self.task_manager.create_task(
            self._kafka_consumer_loop(),
            category="kafka_consumer"
        )

        # Start the processing loop
        processing_task = self.task_manager.create_task(
            self._processing_loop(), 
            category="_processing_loop"
        )
        
        health_monitor = self.task_manager.create_task(
            self._health_monitor(), 
            category="_health_monitor"
        )

        try:
            # Wait for the processing task to complete (or be cancelled)
            await processing_task
        except asyncio.CancelledError:
            logger.info("Main processing task was cancelled")
        except Exception as e:
            logger.error(f"Error in main task loop: {e}")
        finally:
            # Clean up resources
            logger.info("Cleaning up resources...")

            # Close all Milvus clients
            for store_id, client in self.milvus_clients.items():
                try:
                    logger.info(f"Closing Milvus client for store {store_id}")
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing Milvus client for store {store_id}: {e}")

            # Wait for any remaining tasks
            await self.task_manager.wait_for_all(timeout=5.0)

            logger.info("All resources cleaned up")

    async def _kafka_consumer_loop(self):
        """
        Pull frames off Kafka topics matching the pattern and filter to only those
        listed in KAFKA_TOPICS or matching stores in STORE_IDS.
        """
        self.KAFKA_TOPIC_PATTERN = re.compile(os.getenv("KAFKA_TOPIC_PATTERN"))
        self.KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP")
        self.KAFKA_BOOTSTRAP_SERVER = os.getenv("KAFKA_BOOTSTRAP_SERVER")

        allowed_topics_str = os.getenv("KAFKA_TOPICS", "[]")
        allowed_stores_str = os.getenv("STORE_IDS", "[]")

        # Parse allowed topics
        try:
            # Try parsing as JSON first
            self.allowed_topics = json.loads(allowed_topics_str)
            logger.info(f"Loaded allowed topics from config: {self.allowed_topics}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse KAFKA_TOPICS '{allowed_topics_str}' as JSON: {e}")
            # Fall back to comma-separated format
            self.allowed_topics = [t.strip() for t in allowed_topics_str.split(",") if t.strip()]
            logger.info(f"Parsed allowed topics as comma-separated list: {self.allowed_topics}")

        # Parse allowed stores
        try:
            # Try parsing as JSON first
            self.allowed_stores = json.loads(allowed_stores_str)
            logger.info(f"Loaded allowed stores from config: {self.allowed_stores}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse STORE_IDS '{allowed_stores_str}' as JSON: {e}")
            # Fall back to comma-separated format
            self.allowed_stores = [s.strip() for s in allowed_stores_str.split(",") if s.strip()]
            logger.info(f"Parsed allowed stores as comma-separated list: {self.allowed_stores}")

        logger.info(f"KAFKA_BOOTSTRAP_SERVERS={self.KAFKA_BOOTSTRAP_SERVER}, "
                f"KAFKA_CONSUMER_GROUP={self.KAFKA_CONSUMER_GROUP}, "
                f"KAFKA_TOPIC_PATTERN={self.KAFKA_TOPIC_PATTERN.pattern}")
        logger.info(f"Allowed topics: {self.allowed_topics}")
        logger.info(f"Allowed stores: {self.allowed_stores}")

        self.kafka_consumer = KafkaConsumer(
            group_id=os.getenv("KAFKA_CONSUMER_GROUP"),
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVER"),
            auto_offset_reset="latest",
            enable_auto_commit=True,   # whether to commit offsets automatically
            value_deserializer=lambda b: json.loads(b.decode("utf-8"))
        )
        all_kafka_topics = self.kafka_consumer.topics()
        filtered_topics = []

        for topic in all_kafka_topics:
            # First check if it matches our pattern
            if not self.KAFKA_TOPIC_PATTERN.match(topic):
                continue
                
            # Then check if we have explicit topic restrictions
            if self.allowed_topics and topic not in self.allowed_topics:
                # If we have allowed topics but this isn't one of them
                # However, also check if we have allowed stores
                if self.allowed_stores:
                    # For topics like "store-001-frames", extract the store ID
                    parts = topic.split("-")
                    if len(parts) >= 2:
                        store_id = parts[1]
                        if store_id in self.allowed_stores:
                            filtered_topics.append(topic)
                            logger.info(f"Adding topic {topic} due to store ID {store_id} in allowed stores")
            else:
                # Topic is explicitly allowed or we have no restrictions
                filtered_topics.append(topic)
                logger.info(f"Adding topic {topic} - explicitly allowed or no restrictions")
        
        if not filtered_topics:
            logger.warning("No topics matched after filtering! Check your KAFKA_TOPICS and STORE_IDS values.")

        if filtered_topics:
            logger.info(f"Subscribing to filtered topics: {filtered_topics}")
            self.kafka_consumer.subscribe(topics=filtered_topics)
        else:
            logger.warning("No topics to subscribe to after filtering!")
            # You might want to sleep and retry, or exit, depending on your application needs
            await asyncio.sleep(30)
            return

        self.kafka_consumer.poll(timeout_ms=0)

        # Log the actually assigned partitions
        assigned_partitions = self.kafka_consumer.assignment()
        if assigned_partitions:
            assigned_topics = set(tp.topic for tp in assigned_partitions)
            logger.info(f"Consumer assigned to topics: {assigned_topics}")
            logger.info(f"Consumer assigned to partitions: {assigned_partitions}")
        else:
            logger.warning("No partitions assigned to consumer!")
        
        loop = asyncio.get_running_loop()

        # This will block, so run it in a threadpool
        def poll_loop():
            try:
                for msg in self.kafka_consumer:
                    # Extract store_id from topic: e.g. "store-001-frames" → "001"
                    topic_parts = msg.topic.split("-")
                    if len(topic_parts) >= 2:
                        store_id = topic_parts[1]
                    else:
                        logger.warning(f"Unexpected topic format: {msg.topic}, using topic as store_id")
                        store_id = msg.topic
                        
                    # Extract camera_id from message key: e.g. b"camera-101" → 101
                    camera_id_key = msg.key.decode()
                    try:
                        camera_id = int(camera_id_key.split("-")[1])
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Failed to parse camera_id from key '{camera_id_key}': {e}")
                        camera_id = 0  # Default value
                        
                    # Process the frame data
                    frame_data = msg.value
                    try:
                        frame_bytes = bytes.fromhex(frame_data["frame"])
                        timestamp = dict(msg.headers).get("timestamp", datetime.utcnow().isoformat())
                        
                        metadata = {
                            "store_id": store_id,
                            "camera_id": camera_id,
                            "frame_id": frame_data.get("frame_id", None),
                            "timestamp": timestamp,
                            "queued_time": time.time()
                        }
                        
                        # Decode bytes → image
                        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        
                        # Add to buffer (async)
                        asyncio.run_coroutine_threadsafe(
                            self.frame_buffer.add_frame(frame, metadata),
                            loop
                        )
                        self.stats["incoming_frames"] += 1
                    except Exception as e:
                        logger.error(f"Error processing message from topic {msg.topic}: {e}")
            except Exception as e:
                logger.error(f"Error in Kafka consumer loop: {e}", exc_info=True)
                # Signal that the loop has exited so it can be restarted
                asyncio.run_coroutine_threadsafe(
                    self._handle_consumer_error(),
                    loop
                )

        # Schedule the blocking poll in the threadpool
        await loop.run_in_executor(None, poll_loop)

    async def _processing_loop(self):
        """Background task that ensures batch processing happens regularly"""
        logger.info("Starting processing loop")
        last_cleanup_time = time.time()
        cleanup_interval = 60  # Flush remaining features every minute
        while self.running:
            try:
                # If not currently processing, check if we should start
                if not self.processing_busy:
                    buffer_status = self.frame_buffer.get_buffer_status()
                    
                    # Process if we have frames and enough time has passed
                    if (buffer_status['total_frames'] > 0 and 
                        time.time() - self.last_process_time >= self.batch_interval):
                        self.processing_busy = True
                        try:
                            await self.process_batch()
                        except Exception as e:
                            logger.error(f"Error in process_batch: {e}")
                        finally:
                            self.processing_busy = False
                        self.last_process_time = time.time()
                

                # Check if we should do a periodic cleanup of features
                current_time = time.time()
                if current_time - self.last_cache_stats_time >= self.cache_stats_interval:
                    await self._log_cache_statistics()
                    self.last_cache_stats_time = current_time
                
                # Clean up inactive store caches every 5 minutes
                if current_time - last_cleanup_time >= 300:
                    await self._cleanup_inactive_stores()
                    last_cleanup_time = current_time

                if current_time - last_cleanup_time >= cleanup_interval:
                    # Flush features for all camera processors
                    for key, camera_processor in self.camera_processors.items():
                        if hasattr(camera_processor.tracker, 'cleanup'):
                            await camera_processor.tracker.cleanup()
                    last_cleanup_time = current_time
                    logger.info("Performed periodic feature flush for all trackers")

                # Short sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                self.processing_busy = False
                await asyncio.sleep(1)  # Sleep before retrying

    async def _handle_consumer_error(self):
        """Handle Kafka consumer errors and attempt restart"""
        logger.error("Kafka consumer encountered an error, attempting to restart...")
        
        # Close existing consumer if it exists
        if hasattr(self, 'kafka_consumer') and self.kafka_consumer:
            try:
                self.kafka_consumer.close()
            except Exception as e:
                logger.error(f"Error closing Kafka consumer: {e}")
        
        # Wait before restarting
        await asyncio.sleep(5)
        
        # Restart consumer loop
        self.task_manager.create_task(
            self._kafka_consumer_loop(),
            category="kafka_consumer_restart"
        )


    async def _cleanup_inactive_stores(self):
        """Remove caches for stores that haven't been used recently"""
        current_time = time.time()
        inactive_threshold = 600  # 10 minutes

        stores_to_remove = []
        for store_id, last_access in self.store_last_access.items():
            if current_time - last_access > inactive_threshold:
                stores_to_remove.append(store_id)

        for store_id in stores_to_remove:
            logger.info(f"Removing inactive store cache: {store_id}")
            del self.store_cache_managers[store_id]
            del self.store_last_access[store_id]

            # Also clean up milvus client if not needed
            if store_id in self.milvus_clients:
                await self.milvus_clients[store_id].close()
                del self.milvus_clients[store_id]

    async def _health_monitor(self):
        """Monitor overall system health and log status"""
        logger.info("Starting health monitor")
        
        while self.running:
            try:
                # Monitor every 30 seconds
                await asyncio.sleep(30)
                
                # Get buffer status
                buffer_status = self.frame_buffer.get_buffer_status()
                
                # Check memory usage
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                # Get GPU utilization
                gpu_utils = []
                for i, proc in enumerate(self.gpu_processors):
                    if hasattr(proc, 'get_memory_stats'):
                        mem_stats = proc.get_memory_stats()
                        gpu_utils.append(f"GPU{i}:{mem_stats['allocated_mb']:.1f}/{mem_stats['peak_allocated_mb']:.1f}MB")
                
                camera_count = len(buffer_status['per_camera'])
                
                # Log health stats
                logger.info(f"Health: RAM={memory_info.rss/1024/1024:.1f}MB, "
                           f"CPU={cpu_percent:.1f}%, "
                           f"BufferUtil={buffer_status['utilization_percent']:.1f}%, "
                           f"GPUMem=[{', '.join(gpu_utils)}], "
                           f"CameraCount={camera_count}")
                
                # Check for cameras with empty buffers (potential issues)
                for camera_id, buffer_count in buffer_status['per_camera'].items():
                    if buffer_count == 0:
                        logger.warning(f"Camera {camera_id} has empty buffer, may be disconnected")
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def stop(self):
        """Stop the processor"""
        logger.info("Stopping KAfka Processor")
        self.running = False

        for key, camera_processor in self.camera_processors.items():
            try:
                if hasattr(camera_processor.tracker, 'cleanup'):
                    await camera_processor.tracker.cleanup()
                    logger.info(f"Flushed remaining features for camera processor {key}")
            except Exception as e:
                logger.error(f"Error cleaning up tracker for camera {key}: {e}")

        # Close Kafka consumer (if stored on self)
        if getattr(self, "kafka_consumer", None) is not None:
            try:
                self.kafka_consumer.close()
                logger.info("Kafka consumer closed")
            except Exception as e:
                logger.error(f"Error closing Kafka consumer: {e}")

        try:
            await self.frame_buffer.stop()
            logger.info("Frame buffer stopped")
        except Exception as e:
            logger.error(f"Error stopping frame buffer: {e}")
        
        await self.task_manager.wait_for_all(timeout=5.0)

        if not self.processing_busy:
            try:
                self.processing_busy = True
                await asyncio.wait_for(self.process_batch(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for final batch processing")
            except Exception as e:
                logger.error(f"Error in final batch processing: {e}")

        for store_id, client in self.milvus_clients.items():
            try:
                await client.close()
                logger.info(f"Closed Milvus client for store {store_id}")
            except Exception as e:
                logger.error(f"Error closing Milvus client for store {store_id}: {e}")
        
        self.thread_pool.shutdown(wait=False)
        logger.info("Kafka Stream Processor successfully stopped")

async def main():
    """Main entry point for the Kafka processing service"""
    parser = argparse.ArgumentParser(description='Kafka Stream Processor for Retail Analytics')
    parser.add_argument('--config', default='system_config.json', help='Path to camera configuration file')
    parser.add_argument('--batch-size', type=int, default=8, help='Maximum number of frames to process in a batch')
    parser.add_argument('--batch-interval', type=float, default=0.5, help='Maximum time to wait before processing a batch (seconds)')
    parser.add_argument('--fps', type=float, default=5, help='Frames per second to process from each camera')
    parser.add_argument('--send_fps', type=float, default=1, help='How frequently to hit the send_detection function')
    
    args = parser.parse_args()
    
    logger.info("Starting Kafka-backed Stream Processor service")
    
    processor = None
    try:
        with open(args.config, 'r') as f:
            base_config = json.load(f)

        system_config = base_config.get("system", {})
        buffer_config = base_config.get("buffer_settings", {})
        gpu_config = system_config.get("gpu_config", {})
        memory_management = system_config.get("memory_management", {})
        thread_pool_cfg = system_config.get("thread_pool", {})
        thread_pool_size = thread_pool_cfg.get("max_workers", 8)
        processing_config = base_config.get("processing", {})
        result_database_config = base_config.get("result_database", {})

        send_fps = result_database_config.get("send_fps",args.send_fps)


        # Determine GPU settings
        if gpu_config.get("enabled", False):
            # Use the number of devices listed in the gpu_config
            num_processors = gpu_config.get("num_processors", 2)
            # Override batch_size per GPU if provided, else use the command-line argument
            batch_size = gpu_config.get("batch_size_per_gpu", args.batch_size)
            batch_interval = processing_config.get("batch_interval", args.batch_interval)
            fps = processing_config.get("processing_fps", args.fps)
            logger.info(f"batch_size is {batch_size}, batch_interval is {batch_interval}")
        else:
            num_processors = 2  # or any default value if GPUs are not enabled
            batch_size = args.batch_size
        
        processor = KafkaProcessor(
            num_processors=num_processors,
            batch_size=batch_size,
            batch_interval=batch_interval,
            processing_fps=fps,
            frame_buffer_config=buffer_config,
            thread_pool_size=thread_pool_size,
            send_fps = send_fps
        )


        # Create the KafkaProcessor with all relevant settings

        if processor is None:
            logger.error("Processor was not initialized. Exiting.")
            return
    
        await processor.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # Clean up
        await processor.stop()
        logger.info("Kafka-backed Stream Processor service stopped")

if __name__ == "__main__":
    asyncio.run(main())