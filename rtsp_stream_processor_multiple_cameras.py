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
from processor_segment_with_transreid import is_entering_store_percent, has_left_store_percent, filter_duplicate_detections
from processor_segment_with_transreid import crop_without_resize, setup_predictor
from sender import send_detection_data

# Initialize TransReID model
from TransReID.config import cfg
from TransReID.model import make_model
from TransReID.datasets.transforms import build_transforms
from TransReID.processor import extract_features

# Create Detection objects for DeepSORT
from yolov4_deepsort.deep_sort.detection import Detection

# Get classification
from status_checker import improved_human_status

from robust_frame_buffer import RobustFrameBuffer, FrameBufferStats

from task_manager import TaskManager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def log_total_memory(gpu_processors):
    total_allocated = 0
    total_reserved = 0
    for idx, processor in enumerate(gpu_processors):
        device = processor.device  # Ensure each processor has a 'device' attribute.
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        logger.info(
            f"Processor {idx} on {device}: allocated = {allocated/1024**2:.2f} MB, "
            f"reserved = {reserved/1024**2:.2f} MB"
        )
        total_allocated += allocated
        total_reserved += reserved
        
    logger.info(
        f"Total across all processors: allocated = {total_allocated/1024**2:.2f} MB, "
        f"reserved = {total_reserved/1024**2:.2f} MB"
    )

def set_memory_limit(fraction=0.9):
    """Limit GPU memory usage to a fraction of available memory"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_properties = torch.cuda.get_device_properties(device)
        total_memory = device_properties.total_memory
        
        # Set a limit on reserved memory
        max_memory = int(total_memory * fraction)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        
        logger.info(f"Set GPU memory limit to {fraction * 100:.0f}% of total ({max_memory / (1024**3):.2f} GB)")

def get_time_window_list(image_batch):
    """
    Compute the batch time window (from and to timestamps) using list comprehension.
    
    Args:
        image_batch: List of (image, metadata) tuples.
    
    Returns:
        A tuple (from_timestamp, to_timestamp) in ISO format, or (None, None) if no valid timestamps.
    """
    timestamps = []
    for _, metadata in image_batch:
        ts_str = metadata.get('timestamp')
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                timestamps.append(ts)
            except Exception as e:
                logger.error(f"Error parsing timestamp {ts_str}: {e}")
    
    if timestamps:
        return min(timestamps).isoformat(), max(timestamps).isoformat()
    else:
        return None, None

def get_time_window_single_pass(image_batch):
    """
    Compute the batch time window (from and to timestamps) in a single pass.
    
    Args:
        image_batch: List of (image, metadata) tuples.
    
    Returns:
        A tuple (from_timestamp, to_timestamp) in ISO format, or (None, None) if no valid timestamps.
    """
    min_ts = None
    max_ts = None
    for _, metadata in image_batch:
        ts_str = metadata.get('timestamp')
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
            except Exception as e:
                logger.error(f"Error parsing timestamp {ts_str}: {e}")
                continue
            if min_ts is None or ts < min_ts:
                min_ts = ts
            if max_ts is None or ts > max_ts:
                max_ts = ts
    return (min_ts.isoformat() if min_ts else None, 
            max_ts.isoformat() if max_ts else None)

async def initialize_processors(gpu_processors):
    """Initialize processors in sequence to avoid memory spikes"""
    for i, processor in enumerate(gpu_processors):
        logger.info(f"Initializing processor {i}...")
        
        # Force initialization of models if not already done
        if not hasattr(processor, 'model') or processor.model is None:
            # Initialize model here or call a method that does
            pass
            
        await asyncio.sleep(0.5)  # Brief pause between initializations
        
        # Log memory after each initialization
        if processor.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(processor.device) / (1024**2)
            reserved = torch.cuda.memory_reserved(processor.device) / (1024**2)
            logger.info(f"After initializing processor {i}: allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")


async def fetch_camera_config(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()  # raises an exception for non-200 responses
            camera_list = await response.json()
            # config = await response.json()
            return {"cameras": camera_list}

# Import needed to match original code
class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class GPUBatchProcessor:
    """Handles batch processing of images using GPU for shared operations"""
    def __init__(self, max_batch_size=8, device=None, model_config=None):
        self.max_batch_size = max_batch_size
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

    def process_single_frame(self, image, metadata):
        """
        Optimized processing of a single image frame.
        
        Args:
            image: The image frame (numpy array).
            metadata: Associated metadata dictionary.
            
        Returns:
            A tuple (metadata, valid_detections, detection_features), where:
            - valid_detections is a list of detection tuples (bbox, score, class, mask)
            - detection_features is a list of corresponding feature vectors.
        """
        start_time = time.time()
        try:
            # Use the dedicated CUDA stream for asynchronous operations.
            with torch.cuda.stream(self.stream):
                # Run segmentation on the single image.
                outputs = self.seg_predictor(image)
                instances = outputs["instances"]
                # Select detections classified as person (assuming class 0 is person).
                person_indices = (instances.pred_classes == 0).nonzero().flatten()
                if len(person_indices) == 0:
                    # Update statistics and synchronize the stream.
                    self.stats["total_processing_time"] += time.time() - start_time
                    self.stats["total_batches_processed"] += 1
                    self.stats["total_frames_processed"] += 1
                    self._update_memory_stats()
                    self.stream.synchronize()
                    return (metadata, [], [])
                
                # Get boxes, scores, and masks for the detected persons.
                person_boxes = instances.pred_boxes.tensor[person_indices].cpu().numpy()
                person_scores = instances.scores[person_indices].cpu().numpy()
                person_masks = instances.pred_masks[person_indices].cpu().numpy()
                
                # Optionally filter duplicate detections.
                # frame_copy = image.copy()  # Only needed if visualizing or debugging duplicates.
                filtered_boxes, filtered_scores, filtered_masks = filter_duplicate_detections(
                    person_boxes, person_scores, person_masks, image, iou_threshold=0.9
                )
                
                # Define a “green box” (or entrance region) for overlap filtering.
                height, width = image.shape[:2]
                line_y = int(height * 0.10)  # e.g., 10% from the top.
                green_box = [0, line_y, width, height]
                
                valid_detections = []
                detection_crops = []
                # Iterate over filtered detections.
                # Convert the entire frame once to RGB
                # if 'converted_frame' not in locals():
                converted_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                for j, bbox in enumerate(filtered_boxes):
                    if self.is_blacklisted(bbox):
                        continue
                    score = filtered_scores[j]
                    mask = filtered_masks[j]
                    overlap_area, bbox_area = self.calculate_overlap(bbox, green_box)
                    if overlap_area / bbox_area > 0.7:
                        # Convert bbox to [x, y, width, height] format for DeepSORT.
                        tlwh_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        detection = (tlwh_bbox, score, "person", mask)
                        valid_detections.append(detection)
                        
                        # Prepare crop for feature extraction.
                        x, y, w, h = map(int, tlwh_bbox)
                        crop_masked = crop_without_resize(converted_frame, [x, y, x + w, y + h], mask)
                        if crop_masked.size == 0 or w <= 0 or h <= 0:
                            continue
                        # Convert crop to PIL and apply transformation.
                        crop_pil = Image.fromarray(crop_masked)
                        # crop_pil = Image.fromarray(cv2.cvtColor(crop_masked, cv2.COLOR_BGR2RGB))
                        transformed_crop = self.transform(crop_pil).unsqueeze(0)
                        detection_crops.append(transformed_crop)
                
                # If no valid detections remain, return empty results.
                if not valid_detections:
                    self.stats["total_processing_time"] += time.time() - start_time
                    self.stats["total_batches_processed"] += 1
                    self.stats["total_frames_processed"] += 1
                    self._update_memory_stats()
                    self.stream.synchronize()
                    return (metadata, [], [])
                
                # Process detection crops in mini-batches (if needed).
                detection_features = [None] * len(valid_detections)
                if detection_crops:
                    for start_idx in range(0, len(detection_crops), self.max_batch_size):
                        end_idx = min(start_idx + self.max_batch_size, len(detection_crops))
                        mini_batch = detection_crops[start_idx:end_idx]
                        batch_tensor = torch.cat(mini_batch, dim=0).to(self.device)
                        
                        if self.use_half_precision and self.device.type == "cuda":
                            batch_tensor = batch_tensor.half()
                        
                        with torch.no_grad():
                            features = self.extract_features(self.model, batch_tensor).cpu().float().numpy()
                        
                        # Assign extracted features to their corresponding detection.
                        for idx, feat_idx in enumerate(range(start_idx, end_idx)):
                            detection_features[feat_idx] = features[idx].reshape(-1)
                
                # For any detection missing features, insert a zero vector.
                for i in range(len(detection_features)):
                    if detection_features[i] is None:
                        detection_features[i] = np.zeros((768,))
                
                # Update processing stats.
                self.stats["total_processing_time"] += time.time() - start_time
                self.stats["total_batches_processed"] += 1
                self.stats["total_frames_processed"] += 1
                self.stats["total_people_detected"] += len(valid_detections)
                self._update_memory_stats()
                self.stream.synchronize()
                
                return (metadata, valid_detections, detection_features)
        
        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA out of memory error on {self.device}")
            torch.cuda.empty_cache()
            return (metadata, [], [])
        
        except Exception as e:
            logger.error(f"Error processing single frame on {self.device}: {e}", exc_info=True)
            return (metadata, [], [])

    
    def process_batch(self, image_batch):
        """
        Process a batch of images to extract detections and features
        
        Args:
            image_batch: List of (image, metadata) tuples
        
        Returns:
            List of (metadata, detections, features) tuples
        """
        start_time = time.time()
        # logger.info(f"Processing batch of {len(image_batch)} images on GPU")
        batch_results = []
        timestamps = []
        total_people = 0
        
        try:
            # Step 1: Run segmentation on each image
            with torch.cuda.stream(self.stream):
                for i, (image, metadata) in enumerate(image_batch):
                    # Run segmentation model (on GPU)
                    outputs = self.seg_predictor(image)
                    instances = outputs["instances"]
                    person_indices = (instances.pred_classes == 0).nonzero().flatten()
                    
                    # Skip if no people detected
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
                            
                            # Prepare crop for feature extraction
                            x, y, w, h = map(int, tlwh_bbox)
                            crop_masked = crop_without_resize(image, [x, y, x+w, y+h], mask)
                            
                            # Skip empty crops
                            if crop_masked.size == 0 or w <= 0 or h <= 0:
                                continue
                            
                            # Convert to PIL for TransReID
                            crop_pil = Image.fromarray(cv2.cvtColor(crop_masked, cv2.COLOR_BGR2RGB))
                            detection_crops.append(self.transform(crop_pil).unsqueeze(0))
                            detection_indices.append(len(valid_detections) - 1)
                    
                    # Skip if no valid detections
                    if not valid_detections:
                        batch_results.append((metadata, [], []))
                        continue
                    
                    # Step 2: Batch extract features for this image's detections
                    detection_features = [None] * len(valid_detections)
                    
                    if detection_crops:
                        # Process crops in mini-batches to avoid GPU memory issues
                        for start_idx in range(0, len(detection_crops), self.max_batch_size):
                            end_idx = min(start_idx + self.max_batch_size, len(detection_crops))
                            mini_batch = detection_crops[start_idx:end_idx]
                            mini_indices = detection_indices[start_idx:end_idx]
                            
                            # Concatenate crops for batch processing
                            batch_tensor = torch.cat(mini_batch, dim=0).to(self.device)

                            # Convert to half precision if enabled
                            if self.use_half_precision and self.device.type == "cuda":
                                batch_tensor = batch_tensor.half()
                            
                            # Extract features using TransReID
                            with torch.no_grad():
                                features = self.extract_features(self.model, batch_tensor).cpu().float().numpy()
                            
                            # Assign features to detections
                            for idx, feat_idx in enumerate(mini_indices):
                                detection_features[feat_idx] = features[idx].reshape(-1)
                    
                    # Fill in empty features with zeros
                    for i in range(len(detection_features)):
                        if detection_features[i] is None:
                            detection_features[i] = np.zeros((768,))  # TransReID feature dim
                    
                    # Add results for this image
                    batch_results.append((metadata, valid_detections, detection_features))
                    total_people += len(valid_detections)

                    # Update stats
                processing_time = time.time() - start_time
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
    


class CameraProcessor:
    """Handles per-camera tracking and processing"""
    def __init__(self, camera_id, store_id):
        self.camera_id = camera_id
        self.store_id = store_id
        
        # Initialize tracker and status tracking
        self.processor = Segmentation_DeepSort(info_flag=True)
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
    
    def process_frame(self, frame, frame_id, detections, features, timestamp):
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
        self.tracker.update(detection_objs)
        
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
        
        # Save tracker state - only save every 30 frames to reduce I/O
        if self.frame_count % 30 == 0:
            self.tracker.save_global_database()
        
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


class RTSPStreamProcessor:
    """Manages processing for multiple RTSP camera streams"""
    def __init__(self, num_processors = 2, batch_size=8, batch_interval=0.5, processing_fps=5,
    gpu_processors=None, frame_buffer_config=None, thread_pool_size=8, send_fps = 5):
        # self.gpu_processor = GPUBatchProcessor(max_batch_size=batch_size)
        set_memory_limit(fraction=0.9)
        self.num_processors = num_processors
        self.active_processing_tasks = set()

        if gpu_processors is None:
            self.gpu_processors = [
            GPUBatchProcessor(
                max_batch_size=batch_size, 
                device=torch.device("cuda:0"),  # All use the same device
                model_config={"context_id": i}  # Give each a unique context ID
            )
            for i in range(self.num_processors)
        ]
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
        self.camera_streams = {}
        self.running = True

        # Performance tracking
        self.stats = {
            "frames_processed": 0,
            "frames_dropped": 0,
            "processing_times": deque(maxlen=1000),
            "last_stats_time": time.time(),
            "stats_interval": 60  # Log stats every minute
        }
        self.task_manager = TaskManager()
        self.send_interval = 1.0 / send_fps
        self.last_send_time = time.time()
        
        logger.info(f"Initialized RTSPStreamProcessor with {len(self.gpu_processors)} GPU processors")
        
        # Log GPU device details
        for i, proc in enumerate(self.gpu_processors):
            logger.info(f"GPU processor {i}: device={proc.device}")
    
    def _select_processor_for_batch(self):
        """Select the least busy processor for the next batch"""
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
            self.camera_processors[key] = CameraProcessor(camera_id, store_id)
        return self.camera_processors[key]
    
    def add_camera(self, rtsp_url, camera_id, store_id):
        """Add a camera to be processed"""
        logger.info(f"Adding camera {camera_id} in store {store_id} with URL {rtsp_url}")
        self.camera_streams[camera_id] = {
            'url': rtsp_url,
            'store_id': store_id,
            'camera_id': camera_id,
            'frame_count': 0,
            'capture': None,
            'last_frame_time': 0,
            'frame_interval': 1.0 / self.processing_fps  # Time between frames to process
        }
    
    def open_stream(self, camera_id):
        """Open the RTSP stream for a camera"""
        if camera_id not in self.camera_streams:
            logger.error(f"Camera {camera_id} not in camera streams")
            return False
            
        stream_info = self.camera_streams[camera_id]
        
        # If already open, close it first
        if stream_info['capture'] is not None:
            try:
                stream_info['capture'].release()
            except Exception as e:
                logger.error(f"Error closing existing stream: {e}")
        
        # Configure OpenCV capture with RTSP transport
        capture = cv2.VideoCapture(stream_info['url'], cv2.CAP_FFMPEG)
        
        # Set additional parameters for RTSP streaming
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Minimize buffer size to reduce latency
        
        if not capture.isOpened():
            logger.error(f"Failed to open RTSP stream: {stream_info['url']}")
            capture.release() 
            return False
        
        # Reset frame count and store the capture object
        stream_info['frame_count'] = 0
        stream_info['capture'] = capture
        stream_info['last_frame_time'] = time.time()
        
        logger.info(f"Successfully opened RTSP stream for camera {camera_id}")
        return True
    
    async def frame_reader_task(self, camera_id):
        """Task to read frames from a specific camera stream"""
        logger.info(f"Starting frame reader for camera {camera_id}")
        
        if camera_id not in self.camera_streams:
            logger.error(f"Camera {camera_id} not found in camera streams")
            return
            
        stream_info = self.camera_streams[camera_id]

        # Connection retry variables
        max_connection_retries = 5
        connection_retry_count = 0
        connection_retry_delay = 5  # seconds
        
        if not self.open_stream(camera_id):
            logger.error(f"Failed to open stream for camera {camera_id}, retrying in 5 seconds")
            connection_retry_count += 1
            while connection_retry_count < max_connection_retries and not self.open_stream(camera_id):
                logger.error(f"Retry {connection_retry_count}/{max_connection_retries} failed")
                await asyncio.sleep(connection_retry_delay)
                connection_retry_count += 1
            if connection_retry_count >= max_connection_retries:
                logger.error(f"Failed to open stream for camera {camera_id} after {max_connection_retries} attempts")
                return
        # Local frame counters for stats
        frames_read = 0
        frames_skipped = 0
        last_stats_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 10

        while self.running:
            try:
                # Check time since last processed frame
                current_time = time.time()
                time_since_last_frame = current_time - stream_info['last_frame_time']
                
                # Skip if not enough time has passed (to maintain desired FPS)
                if time_since_last_frame < stream_info['frame_interval']:
                    await asyncio.sleep(0.01)  # Short sleep to avoid CPU spin
                    continue

                # Check buffer status - skip reading if buffer is getting very full
                buffer_status = self.frame_buffer.get_buffer_status()
                camera_buffer_size = buffer_status['per_camera'].get(camera_id, 0)
                if camera_buffer_size >= buffer_status['max_size_per_camera'] * 0.9:
                    # Buffer almost full, skip this frame to avoid memory issues
                    frames_skipped += 1
                    await asyncio.sleep(0.05)
                    continue
                
                # Read frame from RTSP stream
                ret, frame = stream_info['capture'].read()
                
                # Handle end of stream or error
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame from camera {camera_id}, reconnecting...")
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Reconnecting to camera {camera_id} after {consecutive_failures} consecutive failures")
                        if not self.open_stream(camera_id):
                            logger.error(f"Failed to reconnect to camera {camera_id}, retrying in 5 seconds")
                            await asyncio.sleep(5)
                            continue
                        consecutive_failures = 0
                    else:
                        # Small delay before retry
                        await asyncio.sleep(0.1)
                        continue
                consecutive_failures = 0
                
                # Update frame count and last frame time
                stream_info['frame_count'] += 1
                stream_info['last_frame_time'] = current_time
                frames_read += 1
                # Create metadata for this frame
                metadata = {
                    'store_id': stream_info['store_id'],
                    'camera_id': camera_id,
                    'frame_id': stream_info['frame_count'],
                    'timestamp': datetime.utcnow().isoformat(),
                    'queued_time': time.time()  # Track when frame was added to buffer
                }
                await self.frame_buffer.add_frame(frame, metadata)
                # Add to buffer for batch processing
                # self.frame_buffer.append((frame, metadata))
                
                # Check if it's time to process a batch
                buffer_status = self.frame_buffer.get_buffer_status()
                if ((not self.processing_busy and (current_time - self.last_process_time >= self.batch_interval)) or 
                    (buffer_status['total_frames'] >= self.gpu_processors[0].max_batch_size)):
                    # self.processing_busy = True
                    # asyncio.create_task(self.process_batch())
                    # self.task_manager.create_task(self.process_batch(), category="batch_processing")
                    # self.active_processing_tasks.add(task)
                    # task.add_done_callback(self._task_done_callback)
                    # await self.process_batch()
                    self.task_manager.create_task(self.process_batch(), category="batch_processing")
                    self.last_process_time = current_time

                # Log reader statistics periodically
                if current_time - last_stats_time > 30:  # Every 30 seconds
                    fps_actual = frames_read / (current_time - last_stats_time)
                    skip_rate = frames_skipped / max(1, frames_read + frames_skipped) * 100
                    buffer_size = buffer_status['per_camera'].get(camera_id, 0)
                    
                    logger.info(f"Camera {camera_id} stats: fps={fps_actual:.2f}, "
                                f"skip_rate={skip_rate:.1f}%, buffer_size={buffer_size}")
                    
                    # Reset counters
                    frames_read = 0
                    frames_skipped = 0
                    last_stats_time = current_time

                # Sleep briefly to avoid hogging CPU
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in frame reader for camera {camera_id}: {e}", exc_info=True)
                await asyncio.sleep(1)  # Sleep before retrying

    def _task_done_callback(self, task):
        """Callback function when a processing task completes."""
        # Remove the task from our tracking set
        self.active_processing_tasks.discard(task)
        
        # Check if it raised any exceptions
        if not task.cancelled() and task.exception() is not None:
            logger.error(f"Processing task failed with error: {task.exception()}")
    
    async def process_batch(self):
        """Process all frames in the current batch"""
        try:
            # Get batch from the robust buffer using fair distribution
            loop = asyncio.get_running_loop()
            current_batch = await self.frame_buffer.get_next_batch(
                max_batch_size=self.gpu_processors[0].max_batch_size,
                strategy='fair'  # Ensures all cameras get processing time
            )
            
            if not current_batch:
                logger.debug("No frames in buffer to process")
                self.processing_busy = False
                return
           # if not self.frame_buffer:
           #     return
           # batch = await self.frame_buffer.get_next_batch(self.gpu_processor.max_batch_size)
           # if not batch:
           #     return  
           # Get current buffer and clear it
           # current_batch = self.frame_buffer.copy()

           # if not current_batch:
           #     return
           
            batch_start_time = time.time()
            # logger.info(f"Processing batch of {len(current_batch)} frames")
                    
            # from_ts, to_ts = get_time_window_list(current_batch)
            # logger.info(f"Batch time window (list approach): from {from_ts} to {to_ts}")
            # Process batch on GPU (segmentation + feature extraction)
            # Use round-robin to select a GPU processor for processing this batch
            processor = self._select_processor_for_batch()
            # processor = self.gpu_processors[self.current_processor_index]
            # self.current_processor_index = (self.current_processor_index + 1) % len(self.gpu_processors)
            batch_results = await loop.run_in_executor(
                None,  # Use default executor
                lambda: processor.process_batch(current_batch)
            )
            
            # Process each result with its camera processor (CPU-intensive)
            tasks = []
            for metadata, detections, features in batch_results:
                if not detections:  # Skip empty results
                    continue
                store_id = metadata['store_id']
                camera_id = metadata['camera_id']
                frame_id = metadata['frame_id']
                timestamp = metadata['timestamp']
                
                # Find the original frame
                original_frame = None
                for frame, meta in current_batch:
                    if meta['camera_id'] == camera_id and meta['frame_id'] == frame_id:
                        original_frame = frame
                        break
                
                if original_frame is None:
                    logger.error(f"Original frame not found for camera {camera_id}, frame {frame_id}")
                    continue
                
                # Get camera processor
                processor = self.get_camera_processor(camera_id, store_id)
                
                # Create task for CPU processing
                task = loop.run_in_executor(
                    self.thread_pool,
                    processor.process_frame,
                    original_frame,
                    frame_id,
                    detections,
                    features,
                    timestamp
                )
                tasks.append((task, metadata))
            
            # Wait for all processing to complete
            send_tasks = []
            results_to_send = []  # Create a list to store all results
            for task, metadata in tasks:
                try:
                    # result, annotated_frame = await task
                    result = await task

                    # Ensure all required fields are present regardless of detection count
                    if "image_url" not in result or not result["image_url"]:
                        result["image_url"] = ""  # Default empty string if not already set

                    if "camera_id" not in result or not result["camera_id"]:
                        result["camera_id"] = ""  # Default empty string if not already set

                    if "is_organised" not in result:
                        result["is_organised"] = True  # Default to True
                        
                    # Make sure date_time is properly set
                    if "date_time" not in result or not result["date_time"]:
                        result["date_time"] = metadata["timestamp"]

                    if "persons" not in result:
                        result["persons"] = []
                    
                    # if result["persons"]:
                    #     for person in result["persons"]:
                    #         if "coords" not in person:
                    #             person["coords"] = {
                    #                 "x": str(person.pop("x_coord", 0)),
                    #                 "y": str(person.pop("y_coord", 0))
                    #                 }
                    
                            # # Ensure all required person fields exist
                            # if "person_id" not in person:
                            #     person["person_id"] = "unknown"
                            # if "type" not in person:
                            #     person["type"] = "unknown"
                            # if "group_id" not in person:
                            #     person["group_id"] = ""

                    results_to_send.append(result)
                    # result["from_datetime"] = from_ts
                    # result["to_datetime"] = to_ts
                    # Calculate processing latency
                    queued_time = metadata.get('queued_time', time.time())
                    total_latency = time.time() - queued_time
                    # Track processing time for stats
                    self.stats["processing_times"].append(total_latency)
                    self.stats["frames_processed"] += 1

                    # Add latency info to result
                    # result['processing_latency'] = total_latency
                    logger.info(f"Processed frame {frame_id} from camera {result['camera_id']} "
                               f"with {result['no_of_people']} people (latency: {total_latency*1000:.1f}ms)")
                    current_time = time.time()
                    if current_time - self.last_send_time >= self.send_interval:
                        send_task = asyncio.create_task(send_detection_data(results_to_send))
                        send_tasks.append(send_task)
                        self.last_send_time = current_time
                    else:
                        logger.debug("Skipping send to maintain configured send rate")
                except Exception as task_error:
                    logger.error(f"Error processing task: {task_error}", exc_info=True)

            # After collecting all results
            if results_to_send:
                success = await send_detection_data(results_to_send)
                if not success:
                    logger.warning("Failed to send batch of results after multiple attempts")

            # Make sure to send any remaining results after the loop
            if results_to_send:
                current_time = time.time()
                if current_time - self.last_send_time >= self.send_interval:
                    send_task = asyncio.create_task(send_detection_data(results_to_send))
                    send_tasks.append(send_task)
                    self.last_send_time = current_time
                    logger.info(f"Sent final batch of {len(results_to_send)} results")
                    results_to_send = []  # Clear the list after sending
                else:
                    # If we need to respect the interval, schedule the send for later
                    wait_time = self.send_interval - (current_time - self.last_send_time)
                    logger.debug(f"Waiting {wait_time:.2f}s before sending final batch of {len(results_to_send)} results")
                    await asyncio.sleep(wait_time)
                    send_task = asyncio.create_task(send_detection_data(results_to_send))
                    send_tasks.append(send_task)
                    self.last_send_time = time.time()
                    results_to_send = []  # Clear the list after sending
                    # Wait for all send tasks to complete
            # if send_tasks:
                # await asyncio.gather(*send_tasks)

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
        
        # Reset counters
        self.stats["frames_processed"] = 0
    
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
        logger.info("Starting RTSP Stream Processor")
        await initialize_processors(self.gpu_processors)
        await self.frame_buffer.start_monitors()
        # processing_task = asyncio.create_task(self._processing_loop())
        processing_task = self.task_manager.create_task(self._processing_loop(), category="_processing_loop")
        
        # Start frame readers for all cameras
        frame_readers = []
        for camera_id in self.camera_streams:
            self.task_manager.create_task(self.frame_reader_task(camera_id), category="frame_reader_task")
            # frame_readers.append(frame_render)
        
        # Start health monitor
        # health_monitor = asyncio.create_task(self._health_monitor())
        self.task_manager.create_task(self._health_monitor(), category="_health_monitor")
    
        # all_tasks = frame_readers + [processing_task, health_monitor]
        try:
            # Note: Even though these tasks are being tracked by the task manager,
            # we still need to await them directly here to keep the run() method running
            # await asyncio.gather(*all_tasks)
            await processing_task
        except asyncio.CancelledError:
            logger.info("Main processing task was cancelled")
        except Exception as e:
            logger.error(f"Error in main task loop: {e}")
            # # Cancel all tasks on error
            # for task in all_tasks:
            #     if not task.done():
            #         task.cancel()
        finally:
            # Wait for any other background tasks managed by the task manager
            # This will handle any other tasks created during execution
            await self.task_manager.wait_for_all(timeout=5.0)

    async def _processing_loop(self):
        """Background task that ensures batch processing happens regularly"""
        logger.info("Starting processing loop")
        
        while self.running:
            try:
                # If not currently processing, check if we should start
                if not self.processing_busy:
                    buffer_status = self.frame_buffer.get_buffer_status()
                    
                    # Process if we have frames and enough time has passed
                    if (buffer_status['total_frames'] > 0 and 
                        time.time() - self.last_process_time >= self.batch_interval):
                        self.processing_busy = True
                        await self.process_batch()
                        self.last_process_time = time.time()
                
                # Short sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Sleep before retrying
    
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
                
                # Log health stats
                logger.info(f"Health: RAM={memory_info.rss/1024/1024:.1f}MB, "
                           f"CPU={cpu_percent:.1f}%, "
                           f"BufferUtil={buffer_status['utilization_percent']:.1f}%, "
                           f"GPUMem=[{', '.join(gpu_utils)}], "
                           f"CameraCount={len(self.camera_streams)}")
                
                # Check for cameras with empty buffers (potential issues)
                for camera_id, buffer_count in buffer_status['per_camera'].items():
                    if buffer_count == 0 and camera_id in self.camera_streams:
                        logger.warning(f"Camera {camera_id} has empty buffer, may be disconnected")
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def stop(self):
        """Stop the processor"""
        logger.info("Stopping RTSP Stream Processor")
        self.running = False
        await self.task_manager.wait_for_all(timeout=5.0)
        # Wait for all active processing tasks to complete
        # if self.active_processing_tasks:
        #     logger.info(f"Waiting for {len(self.active_processing_tasks)} active processing tasks to complete")
        #     await asyncio.gather(*self.active_processing_tasks, return_exceptions=True)
            
        try:
            await self.frame_buffer.stop()
        except Exception as e:
            logger.error(f"Error stopping frame buffer: {e}")
        # Release all camera captures
        for camera_id, stream_info in self.camera_streams.items():
            if stream_info['capture'] is not None:
                try:
                    stream_info['capture'].release()
                    logger.info(f"Released camera {camera_id}")
                except Exception as e:
                    logger.error(f"Error releasing camera {camera_id}: {e}")
        
        # Process any remaining frames
        if not self.processing_busy:
            try:
                self.processing_busy = True
                await asyncio.wait_for(self.process_batch(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for final batch processing")
            except Exception as e:
                logger.error(f"Error in final batch processing: {e}")
        self.thread_pool.shutdown(wait=False)
        
        logger.info("RTSP Stream Processor successfully stopped")

class FrameByFrameProcessor(RTSPStreamProcessor):
    """Processes frames individually rather than in batches"""
    
    def __init__(self, processing_fps=5, gpu_processors=None, num_processors = 2, frame_buffer_config=None, thread_pool_size=8, send_fps = 1,):
        # Call parent constructor but with batch_size=1
        super().__init__(
            batch_size=1, 
            batch_interval=0.001, 
            num_processors = num_processors,
            processing_fps=processing_fps,
            gpu_processors=gpu_processors,
            frame_buffer_config=frame_buffer_config,
            thread_pool_size=thread_pool_size,
            send_fps = send_fps
        )        
        # Override the thread pool to be slightly larger
        # self.thread_pool = ThreadPoolExecutor(max_workers=8)
        logger.info("Initialized FrameByFrameProcessor for immediate frame processing")
    
    async def continuous_frame_processor(self):
        """Continuously process frames from the buffer without creating new tasks each time."""
        loop = asyncio.get_running_loop()
        while self.running:
            # Fetch a single frame (or a small batch) from the buffer.
            frame_batch = await self.frame_buffer.get_next_batch(1, strategy='oldest')
            if not frame_batch:
                await asyncio.sleep(0.001)
                continue

            # Unpack the frame and metadata.
            frame, metadata = frame_batch[0]
            
            # Select a GPU processor (for simplicity, we assume one processor here)
            processor = self._select_processor_for_batch()
            
            # Process the frame on the GPU
            metadata, detections, features = await loop.run_in_executor(
                None,
                lambda: processor.process_single_frame(frame, metadata)
            )

            # If no detections, skip to the next frame
            if not detections:
                continue

            # Process with the camera-specific tracker on the CPU
            cam_processor = self.get_camera_processor(metadata['camera_id'], metadata['store_id'])
            result, annotated_frame = await loop.run_in_executor(
                self.thread_pool,
                cam_processor.process_frame,
                frame,
                metadata['frame_id'],
                detections,
                features,
                metadata['timestamp']
            )

            # Ensure all required fields are present regardless of detection count
            if "image_url" not in result or not result["image_url"]:
                result["image_url"] = ""  # Default empty string if not already set

            if "camera_id" not in result or not result["camera_id"]:
                result["camera_id"] = ""  # Default empty string if not already set

            if "is_organised" not in result:
                result["is_organised"] = True  # Default to True
                
            # Make sure date_time is properly set
            if "date_time" not in result or not result["date_time"]:
                result["date_time"] = metadata["timestamp"]

            if "persons" not in result:
                result["persons"] = []
                    

            # Immediately dispatch the result
            await send_detection_data([result])
            logger.info(f"Processed frame {metadata['frame_id']} from camera {metadata['camera_id']}")

    async def run(self):
        """Optimized run method using a continuous frame processor."""
        logger.info("Starting Optimized Frame-By-Frame Processor")
        await initialize_processors(self.gpu_processors)
        await self.frame_buffer.start_monitors()
        # Start the optimized continuous processing loop once
        self.task_manager.create_task(self.continuous_frame_processor(), category="continuous_processor")
        # Optionally, start health monitoring or other background tasks
        self.task_manager.create_task(self._health_monitor(), category="health_monitor")
        await self.task_manager.wait_for_all(timeout=5.0)
    
    async def frame_reader_task(self, camera_id):
        """Override the frame reader to process each frame immediately"""
        logger.info(f"Starting frame-by-frame reader for camera {camera_id}")
        
        if camera_id not in self.camera_streams:
            logger.error(f"Camera {camera_id} not found in camera streams")
            return
            
        stream_info = self.camera_streams[camera_id]

        # Try to open the stream initially with retries
        connection_retry_count = 0
        max_connection_retries = 5
        connection_retry_delay = 5  # seconds
        
        if not self.open_stream(camera_id):
            logger.error(f"Failed to open stream for camera {camera_id}, retrying...")
            connection_retry_count += 1
            
            while connection_retry_count < max_connection_retries and not self.open_stream(camera_id):
                logger.error(f"Retry {connection_retry_count}/{max_connection_retries} failed")
                await asyncio.sleep(connection_retry_delay)
                connection_retry_count += 1
            
            if connection_retry_count >= max_connection_retries:
                logger.error(f"Failed to open stream for camera {camera_id} after {max_connection_retries} attempts")
                return
    
            # Track stats
        frames_read = 0
        frames_processed = 0
        frames_skipped = 0
        last_stats_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running:
            try:
                # Check time since last processed frame
                current_time = time.time()
                time_since_last_frame = current_time - stream_info['last_frame_time']
                
                # Skip if not enough time has passed (to maintain desired FPS)
                if time_since_last_frame < stream_info['frame_interval']:
                    await asyncio.sleep(0.01)  # Short sleep to avoid CPU spin
                    continue
                
                # Read frame from RTSP stream
                ret, frame = stream_info['capture'].read()
                
                # Handle end of stream or error
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame from camera {camera_id},  ({consecutive_failures}/{max_consecutive_failures})")
                    if not self.open_stream(camera_id):
                        logger.error(f"Failed to reconnect to camera {camera_id}, retrying in 5 seconds")

                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Reconnecting to camera {camera_id} after {consecutive_failures} consecutive failures")
                        if not self.open_stream(camera_id):
                            logger.error(f"Failed to reconnect to camera {camera_id}, retrying in {connection_retry_delay} seconds")
                            await asyncio.sleep(connection_retry_delay)
                            continue
                        consecutive_failures = 0
                    else:
                        # Small delay before retry
                        await asyncio.sleep(0.1)
                        continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                
                # Update frame count and last frame time
                stream_info['frame_count'] += 1
                stream_info['last_frame_time'] = current_time
                frames_read += 1
                # Create metadata for this frame
                metadata = {
                    'store_id': stream_info['store_id'],
                    'camera_id': camera_id,
                    'frame_id': stream_info['frame_count'],
                    'timestamp': datetime.utcnow().isoformat(),
                    'queued_time': time.time()
                }
                added = await self.frame_buffer.add_frame(frame, metadata)
            
                if added:
                    # Process this frame immediately
                    if not self.processing_busy:
                        self.processing_busy = True
                        # asyncio.create_task(self.process_single_frame())
                        self.task_manager.create_task(self.process_single_frame(), category="process_single_frame")
                        frames_processed += 1
                    else:
                        # If busy, skip individual processing and wait for batch
                        frames_skipped += 1
                
                # Log reader statistics periodically
                if current_time - last_stats_time > 30:  # Every 30 seconds
                    fps_actual = frames_read / (current_time - last_stats_time)
                    process_rate = frames_processed / max(1, frames_read) * 100
                    skip_rate = frames_skipped / max(1, frames_read) * 100
                    
                    logger.info(f"Camera {camera_id} stats: fps={fps_actual:.2f}, "
                                f"process_rate={process_rate:.1f}%, "
                                f"skip_rate={skip_rate:.1f}%")
                    
                    # Reset counters
                    frames_read = 0
                    frames_processed = 0
                    frames_skipped = 0
                    last_stats_time = current_time
                
                # Sleep briefly to avoid hogging CPU
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in frame reader for camera {camera_id}: {e}", exc_info=True)
                await asyncio.sleep(1)  # Sleep before retrying
    
    
    async def process_single_frame(self):
        """Process a single frame immediately"""
        try:
            # Create a batch with just this one frame
            single_frame_batch = await self.frame_buffer.get_next_batch(1, strategy='oldest')

            if not single_frame_batch:
                self.processing_busy = False
                return
            frame, metadata = single_frame_batch[0]
            # Select GPU processor using round-robin
            processor = self.gpu_processors[self.current_processor_index]
            self.current_processor_index = (self.current_processor_index + 1) % len(self.gpu_processors)
            
            # Process on selected GPU
            loop = asyncio.get_running_loop()
            metadata, detections, features = await loop.run_in_executor(
                None,
                lambda: processor.process_single_frame(frame, metadata)
            )
            
            # # Should only have one result
            # if not batch_results:
            #     self.processing_busy = False
            #     return
                
            # metadata, detections, features = batch_results[0]
            
            # Skip if no detections
            if not detections:
                self.processing_busy = False
                return
                
            store_id = metadata['store_id']
            camera_id = metadata['camera_id']
            frame_id = metadata['frame_id']
            timestamp = metadata['timestamp']
            
            # Get the original frame
            # frame, _ = single_frame_batch[0]
            
            # Get camera processor
            processor = self.get_camera_processor(camera_id, store_id)
            
            # Process with camera-specific tracker
            # result, annotated_frame = await loop.run_in_executor(
            result = await loop.run_in_executor(
                self.thread_pool,
                processor.process_frame,
                frame,
                frame_id,
                detections,
                features,
                timestamp
            )
            
            # Calculate processing latency
            queued_time = metadata.get('queued_time', time.time())
            total_latency = time.time() - queued_time
            
            # Track processing time for stats
            self.stats["processing_times"].append(total_latency)
            self.stats["frames_processed"] += 1
            
            # Add latency info to result
            # result['processing_latency'] = total_latency
            
            # Send result
            current_time = time.time()
            if current_time - self.last_send_time >= self.send_interval:
                success = await send_detection_data([result])
                if not success:
                    logger.warning(f"Failed to send detection data for frame {frame_id}")
                self.last_send_time = current_time
            else:
                logger.debug("Skipping send to maintain configured send rate")
            
            logger.info(f"Processed frame {frame_id} from camera {camera_id}: "
                      f"{len(detections)} detections, latency: {total_latency*1000:.1f}ms")
            
            # Log processing stats periodically
            current_time = time.time()
            if current_time - self.stats["last_stats_time"] > self.stats["stats_interval"]:
                self._log_processing_stats()
                self.stats["last_stats_time"] = current_time
            
            # Check if there are more frames to process
            buffer_status = self.frame_buffer.get_buffer_status()
            if buffer_status['total_frames'] > 0:
                # Process next frame immediately
                # asyncio.create_task(self.process_single_frame())
                self.task_manager.create_task(self.process_single_frame(), category="process_single_frame")
            else:
                self.processing_busy = False
                
        except Exception as e:
            logger.error(f"Error in process_single_frame: {e}", exc_info=True)
            self.processing_busy = False


async def main():
    """Main entry point for the RTSP processing service"""
    parser = argparse.ArgumentParser(description='RTSP Stream Processor for Retail Analytics')
    parser.add_argument('--config', default='camera_config.json', help='Path to camera configuration file')
    parser.add_argument('--batch-size', type=int, default=8, help='Maximum number of frames to process in a batch')
    parser.add_argument('--batch-interval', type=float, default=0.5, help='Maximum time to wait before processing a batch (seconds)')
    parser.add_argument('--fps', type=float, default=5, help='Frames per second to process from each camera')
    parser.add_argument('--send_fps', type=float, default=1, help='How frequently to hit the send_detection function')
    
    args = parser.parse_args()
    
    logger.info("Starting RTSP Stream Processor service")
    
    # # Create processor
    # processor = RTSPStreamProcessor(
    #     batch_size=args.batch_size,
    #     batch_interval=args.batch_interval,
    #     processing_fps=args.fps
    # )
    
    # Load camera configuration
    processor = None
    try:
        camera_config_remote = await fetch_camera_config("http://genfied-api.xperie.nz:8000/api/v1/ai-server/config")
        with open(args.config, 'r') as f:
            base_config = json.load(f)
        base_config['cameras'] = camera_config_remote.get('cameras', [])

        system_config = base_config.get("system", {})
        buffer_config = base_config.get("buffer_settings", {})
        gpu_config = system_config.get("gpu_config", {})
        memory_management = system_config.get("memory_management", {})
        thread_pool_config = system_config.get("thread_pool", {})
        thread_pool_size = thread_pool_config.get("max_workers", 8)
        send_fps = args.send_fps
        


        # Determine GPU settings
        if gpu_config.get("enabled", False):
            # Use the number of devices listed in the gpu_config
            num_processors = gpu_config.get("num_processors", 2)
            # Override batch_size per GPU if provided, else use the command-line argument
            batch_size = gpu_config.get("batch_size_per_gpu", args.batch_size)
        else:
            num_processors = 2  # or any default value if GPUs are not enabled
            batch_size = args.batch_size
        
        processor = RTSPStreamProcessor(
            num_processors=num_processors,
            batch_size=batch_size,
            batch_interval=args.batch_interval,
            processing_fps=args.fps,
            frame_buffer_config=buffer_config,
            thread_pool_size=thread_pool_size,
            send_fps = send_fps
        )

        # Add each camera
        try:

            for camera in base_config['cameras']:
                processor.add_camera(
                    rtsp_url=camera['rtsp_url'],
                    camera_id=camera['camera_id'],
                    store_id=camera['store_id']
                )
        except Exception as e:
            logger.error(f"Error loading camera configuration: {e}", exc_info=True)
            # default_config = {
            #     "cameras": [
            #         {
            #             "rtsp_url": "rtsp://admin:password123@192.168.1.101:554/stream1",
            #             "camera_id": "camera-001",
            #             "store_id": "store-001"
            #         },
            #         {
            #             "rtsp_url": "rtsp://admin:password123@192.168.1.102:554/stream1",
            #             "camera_id": "camera-002",
            #             "store_id": "store-001"
            #         }
            #     ]
            # }

            # logger.info("Using default camera configuration:")
            # for camera in default_config['cameras']:
            #     logger.info(f"  - Camera {camera['camera_id']} in store {camera['store_id']}: {camera['rtsp_url']}")
            #     processor.add_camera(
            #         rtsp_url=camera['rtsp_url'],
            #         camera_id=camera['camera_id'],
            #         store_id=camera['store_id']
            #     )
    
    except Exception as e:
        logger.error(f"Error reading the config file: {e}")

        # Create the RTSPStreamProcessor with all relevant settings

    if processor is None:
        logger.error("Processor was not initialized. Exiting.")
        return
    
    try:
        # Run the processor
        await processor.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # Clean up
        await processor.stop()
        logger.info("RTSP Stream Processor service stopped")

# Modified main function to use the frame-by-frame processor
async def frame_by_frame_main():
    """Main entry point for frame-by-frame RTSP processing service"""
    parser = argparse.ArgumentParser(description='Frame-by-Frame RTSP Stream Processor for Retail Analytics')
    parser.add_argument('--config', default='camera_config.json', help='Path to camera configuration file')
    parser.add_argument('--fps', type=float, default=30, help='Frames per second to process from each camera')
    
    args = parser.parse_args()
    
    logger.info("Starting Frame-by-Frame RTSP Stream Processor service")
    processor = None
    
    # Create processor
    # processor = FrameByFrameProcessor(processing_fps=args.fps)
    
    # Load camera configuration
    try:
        camera_config_remote = await fetch_camera_config("http://genfied-api.xperie.nz:8000/api/v1/ai-server/config")
        with open(args.config, 'r') as f:
            base_config = json.load(f)
        base_config['cameras'] = camera_config_remote.get('cameras', [])

        system_config = base_config.get("system", {})
        buffer_config = base_config.get("buffer_settings", {})
        gpu_config = system_config.get("gpu_config", {})
        memory_management = system_config.get("memory_management", {})
        thread_pool_config = system_config.get("thread_pool", {})
        thread_pool_size = thread_pool_config.get("max_workers", 8)
        send_fps = args.send_fps

        # Determine GPU settings
        if gpu_config.get("enabled", False):
            # Use the number of devices listed in the gpu_config
            num_processors = gpu_config.get("num_processors", 2)
            # Override batch_size per GPU if provided, else use the command-line argument
            batch_size = gpu_config.get("batch_size_per_gpu", args.batch_size)
        else:
            num_processors = 2  # or any default value if GPUs are not enabled
            batch_size = args.batch_size
        
        processor = FrameByFrameProcessor(
            num_processors=num_processors,
            processing_fps=args.fps,
            frame_buffer_config=buffer_config,
            thread_pool_size=thread_pool_size,
            send_fps = send_fps
        )
        try:
            # Add each camera
            for camera in base_config['cameras']:
                processor.add_camera(
                    rtsp_url=camera['rtsp_url'],
                    camera_id=camera['camera_id'],
                    store_id=camera['store_id']
                )
        except Exception as e:
            logger.error(f"Error loading camera configuration: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error loading camera configuration: {e}", exc_info=True)
        # Use default configuration
        # (Same default configuration as in the original code)

    if processor is None:
        logger.error("Processor was not initialized. Exiting.")
        return
        
    try:
        # Run the processor
        await processor.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # Clean up
        await processor.stop()
        logger.info("Frame-by-Frame RTSP Stream Processor service stopped")

if __name__ == "__main__":
    asyncio.run(main())