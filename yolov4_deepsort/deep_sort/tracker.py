# tracker.py
import numpy as np
import os
import pickle
from . import kalman_filter_anuj
from . import linear_assignment
from . import iou_matching
from .track import Track
from .track import TrackState
import logging
import os
from collections import defaultdict
from milvus_read_client import MilvusReIDClient
from .tracker_utils import calculate_cosine_distance, compute_iou, overlap_ratio_single_box, is_entering_store_percent
import time
# Configure logger at the top of your module (or in a separate config module)
LOG_FILENAME = "tracker.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
from typing import List, Dict, Tuple, Union, Optional  # Add List to existing import


class AsyncTracker:
    """
    This is the Asynchronous multi-target tracker with shared store cache.
    keeping the max_age forces the features to be checked with the features in the global database as quickly as possible. 
    """

    def __init__(self, metric, camera_id, store_id, milvus_client, store_cache=None, max_iou_distance=0.7, max_age=3, n_init=5, matching_threshold=0.5):
        self.metric = metric
        self.max_age = max_age
        self.n_init = n_init
        self.matching_threshold = matching_threshold  # Set the matching threshold
        self.store_id = store_id
        self.store_cache = store_cache
        self.use_shared_cache = False
        if self.use_shared_cache:
            logger.info(f"[Tracker-{camera_id}] Using shared store cache for store {self.store_id}")
        else:
            logger.warning(f"[Tracker-{camera_id}] No shared cache provided, will use individual queries")
        # logger.debug(f"MATCHING THRESHOLD IS {self.matching_threshold}")
        # logger.debug(f"METRIC MATCHING THRESHOLD IS {self.metric.matching_threshold}")
        self.kf = kalman_filter_anuj.KalmanFilter(dt = 1/25)
        self.tracks = []
        self._next_id = 1
        self.milvus_client = milvus_client
        self.max_iou_distance = max_iou_distance
        logger.debug(f"Successfully receievd milvus client from processor to upate values for store {self.store_id}")
        self.camera_id = camera_id
        self._feature_batch = {
            'track_ids': [],
            'embeddings': [],
            'store_ids': [],
            'camera_ids': [],
            'timestamps': []
        } #Created for batch fetaure insertion. Instead of inserting the features everytime a track gets confirmed/created, we store them temporarily and insert them in batches.
        try:
            self._batch_size = int(os.environ.get("MILVUS_BATCH_SIZE", "32"))
            self._batch_interval = float(os.environ.get("MILVUS_BATCH_INTERVAL", "1.0"))
             # Validate values
            if self._batch_size < 1:
                logger.warning(f"Invalid MILVUS_BATCH_SIZE: {self._batch_size}, using default 20")
                self._batch_size = 20

            if self._batch_interval < 0.1:
                logger.warning(f"Invalid MILVUS_BATCH_INTERVAL: {self._batch_interval}, using default 1.0")
                self._batch_interval = 1.0
            logger.debug(f"Milvus batch configuration: size={self._batch_size}, interval={self._batch_interval}s")
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing batch configuration from environment: {e}")
            logger.debug("Using default batch configuration: size=20, interval=1.0s")
            self._batch_size = 20
            self._batch_interval = 1.0
        self._last_batch_time = time.time()
        # Load or create the global database
        self.retired_ids = set()

    async def _search_with_shared_cache(self, features: List[np.ndarray], exclude_ids: set = None,
                                        threshold: float = None, top_k: int = 10):
            """
            Perform batch search using shared store cache.
            """
            if not self.use_shared_cache:
                # Fallback to regular Milvus search
                return await self.milvus_client.search_embeddings_batch(
                    embeddings_list=features,
                    top_k=top_k,
                    store_id=self.store_id
                )
            
            # Use shared cache for batch search
            return await self.store_cache.batch_search_in_cache(
                features=features,
                exclude_ids=exclude_ids,
                threshold=threshold,
                top_k=top_k
            )

    async def mark_track_as_left(self, track):
        """Mark a track as left and retire its ID."""
        try:
            await self.milvus_client.delete_track(track_id=track.track_id, store_id=self.store_id)
            logger.debug(f"Deleted track {track.track_id} from Milvus collection.")
        except Exception as e:
            logger.error(f"Error deleting track {track.track_id} from Milvus: {e}")
        
        self.retired_ids.add(track.track_id)
        # Mark the track as deleted so it will be removed from active tracking
        track.state = TrackState.Deleted
        logger.debug(f"Track {track.track_id} retired and removed from global database.")
    
    async def _insert_feature_into_milvus(self, track, detection):
        """
        Add feature to batch for later insertion.

        Args:
            track: The Track object being updated
            detection: The Detection object with the new feature
        """
        if track.state == TrackState.Confirmed and detection.feature is not None:
            camera_id = getattr(detection, "camera_id", 0)
            timestamp = getattr(detection, "timestamp", 0)
            self._feature_batch['track_ids'].append(track.track_id)
            self._feature_batch['embeddings'].append(detection.feature)
            self._feature_batch['store_ids'].append(self.store_id)
            self._feature_batch['camera_ids'].append(camera_id)
            self._feature_batch['timestamps'].append(timestamp)
            # Check if batch should be sent
            current_len = len(self._feature_batch['track_ids'])
            logger.debug(f"current length of batches of features to be inserted: {current_len}")
            if (current_len >= self._batch_size or
                time.time() - self._last_batch_time > self._batch_interval):
                logger.debug("flushing data now")
                await self._flush_feature_batch()
    
    async def _flush_feature_batch(self):
        """Flush the feature batch to Milvus"""
        if not self._feature_batch['track_ids']:
            return  # Empty batch
            
        batch_size = len(self._feature_batch['track_ids'])
        start_time = time.time()
        try:
            await self.milvus_client.insert_embeddings_batch(
                track_ids=self._feature_batch['track_ids'],
                embeddings=self._feature_batch['embeddings'],
                store_ids=self._feature_batch['store_ids'],
                camera_ids=self._feature_batch['camera_ids'],
                timestamps=self._feature_batch['timestamps']
            )
            elapsed = time.time() - start_time
            logger.debug(f"[Milvus] Batch inserted {batch_size} features in {elapsed:.3f}s ({batch_size/elapsed:.1f} features/sec)")
        except Exception as e:
            logger.error(f"[Milvus] Failed to batch insert {batch_size} features: {e}")
        
        # Reset batch
        self._feature_batch = {
            'track_ids': [],
            'embeddings': [],
            'store_ids': [],
            'camera_ids': [],
            'timestamps': []
        }
        self._last_batch_time = time.time()

    # Add cleanup method
    async def cleanup(self):
        """Clean up and flush any remaining features"""
        await self._flush_feature_batch()


    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)
    
    async def _find_next_best_match(self, feature, assigned_ids, max_candidates=5, distance_threshold=0.7):
        """
        Find the next best match from Milvus, excluding already assigned IDs,
        and return both the best track ID and its distance.

        Args:
            feature: Feature vector of the detection.
            assigned_ids: Set of track IDs already assigned in the current frame.
            max_candidates: Maximum number of candidates to consider.
            distance_threshold: Maximum cosine distance threshold for a valid match.

        Returns:
            Tuple[int or None, float]: (track_id, distance) or (None, inf) if no good match found.
        """
        try:
            results = await self.milvus_client.search_embedding(
                query_embedding=feature,
                top_k=max_candidates * 2,
                store_filter=self.store_id
            )

            if not isinstance(assigned_ids, set):
                assigned_ids = set(assigned_ids)

            # Filter and sort
            unassigned_matches = [
                (track_id, distance)
                for track_id, distance in results
                if track_id not in assigned_ids
            ]
            unassigned_matches.sort(key=lambda x: x[1])

            for track_id, distance in unassigned_matches[:max_candidates]:
                if distance < distance_threshold:
                    logger.debug(f"[Milvus] Best unassigned match: track_id={track_id}, distance={distance}")
                    return track_id, distance

            logger.debug("[Milvus] No unassigned track ID below threshold.")
            return None, float('inf')

        except Exception as e:
            logger.error(f"[Milvus] Error in _find_next_best_match: {e}")
            return None, float('inf')

    def _deduplicate_tracks(self):
        """
        Remove duplicate track objects with the same track_id,
        keeping only the one with the highest n_hits count.
        """
        # Group tracks by track_id
        tracks_by_id = defaultdict(list)
        for track in self.tracks:
            tracks_by_id[track.track_id].append(track)
        
        # Find duplicates and keep only the best one
        tracks_to_keep = []
        for track_id, track_list in tracks_by_id.items():
            if len(track_list) > 1:
                # Multiple tracks with the same ID found
                logger.debug(f"Found {len(track_list)} duplicate tracks for ID {track_id}")
                
                # Sort by n_hits (descending) to keep the one with highest count
                track_list.sort(key=lambda t: t.hits, reverse=True)
                
                # Keep the track with highest hits
                best_track = track_list[0]
                tracks_to_keep.append(best_track)
                
                logger.debug(f"Keeping track with ID {track_id} and n_hits={best_track.hits}, " 
                            f"removing {len(track_list)-1} duplicates")
            else:
                # Just one track with this ID, no duplicates
                tracks_to_keep.append(track_list[0])
        
        # Update the tracks list
        self.tracks = tracks_to_keep

    async def update(self, detections):
        """
        Comprehensive update method using shared store cache for searches with location constraint for new ID creation.
        Includes an exception for the first 250 frames (10 seconds) to handle abrupt camera startup.
        
        Combines robust features from original competition-based matching while maintaining
        the location constraint system for stable tracking.
        
        Key constraint: No new ID can be created below y1 = 270, forcing the system
        to choose the best match for detections in the center/bottom of the frame,
        EXCEPT during the first 250 frames where new IDs are allowed anywhere.
        """
        frame_start = time.time()
        operation_times = []
        milvus_ops_count = 0

        def log_operation(operation_name, start_time):
            elapsed = time.time() - start_time
            operation_times.append((operation_name, elapsed))
            logger.warning(f"[TIMING] {operation_name}: {elapsed:.3f}s")
            return time.time()
        
        logger.warning(f"=== FRAME START: {len(detections)} detections ===")
        # Step 1: Identify potentially overlapping detections
        self._deduplicate_tracks()
        overlap_threshold = 0.25  # IoU threshold
        num_dets = len(detections)
        overlap_groups = {}  # Track which detections overlap with others
        
        # Threshold for where new IDs can be created - no new IDs below this y-coordinate
        new_id_y_threshold = 270
        
        # Track frame count as a class attribute if it doesn't exist
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        else:
            self._frame_count += 1
        
        # Check if we're in the startup grace period (first 10 seconds / 250 frames)
        startup_grace_period = self._frame_count < 251
        if startup_grace_period:
            logger.debug(f"In startup grace period (frame {self._frame_count}/250): allowing new IDs anywhere in frame")
        
        # Find all overlapping pairs of detections
        for i in range(num_dets):
            box_i = detections[i].to_tlbr()
            entering_i, frac_i = is_entering_store_percent(box_i, line_y=108, threshold=0.60)
            
            if i not in overlap_groups:
                overlap_groups[i] = []
                
            for j in range(i + 1, num_dets):
                box_j = detections[j].to_tlbr()
                
                # Check overlap using both methods for robustness
                iou = compute_iou(box_i, box_j)
                overlap_A, fraction_A = overlap_ratio_single_box(box_i, box_j, ratio_threshold=0.30)
                overlap_B, fraction_B = overlap_ratio_single_box(box_j, box_i, ratio_threshold=0.30)
                
                if (iou > overlap_threshold) or overlap_A or overlap_B:
                    logger.debug(f"Detected overlap between boxes {box_i} and {box_j}: IoU={iou:.2f}, fractions: {fraction_A:.2f}, {fraction_B:.2f}")
                    
                    if j not in overlap_groups:
                        overlap_groups[j] = []
                    
                    overlap_groups[i].append(j)
                    overlap_groups[j].append(i)
        
        # Step 2: Run the standard matching cascade first
        # Update metric first
        logger.debug(f"Before metric update: Available samples: {list(self.metric.samples.keys())}")
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features_to_add = []
        targets_to_add = []
        
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            if track.features:  # Only process tracks that have features
                features_to_add.extend(track.features)
                targets_to_add.extend([track.track_id] * len(track.features))

        if features_to_add and targets_to_add:
            self.metric.partial_fit(np.asarray(features_to_add), np.asarray(targets_to_add), active_targets)
            # Now we can safely clear the features since they're in the metric
            for track in self.tracks:
                if track.is_confirmed() and track.features:
                    track.features = []
        else:
            logger.debug("No features to update metric with")

        logger.debug(f"Before _match: Available metric samples: {list(self.metric.samples.keys())}")
        logger.debug(f"Active tracks: {[(t.track_id, t.is_confirmed(), len(t.features)) for t in self.tracks]}")

        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        
        # Create a dict mapping track_idx to track object for easier reference
        logger.debug(f"length of self.tracks is {len(self.tracks)}")
        track_idx_to_track = {idx: self.tracks[idx] for idx in range(len(self.tracks))}
        logger.debug("track_idx_to_track is: " + ", ".join([f"{k}: {v.track_id}" for k, v in track_idx_to_track.items()]))


        # Group the track objects by their track.track_id
        tracks_by_id = defaultdict(list)
        for idx, track in track_idx_to_track.items():
            tracks_by_id[track.track_id].append(track)
        
        # Log the grouping: print each track ID and the number of objects for that ID.
        logger.debug("track_idx_to_track grouped by track.track_id: " +
                    ", ".join([f"{tid}: {len(lst)}" for tid, lst in tracks_by_id.items()]))

        
        # First process the good matches from the cascade directly
        for track_idx, detection_idx in matches:
            # track = track_idx_to_track[track_idx]
            detection = detections[detection_idx]
            track_id = track_idx_to_track[track_idx].track_id

            # For all duplicate tracks with the same track_id, run update:
            for track in tracks_by_id[track_id]:
                # If no features exist yet, update immediately:
                if not track.features:
                    track.update(self.kf, detection)
                    await self._insert_feature_into_milvus(track, detection)
                    logger.debug(f"Initial feature update for track_id {track.track_id} with detection {detection_idx}")
                    continue

                # Otherwise, calculate the distance from the detection feature to the track's latest feature.
                distance = calculate_cosine_distance(detection.feature, track.features[-1])
                logger.debug(f"Distance for track_id {track.track_id} with detection {detection_idx} is {distance}")

                # If the distance is below threshold, update the track:
                if distance < self.matching_threshold:
                    track.update(self.kf, detection)
                    # Immediately update metric for this track
                    if track.is_confirmed() and detection.feature is not None:
                        self.metric.partial_fit(
                            np.array([detection.feature]),
                            np.array([track.track_id]),
                            [track.track_id]
                        )
                    await self._insert_feature_into_milvus(track, detection)
                    logger.debug(f"Direct update for good match: track_id {track.track_id} with detection {detection_idx}, distance {distance}")
                    if detection_idx in unmatched_detections:
                        unmatched_detections.remove(detection_idx)
                    # if track_idx in unmatched_tracks:
                    #     unmatched_tracks.remove(track_idx)
                    for idx, track_obj in track_idx_to_track.items():
                                if track_obj.track_id == track.track_id and idx in unmatched_tracks:
                                    unmatched_tracks.remove(idx)
                                
                
                # Remove this detection and track from further processing
                if detection_idx in unmatched_detections:
                    unmatched_detections.remove(detection_idx)
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)
        
        # Step 3: Collect all potential ID matches for all remaining detections with scores
        potential_matches = {}  # {detection_idx: [(track_id, distance, is_visible), ...]}
        entering_detections = set()  # Track which detections are entering
        center_detections = set()    # Detections below the new ID threshold

        # Collect features for batch search
        search_features = []
        search_indices = []

        
        # Process all remaining detections to find ALL potential ID matches
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            bbox = detection.to_tlbr()
            
            # Check if this detection represents someone entering
            entering, fraction = is_entering_store_percent(bbox, line_y=108, threshold=0.60)
            
            if entering:
                logger.debug(f"Detection with bbox {bbox} is entering the store (fraction: {fraction}).")
                # Mark this detection as entering
                entering_detections.add(det_idx)
                continue
            
            # Check if this detection is below the threshold line (and we're not in startup period)
            if bbox[1] > new_id_y_threshold and not startup_grace_period:
                center_detections.add(det_idx)
                logger.debug(f"Detection with bbox {bbox} is below threshold y={new_id_y_threshold}, must use existing ID.")
            

            search_features.append(detection.feature)
            search_indices.append(det_idx)

        # Step 4: PERFORM BATCH SEARCH for all unmatched detections at once
        potential_matches = {}  # {detection_idx: [(track_id, distance, is_visible), ...]}

        if search_features:
            logger.info(f"Performing batch search for {len(search_features)} unmatched detections")
            milvus_ops_count += 1
            if self.use_shared_cache:
                logger.info(f"SEARCHING IN CACHE FIRST")
                # Search in shared cache first
                op_start = time.time()
                cache_results = await self.store_cache.batch_search_in_cache(
                    features=search_features,
                    exclude_ids=set(),  # Don't exclude any IDs initially
                    threshold=self.matching_threshold,
                    top_k=20
                )
                log_operation("search_embeddings_batch", op_start)
                
                # If we need more results, fall back to Milvus
                batch_results = []
                for i, (cache_result, feature) in enumerate(zip(cache_results, search_features)):
                    if len(cache_result) < 1:  # Need more candidates
                        logger.info("len(cache_result) < 1")
                        milvus_ops_count += 1
                        # Search Milvus for additional results
                        op_start = time.time()
                        milvus_result = await self.milvus_client.search_embedding(
                            query_embedding=feature,
                            top_k=10,
                            store_filter=self.store_id
                        )
                        log_operation("search_embeddings", op_start)
                        # Merge results
                        seen_ids = {r[0] for r in cache_result}
                        for track_id, distance in milvus_result:
                            if track_id not in seen_ids:
                                cache_result.append((track_id, distance))
                        
                        # Re-sort merged results
                        cache_result.sort(key=lambda x: x[1])
                        cache_result = cache_result[:20]
                    logger.info(f"Using the cache data")
                    batch_results.append(cache_result)
            else:
                op_start = time.time()
                batch_results = await self.milvus_client.search_embeddings_batch(
                    embeddings_list = search_features,
                    top_k = 20,
                    store_id = self.store_id
                )
                log_operation("search_embeddings_batch", op_start)
                # Process batch results
                for i, (det_idx, results) in enumerate(zip(search_indices, batch_results)):
                    potential_matches[det_idx] = []

                    # Check visible tracks for this detection
                    visible_matches = []
                    for track in self.tracks:
                        if track.is_confirmed() and track.time_since_update <= 1 and track.features:
                            distance = calculate_cosine_distance(search_features[i], track.features[-1])
                            if distance < self.matching_threshold:
                                visible_matches.append((track.track_id, distance, True))
                    
                    visible_ids = {vm[0] for vm in visible_matches}
                    milvus_matches = [(track_id, distance, False) 
                                for track_id, distance in results 
                                if track_id not in visible_ids and distance < self.matching_threshold]

                    all_matches = visible_matches + milvus_matches
                    all_matches.sort(key=lambda x: x[1])
                    potential_matches[det_idx] = all_matches[:10]  # Keep top 10 matches
                    logger.debug(f"Detection {det_idx} has {len(potential_matches[det_idx])} potential matches")
            
        # Step 5: Process detections based on whether they are part of overlapping groups or not
        assigned_detections = set()
        assigned_ids = set()
        final_assignments = {}  # {det_idx: (track_id, is_new)}

        # Keep track of which track IDs are already active in the current frame
        active_track_ids = set(track.track_id for track in self.tracks if track.is_confirmed())
        logger.debug(f"active track ids are {active_track_ids}")

        for track_idx, detection_idx in matches:
            if detection_idx not in unmatched_detections:  # Only consider successful matches
                track_id = track_idx_to_track[track_idx].track_id
                assigned_ids.add(track_id)
                assigned_detections.add(detection_idx)
                final_assignments[detection_idx] = (track_id, track_id not in active_track_ids)
                logger.debug(f"Final assignment: track ID {track_id} assigned to detection {detection_idx}")

        overlap_detections = set()
        for det_idx in unmatched_detections:
            if det_idx in overlap_groups and len(overlap_groups[det_idx]) > 0:
                overlap_detections.add(det_idx)
        
        # Sort overlapping detections by a heuristic (number of overlaps)
        overlap_detections_list = sorted(
            list(overlap_detections), 
            key=lambda idx: len(overlap_groups[idx]), 
            reverse=True
        )
        
        # Process overlapping detections first with explicit handling
        for det_idx in overlap_detections_list:
            if det_idx in assigned_detections:
                continue
                
            detection = detections[det_idx]
            bbox = detection.to_tlbr()
            
            # If detection is entering, assign new ID
            if det_idx in entering_detections:
                # Force new ID for entering detections
                logger.debug(f"Overlapping detection with bbox {bbox} is entering the store. Assigning new ID.")
                new_id = self._next_id
                assigned_ids.add(new_id)
                assigned_detections.add(det_idx)
                final_assignments[det_idx] = (new_id, True)
                logger.debug(f"Assigning det id {det_idx} to track id {new_id} because overlapping box in entering state")
                self._next_id += 1
                continue
            
            if det_idx in potential_matches and potential_matches[det_idx]:
                matched_track_id = None
                for track_id, distance, is_visible in potential_matches[det_idx]:
                    if track_id not in assigned_ids:
                        matched_track_id = track_id
                        break
                
                if matched_track_id is not None:
                    # Handle ID competition if needed
                    assigned_ids.add(matched_track_id)
                    assigned_detections.add(det_idx)
                    is_new_track = matched_track_id not in active_track_ids
                    final_assignments[det_idx] = (matched_track_id, is_new_track)
                    logger.debug(f"Assigned ID {matched_track_id} to overlapping detection {det_idx}")
                else:
                    if det_idx in center_detections and not startup_grace_period:
                        op_start = time.time()
                        # Center detection must use existing ID
                        alt_id = await self._find_best_match_regardless_of_threshold(detection.feature, assigned_ids)
                        log_operation("find_best_match_fallback", op_start)
                        milvus_ops_count += 1
                        if alt_id is not None:
                            logger.debug(f"Center detection {det_idx} gets forced match with ID {alt_id}")
                            final_assignments[det_idx] = (alt_id, False)
                        else:
                            # Fallback to oldest ID
                            try:
                                op_start = time.time()
                                all_ids = await self.milvus_client.get_all_track_ids(store_id=self.store_id)
                                log_operation("get_all_track_ids", op_start)
                                milvus_ops_count += 1
                                alt_id = min(all_ids) if all_ids else self._next_id
                            except Exception as e:
                                logger.error(f"Error fetching all track IDs from Milvus: {e}")
                                alt_id = self._next_id
                            
                            logger.debug(f"Center detection {det_idx} gets fallback ID {alt_id}")
                            final_assignments[det_idx] = (alt_id, False)
                        
                        assigned_detections.add(det_idx)
                        assigned_ids.add(alt_id)
                    else:
                        new_id = self._next_id
                        assigned_ids.add(new_id)
                        assigned_detections.add(det_idx)
                        final_assignments[det_idx] = (new_id, True)
                        logger.debug(f"Assigned new ID {new_id} to overlapping detection {det_idx}")
                        self._next_id += 1
            
        # Add a conditional flush if batch is above a certain size
        if len(self._feature_batch['track_ids']) >= self._batch_size // 2:
            await self._flush_feature_batch()
        # SECOND: For remaining non-overlapping detections, use competition-based approach
        # For each track ID, find the detection that has the best match score
        id_to_best_detection = {}  # {track_id: (detection_idx, distance)}
        
        # Only consider non-overlapping, unassigned detections
        unassigned_non_overlapping = [
            det_idx for det_idx in unmatched_detections 
            if det_idx not in assigned_detections and det_idx not in overlap_detections
        ]
        
        # First pass: Find the best detection for each ID among non-overlapping detections
        for det_idx in unassigned_non_overlapping:
            if det_idx in entering_detections:
                continue  # Skip entering detections in this pass
                
            if det_idx in potential_matches and potential_matches[det_idx]:
                for track_id, distance, is_visible in potential_matches[det_idx]:
                    if track_id is not None and track_id not in assigned_ids:
                        if track_id not in id_to_best_detection or distance < id_to_best_detection[track_id][1]:
                            id_to_best_detection[track_id] = (det_idx, distance)
        
        # Second pass: Assign non-overlapping detections based on competition results
        # First handle ID winners - these get priority
        for track_id, (det_idx, distance) in id_to_best_detection.items():
            if det_idx not in assigned_detections and track_id not in assigned_ids:
                # For existing tracks, mark as update (False)
                # For new tracks from the database, mark as new (True)
                is_new_track = track_id not in active_track_ids
                final_assignments[det_idx] = (track_id, is_new_track)
                assigned_detections.add(det_idx)
                assigned_ids.add(track_id)
                logger.debug(f"ID competition winner: detection {det_idx} gets track_id {track_id} with distance {distance}")
        
        # Now handle remaining unassigned detections
        for det_idx in unassigned_non_overlapping:
            if det_idx in assigned_detections:
                continue  # Already assigned
            
            # If detection is entering, assign new ID
            if det_idx in entering_detections:
                # Assign new ID for entering detection
                new_id = self._next_id
                final_assignments[det_idx] = (new_id, True)  # New ID
                assigned_ids.add(new_id)
                assigned_detections.add(det_idx)
                logger.debug(f"Entering detection {det_idx} gets new track_id {new_id}")
                self._next_id += 1
                continue
            
            # Check if it's below the threshold line
            detection = detections[det_idx]
            bbox = detection.to_tlbr()
            is_center = det_idx in center_detections
            
            # For non-entering detections, try to find best available ID
            best_available_id = None
            best_distance = float('inf')
            
            if det_idx in potential_matches and potential_matches[det_idx]:
                for track_id, distance, _ in potential_matches[det_idx]:
                    if track_id is not None and track_id not in assigned_ids and distance < best_distance:
                        best_available_id = track_id
                        best_distance = distance
            
            if best_available_id is not None and best_distance <= self.matching_threshold:
                # Assign next best available ID
                is_new_track = best_available_id not in active_track_ids
                final_assignments[det_idx] = (best_available_id, is_new_track)
                assigned_ids.add(best_available_id)
                assigned_detections.add(det_idx)
                logger.debug(f"Alternative match: detection {det_idx} gets track_id {best_available_id} with distance {best_distance}")
            else:
                # No good match available
                if is_center and not startup_grace_period:
                    # Center detection must use existing ID - find any usable match
                    alt_id = await self._find_best_match_regardless_of_threshold(detection.feature, assigned_ids)
                    
                    if alt_id is not None:
                        logger.debug(f"Center detection {det_idx} forced to use existing ID {alt_id}")
                        final_assignments[det_idx] = (alt_id, False)
                    else:
                        # Fallback to oldest ID
                        try:
                            op_start = time.time()
                            all_ids = await self.milvus_client.get_all_track_ids(store_id=self.store_id)
                            log_operation("get_all_track_ids", op_start)
                            milvus_ops_count += 1
                            alt_id = min(all_ids) if all_ids else self._next_id
                        except Exception as e:
                            logger.error(f"Error fetching all track IDs from Milvus: {e}")
                            alt_id = self._next_id
                        logger.debug(f"Center detection {det_idx} gets fallback ID {alt_id}")
                        final_assignments[det_idx] = (alt_id, False)
                    
                    assigned_detections.add(det_idx)
                    assigned_ids.add(alt_id)
                else:
                    # Can assign new ID
                    new_id = self._next_id
                    final_assignments[det_idx] = (new_id, True)  # New ID
                    assigned_ids.add(new_id)
                    assigned_detections.add(det_idx)
                    logger.debug(f"No good match found: detection {det_idx} gets new track_id {new_id}")
                    if is_center and startup_grace_period:
                        logger.debug(f"  (Note: Allowing new ID below threshold because in startup grace period)")
                    self._next_id += 1
        
        # Step 6: Apply all modifications to the track list
        tracks_to_update = {}  # {track_id: (detection, is_new_track)}
        tracks_to_mark_missed = set()  # Set of track_ids to mark as missed
        
        # Process remaining unmatched tracks
        for track_idx in unmatched_tracks:
            track_id = track_idx_to_track[track_idx].track_id
            if track_id not in assigned_ids:
                tracks_to_mark_missed.add(track_id)
        
        # Process all final assignments to create new tracks or update existing ones
        for det_idx, (track_id, is_new_track) in final_assignments.items():
            detection = detections[det_idx]
            
            # Look for existing track with this ID
            existing_track = None
            for track in self.tracks:
                if track.track_id == track_id and track.is_confirmed():
                    existing_track = track
                    break
            
            if existing_track is not None and not is_new_track:
                # Update existing track
                tracks_to_update[track_id] = (detection, False)
            else:
                # Create new track
                tracks_to_update[track_id] = (detection, True)
        
        for track in self.tracks:
            if track.track_id in tracks_to_update:
                detection, is_new = tracks_to_update[track.track_id]
                if not is_new:
                    # Update this existing track
                    track.update(self.kf, detection)
                    # Immediately update metric for this track
                    if track.is_confirmed() and detection.feature is not None:
                        self.metric.partial_fit(
                            np.array([detection.feature]),
                            np.array([track.track_id]),
                            [track.track_id]
                        )
                    await self._insert_feature_into_milvus(track, detection)
                    logger.debug(f"Updated existing track {track.track_id}")
                    # Remove from tracks_to_update so we don't create a duplicate
                    del tracks_to_update[track.track_id]
            elif track.track_id in tracks_to_mark_missed:
                # Mark this track as missed
                track.mark_missed()
                logger.debug(f"Marked track {track.track_id} as missed")
        
        # Now create any new tracks
        for new_id, (detection, is_new) in tracks_to_update.items():
            existing_track_with_id = any(t.track_id == new_id for t in self.tracks)
            if not existing_track_with_id:
                mean, covariance = self.kf.initiate(detection.to_xyah())
                class_name = detection.get_class()
            
                new_track = Track(
                    mean, covariance, new_id, self.n_init, self.max_age,
                    detection.feature, class_name
                )
            
                try:
                    logger.info("HERE")
                    op_start = time.time()
                    db_features = await self.milvus_client.get_features_by_track_id(
                        track_id = new_id,
                        store_id=self.store_id
                    )
                    log_operation(f"get_features_track_{new_id}", op_start)
                    logger.info(f"store id is {self.store_id}")
                    milvus_ops_count += 1
                    if db_features:
                        new_track.features = db_features
                        logger.debug(f"Created track with ID {new_id} (reidentified from Milvus with {len(new_track.features)} features)")
                    else:
                        logger.debug(f"Created new track with ID {new_id} (no features found in Milvus)")
                except Exception as e:
                    logger.error(f"Failed to fetch features from Milvus for track_id {new_id} in store {self.store_id}: {e}")
                    logger.debug(f"Created new track with ID {new_id} (fallback)")
            
                # Add the new track
                self.tracks.append(new_track)
            else:
                existing_tracks = [t for t in self.tracks if t.track_id == new_id]
                best_track = max(existing_tracks, key=lambda t: t.hits)
                try:
                    op_start = time.time()
                    db_features = await self.milvus_client.get_features_by_track_id(
                        track_id = new_id,
                        store_id=self.store_id,
                    )
                    log_operation(f"get_features_track_{new_id}", op_start)
                    milvus_ops_count += 1
                    if db_features:
                        if not best_track.features:
                            best_track.features = db_features
                        elif len(db_features) > len(best_track.features):
                            added = 0
                            for db_feature in db_features:
                                if not any(np.array_equal(db_feature, f) for f in best_track.features):
                                    best_track.features.append(db_feature)
                                    added += 1
                            logger.debug(f"[Milvus] Merged {added} new features for track {new_id}. Total now: {len(best_track.features)}")
                    else:
                        logger.debug(f"[Milvus] No features found to restore for track {new_id}")
                except Exception as e:
                    logger.error(f"[Milvus] Failed to fetch/merge features for track {new_id}: {e}")
                best_track.update(self.kf, detection)
                await self._insert_feature_into_milvus(best_track, detection)
                logger.debug(f"Updated existing track ID {new_id} with n_hits={best_track.hits} instead of creating duplicate")
        
        # Step 7: Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        self._deduplicate_tracks()
        
        # Step 8: Update distance metric
        """
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
        
        if features and targets:
            self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

            for track in self.tracks:
                if track.is_confirmed():
                    track.features = []
                    """
        
        # Check if it's time to flush the batch by time
        current_time = time.time()
        if current_time - self._last_batch_time > self._batch_interval:
            await self._flush_feature_batch()
        logger.info(f"Frame processed with {milvus_ops_count} Milvus operations")
        total_frame_time = time.time() - frame_start
        total_milvus_time = sum(elapsed for _, elapsed in operation_times)
        non_milvus_time = total_frame_time - total_milvus_time
        
        logger.warning(f"=== FRAME SUMMARY ===")
        logger.warning(f"Total frame time: {total_frame_time:.3f}s")
        logger.warning(f"Total Milvus time: {total_milvus_time:.3f}s ({len(operation_times)} ops)")
        logger.warning(f"Non-Milvus time: {non_milvus_time:.3f}s")
        logger.warning(f"Milvus operations: {[f'{name}:{time:.3f}s' for name, time in operation_times]}")
        logger.warning(f"========================")
    async def _find_best_match_regardless_of_threshold(self, feature, assigned_ids, max_candidates=10):
        """
        Find the best match regardless of threshold from Milvus, used for center detections. Uses shared cache when available

        Args:
            feature: Feature vector to match
            assigned_ids: Set of already assigned IDs to exclude
            max_candidates: Maximum number of candidates to consider

        Returns:
            track_id: Best matching track ID, or None if no unassigned IDs exist
        """
        try:
            if self.use_shared_cache:
                results = await self.store_cache.search_in_cache(
                    feature=feature,
                    exclude_ids=assigned_ids,
                    threshold=None,  # No threshold - we want best match
                    top_k=max_candidates * 2
                )
                
                if results:
                    logger.info(f"FOUND BEST MATCH FROM CACHE SUCCESSFULLY")
                    return results[0][0]  # Return best track_id

            # Get candidate track IDs and their features from Milvus
            op_start = time.time()
            results = await self.milvus_client.search_embedding(
                query_embedding=feature,
                top_k=max_candidates * 5,
                store_filter=self.store_id
            )
            log_operation(f"search_embedding", op_start)
            # Filter and return best
            for track_id, distance in results:
                if track_id not in assigned_ids:
                    return track_id
            
            return None

        except Exception as e:
            logger.error(f"[Tracker] Error in _find_best_match_regardless_of_threshold: {e}")
            return None
          
    def _calculate_distance(self, feature, track):
        """
        Calculate the feature distance between a detection and a track.
        
        Args:
            feature: The feature vector of the detection
            track: The track object
            
        Returns:
            float: The distance score (lower is better)
        """
        if not track.features:
            return float('inf')
        
        # Use the track's last feature for comparison
        cost_matrix = self.metric.distance(
            np.array([feature]), 
            np.array([track.features[-1]])
        )
        return cost_matrix[0, 0]

    async def _calculate_distance_to_id(self, feature, track_id):
        """
        Calculate the distance between a feature vector and a specific track ID using Milvus.

        Args:
            feature: Feature vector to compare
            track_id: Track ID to compare against

        Returns:
            float: The best distance score (lower is better)
        """
        # First check if this ID is in active tracks
        for track in self.tracks:
            if track.track_id == track_id and track.is_confirmed():
                if track.features:
                    distance = calculate_cosine_distance(feature, track.features[-1])
                    logger.debug(f"[Tracker] distance to active track ID {track_id}: {distance}")
                    return distance

        # Fallback to Milvus if not found in active tracks
        try:
            db_features = await self.milvus_client.get_features_by_track_id(track_id, self.store_id)
            milvus_ops_count += 1
            if db_features:
                best_distance = min(calculate_cosine_distance(feature, f) for f in db_features)
                logger.debug(f"[Tracker] distance to track ID {track_id} from Milvus: {best_distance}")
                return best_distance
            else:
                logger.debug(f"[Tracker] no features found in Milvus for track ID {track_id}")
        except Exception as e:
            logger.error(f"[Tracker] error querying Milvus for track ID {track_id}: {e}")

        return float('inf')

    async def _find_all_potential_matches(self, feature, bbox, max_candidates=10):
        """
        Find all potential ID matches for a detection from the global database.
        
        Args:
            feature: The feature vector of the detection
            bbox: The bounding box of the detection
            max_candidates: Maximum number of candidates to return
            
        Returns:
            list: List of tuples (track_id, distance, is_visible)
        """
        # Check if the bounding box is in the center of the frame
        in_center = False
        frame_width, frame_height = 1920, 1080  # Adjust based on your frame size
        center_x, center_y = frame_width // 2, frame_height // 2
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        
        # Define center region (e.g., middle 50% of frame)
        center_margin_x = frame_width * 0.25
        center_margin_y = frame_height * 0.25
        if (abs(bbox_center_x - center_x) < center_margin_x and
            abs(bbox_center_y - center_y) < center_margin_y):
            in_center = True
        
        # Use a more strict threshold for matches in the center
        threshold = 0.3 if in_center else 0.4
        
        # First, check visible tracks
        visible_matches = []
        for track in self.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                # Calculate the feature distance using our custom function
                if track.features:  # Make sure track has features
                    distance = calculate_cosine_distance(feature, track.features[-1])
                    
                    if distance < threshold:
                        visible_matches.append((track.track_id, distance, True))

        visible_ids = {vm[0] for vm in visible_matches}
    
        # Query Milvus for top potential matches for this store
        milvus_matches = await self.milvus_client.search_embedding(
            query_embedding=feature,
            top_k=max_candidates * 2,
            store_filter=self.store_id
        )


        # Then, check all IDs in the global database
        database_matches = []
        for track_id, distance in milvus_matches:
            if track_id not in visible_ids and distance < threshold:
                database_matches.append((track_id, distance, False))

        # Combine both and return top N sorted by distance
        all_matches = visible_matches + database_matches
        all_matches.sort(key=lambda x: x[1])

        return all_matches[:max_candidates]

    def _match(self, detections):
        """Match detections to tracks."""
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            targets = [int(t) for t in targets]

            cost_matrix = self.metric.distance(features, targets)
            logger.debug(f"Cost matrix is {cost_matrix}")
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        logger.debug(f"confirmed_tracks are {confirmed_tracks}")
        # Associate confirmed tracks using appearance features.
        # logger.debug("MAT")
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        # Combine matches and unmatched lists
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        unmatched_detections = list(set(unmatched_detections))

        return matches, unmatched_tracks, unmatched_detections
    
    async def _match_with_global_database_all_tracks_considered(self, detection_feature, detection_bbox, center_bbox=(0, 108, 1152, 800)):
        """
        Match a detection with tracks in the global database.
        Returns the track_id with the smallest distance if it's below the threshold.
        Unlike the original version, this method checks all tracks regardless of their
        active/visible status.
        """
        def is_inside_store_center(bbox, center_bbox):
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            return (center_bbox[0] <= cx <= center_bbox[2]) and (center_bbox[1] <= cy <= center_bbox[3])

        min_distance = float('inf')  # Initialize with a large value
        matched_track_id = None  # Track ID of the best match
        default_threshold = 0.4
        increased_threshold = 0.5

        if is_inside_store_center(detection_bbox, center_bbox):
            threshold = increased_threshold
            logger.debug(f"Detection bbox {detection_bbox} is in the center; using increased threshold {threshold}.")
        else:
            threshold = default_threshold
            logger.debug(f"Detection bbox {detection_bbox} is not in the center; using default threshold {threshold}.")

        # Iterate through all track_ids in the global database without skipping any
            # Perform search on Milvus
        results = await self.milvus_client.search_embedding(
            query_embedding=detection_feature,
            top_k=10,
            store_filter=self.store_id
        )
        best_track_id = None
        best_distance = float('inf')
        for track_id, distance in results:
            logger.debug(f"Candidate match from Milvus: track_id={track_id}, distance={distance}")
            if distance < threshold and distance < best_distance:
                best_track_id = track_id
                best_distance = distance
        if best_track_id is not None:
            logger.debug(f"Best match: track_id={best_track_id}, distance={best_distance}")
            return best_track_id
        else:    
            logger.debug("No match found in Milvus database below threshold.")
            return None

    def _match_with_global_database(self, detection_feature, detection_bbox, center_bbox = (0, 108, 1152, 800)):
        """
        Match a detection with tracks in the global database.
        Returns the track_id with the smallest distance if it's below the threshold (0.1).
        Only checks tracks that are not currently active in the scene.
        """
        def is_inside_store_center(bbox, center_bbox):
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            return (center_bbox[0] <= cx <= center_bbox[2]) and (center_bbox[1] <= cy <= center_bbox[3])

        min_distance = float('inf')  # Initialize with a large value
        matched_track_id = None  # Track ID of the best match
        default_threshold = 0.4
        increased_threshold = 0.6
        if is_inside_store_center(detection_bbox, center_bbox):
            threshold = increased_threshold
            logger.debug(f"Detection bbox {detection_bbox} is in the center; using increased threshold {threshold}.")
        else:
            threshold = default_threshold
            logger.debug(f"Detection bbox {detection_bbox} is not in the center; using default threshold {threshold}.")

        # store_center_x1, store_center_y1, store_center_x2, store_center_y2 = (0, 270, 1152, 800)

        # Get list of track IDs that are both confirmed AND currently visible in the scene
        visible_confirmed_track_ids = [
            track.track_id for track in self.tracks 
            if track.is_confirmed() and track.time_since_update == 0
        ]
        # Get the list of currently active track IDs
        # active_track_ids = [track.track_id for track in self.tracks if track.is_confirmed()]

        results = self.milvus_client.search_embedding(
            query_embedding=detection_feature,
            top_k=10,
            store_filter=self.store_id
        )
        best_track_id = None
        best_distance = float('inf')
        for track_id, distance in results:
            # Skip if the track_id is currently active
            if track_id in visible_confirmed_track_ids:
                logger.debug(f"Skipping track_id {track_id} because it is confirmed and currently visible in scene")
                continue

            # Initialize a list to store distances less than 0.1 for this track_id
            valid_distances = []

            if distance < threshold and distance < best_distance:
                best_track_id = track_id
                best_distance = distance
        if best_track_id is not None:
            logger.debug(f"Best match: track_id={best_track_id}, distance={best_distance}")
            return best_track_id
        else:
            logger.debug("No suitable match found in Milvus. Assigning new track_id.")
            return None