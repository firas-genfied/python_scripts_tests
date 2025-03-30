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

# Configure logger at the top of your module (or in a separate config module)
LOG_FILENAME = "tracker.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_cosine_distance(feature_a, feature_b):
    """
    Calculate cosine distance between two feature vectors.
    Handles various input formats safely including None values.
    
    Args:
        feature_a: First feature vector in any numpy array format
        feature_b: Second feature vector in any numpy array format
    
    Returns:
        float: Cosine distance (0 to 2, where 0 is identical)
    """
    # Handle None inputs
    if feature_a is None or feature_b is None:
        return float('inf')  # Maximum distance for None features
        
    # Ensure features are flattened to 1D
    feature_a = np.asarray(feature_a).flatten()
    feature_b = np.asarray(feature_b).flatten()
    
    # Handle zero-norm vectors
    norm_a = np.linalg.norm(feature_a)
    norm_b = np.linalg.norm(feature_b)
    
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 1.0  # Maximum dissimilarity for zero vectors
    
    # Calculate cosine similarity
    cosine_similarity = np.dot(feature_a, feature_b) / (norm_a * norm_b)
    
    # Clamp to [-1, 1] to handle numerical errors
    cosine_similarity = max(min(cosine_similarity, 1.0), -1.0)
    
    # Convert to distance (0 to 2)
    return 1.0 - cosine_similarity

def compute_iou(boxA, boxB):
    """
    Compute the Intersection-over-Union (IoU) of two bounding boxes.
    Each box is [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if (boxAArea + boxBArea - interArea) == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def overlap_ratio_single_box(boxA, boxB, ratio_threshold=0.5):
    """
    Computes the fraction of boxA's area that is overlapped by boxB.
    If the fraction is greater than or equal to ratio_threshold, returns True.

    Args:
        boxA (list or tuple): [x1, y1, x2, y2] for the first bounding box.
        boxB (list or tuple): [x1, y1, x2, y2] for the second bounding box.
        ratio_threshold (float): The threshold for the overlap fraction. Default is 0.5.

    Returns:
        (bool, float): Tuple where the first element is True if the fraction of boxA
                       overlapped by boxB is >= ratio_threshold, and the second element
                       is the actual fraction.
    """
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    # Compute area of boxA
    areaA = max(0, xA2 - xA1) * max(0, yA2 - yA1)
    if areaA <= 0:
        return False, 0.0

    # Compute intersection coordinates
    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection_area = inter_w * inter_h

    if intersection_area <= 0:
        return False, 0.0

    overlap_fraction = intersection_area / float(areaA)
    return (overlap_fraction >= ratio_threshold, overlap_fraction)


def is_entering_store_percent(bbox, line_y, threshold=0.60):
    """
    Determines if more than a given percentage of the bounding box is below a horizontal line.

    In image coordinates (y increases downward):
      - If the entire bbox is below the line (y1 >= line_y), fraction = 1.0.
      - If the entire bbox is above the line (y2 < line_y), fraction = 0.0.
      - Otherwise, fraction = (y2 - line_y) / (y2 - y1).

    Args:
        bbox (list or tuple): The bounding box [x1, y1, x2, y2].
        line_y (int): The y-coordinate of the horizontal line marking the store entrance.
        threshold (float): The fraction threshold (default 0.70 means 70%).

    Returns:
        (bool, float): A tuple where the first element is True if the fraction below the line
                       is greater than the threshold (i.e. the person is inside the store), 
                       and the second element is the calculated fraction.
    """
    y1 = bbox[1]
    y2 = bbox[3]
    height = y2 - y1
    if height <= 0:
        return False, 0.0

    # If the entire bbox has entered
    if y1 >= line_y:
        fraction_below = 1.0
        return False,fraction_below
    # If the entire bbox has not entered
    elif y2 <= line_y:
        fraction_below = 0.0
        return False,fraction_below
    else:
        fraction_below = (y2 - line_y) / height

    return fraction_below > threshold and fraction_below < 0.9, fraction_below

class Tracker:
    """
    This is the multi-target tracker.
    keeping the max_age forces the features to be checked with the features in the global database as quickly as possible. 
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=3, n_init=5, matching_threshold=0.5):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.matching_threshold = matching_threshold  # Set the matching threshold
        # logger.info(f"MATCHING THRESHOLD IS {self.matching_threshold}")
        # logger.info(f"METRIC MATCHING THRESHOLD IS {self.metric.matching_threshold}")
        self.kf = kalman_filter_anuj.KalmanFilter(dt = 1/25)
        self.tracks = []
        self._next_id = 1

        # Load or create the global database
        self.global_database_file = "global_database_anuj.pkl"
        if os.path.exists(self.global_database_file):
            with open(self.global_database_file, "rb") as f:
                self.global_database = pickle.load(f)
            # logger.info(f"Loaded global database from {self.global_database_file}.")
        else:
            self.global_database = {}
            logger.info(f"Created new global database.")
        self.retired_ids = set()

    def mark_track_as_left(self, track):
        """Mark a track as left and retire its ID."""
        # Remove from global database if present
        if track.track_id in self.global_database:
            del self.global_database[track.track_id]
        # Add to retired IDs so it is not reused
        self.retired_ids.add(track.track_id)
        # Mark the track as deleted so it will be removed from active tracking
        track.state = TrackState.Deleted
        logger.info(f"Track {track.track_id} retired and removed from global database.")

    def save_global_database(self):
        """Save the global database to a file."""
        with open(self.global_database_file, "wb") as f:
            pickle.dump(self.global_database, f)
        logger.info(f"Saved global database to {self.global_database_file}.")

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def _find_next_best_match(self, feature, bbox, assigned_ids, max_candidates=5, distance_threshold=0.7):
        """
        Find the next best match from the global database, excluding already assigned IDs,
        by filtering out the track IDs and then proceeding with the normal distance computation.
        
        Args:
            feature: Feature vector of the detection.
            bbox: Bounding box of the detection in tlbr format.
            assigned_ids: Set of track IDs already assigned in the current frame.
            max_candidates: Maximum number of candidates to consider.
            distance_threshold: Maximum cosine distance threshold for a valid match.
            
        Returns:
            track_id: Next best matching track ID, or None if no good match found.
        """
        valid_track_ids = []
        valid_features = []
        
        # Build lists of track_ids and corresponding features, filtering out assigned IDs.
        database_copy = dict(self.global_database)
        for track_id, track_data in database_copy.items():
            if track_id in assigned_ids:
                continue
            track_features = track_data["features"]
            if not track_features:
                continue
            
            # We assume here you want to compare the new detection feature against all features for the track.
            valid_track_ids.append(track_id)
            valid_features.append(np.array(track_features))
        
        if not valid_track_ids:
            return None

        # Compute the cost for each track using your metric's distance method.
        # Here, we compute distances per track by passing the detection feature and each track's features.
        candidates = []
        for track_id, track_feats in zip(valid_track_ids, valid_features):
            # Note: self.metric.distance expects a matrix of features and a list/array of target IDs.
            # We call the underlying metric function directly (e.g., _nn_cosine_distance) to compute the distance.
            cost_matrix = self.metric._metric(np.array([feature]), track_feats)
            cost = np.min(cost_matrix)  # best cost for this track
            candidates.append((track_id, cost))
        
        # Sort candidates by cost
        candidates.sort(key=lambda x: x[1])
        
        # Check the top candidates against the threshold
        for track_id, cost in candidates[:max_candidates]:
            if cost < distance_threshold:
                return track_id

        return None

    # def update(self, detections): #best one so far
    #     """
    #     Modified update method that handles overlapping detections while maintaining
    #     robust tracking performance and preventing duplicate ID assignment.
    #     """
    #     # Step 1: Identify potentially overlapping detections
    #     overlap_threshold = 0.25  # IoU threshold
    #     num_dets = len(detections)
    #     overlap_groups = {}  # Track which detections overlap with others
        
    #     # Find all overlapping pairs of detections
    #     for i in range(num_dets):
    #         box_i = detections[i].to_tlbr()
    #         entering_i, frac_i = is_entering_store_percent(box_i, line_y=108, threshold=0.60)  # Use original threshold
            
    #         if i not in overlap_groups:
    #             overlap_groups[i] = []
                
    #         for j in range(i + 1, num_dets):
    #             box_j = detections[j].to_tlbr()
                
    #             # Check overlap using both methods for robustness
    #             iou = compute_iou(box_i, box_j)
    #             overlap_A, fraction_A = overlap_ratio_single_box(box_i, box_j, ratio_threshold=0.30)
    #             overlap_B, fraction_B = overlap_ratio_single_box(box_j, box_i, ratio_threshold=0.30)
                
    #             if (iou > overlap_threshold) or overlap_A or overlap_B:
    #                 logger.info(f"Detected overlap between boxes {box_i} and {box_j}: IoU={iou:.2f}, fractions: {fraction_A:.2f}, {fraction_B:.2f}")
                    
    #                 if j not in overlap_groups:
    #                     overlap_groups[j] = []
                    
    #                 overlap_groups[i].append(j)
    #                 overlap_groups[j].append(i)
        
    #     # Step 2: Run the standard matching cascade first
    #     matches, unmatched_tracks, unmatched_detections = self._match(detections)
        
    #     # Step 3: Update track set for matched detections
    #     for track_idx, detection_idx in matches:
    #         self.tracks[track_idx].update(self.kf, detections[detection_idx], self.global_database)
    #         logger.info(f"Matched detection to track_id {self.tracks[track_idx].track_id}.")
        
    #     # Step 4: Mark unmatched tracks as missed
    #     for track_idx in unmatched_tracks:
    #         self.tracks[track_idx].mark_missed()
        
    #     # Step 5: Process unmatched detections with special handling for overlaps
    #     remaining_unmatched = []
        
    #     # Keep track of which IDs have been assigned in this frame
    #     assigned_ids = set()
    #     # Add all matched track IDs to the assigned set
    #     for track_idx, _ in matches:
    #         assigned_ids.add(self.tracks[track_idx].track_id)
        
    #     # First handle detections that are part of overlap groups
    #     overlap_detections = set()
    #     for det_idx in unmatched_detections:
    #         if det_idx in overlap_groups and len(overlap_groups[det_idx]) > 0:
    #             overlap_detections.add(det_idx)
        
    #     # Sort overlapping detections by a heuristic to process most likely matches first
    #     # (This helps ensure better assignments when multiple overlapping detections compete for IDs)
    #     overlap_detections_list = list(overlap_detections)
    #     overlap_detections_list.sort(key=lambda idx: len(overlap_groups[idx]), reverse=True)
        
    #     # Process overlap detections first
    #     for det_idx in overlap_detections_list:
    #         detection = detections[det_idx]
    #         bbox = detection.to_tlbr()
    #         entering, fraction = is_entering_store_percent(bbox, line_y=108, threshold=0.60)
            
    #         if entering:
    #             # Force new ID for entering detections
    #             logger.info(f"Overlapping detection with bbox {bbox} is entering the store (fraction: {fraction}). Assigning new ID.")
    #             matched_track_id = None
    #         else:
    #             # Use more comprehensive matching for overlapping detections
    #             logger.info(f"Matching overlapping detection {bbox} with global database.")
    #             matched_track_id = self._match_with_global_database_all_tracks_considered(detection.feature, bbox)
                
    #             # Check if this ID is already assigned in this frame
    #             if matched_track_id is not None and matched_track_id in assigned_ids:
    #                 logger.warning(f"ID {matched_track_id} already assigned in this frame. Finding next best match.")
    #                 # Find the next best match from the database
    #                 matched_track_id = self._find_next_best_match(detection.feature, bbox, assigned_ids)
                    
    #                 if matched_track_id is not None:
    #                     logger.info(f"Found alternative match: track_id {matched_track_id}")
    #                 else:
    #                     logger.info("No good alternative match found. Will assign new ID.")
            
    #         if matched_track_id is not None:
    #             # Re-identify the track using the database
    #             mean, covariance = self.kf.initiate(detection.to_xyah())
    #             class_name = detection.get_class()
    #             self.tracks.append(Track(
    #                 mean, covariance, matched_track_id, self.n_init, self.max_age,
    #                 detection.feature, class_name))
                
    #             # Always ensure feature history is maintained
    #             self.tracks[-1].features = self.global_database[matched_track_id]["features"]
    #             logger.info(f"Re-identified overlapping detection as track_id {matched_track_id} with {len(self.tracks[-1].features)} features.")
                
    #             # Mark this ID as assigned
    #             assigned_ids.add(matched_track_id)
    #         else:
    #             # If no match found or all potential matches already assigned, create new track
    #             mean, covariance = self.kf.initiate(detection.to_xyah())
    #             class_name = detection.get_class()
    #             self.tracks.append(Track(
    #                 mean, covariance, self._next_id, self.n_init, self.max_age,
    #                 detection.feature, class_name))
    #             logger.info(f"Assigned new track_id {self._next_id} for overlapping detection.")
                
    #             # Mark this ID as assigned
    #             assigned_ids.add(self._next_id)
    #             self._next_id += 1
        
    #     # Process remaining unmatched detections using standard approach
    #     for det_idx in unmatched_detections:
    #         if det_idx not in overlap_detections:
    #             remaining_unmatched.append(det_idx)
        
    #     for det_idx in remaining_unmatched:
    #         detection = detections[det_idx]
    #         bbox = detection.to_tlbr()
    #         entering, fraction = is_entering_store_percent(bbox, line_y=108, threshold=0.60)
            
    #         if entering:
    #             logger.info(f"Detection with bbox {bbox} is entering the store (fraction: {fraction}).")
    #             matched_track_id = None
    #         else:
    #             matched_track_id = self._match_with_global_database(detection.feature, bbox)
                
    #             # Check if this ID is already assigned in this frame
    #             if matched_track_id is not None and matched_track_id in assigned_ids:
    #                 logger.warning(f"ID {matched_track_id} already assigned in this frame. Finding next best match.")
    #                 # Find the next best match from the database
    #                 matched_track_id = self._find_next_best_match(detection.feature, bbox, assigned_ids)
                    
    #                 if matched_track_id is not None:
    #                     logger.info(f"Found alternative match: track_id {matched_track_id}")
    #                 else:
    #                     logger.info("No good alternative match found. Will assign new ID.")
            
    #         if matched_track_id is not None:
    #             # Re-identify track using database
    #             mean, covariance = self.kf.initiate(detection.to_xyah())
    #             class_name = detection.get_class()
    #             self.tracks.append(Track(
    #                 mean, covariance, matched_track_id, self.n_init, self.max_age,
    #                 detection.feature, class_name))
                
    #             # Always ensure feature history is maintained
    #             self.tracks[-1].features = self.global_database[matched_track_id]["features"]
    #             logger.info(f"Re-identified track_id {matched_track_id} from global database with {len(self.tracks[-1].features)} features.")
                
    #             # Mark this ID as assigned
    #             assigned_ids.add(matched_track_id)
    #         else:
    #             # Assign new track ID
    #             mean, covariance = self.kf.initiate(detection.to_xyah())
    #             class_name = detection.get_class()
    #             self.tracks.append(Track(
    #                 mean, covariance, self._next_id, self.n_init, self.max_age,
    #                 detection.feature, class_name))
    #             logger.info(f"Assigned new track_id {self._next_id} for detection.")
                
    #             # Mark this ID as assigned
    #             assigned_ids.add(self._next_id)
    #             self._next_id += 1
        
    #     # Remove deleted tracks
    #     self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
    #     # Update distance metric
    #     active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
    #     features, targets = [], []
    #     for track in self.tracks:
    #         if not track.is_confirmed():
    #             continue
    #         features += track.features
    #         targets += [track.track_id for _ in track.features]
    #         track.features = []
        
    #     if features:
    #         self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)


    # def update(self, detections): #untested thoroughly but working
    #     """
    #     Comprehensive update method that handles ID competition between detections,
    #     new entries, overlapping detections, and maintains tracking consistency.
        
    #     This implementation avoids index invalidation by collecting all track modifications
    #     and applying them only after all processing is complete, while maintaining
    #     track ID consistency across frames.
    #     """
    #     # Step 1: Identify potentially overlapping detections
    #     overlap_threshold = 0.25  # IoU threshold
    #     num_dets = len(detections)
    #     overlap_groups = {}  # Track which detections overlap with others
        
    #     # Find all overlapping pairs of detections
    #     for i in range(num_dets):
    #         box_i = detections[i].to_tlbr()
    #         entering_i, frac_i = is_entering_store_percent(box_i, line_y=108, threshold=0.60)
            
    #         if i not in overlap_groups:
    #             overlap_groups[i] = []
                
    #         for j in range(i + 1, num_dets):
    #             box_j = detections[j].to_tlbr()
                
    #             # Check overlap using both methods for robustness
    #             iou = compute_iou(box_i, box_j)
    #             overlap_A, fraction_A = overlap_ratio_single_box(box_i, box_j, ratio_threshold=0.30)
    #             overlap_B, fraction_B = overlap_ratio_single_box(box_j, box_i, ratio_threshold=0.30)
                
    #             if (iou > overlap_threshold) or overlap_A or overlap_B:
    #                 logger.info(f"Detected overlap between boxes {box_i} and {box_j}: IoU={iou:.2f}, fractions: {fraction_A:.2f}, {fraction_B:.2f}")
                    
    #                 if j not in overlap_groups:
    #                     overlap_groups[j] = []
                    
    #                 overlap_groups[i].append(j)
    #                 overlap_groups[j].append(i)
        
    #     # Step 2: Run the standard matching cascade first
    #     matches, unmatched_tracks, unmatched_detections = self._match(detections)
        
    #     # Create a dict mapping track_idx to track object for easier reference
    #     track_idx_to_track = {idx: self.tracks[idx] for idx in range(len(self.tracks))}
        
    #     # First process the good matches from the cascade directly
    #     for track_idx, detection_idx in matches:
    #         track = track_idx_to_track[track_idx]
    #         detection = detections[detection_idx]
            
    #         # Skip tracks with no features or initialize them
    #         if not track.features:
    #             # First update with this feature
    #             track.update(self.kf, detection, self.global_database)
    #             logger.info(f"Initial feature update for track_id {track.track_id} with detection {detection_idx}")
    #             continue
            
    #         # Calculate distance to check if this is a good match
    #         distance = calculate_cosine_distance(detection.feature, track.features[-1])
    #         logger.info(f"distance for id {track_idx} with detection is {distance} ")
            
    #         # If this is a good match (distance below threshold), update track directly
    #         if distance < self.matching_threshold:
    #             # Update the track directly
    #             track.update(self.kf, detection, self.global_database)
    #             logger.info(f"Direct update for good match: track_id {track.track_id} with detection {detection_idx}, distance {distance}")
                        
    #             # Remove this detection and track from further processing
    #             if detection_idx in unmatched_detections:
    #                 unmatched_detections.remove(detection_idx)
    #             if track_idx in unmatched_tracks:
    #                 unmatched_tracks.remove(track_idx)
        
    #     # Step 3: Collect all potential ID matches for all remaining detections with scores
    #     potential_matches = {}  # {detection_idx: [(track_id, distance, is_entering, is_visible), ...]}
    #     entering_detections = set()  # Track which detections are entering
        
    #     # Process all remaining detections to find ALL potential ID matches
    #     for det_idx in unmatched_detections:
    #         detection = detections[det_idx]
    #         bbox = detection.to_tlbr()
            
    #         # Check if this detection represents someone entering
    #         entering, fraction = is_entering_store_percent(bbox, line_y=108, threshold=0.60)
            
    #         if entering:
    #             logger.info(f"Detection with bbox {bbox} is entering the store (fraction: {fraction}).")
    #             # Mark this detection as entering
    #             entering_detections.add(det_idx)
    #             continue
            
    #         # For non-entering detections, find all potential ID matches from the database
    #         database_matches = self._find_all_potential_matches(detection.feature, bbox)
            
    #         if det_idx not in potential_matches:
    #             potential_matches[det_idx] = []
            
    #         # Add all database matches to potential matches list
    #         for db_id, db_distance, is_visible in database_matches:
    #             if db_distance <= self.matching_threshold:  # Only add if within threshold
    #                 potential_matches[det_idx].append((db_id, db_distance, is_visible))
    #                 logger.info(f"Potential match: detection {det_idx} with database ID {db_id}, distance {db_distance}, visible: {is_visible}")
        
    #     # Step 4: Process detections based on whether they are part of overlapping groups or not
    #     assigned_detections = set()
    #     assigned_ids = set()
    #     final_assignments = {}  # {det_idx: (track_id, is_new)}
        
    #     # Keep track of which track IDs are already active in the current frame
    #     active_track_ids = set(track.track_id for track in self.tracks if track.is_confirmed())
        
    #     # FIRST: Process overlapping detections separately with special handling
    #     overlap_detections = set()
    #     for det_idx in unmatched_detections:
    #         if det_idx in overlap_groups and len(overlap_groups[det_idx]) > 0:
    #             overlap_detections.add(det_idx)
        
    #     # Sort overlapping detections by a heuristic (number of overlaps)
    #     overlap_detections_list = sorted(
    #         list(overlap_detections), 
    #         key=lambda idx: len(overlap_groups[idx]), 
    #         reverse=True
    #     )
        
    #     # Process overlapping detections first with explicit handling
    #     for det_idx in overlap_detections_list:
    #         if det_idx in assigned_detections:
    #             continue
                
    #         detection = detections[det_idx]
    #         bbox = detection.to_tlbr()
            
    #         # If detection is entering, assign new ID
    #         if det_idx in entering_detections:
    #             # Force new ID for entering detections
    #             logger.info(f"Overlapping detection with bbox {bbox} is entering the store. Assigning new ID.")
    #             new_id = self._next_id
    #             assigned_ids.add(new_id)
    #             assigned_detections.add(det_idx)
    #             final_assignments[det_idx] = (new_id, True)
    #             self._next_id += 1
    #             continue
            
    #         # Use more comprehensive matching specifically for overlapping detections
    #         logger.info(f"Matching overlapping detection {bbox} with global database.")
    #         matched_track_id = self._match_with_global_database_all_tracks_considered(detection.feature, bbox)
            
    #         # Check if this ID is already assigned in this frame
    #         if matched_track_id is not None and matched_track_id in assigned_ids:
    #             logger.warning(f"ID {matched_track_id} already assigned in this frame. Competition check needed.")
                
    #             # Find which detection already has this ID
    #             existing_det_idx = next(d_idx for d_idx, (t_id, _) in final_assignments.items() if t_id == matched_track_id)
    #             existing_detection = detections[existing_det_idx]
                
    #             # Calculate distances for both detections to this ID
    #             current_distance = self._calculate_distance_to_id(detection.feature, matched_track_id)
    #             existing_distance = self._calculate_distance_to_id(existing_detection.feature, matched_track_id)
                
    #             logger.info(f"Distance comparison: Current detection ({det_idx}): {current_distance}, " +
    #                     f"Existing detection ({existing_det_idx}): {existing_distance}")
                
    #             if current_distance < existing_distance:
    #                 # Current detection has a better match - reassign the existing detection
    #                 logger.info(f"Current detection {det_idx} has a better match with ID {matched_track_id}. " +
    #                         f"Reassigning detection {existing_det_idx}.")
                    
    #                 # Remove the existing assignment
    #                 assigned_detections.remove(existing_det_idx)
    #                 del final_assignments[existing_det_idx]
                    
    #                 # We'll keep the matched_track_id for the current detection
    #                 # The existing detection will be processed again later
    #             else:
    #                 # Existing detection has a better match - find an alternative for the current detection
    #                 logger.info(f"Existing detection {existing_det_idx} has a better match with ID {matched_track_id}. " +
    #                         f"Finding alternative for detection {det_idx}.")
                    
    #                 # Find the next best match from the database
    #                 matched_track_id = self._find_next_best_match(detection.feature, bbox, assigned_ids)
                    
    #                 if matched_track_id is not None:
    #                     logger.info(f"Found alternative match: track_id {matched_track_id}")
    #                 else:
    #                     logger.info("No good alternative match found. Will assign new ID.")
            
    #         if matched_track_id is not None:
    #             # Assign track ID
    #             assigned_ids.add(matched_track_id)
    #             assigned_detections.add(det_idx)
                
    #             # For existing tracks, mark as update (False)
    #             # For new tracks from the database, mark as new (True)
    #             is_new_track = matched_track_id not in active_track_ids
    #             final_assignments[det_idx] = (matched_track_id, is_new_track)
    #         else:
    #             # Assign new ID only if no match found
    #             new_id = self._next_id
    #             assigned_ids.add(new_id)
    #             assigned_detections.add(det_idx)
    #             final_assignments[det_idx] = (new_id, True)
    #             self._next_id += 1
        
    #     # SECOND: For remaining non-overlapping detections, use competition-based approach
    #     # For each track ID, find the detection that has the best match score
    #     id_to_best_detection = {}  # {track_id: (detection_idx, distance)}
        
    #     # Only consider non-overlapping, unassigned detections
    #     unassigned_non_overlapping = [
    #         det_idx for det_idx in unmatched_detections 
    #         if det_idx not in assigned_detections and det_idx not in overlap_detections
    #     ]
        
    #     # First pass: Find the best detection for each ID among non-overlapping detections
    #     for det_idx in unassigned_non_overlapping:
    #         if det_idx in entering_detections:
    #             continue  # Skip entering detections in this pass
                
    #         if det_idx in potential_matches and potential_matches[det_idx]:
    #             for track_id, distance, is_visible in potential_matches[det_idx]:
    #                 if track_id is not None and track_id not in assigned_ids:
    #                     if track_id not in id_to_best_detection or distance < id_to_best_detection[track_id][1]:
    #                         id_to_best_detection[track_id] = (det_idx, distance)
        
    #     # Second pass: Assign non-overlapping detections based on competition results
    #     # First handle ID winners - these get priority
    #     for track_id, (det_idx, distance) in id_to_best_detection.items():
    #         if det_idx not in assigned_detections and track_id not in assigned_ids:
    #             # For existing tracks, mark as update (False)
    #             # For new tracks from the database, mark as new (True)
    #             is_new_track = track_id not in active_track_ids
    #             final_assignments[det_idx] = (track_id, is_new_track)
    #             assigned_detections.add(det_idx)
    #             assigned_ids.add(track_id)
    #             logger.info(f"ID competition winner: detection {det_idx} gets track_id {track_id} with distance {distance}")
        
    #     # Now handle remaining unassigned detections
    #     for det_idx in unassigned_non_overlapping:
    #         if det_idx in assigned_detections:
    #             continue  # Already assigned
            
    #         # If detection is entering, assign new ID
    #         if det_idx in entering_detections:
    #             # Assign new ID for entering detection
    #             new_id = self._next_id
    #             final_assignments[det_idx] = (new_id, True)  # New ID
    #             assigned_ids.add(new_id)
    #             assigned_detections.add(det_idx)
    #             logger.info(f"Entering detection {det_idx} gets new track_id {new_id}")
    #             self._next_id += 1
    #             continue
            
    #         # For non-entering detections, try to find best available ID
    #         best_available_id = None
    #         best_distance = float('inf')
            
    #         if det_idx in potential_matches and potential_matches[det_idx]:
    #             for track_id, distance, _ in potential_matches[det_idx]:
    #                 if track_id is not None and track_id not in assigned_ids and distance < best_distance:
    #                     best_available_id = track_id
    #                     best_distance = distance
            
    #         if best_available_id is not None and best_distance <= self.matching_threshold:
    #             # Assign next best available ID
    #             is_new_track = best_available_id not in active_track_ids
    #             final_assignments[det_idx] = (best_available_id, is_new_track)
    #             assigned_ids.add(best_available_id)
    #             assigned_detections.add(det_idx)
    #             logger.info(f"Alternative match: detection {det_idx} gets track_id {best_available_id} with distance {best_distance}")
    #         else:
    #             # No good match available, assign new ID
    #             new_id = self._next_id
    #             final_assignments[det_idx] = (new_id, True)  # New ID
    #             assigned_ids.add(new_id)
    #             assigned_detections.add(det_idx)
    #             logger.info(f"No good match found: detection {det_idx} gets new track_id {new_id}")
    #             self._next_id += 1
        
    #     # Step 5: Collect all remaining track modifications for batch application
    #     tracks_to_update = {}  # {track_id: (detection, is_new_track)}
    #     tracks_to_mark_missed = set()  # Set of track_ids to mark as missed
        
    #     # Process remaining unmatched tracks
    #     for track_idx in unmatched_tracks:
    #         track_id = track_idx_to_track[track_idx].track_id
    #         if track_id not in assigned_ids:
    #             tracks_to_mark_missed.add(track_id)
        
    #     # Process all final assignments to create new tracks or update existing ones
    #     for det_idx, (track_id, is_new_track) in final_assignments.items():
    #         detection = detections[det_idx]
            
    #         # Look for existing track with this ID
    #         existing_track = None
    #         for track in self.tracks:
    #             if track.track_id == track_id and track.is_confirmed():
    #                 existing_track = track
    #                 break
            
    #         if existing_track is not None and not is_new_track:
    #             # Update existing track
    #             tracks_to_update[track_id] = (detection, False)
    #         else:
    #             # Create new track
    #             tracks_to_update[track_id] = (detection, True)
        
    #     # Step 6: Apply all modifications to the track list
    #     # First, update existing tracks and mark tracks as missed
    #     for track in self.tracks:
    #         if track.track_id in tracks_to_update:
    #             detection, is_new = tracks_to_update[track.track_id]
    #             if not is_new:
    #                 # Update this existing track
    #                 track.update(self.kf, detection, self.global_database)
    #                 logger.info(f"Updated existing track {track.track_id}")
    #                 # Remove from tracks_to_update so we don't create a duplicate
    #                 del tracks_to_update[track.track_id]
    #         elif track.track_id in tracks_to_mark_missed:
    #             # Mark this track as missed
    #             track.mark_missed()
    #             logger.info(f"Marked track {track.track_id} as missed")
        
    #     # Now create any new tracks
    #     for track_id, (detection, _) in tracks_to_update.items():
    #         mean, covariance = self.kf.initiate(detection.to_xyah())
    #         class_name = detection.get_class()
            
    #         new_track = Track(
    #             mean, covariance, track_id, self.n_init, self.max_age,
    #             detection.feature, class_name
    #         )
            
    #         # Copy features from global database if available
    #         if track_id in self.global_database:
    #             new_track.features = self.global_database[track_id]["features"]
    #             logger.info(f"Created track with ID {track_id} (reidentified from database with {len(new_track.features)} features)")
    #         else:
    #             logger.info(f"Created new track with ID {track_id}")
            
    #         # Add the new track
    #         self.tracks.append(new_track)
        
    #     # Step 7: Remove deleted tracks
    #     self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
    #     # Step 8: Update distance metric
    #     active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
    #     features, targets = [], []
    #     for track in self.tracks:
    #         if not track.is_confirmed():
    #             continue
    #         features += track.features
    #         targets += [track.track_id for _ in track.features]
    #         track.features = []
        
    #     if features:
    #         self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)


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
                logger.info(f"Found {len(track_list)} duplicate tracks for ID {track_id}")
                
                # Sort by n_hits (descending) to keep the one with highest count
                track_list.sort(key=lambda t: t.hits, reverse=True)
                
                # Keep the track with highest hits
                best_track = track_list[0]
                tracks_to_keep.append(best_track)
                
                logger.info(f"Keeping track with ID {track_id} and n_hits={best_track.hits}, " 
                            f"removing {len(track_list)-1} duplicates")
            else:
                # Just one track with this ID, no duplicates
                tracks_to_keep.append(track_list[0])
        
        # Update the tracks list
        self.tracks = tracks_to_keep

    def update(self, detections):
        """
        Comprehensive update method with location constraint for new ID creation.
        Includes an exception for the first 250 frames (10 seconds) to handle abrupt camera startup.
        
        Combines robust features from original competition-based matching while maintaining
        the location constraint system for stable tracking.
        
        Key constraint: No new ID can be created below y1 = 270, forcing the system
        to choose the best match for detections in the center/bottom of the frame,
        EXCEPT during the first 250 frames where new IDs are allowed anywhere.
        """
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
            logger.info(f"In startup grace period (frame {self._frame_count}/250): allowing new IDs anywhere in frame")
        
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
                    logger.info(f"Detected overlap between boxes {box_i} and {box_j}: IoU={iou:.2f}, fractions: {fraction_A:.2f}, {fraction_B:.2f}")
                    
                    if j not in overlap_groups:
                        overlap_groups[j] = []
                    
                    overlap_groups[i].append(j)
                    overlap_groups[j].append(i)
        
        # Step 2: Run the standard matching cascade first
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        
        # Create a dict mapping track_idx to track object for easier reference
        logger.info(f"length of self.tracks is {len(self.tracks)}")
        track_idx_to_track = {idx: self.tracks[idx] for idx in range(len(self.tracks))}
        logger.info("track_idx_to_track is: " + ", ".join([f"{k}: {v.track_id}" for k, v in track_idx_to_track.items()]))


        # Group the track objects by their track.track_id
        tracks_by_id = defaultdict(list)
        for idx, track in track_idx_to_track.items():
            tracks_by_id[track.track_id].append(track)
        
        # Log the grouping: print each track ID and the number of objects for that ID.
        logger.info("track_idx_to_track grouped by track.track_id: " +
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
                    track.update(self.kf, detection, self.global_database)
                    logger.info(f"Initial feature update for track_id {track.track_id} with detection {detection_idx}")
                    continue

                # Otherwise, calculate the distance from the detection feature to the track's latest feature.
                distance = calculate_cosine_distance(detection.feature, track.features[-1])
                logger.info(f"Distance for track_id {track.track_id} with detection {detection_idx} is {distance}")

                # If the distance is below threshold, update the track:
                if distance < self.matching_threshold:
                    track.update(self.kf, detection, self.global_database)
                    logger.info(f"Direct update for good match: track_id {track.track_id} with detection {detection_idx}, distance {distance}")
                    if detection_idx in unmatched_detections:
                        unmatched_detections.remove(detection_idx)
                    # if track_idx in unmatched_tracks:
                    #     unmatched_tracks.remove(track_idx)
                    for idx, track_obj in track_idx_to_track.items():
                                if track_obj.track_id == track.track_id and idx in unmatched_tracks:
                                    unmatched_tracks.remove(idx)
                                
            # # Skip tracks with no features or initialize them
            # if not track.features:
            #     # First update with this feature
            #     track.update(self.kf, detection, self.global_database)
            #     logger.info(f"Initial feature update for track_id {track.track_id} with detection {detection_idx}")
            #     continue
            
            # # Calculate distance to check if this is a good match
            # distance = calculate_cosine_distance(detection.feature, track.features[-1])
            # logger.info(f"distance for id {track_idx} whose track id is {track.track_id}, with detection is {distance}")
            
            # # If this is a good match (distance below threshold), update track directly
            # if distance < self.matching_threshold:
            #     # Update the track directly
            #     track.update(self.kf, detection, self.global_database)
            #     logger.info(f"Direct update for good match: track_id {track.track_id} with detection {detection_idx}, distance {distance}")
                
                # Remove this detection and track from further processing
                if detection_idx in unmatched_detections:
                    unmatched_detections.remove(detection_idx)
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)
        
        # Step 3: Collect all potential ID matches for all remaining detections with scores
        potential_matches = {}  # {detection_idx: [(track_id, distance, is_visible), ...]}
        entering_detections = set()  # Track which detections are entering
        center_detections = set()    # Detections below the new ID threshold
        
        # Process all remaining detections to find ALL potential ID matches
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            bbox = detection.to_tlbr()
            
            # Check if this detection represents someone entering
            entering, fraction = is_entering_store_percent(bbox, line_y=108, threshold=0.60)
            
            if entering:
                logger.info(f"Detection with bbox {bbox} is entering the store (fraction: {fraction}).")
                # Mark this detection as entering
                entering_detections.add(det_idx)
                continue
            
            # Check if this detection is below the threshold line (and we're not in startup period)
            if bbox[1] > new_id_y_threshold and not startup_grace_period:
                center_detections.add(det_idx)
                logger.info(f"Detection with bbox {bbox} is below threshold y={new_id_y_threshold}, must use existing ID.")
            
            # For non-entering detections, find all potential ID matches from the database
            database_matches = self._find_all_potential_matches(detection.feature, bbox)
            
            if det_idx not in potential_matches:
                potential_matches[det_idx] = []
            
            # Add all database matches to potential matches list
            for db_id, db_distance, is_visible in database_matches:
                if db_distance <= self.matching_threshold:  # Only add if within threshold
                    potential_matches[det_idx].append((db_id, db_distance, is_visible))
                    logger.info(f"Potential match: detection {det_idx} with database ID {db_id}, distance {db_distance}, visible: {is_visible}")
        
        # Step 4: Process detections based on whether they are part of overlapping groups or not
        assigned_detections = set()
        assigned_ids = set()
        final_assignments = {}  # {det_idx: (track_id, is_new)}
        
        # Keep track of which track IDs are already active in the current frame
        active_track_ids = set(track.track_id for track in self.tracks if track.is_confirmed())
        logger.info(f"active track ids are {active_track_ids}")
        
        # For matched detections, track the assigned IDs
        for track_idx, detection_idx in matches:
            if detection_idx not in unmatched_detections:  # Only consider successful matches
                track_id = track_idx_to_track[track_idx].track_id
                assigned_ids.add(track_id)
                assigned_detections.add(detection_idx)
                final_assignments[detection_idx] = (track_id, track_id not in active_track_ids)
                logger.info(f"Final assignment: track ID {track_id} assigned to detection {detection_idx}. Is it an active track: {track_id not in active_track_ids}")

                logger.info(f"track ID {track_id} with det id {detection_idx} was matched in the matching cascade process but not a final assignment")
        
        # FIRST: Process overlapping detections separately with special handling
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
                logger.info(f"Overlapping detection with bbox {bbox} is entering the store. Assigning new ID.")
                new_id = self._next_id
                assigned_ids.add(new_id)
                assigned_detections.add(det_idx)
                final_assignments[det_idx] = (new_id, True)
                logger.info(f"Assigning det id {det_idx} to track id {new_id} because overlapping box in entering state")
                self._next_id += 1
                continue
            
            # Use more comprehensive matching specifically for overlapping detections
            logger.info(f"Matching overlapping detection {bbox} with global database.")
            matched_track_id = self._match_with_global_database_all_tracks_considered(detection.feature, bbox)
            
            # Check if this ID is already assigned in this frame
            if matched_track_id is not None and matched_track_id in assigned_ids:
                logger.warning(f"ID {matched_track_id} already assigned in this frame. Competition check needed.")
                
                # Find which detection already has this ID
                existing_det_idx = next(d_idx for d_idx, (t_id, _) in final_assignments.items() if t_id == matched_track_id)
                existing_detection = detections[existing_det_idx]
                
                # Calculate distances for both detections to this ID
                current_distance = self._calculate_distance_to_id(detection.feature, matched_track_id)
                existing_distance = self._calculate_distance_to_id(existing_detection.feature, matched_track_id)
                
                logger.info(f"Distance comparison: Current detection ({det_idx}): {current_distance}, " +
                        f"Existing detection ({existing_det_idx}): {existing_distance}")
                
                if current_distance < existing_distance:
                    # Current detection has a better match - reassign the existing detection
                    logger.info(f"Current detection {det_idx} has a better match with ID {matched_track_id}. " +
                            f"Reassigning detection {existing_det_idx}.")
                    
                    # Remove the existing assignment
                    assigned_detections.remove(existing_det_idx)
                    del final_assignments[existing_det_idx]
                    
                    # We'll keep the matched_track_id for the current detection
                    # The existing detection will be processed again later
                else:
                    # Existing detection has a better match - find an alternative for the current detection
                    logger.info(f"Existing detection {existing_det_idx} has a better match with ID {matched_track_id}. " +
                            f"Finding alternative for detection {det_idx}.")
                    
                    # Find the next best match from the database
                    matched_track_id = self._find_next_best_match(detection.feature, bbox, assigned_ids)
                    
                    if matched_track_id is not None:
                        logger.info(f"Found alternative match: track_id {matched_track_id}")
                    else:
                        logger.info("No good alternative match found. Will assign new ID.")
            
            # After competition resolution, process the final assignment
            if matched_track_id is not None:
                # For center detections: enforce using existing ID
                if det_idx in center_detections and matched_track_id not in active_track_ids and not startup_grace_period:
                    logger.info(f"Center detection matched with inactive ID {matched_track_id}. Ensuring it is a good match.")
                    # Verify this is a sufficiently good match
                    distance = self._calculate_distance_to_id(detection.feature, matched_track_id)
                    if distance > self.matching_threshold * 1.5:  # Relax threshold a bit for center detections
                        # If not a good match, find the best available active ID
                        alternative_id = self._find_best_match_regardless_of_threshold(detection.feature, assigned_ids)
                        if alternative_id is not None:
                            matched_track_id = alternative_id
                            logger.info(f"Using better alternative active ID {matched_track_id} for center detection.")
                
                # Assign track ID
                assigned_ids.add(matched_track_id)
                assigned_detections.add(det_idx)
                logger.info(f"track ID {matched_track_id} was not an already assigned ID and is not a final assignment yet")
                
                # For existing tracks, mark as update (False)
                # For new tracks from the database, mark as new (True)
                is_new_track = matched_track_id not in active_track_ids
                final_assignments[det_idx] = (matched_track_id, is_new_track)
                logger.info(f"Assigned ID {matched_track_id} to overlapping detection {det_idx} (new track: {is_new_track}) and is now final")
            else:
                # No match found - check if new ID is allowed
                if det_idx in center_detections and not startup_grace_period:
                    # Center detection must use existing ID - find best available
                    alt_id = self._find_best_match_regardless_of_threshold(detection.feature, assigned_ids)
                    
                    if alt_id is not None:
                        logger.info(f"Center detection {det_idx} gets forced match with ID {alt_id}")
                        final_assignments[det_idx] = (alt_id, False)
                    else:
                        # Fallback to oldest ID if no good match
                        alt_id = min(self.global_database.keys()) if self.global_database else self._next_id
                        logger.info(f"Center detection {det_idx} gets fallback ID {alt_id}")
                        final_assignments[det_idx] = (alt_id, False)
                    
                    assigned_detections.add(det_idx)
                    assigned_ids.add(alt_id)
                    logger.info(f"Adding {alt_id} to assigned IDs dict and is a final assignment")
                else:
                    # Can assign new ID
                    new_id = self._next_id
                    assigned_ids.add(new_id)
                    assigned_detections.add(det_idx)
                    final_assignments[det_idx] = (new_id, True)
                    logger.info(f"Assigned new ID {new_id} to overlapping detection {det_idx} and is a final assignment")
                    self._next_id += 1
        
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
                logger.info(f"ID competition winner: detection {det_idx} gets track_id {track_id} with distance {distance}")
        
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
                logger.info(f"Entering detection {det_idx} gets new track_id {new_id}")
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
                logger.info(f"Alternative match: detection {det_idx} gets track_id {best_available_id} with distance {best_distance}")
            else:
                # No good match available
                if is_center and not startup_grace_period:
                    # Center detection must use existing ID - find any usable match
                    alt_id = self._find_best_match_regardless_of_threshold(detection.feature, assigned_ids)
                    
                    if alt_id is not None:
                        logger.info(f"Center detection {det_idx} forced to use existing ID {alt_id}")
                        final_assignments[det_idx] = (alt_id, False)
                    else:
                        # Fallback to oldest ID
                        alt_id = min(self.global_database.keys()) if self.global_database else self._next_id
                        logger.info(f"Center detection {det_idx} gets fallback ID {alt_id}")
                        final_assignments[det_idx] = (alt_id, False)
                    
                    assigned_detections.add(det_idx)
                    assigned_ids.add(alt_id)
                else:
                    # Can assign new ID
                    new_id = self._next_id
                    final_assignments[det_idx] = (new_id, True)  # New ID
                    assigned_ids.add(new_id)
                    assigned_detections.add(det_idx)
                    logger.info(f"No good match found: detection {det_idx} gets new track_id {new_id}")
                    if is_center and startup_grace_period:
                        logger.info(f"  (Note: Allowing new ID below threshold because in startup grace period)")
                    self._next_id += 1
        
        # Step 5: Collect all track modifications for batch application
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
        
        # Step 6: Apply all modifications to the track list
        # First, update existing tracks and mark tracks as missed
        for track in self.tracks:
            if track.track_id in tracks_to_update:
                detection, is_new = tracks_to_update[track.track_id]
                if not is_new:
                    # Update this existing track
                    track.update(self.kf, detection, self.global_database)
                    logger.info(f"Updated existing track {track.track_id}")
                    # Remove from tracks_to_update so we don't create a duplicate
                    del tracks_to_update[track.track_id]
            elif track.track_id in tracks_to_mark_missed:
                # Mark this track as missed
                track.mark_missed()
                logger.info(f"Marked track {track.track_id} as missed")
        
        # Now create any new tracks
        for track_id, (detection, _) in tracks_to_update.items():
            existing_track_with_id = any(t.track_id == track_id for t in self.tracks)
            if not existing_track_with_id:
                mean, covariance = self.kf.initiate(detection.to_xyah())
                class_name = detection.get_class()
            
                new_track = Track(
                    mean, covariance, track_id, self.n_init, self.max_age,
                    detection.feature, class_name
                )
            
                # Copy features from global database if available
                if track_id in self.global_database:
                    new_track.features = self.global_database[track_id]["features"]
                    logger.info(f"Created track with ID {track_id} (reidentified from database with {len(new_track.features)} features)")
                else:
                    logger.info(f"Created new track with ID {track_id}")
            
                # Add the new track
                self.tracks.append(new_track)
            else:
                existing_tracks = [t for t in self.tracks if t.track_id == track_id]
                best_track = max(existing_tracks, key=lambda t: t.hits)
                # best_track.update(self.kf, detection, self.global_database)
                # logger.info(f"Updated existing track ID {track_id} with n_hits={best_track.hits} instead of creating duplicate")
                if track_id in self.global_database and self.global_database[track_id]["features"]:
                    if not best_track.features:
                        # If track has no features, copy all from database
                        best_track.features = self.global_database[track_id]["features"].copy()
                        logger.info(f"Restored {len(best_track.features)} features from database for track {track_id}")
                    elif len(self.global_database[track_id]["features"]) > len(best_track.features):
                        # If database has more features, merge them with deduplication
                        db_features = self.global_database[track_id]["features"]
                        # Only add features that aren't already in the track
                        for db_feature in db_features:
                            is_duplicate = False
                            for existing_feature in best_track.features:
                                if np.array_equal(db_feature, existing_feature):
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                best_track.features.append(db_feature)
                            # if db_feature not in best_track.features:
                            #     best_track.features.append(db_feature)
                        logger.info(f"Merged features from database for track {track_id}, now has {len(best_track.features)} features")

                # Update the best track with the detection (from option C)
                best_track.update(self.kf, detection, self.global_database)
                logger.info(f"Updated existing track ID {track_id} with n_hits={best_track.hits} instead of creating duplicate")
        
        # Step 7: Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        self._deduplicate_tracks()
        
        # Step 8: Update distance metric
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        
        if features:
            self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)


    def _find_best_match_regardless_of_threshold(self, feature, assigned_ids, max_candidates=10):
        """
        Find the best match regardless of threshold, used for center detections.
        
        Args:
            feature: Feature vector to match
            assigned_ids: Set of already assigned IDs to exclude
            max_candidates: Maximum number of candidates to consider
        
        Returns:
            track_id: Best matching track ID, or None if no unassigned IDs exist
        """
        all_matches = []
        
        # Check all track IDs in the global database
        database_copy = dict(self.global_database)
        for track_id, data in database_copy.items():
            # Skip if already assigned
            if track_id in assigned_ids:
                continue
                
            # Calculate best match with this ID
            best_distance = float('inf')
            for db_feature in data["features"]:
                distance = calculate_cosine_distance(feature, db_feature)
                best_distance = min(best_distance, distance)
                
            # Add to candidates list regardless of threshold
            all_matches.append((track_id, best_distance))
        
        # If no unassigned IDs, return None
        if not all_matches:
            return None
            
        # Sort by distance and return best match
        all_matches.sort(key=lambda x: x[1])
        return all_matches[0][0]  # Return the ID with lowest distance
                
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
        
    def _calculate_distance_directly(self, feature, track):
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
        
        # Use the last feature from the track and our custom distance function
        return calculate_cosine_distance(feature, track.features[-1]) 


    def _calculate_distance_to_id(self, feature, track_id):
        """
        Calculate the distance between a feature vector and a specific track ID.
        
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
                    # Use our direct calculation with the last feature
                    distance = calculate_cosine_distance(feature, track.features[-1])
                    logger.info(f"distance to id is {distance}")
                    return distance
        
        # If not in active tracks or no features, check global database
        if track_id in self.global_database and "features" in self.global_database[track_id]:
            db_features = self.global_database[track_id]["features"]
            if len(db_features) > 0:
                # Find the best match among all features for this ID
                best_distance = float('inf')
                for db_feature in db_features:
                    distance = calculate_cosine_distance(feature, db_feature)
                    best_distance = min(best_distance, distance)
                
                return best_distance
        
        # If no features found, return infinity
        return float('inf')

    def _find_all_potential_matches(self, feature, bbox, max_candidates=10):
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
        
        # Then, check all IDs in the global database
        database_matches = []
        database_copy = dict(self.global_database)
        for db_id, db_info in database_copy.items():
            # Skip IDs that are already in the visible matches
            if any(vm[0] == db_id for vm in visible_matches):
                continue
            
            # Calculate distances to all features for this ID
            if "features" in db_info and db_info["features"]:
                # Find the best match among all features for this ID
                best_distance = float('inf')
                for db_feature in db_info["features"]:
                    distance = calculate_cosine_distance(feature, db_feature)
                    best_distance = min(best_distance, distance)
                
                if best_distance < threshold:
                    database_matches.append((db_id, best_distance, False))
        
        # Combine and sort all matches by distance (ascending)
        all_matches = visible_matches + database_matches
        all_matches.sort(key=lambda x: x[1])
        
        # Return top candidates
        return all_matches[:max_candidates]

    def _match(self, detections):
        """Match detections to tracks."""
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            logger.info(f"Cost matrix is {cost_matrix}")
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        logger.info(f"confirmed_tracks are {confirmed_tracks}")
        # Associate confirmed tracks using appearance features.
        # logger.info("MAT")
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
    
    def _match_with_global_database_all_tracks_considered(self, detection_feature, detection_bbox, center_bbox=(0, 108, 1152, 800)):
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
            logger.info(f"Detection bbox {detection_bbox} is in the center; using increased threshold {threshold}.")
        else:
            threshold = default_threshold
            logger.info(f"Detection bbox {detection_bbox} is not in the center; using default threshold {threshold}.")

        # Iterate through all track_ids in the global database without skipping any
        database_copy = dict(self.global_database)
        for track_id, data in database_copy.items():
            valid_distances = []
            for db_feature in data["features"]:
                # Flatten features to ensure they are 1D arrays
                detection_feature_arr = np.asarray(detection_feature).flatten()
                db_feature_arr = np.asarray(db_feature).flatten()

                # Calculate cosine distance between detection feature and database feature
                dot_product = np.dot(detection_feature_arr, db_feature_arr)
                norm_detection = np.linalg.norm(detection_feature_arr)
                norm_db = np.linalg.norm(db_feature_arr)
                cosine_similarity = dot_product / (norm_detection * norm_db)
                cosine_distance = 1 - cosine_similarity

                # Only consider distances below the threshold
                if cosine_distance < threshold:
                    valid_distances.append(cosine_distance)

            if valid_distances:
                smallest_distance_for_track = min(valid_distances)
                logger.info(f"Track ID {track_id} has valid distances: {valid_distances}. Smallest: {smallest_distance_for_track}")
                if smallest_distance_for_track < min_distance:
                    min_distance = smallest_distance_for_track
                    matched_track_id = track_id
            else:
                logger.info(f"Track ID {track_id} did not yield valid distances under threshold.")

        if matched_track_id is not None:
            logger.info(f"Matched track_id {matched_track_id} with distance {min_distance}.")
        else:
            logger.info("No match found in global database. Assigning new track_id.")

        return matched_track_id


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
            logger.info(f"Detection bbox {detection_bbox} is in the center; using increased threshold {threshold}.")
        else:
            threshold = default_threshold
            logger.info(f"Detection bbox {detection_bbox} is not in the center; using default threshold {threshold}.")

        # store_center_x1, store_center_y1, store_center_x2, store_center_y2 = (0, 270, 1152, 800)

        # Get list of track IDs that are both confirmed AND currently visible in the scene
        visible_confirmed_track_ids = [
            track.track_id for track in self.tracks 
            if track.is_confirmed() and track.time_since_update == 0
        ]
        # Get the list of currently active track IDs
        # active_track_ids = [track.track_id for track in self.tracks if track.is_confirmed()]

        # Iterate through all track_ids in the global database
        for track_id, data in self.global_database.items():
            # Skip if the track_id is currently active
            if track_id in visible_confirmed_track_ids:
                logger.info(f"Skipping track_id {track_id} because it is confirmed and currently visible in scene")
                continue

            # Initialize a list to store distances less than 0.1 for this track_id
            valid_distances = []

            # Iterate through all features for this track_id
            for db_feature in data["features"]:
                # Ensure the features are in the correct format (1D arrays)
                detection_feature = np.asarray(detection_feature).flatten()
                db_feature = np.asarray(db_feature).flatten()

                # Calculate cosine distance between detection feature and database feature
                dot_product = np.dot(detection_feature, db_feature)
                norm_detection = np.linalg.norm(detection_feature)
                norm_db = np.linalg.norm(db_feature)
                cosine_similarity = dot_product / (norm_detection * norm_db)
                cosine_distance = 1 - cosine_similarity

                # Only consider distances less than 0.3
                if cosine_distance < threshold:
                    valid_distances.append(cosine_distance)

            # If there are valid distances for this track_id, find the smallest one
            if valid_distances:
                smallest_distance_for_track = min(valid_distances)
                logger.info(f"Track ID {track_id} has valid distances: {valid_distances}. Smallest: {smallest_distance_for_track}")

                # Check if this track_id has the smallest overall distance
                if smallest_distance_for_track < min_distance:
                    min_distance = smallest_distance_for_track
                    matched_track_id = track_id
            else:
                logger.info(f"Smallest distance with {track_id} is more than 0.5, so we are skipping it.")

        if matched_track_id is not None:
            logger.info(f"Matched track_id {matched_track_id} with distance {min_distance}.")
        else:
            logger.info("No match found in global database. Assigning new track_id.")

        return matched_track_id