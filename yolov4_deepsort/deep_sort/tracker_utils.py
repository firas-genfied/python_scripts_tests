import numpy as np

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