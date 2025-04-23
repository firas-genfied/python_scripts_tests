import numpy as np
import cv2
import os 
import logging
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

def filter_duplicate_detections(boxes, scores, masks, frame_copy, iou_threshold=0.7):
    """
    Filter out duplicate detections with high IoU.
    Keep the detection with the higher confidence score.
    Additionally, draw the suppressed (duplicate) boxes in red on frame_copy.
    
    Args:
        boxes (np.ndarray): Array of bounding boxes.
        scores (np.ndarray): Array of confidence scores.
        masks (np.ndarray): Array of masks.
        frame_copy (np.ndarray): The image frame on which to draw discarded boxes.
        iou_threshold (float): IoU threshold for considering two boxes as duplicates.
        
    Returns:
        filtered_boxes, filtered_scores, filtered_masks: The filtered detection outputs.
    """
    indices = list(range(len(boxes)))
    suppressed = set()
    indices.sort(key=lambda i: scores[i], reverse=True)
    keep = []
    
    for i in indices:
        if i in suppressed:
            # Draw suppressed detection in red (if not already drawn in the inner loop)
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            continue
        
        keep.append(i)
        for j in indices:
            if j <= i or j in suppressed:
                continue
            iou = compute_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                suppressed.add(j)
                # Draw the suppressed box in red
                x1, y1, x2, y2 = map(int, boxes[j])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                logger.info(f"suppressed box {x1}, {y1}, {x2}, {y2}. Had an IoU of {iou}")
    
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    filtered_masks = masks[keep]
    return filtered_boxes, filtered_scores, filtered_masks

def crop_without_resize(image, bbox, mask):
    """
    Crop the image using the given bbox and apply the mask without resizing or padding.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2]
    # Apply mask: set pixels where mask==0 to black.
    cropped_img[cropped_mask == 0] = 0
    return cropped_img

def has_left_store_percent(bbox, line_y, threshold=0.20):
    """
    Determines if more than a given percentage of the bounding box is above a horizontal line.

    In image coordinates (y increases downward):
      - If the entire bbox is above the line (y2 < line_y), fraction = 1.0.
      - If the entire bbox is below the line (y1 >= line_y), fraction = 0.0.
      - Otherwise, fraction = (line_y - y1) / (y2 - y1).

    Args:
        bbox (list or tuple): The bounding box [x1, y1, x2, y2].
        line_y (int): The y-coordinate of the horizontal line.
        threshold (float): The fraction threshold (default 0.30 for 30%).

    Returns:
        (bool, float): A tuple where the first element is True if the fraction above the line 
                       is greater than the threshold, and the second element is the fraction.
    """
    y1 = bbox[1]
    y2 = bbox[3]
    height = y2 - y1
    if height <= 0:
        return False, 0.0

    # If the entire box is above the line.
    if y2 < line_y:
        fraction_above = 1.0
    # If the entire box is below the line.
    elif y1 >= line_y:
        fraction_above = 0.0
    else:
        fraction_above = (line_y - y1) / height

    return fraction_above > threshold, fraction_above

def is_entering_store_percent(bbox, line_y, threshold=0.70):
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

    return fraction_below > threshold, fraction_below