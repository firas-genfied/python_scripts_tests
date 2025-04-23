import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
transreid_path = os.path.join(current_dir, "TransReID")
if transreid_path not in sys.path:
    sys.path.append(transreid_path)
from TransReID.config import cfg
from TransReID.model import make_model
from TransReID.datasets.transforms import build_transforms
from TransReID.processor import extract_features
import torch
import json
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from yolov4_deepsort.deep_sort import preprocessing, nn_matching
from yolov4_deepsort.deep_sort.detection import Detection
from yolov4_deepsort.deep_sort.tracker import AsyncTracker
from yolov4_deepsort.tools import generate_detections as gdet
from ultralytics import YOLO
from status_checker import improved_human_status
from PIL import Image
cfg.merge_from_file("/home/azureuser/workspace/Genfied/TransReID/configs/Market/vit_transreid.yml")  # Use the MSMT17 config
cfg.TEST.WEIGHT = "/home/azureuser/workspace/Genfied/TransReID/models/vit_base_msmt.pth"  # Fine-tuned model weights
cfg.freeze()

# === 2️⃣ Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MSMT17 has 4,101 IDs so we pass num_class=4101
model = make_model(cfg, num_class = 1041, camera_num=0, view_num=0).to(device)
model.load_param(cfg.TEST.WEIGHT)
model.eval()

transform = build_transforms(cfg, is_train=False)


class YOLOv8_DeepSort:
    def __init__(self, info_flag=True):
        # Initialize YOLOv8 model
        self.yolo_model = YOLO("yolov8l.pt", verbose = False)  # Changed to yolov8l.pt as in object_tracker.py
        
        # Initialize DeepSORT
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 0.8
        self.model_filename = 'yolov4_deepsort/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)
        
        # Flag for detailed information
        self.info_flag = info_flag

        self.person_status = {}
        self.recent_entries = {"entries": [], "classified": {}}
        self.previous_positions = {}  # Stores previous coordinates for direction checking
        self.counted_track_ids = set() 
        self.singles = 0
        self.couples = 0
        self.groups = 0
    
    def is_inside_roi(self,x, y, roi_x1, roi_y1, roi_x2, roi_y2):
        return roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2

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

    def process(self, video_path: str, output_path: str = 'output_video.mp4', desired_fps = 1, images_folder="images/cam1"):
        """
        Process the video frame by frame and return detections for each frame.
        Also saves the processed video with detections.
        """
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise ValueError("Unable to open video file")
        
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        # Get video properties for output
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(vid.get(cv2.CAP_PROP_FPS))
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(round(original_fps / desired_fps))

        print(f"\nVideo Properties:")
        print(f"Original Resolution: {width}x{height}")
        print(f"Total Frames: {total_frames}")
        print(f"FPS: {original_fps}\n")
        print(f"Detection/Tracking will run every {frame_interval} frame(s).\n")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
        # print(f"Processing every {frame_interval} frames to achieve {desired_fps} fps sampling rate.")

        frame_num = 0
        all_frames_data = []
        start_time = datetime.utcnow()  # Start time for the video

        # Store the last computed overlay information so that we can re-use it on in-between frames.
        last_active_tracks = []
        last_entity_coordinates = []
        last_frame_overlay = None  # You could store a copy of the last frame with overlays

        roi_x1, roi_y1, roi_x2, roi_y2 = (768, 270, 1152, 800)  # Entrance Region

        while True:
            ret, frame = vid.read()
            if not ret:
                break
            frame_copy = frame.copy()
            if frame_num % frame_interval == 0:
                frame_start_time = time.time()
                print(f"processing frame {frame_num}")
            #     image_filename = os.path.join(images_folder, f"frame_{frame_num:04d}.jpg")
            #     cv2.imwrite(image_filename, frame)

            # if frame_num % frame_interval != 0:
            #     frame_num += 1
            #     continue

            # frame_start_time = time.time()  # For FPS calculation
            
                current_height, current_width = frame.shape[:2]
                # print(f"\nProcessing Frame #{frame_num}")
                # print(f"Current Frame Resolution: {current_width}x{current_height}")

                # Ensure consistent resolution
                if current_width != width or current_height != height:
                    frame = cv2.resize(frame, (width, height))
                    print(f"Frame resized to match original resolution: {width}x{height}")

                # Calculate the timestamp based on frame number and FPS
                current_time_video = start_time + timedelta(seconds=(frame_num / original_fps))
                current_timestamp = current_time_video.timestamp() 

                # Convert BGR to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Define the green box dimensions (10% smaller on all sides)
                frame_height, frame_width, _ = frame.shape
                shrink_percentage = 0.10
                x1 = int(frame_width * shrink_percentage)
                y1 = int(frame_height * shrink_percentage)
                x2 = int(frame_width * (1 - shrink_percentage))
                y2 = int(frame_height * (1 - shrink_percentage))
                # Draw the green box on the frame
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Run YOLO inference on the full frame
                yolo_results = self.yolo_model(frame_rgb, stream=True)

                detections = []
                person_count = 0
                for result in yolo_results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    for i, class_id in enumerate(class_ids):
                        class_name = self.yolo_model.names[class_id]
                        if class_name == "person" and scores[i] >= 0.1:
                            person_count += 1
                            bbox = boxes[i]  # [x1, y1, x2, y2]
                            cv2.rectangle(frame_copy,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(255, 255, 255), 2)

                    print(f"Number of persons detected in this frame {frame_num}: {person_count}")

                    for i, class_id in enumerate(class_ids):
                        class_name = self.yolo_model.names[class_id]
                        if class_name == "person" and scores[i] >= 0.1:
                            bbox = boxes[i]
                            # Calculate overlap area with the green box
                            green_box = [x1, y1, x2, y2]
                            overlap_area, bbox_area = self.calculate_overlap(bbox, green_box)

                            # If majority of the bounding box is inside the green box, add to detections
                            if overlap_area / bbox_area > 0.5:
                                # Convert to TLWH format for DeepSORT
                                tlwh_bbox = [
                                    bbox[0],
                                    bbox[1],
                                    bbox[2] - bbox[0],  # width
                                    bbox[3] - bbox[1]   # height
                                ]
                                confidence = scores[i]
                                detections.append((tlwh_bbox, confidence, class_name))

                # Get features
                # features = self.encoder(frame_rgb, [d[0] for d in detections])
                body_features = []
                for (tlwh_bbox, confidence, class_name) in detections:
                    x, y, w, h = map(int, tlwh_bbox)
                    crop = frame_rgb[y:y+h, x:x+w]
                    if crop.size == 0:
                        body_features.append(np.zeros((1,)))  # You may want to set the correct shape
                        continue
                    crop_pil = Image.fromarray(crop)
                    crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = extract_features(model, crop_tensor).cpu().numpy()
                        feat = feat.reshape(-1)  # or np.squeeze(feat) if that gives you the correct shape
                    body_features.append(feat)

                # Create Detection objects
                detections = [Detection(bbox, score, class_name, feature) 
                            for (bbox, score, class_name), feature in zip(detections, body_features)]

                # Process detections
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.class_name for d in detections])
                indices = preprocessing.non_max_suppression(boxes, classes, self.nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Update tracker
                self.tracker.predict()
                self.tracker.update(detections)

                # Initialize color map for visualization
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                active_tracks = []
                entity_coordinates = []

                # Print frame header for detailed info
                # if self.info_flag:
                    # print(f"\nFrame {frame_num} Tracking Details:")

                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    bbox = track.to_tlbr()
                    top_left = (bbox[0], bbox[1])
                    top_right = (bbox[2], bbox[1])
                    bottom_left = (bbox[0], bbox[3])
                    bottom_right = (bbox[2], bbox[3])
                    bbox_center_x = int(round((bbox[0] + bbox[2]) / 2))
                    bbox_center_y = int(round((bbox[1] + bbox[3]) / 2))
                    bbox_y_80 = int(round(bbox[1] + 0.8 * (bbox[3] - bbox[1])))

                    classification, is_final = improved_human_status(track, current_timestamp, self.recent_entries, self.previous_positions, roi_y1)#, entrance_direction='top')
                    if classification:
                        if "_" in classification:
                            classification_type, group_id_str = classification.split("_", 1)
                            group_id = int(group_id_str)
                            print(f"classification_type for {track.track_id} is {classification_type} and id is {group_id}")
                        else:
                            classification_type = classification
                            group_id = None
                            print(f"classification_type for {track.track_id} is {classification_type} and id is {group_id}")
                    else:
                        classification_type = None
                        group_id = None
                    # print(f"Classification result for Track ID {track.track_id}:")
                    # print(f"  Type: {classification_type}, Group ID: {group_id}")
                    # print(f"  Is it finalized: {is_final}")
                    # Update counters if classification is finalized
                    if is_final and track.track_id not in self.counted_track_ids:
                    # if classification and track.track_id not in self.recent_entries["classified"]:
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
                        print(f"finalizing status of {track.track_id} to {classification_type}")
                        self.person_status[track.track_id]["status"] = classification_type if classification_type else ""
                        self.person_status[track.track_id]["group_id"] = str(group_id) if group_id is not None else ""
                    else:
                        # For non-finalized tracks, leave the status and group_id as empty strings.
                        self.person_status[track.track_id]["status"] = ""
                        self.person_status[track.track_id]["group_id"] = ""

                # entity_coordinates.append({
                #     "person_id": str(track.track_id),
                #     "x_coord": bbox_center_x,
                #     "y_coord": bbox_center_y,
                #     "status": classification if classification else "undetermined"
                # })

                # active_tracks.append(track.track_id)

                # if (roi_x1 <= bbox_center_x <= roi_x2 and roi_y1 <= bbox_center_y <= roi_y2):
                    # if any(self.is_inside_roi(x, y, roi_x1, roi_y1, roi_x2, roi_y2) for x, y in [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3])]):
                        # Print detailed tracking information
                        # if self.info_flag:
                            # print(f"Tracker ID: {track.track_id}")
                            # print(f"Class: {track.get_class()}")
                            # print(f"BBox Coords (xmin, ymin, xmax, ymax): ({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])})")
                            # print(f"Center Coordinates (x, y): ({bbox_center_x}, {bbox_center_y})")
                            # print(f"80y Offset Coordinates (x, y): ({bbox_center_x}, {bbox_y_80})")

                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]

                    # Draw bounding box
                    cv2.rectangle(frame_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                    # Prepare text for display (ID and coordinates)
                    id_text = f"{track.get_class()}-{track.track_id}"
                    status_text = f"Status: {classification}" if classification else "Status: undetermined"
                    coord_text = f"({bbox_center_x}, {bbox_y_80})"

                    # Calculate text sizes for positioning
                    (id_width, id_height), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    (status_width, status_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    id_text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
                    coord_text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]

                    # Draw background rectangles and texts
                    cv2.rectangle(frame_copy, 
                                (int(bbox[0]), int(bbox[1]-35)), 
                                (int(bbox[0])+id_width, int(bbox[1]-5)), 
                                color, -1)
                    cv2.rectangle(frame_copy, 
                                (int(bbox[0]), int(bbox[1]-70)), 
                                (int(bbox[0])+status_width, int(bbox[1]-40)), 
                                color, -1)

                    cv2.putText(frame_copy, id_text,
                                (int(bbox[0]), int(bbox[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                    cv2.putText(frame_copy, status_text,
                                (int(bbox[0]), int(bbox[1]-45)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

                    active_tracks.append(track.track_id)
                    if bbox_center_x is not None and bbox_center_y is not None:
                        entity_coordinates.append({
                            "person_id": str(track.track_id),
                            "x_coord": bbox_center_x,
                            "y_coord": bbox_y_80,
                            "status": self.person_status.get(track.track_id, {}).get("status", "undetermined"),
                            "group_id": self.person_status.get(track.track_id, {}).get("group_id", "")
                        })
                # entity_coordinates.append({
                #     "person_id": str(track.track_id),
                #     "x_coord": bbox_center_x,
                #     "y_coord": bbox_y_80,
                #     "status": classification if classification else "undetermined"
                # })
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)  # Red for entrance ROI
            # Calculate FPS
                last_active_tracks = active_tracks
                last_entity_coordinates = entity_coordinates
                processing_fps = 1.0 / (time.time() - frame_start_time)
                # print(f"Processing FPS for key frame: {processing_fps:.2f}")
            # calculated_fps = 1.0 / (time.time() - frame_start_time)
            # fps = 1.0 / (time.time() - frame_start_time)
            # print(f"FPS: {fps:.2f}")

            # Print total number of people in the frame
                # if self.info_flag:
                #     print(f"\nTotal people in frame: {len(active_tracks)}")
                #     if active_tracks:
                #         print(f"Active tracker IDs: {active_tracks}")

                frame_data = {
                    "camera_id": 1,
                    "frame_number": frame_num,
                    "resolution": f"{width}x{height}",
                    "fps": round(processing_fps, 2),
                    "image_url": "",
                    "is_organised": True,
                    "no_of_people": len(active_tracks),
                    "current_time": current_time_video.isoformat(),
                    "ENTITY_COORDINATES": entity_coordinates
                }

                all_frames_data.append(frame_data)
            else:
                self.tracker.predict()
                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    color = (0, 255, 255)  # For example, a different color for predicted boxes
                    cv2.rectangle(frame_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame_copy, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

            # Write the frame to output video
            out.write(frame_copy)
            frame_num += 1

            # Display the frame
            # plt.figure(figsize=(10, 10))
            # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.title(f"Frame {frame_num} - People Count: {len(active_tracks)}")
            # plt.show()
            # plt.close()

        # Release everything
        vid.release()
        out.release()
        self.tracker.save_global_database()

        # Save tracking results to JSON
        videofile_name = "video1long_supershot_cropped_lady"
        output_json_file = f"tracking_results_{videofile_name}_{desired_fps}fps.json"
        output_file = os.path.join(os.getcwd(), output_json_file)
        with open(output_file, "w") as f:
            json.dump(all_frames_data, f, indent=4)

        # return all_frames_data
        return {
            "singles": self.singles,
            "couples": self.couples,
            "groups": self.groups,
            "all_frames_data": all_frames_data
        }


def process_video(file_path: str, output_path: str = 'output_video.mp4', info_flag=True, desired_fps = 1, images_folder="images/cam1"):
    """
    Process the video and extract required information.
    Args:
        file_path: Path to input video
        output_path: Path to save processed video
        info_flag: Whether to print detailed information
    """
    model = YOLOv8_DeepSort(info_flag=info_flag)
    results = model.process(file_path, output_path, desired_fps, images_folder)
    singles, couples, groups = results["singles"], results["couples"], results["groups"]
    print(f"singles count {singles}, couples count: {couples}, groups: {groups}")
    return results

if __name__ == "__main__":
    videofile_name = "video1long_supershot_cropped_lady"
    video_path = f"/home/azureuser/workspace/Genfied/input_videos/{videofile_name}.mp4"
    output_path = f"/home/azureuser/workspace/Genfied/input_videos/result_{videofile_name}_Transreid_lowconf.mp4"
    desired_fps = 25
    results = process_video(video_path, output_path, info_flag=True, desired_fps = desired_fps, images_folder="images/video2long")
    output_json_file = f"tracking_results_{videofile_name}_{desired_fps}fps.json"
    print(f"Processing complete. Results saved in {output_json_file}")
    print(f"Processed video saved as '{output_path}'")