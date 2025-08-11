#!/usr/bin/env python3
import os, json, time, base64, argparse, logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from collections import defaultdict

import cv2
import numpy as np
import torch
from kafka import KafkaConsumer
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta

# Optional heavy models only if needed
def maybe_load_detector(enable):
    if not enable:
        return None, None
    from processor_segment_with_transreid import setup_predictor
    pred = setup_predictor()
    model = pred.model.eval()
    return pred, model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("kafka_image_sampler")

# ---------- AWS Secrets ----------
def fetch_kafka_creds():
    region = (
        os.getenv("KAFKA_AWS_SECRETS_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "eu-west-3"
    )
    secret_name = os.getenv("KAFKA_CREDENTIALS_SECRET_NAME", "AmazonMSK_genfied-kafka-consumer")
    client = boto3.client("secretsmanager", region_name=region)
    try:
        resp = client.get_secret_value(SecretId=secret_name)
        sec = json.loads(resp["SecretString"])
        return sec["username"], sec["password"]
    except (ClientError, KeyError, json.JSONDecodeError) as e:
        log.error(f"Failed to fetch Kafka creds: {e}")
        raise

# ---------- Drawing ----------
def draw_top_right_label(img, text, pad=10, scale=0.8, thickness=2):
    """Draw text at the top-right corner with a white background box."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    h, w = img.shape[:2]
    x2 = w - pad
    y1 = pad + th + 4
    x1 = x2 - tw - 10
    y2 = y1 + 6
    cv2.rectangle(img, (x1-2, y1-th-8), (x2+2, y2), (255,255,255), -1)
    cv2.putText(img, text, (x1+2, y1-6), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness, cv2.LINE_AA)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamp_for_name(iso_ts: Optional[str]) -> str:
    t = iso_ts or datetime.utcnow().isoformat()
    return t.replace(":", "-").replace(".", "_")

# ---------- Detection ----------
def detect_person_bboxes(detector, image_bgr):
    """
    Returns list of bboxes [(x1,y1,x2,y2), ...] for person class (0),
    in ORIGINAL image coordinates (Detectron2 handles mapping back).
    """
    if detector is None:
        return []
    pred = detector
    h, w = image_bgr.shape[:2]
    tfm = pred.aug.get_transform(image_bgr)
    timg = tfm.apply_image(image_bgr).astype("float32").transpose(2,0,1)
    timg = torch.as_tensor(timg)
    with torch.no_grad():
        out = pred.model([{"image": timg, "height": h, "width": w}])[0]
    inst = out["instances"]
    idx = (inst.pred_classes == 0).nonzero().flatten()
    if len(idx) == 0:
        return []
    boxes = inst.pred_boxes.tensor[idx].cpu().numpy().astype(int).tolist()
    return boxes

def draw_bboxes(img, boxes, color=(0,255,255), thick=2):
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)

def scale_boxes(boxes, sx, sy):
    return [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)) for (x1,y1,x2,y2) in boxes]

# ---------- Decoding ----------
def decode_message_to_image_and_meta(msg_dict):
    """Same function but with image identifier in logs"""
    if not isinstance(msg_dict, dict):
        return None, {}
        
    # Metadata passthrough
    meta_keys = ("stream_name","store_id","camera_id","node_ip","processor_id",
                 "timestamp","sequence_number","width","height","format","fps","original_size")
    metadata = {k: msg_dict.get(k) for k in meta_keys}
    
    # CREATE IDENTIFIER EARLY
    store_id = str(metadata.get("store_id", "unknown"))
    camera_id = str(metadata.get("camera_id", "unknown"))
    timestamp = metadata.get("timestamp", "")
    ts_name = timestamp_for_name(timestamp)
    image_id = f"store{store_id}_cam{camera_id}_{ts_name}"
    
    frame_data = msg_dict.get("frame_data")
    if not frame_data or not isinstance(frame_data, str):
        log.warning(f"[{image_id}] No valid frame_data")
        return None, metadata
        
    try:
        width = metadata.get("width", 0)
        height = metadata.get("height", 0)
        expected_raw_size = width * height * 3
        img_bytes = base64.b64decode(frame_data)
        decoded_size = len(img_bytes)
        
        # INCLUDE IMAGE_ID IN ALL LOGS
        log.info(f"[{image_id}] Image {width}x{height}: base64={len(frame_data)} chars, "
                f"decoded={decoded_size} bytes, expected={expected_raw_size} bytes, "
                f"ratio={decoded_size/expected_raw_size:.2f}")

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            std_dev = np.std(img)
            log.info(f"[{image_id}] Decode SUCCESS: shape={img.shape}, std_dev={std_dev:.1f}")
        else:
            log.warning(f"[{image_id}] Decode FAILED: cv2.imdecode returned None")
            
        return img, metadata
    except Exception as e:
        log.warning(f"[{image_id}] Failed to decode base64 frame: {e}")
        return None, metadata

# ---------- Sampling policies ----------
class Sampler:
    def __init__(self, mode, interval_seconds=None, every_n=None):
        self.mode = mode
        self.interval = interval_seconds
        self.every_n = every_n
        self.last_time: Dict[Tuple[str, str], datetime] = {}
        self.counts: defaultdict[Tuple[str, str], int] = defaultdict(int)


    def should_save(self, store_id, camera_id, now: datetime) -> bool:
        key = (store_id, camera_id)
        if self.mode == "time":
            lt = self.last_time.get(key)
            if lt is None or (now - lt) >= timedelta(seconds=self.interval):
                self.last_time[key] = now
                return True
            return False
        else:  # count
            self.counts[key] += 1
            if self.counts[key] % self.every_n == 0:
                return True
            return False

# ---------- Main consumer ----------
def main():
    ap = argparse.ArgumentParser(description="Kafka image quality sampler")
    ap.add_argument("--store-ids", required=True,
                    help="Comma-separated store IDs to sample (e.g., '114,121')")
    ap.add_argument("--topic-prefix", default="store-",
                    help="Topic name prefix. Topic per store assumed as '<prefix><store_id>'")
    ap.add_argument("--mode", choices=["time","count"], default="time",
                    help="Sampling mode: time or count")
    ap.add_argument("--interval-seconds", type=int, default=60,
                    help="Save period per camera when --mode=time")
    ap.add_argument("--every-n", type=int, default=100,
                    help="Save every Nth frame per camera when --mode=count")
    ap.add_argument("--output-root", default="/data",
                    help="Root folder mounted in the pod to store images")
    ap.add_argument("--resize-width", type=int, default=1280)
    ap.add_argument("--resize-height", type=int, default=720)
    ap.add_argument("--enable-bboxes", action="store_true",
                    help="Run detector and draw person boxes (GPU)")

    ap.add_argument("--group-id", default="image-quality-sampler")

    ap.add_argument("--max-minutes", type=int, default=2,
                    help="Stop after N minutes (default: 15). Set 0 to run indefinitely.")

    args = ap.parse_args()

    store_ids = [s.strip() for s in args.store_ids.split(",") if s.strip()]
    topics = [f"{args.topic_prefix}{sid}" for sid in store_ids]

    username, password = fetch_kafka_creds()
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").split(",")
    if not bootstrap or bootstrap == [""]:
        raise RuntimeError("KAFKA_BOOTSTRAP_SERVERS is required")

    security_protocol = os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
    sasl_mechanism = os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512")

    log.info(f"Stores: {store_ids} | Topics: {topics}")
    log.info(f"Saving under: {args.output_root}")
    ensure_dir(args.output_root)

    # Model: only if needed
    detector, model = maybe_load_detector(args.enable_bboxes)

    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=bootstrap,
        security_protocol=security_protocol,
        sasl_mechanism=sasl_mechanism,
        sasl_plain_username=username,
        sasl_plain_password=password,
        group_id=args.group_id,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="latest",
        consumer_timeout_ms=10_000,
        session_timeout_ms=30_000,
        heartbeat_interval_ms=10_000,
        max_poll_records=10,
    )

    sampler = Sampler(
        mode=args.mode,
        interval_seconds=args.interval_seconds,
        every_n=args.every_n
    )

    deadline = None
    if args.max_minutes and args.max_minutes > 0:
        deadline = datetime.utcnow() + timedelta(minutes=args.max_minutes)
        log.info(f"Sampler will stop after ~{args.max_minutes} minutes (at {deadline.isoformat()} UTC)")


    try:
        for msg in consumer:

            if deadline and datetime.utcnow() >= deadline:
                log.info("Max runtime reached. Stopping gracefully...")
                break

            payload = msg.value
            img, meta = decode_message_to_image_and_meta(payload)
            if img is None:
                continue

            store_id = str(meta.get("store_id") or "")
            camera_id = str(meta.get("camera_id") or "unknown")
            if store_id not in store_ids:
                continue

            now = datetime.utcnow()
            if not sampler.should_save(store_id, camera_id, now):
                continue

            # Build paths
            leaf_dir = os.path.join(args.output_root, f"store-{store_id}", f"{camera_id}")
            ensure_dir(leaf_dir)

            # Names
            ts = timestamp_for_name(meta.get("timestamp"))
            base = f"{ts}_store{store_id}_cam{camera_id}"
            p_orig   = os.path.join(leaf_dir, f"{base}_orig.jpg")
            p_resz   = os.path.join(leaf_dir, f"{base}_resized.jpg")
            p_bbox_o = os.path.join(leaf_dir, f"{base}_bboxes.jpg")          # original-sized with boxes
            p_bbox_r = os.path.join(leaf_dir, f"{base}_bboxes_resized.jpg")  # resized version with boxes

            # Original (with label only)
            orig = img.copy()
            draw_top_right_label(orig, f"STORE {store_id} | CAMERA {camera_id}")
            ok = cv2.imwrite(p_orig, orig, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok: log.warning(f"Failed to write {p_orig}")

            # Plain resized (with label)
            resized = cv2.resize(img, (args.resize_width, args.resize_height))
            draw_top_right_label(resized, f"STORE {store_id} | CAMERA {camera_id}")
            ok = cv2.imwrite(p_resz, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok: log.warning(f"Failed to write {p_resz}")

            # BBoxes (optional): draw on ORIGINAL-SIZED canvas, then resize AFTER drawing
            boxes_original = []
            boxes_resized  = []
            if args.enable_bboxes:
                boxes_original = detect_person_bboxes(detector, img)  # original coordinates
                bbox_img_orig = img.copy()
                if boxes_original:
                    draw_bboxes(bbox_img_orig, boxes_original, (0,255,255), 2)
                draw_top_right_label(bbox_img_orig, f"STORE {store_id} | CAMERA {camera_id}")
                ok = cv2.imwrite(p_bbox_o, bbox_img_orig, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok: log.warning(f"Failed to write {p_bbox_o}")

                # Now make a resized artifact AFTER drawing to preserve alignment
                bbox_img_resized = cv2.resize(bbox_img_orig, (args.resize_width, args.resize_height))
                # Optional: label again (already present, but harmless)
                draw_top_right_label(bbox_img_resized, f"STORE {store_id} | CAMERA {camera_id}")
                cv2.imwrite(p_bbox_r, bbox_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # For completeness, also provide scaled box coordinates for the resized canvas
                h0, w0 = img.shape[:2]
                sx, sy = args.resize_width / w0, args.resize_height / h0
                boxes_resized = scale_boxes(boxes_original, sx, sy)
            else:
                # Still create bbox files for a consistent triplet, even when detector is off
                bbox_img_orig = img.copy()
                draw_top_right_label(bbox_img_orig, f"STORE {store_id} | CAMERA {camera_id}")
                ok = cv2.imwrite(p_bbox_r, bbox_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok: log.warning(f"Failed to write {p_bbox_r}")

                bbox_img_resized = cv2.resize(bbox_img_orig, (args.resize_width, args.resize_height))
                draw_top_right_label(bbox_img_resized, f"STORE {store_id} | CAMERA {camera_id}")
                cv2.imwrite(p_bbox_r, bbox_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Optional small metadata file per set
            meta_path = os.path.join(leaf_dir, f"{base}.json")
            meta_out = {
                "kafka": {
                    "topic": msg.topic, "partition": msg.partition, "offset": msg.offset
                },
                "metadata": meta,
                "detections": {
                    "boxes_original": boxes_original,         # in original image coords
                    "boxes_resized": boxes_resized,           # scaled to (resize_width, resize_height)
                },
                "outputs": {
                    "orig":   os.path.basename(p_orig),
                    "resized": os.path.basename(p_resz),
                    "bboxes_original": os.path.basename(p_bbox_o),
                    "bboxes_resized": os.path.basename(p_bbox_r),
                }
            }
            with open(meta_path, "w") as f:
                json.dump(meta_out, f, indent=2)

            log.info(f"Saved images for store={store_id} camera={camera_id} at {leaf_dir}")

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        try:
            consumer.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
