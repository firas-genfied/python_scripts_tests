import cv2
import numpy as np
import os
import logging
import json
import argparse
from datetime import datetime
import torch
from PIL import Image
import time
import base64
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import io
import subprocess
import tempfile

# Import your custom modules (assuming they exist)
from processor_segment_with_transreid import setup_predictor
from TransReID.config import cfg
from TransReID.model import make_model
from TransReID.datasets.transforms import build_transforms
from TransReID.processor import extract_features
from utils.detection_utils import (
    crop_without_resize, 
    filter_duplicate_detections
)

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class KafkaCredentialsManager:
    def __init__(self):
        # pick region from env or default
        self.region = (
            os.getenv("KAFKA_AWS_SECRETS_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "eu-west-3"
        )
        self.secret_name = os.getenv(
            "KAFKA_CREDENTIALS_SECRET_NAME",
            "AmazonMSK_genfied-kafka-consumer"
        )
        self.credentials = None

    def get_kafka_credentials(self):
        """Retrieve Kafka credentials from AWS Secrets Manager."""
        if self.credentials:
            return self.credentials

        client = boto3.client("secretsmanager", region_name=self.region)
        try:
            resp = client.get_secret_value(SecretId=self.secret_name)
            secret = json.loads(resp["SecretString"])
            self.credentials = {
                "username": secret["username"],
                "password": secret["password"]
            }
            logger.info(f"Retrieved Kafka creds for user={self.credentials['username']}")
            return self.credentials
        except (ClientError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"Unable to load Kafka credentials: {e}")
            raise


class KafkaImageProcessor:
    """Kafka-enabled processor for real-time image analysis"""
    
    def __init__(self, kafka_bootstrap_servers="35.181.243.135:29092", 
                 kafka_topic="store-109", device=None, output_dir="kafka_output"):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.output_dir = output_dir
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize Kafka consumer
        self.consumer = None
        self.setup_kafka_consumer()
        
        # Initialize the segmentation model (Detectron2)
        self.seg_predictor = setup_predictor()
        self.seg_model = self.seg_predictor.model
        self.seg_model.eval()
        self.aug = self.seg_predictor.aug
        logger.info("Segmentation model initialized")
        
        # Initialize TransReID model
        self.model = make_model(cfg, num_class=1041, camera_num=0, view_num=0).to(self.device)
        self.model.load_param(cfg.TEST.WEIGHT)
        self.model.eval()
        self.transform = build_transforms(cfg, is_train=False)
        self.extract_features = extract_features
        logger.info("TransReID model initialized")
        
        # Frame counter for saving unique files
        self.frame_counter = 0

    def setup_kafka_consumer(self):
        """Setup Kafka consumer with proper configuration"""
        try:
            # self.consumer = KafkaConsumer(
            #     self.kafka_topic,
            #     bootstrap_servers=[self.kafka_bootstrap_servers],
            #     auto_offset_reset='latest',  # Start from latest messages
            #     enable_auto_commit=True,
            #     group_id='image_processor_group',
            #     value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None,  # Try JSON first
            #     consumer_timeout_ms=1000,  # Timeout after 1 second of no messages
            #     max_poll_records=1,  # Process one message at a time
            #     session_timeout_ms=30000,
            #     heartbeat_interval_ms=10000
            # )
            self.consumer = KafkaConsumer(
                self.kafka_topic,
                bootstrap_servers=[self.kafka_bootstrap_servers],
                security_protocol=security,
                sasl_mechanism=mechanism,
                sasl_plain_username=creds["username"],
                sasl_plain_password=creds["password"],
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='image_processor_group',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None,
                consumer_timeout_ms=1000,
                max_poll_records=1,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000
            )
            logger.info(f"Kafka consumer initialized for topic: {self.kafka_topic}")
            logger.info(f"Bootstrap servers: {self.kafka_bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            # Try with raw bytes deserializer as fallback
            try:
                self.consumer = KafkaConsumer(
                    self.kafka_topic,
                    bootstrap_servers=[self.kafka_bootstrap_servers],
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    group_id='image_processor_group',
                    value_deserializer=lambda m: m,  # Keep as bytes
                    consumer_timeout_ms=1000,
                    max_poll_records=1,
                    session_timeout_ms=30000,
                    heartbeat_interval_ms=10000
                )
                logger.info("Kafka consumer initialized with bytes deserializer")
            except Exception as e2:
                logger.error(f"Failed to initialize Kafka consumer with fallback: {e2}")
                raise

    def decode_with_ffmpeg(self, image_data, width, height):
        """Try to decode image data using FFmpeg with various codecs"""
        # List of codecs to try
        codecs_to_try = [
            'mjpeg',      # Motion JPEG
            'h264',       # H.264
            'hevc',       # H.265/HEVC  
            'mpeg4',      # MPEG-4
            'libx264',    # x264 encoder/decoder
            'libx265',    # x265 encoder/decoder
            'rawvideo',   # Raw video
            'png',        # PNG
            'bmp',        # BMP
            'tiff',       # TIFF
        ]
        
        for codec in codecs_to_try:
            try:
                logger.info(f"Trying FFmpeg codec: {codec}")
                
                # Create temporary files
                with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as input_file:
                    input_file.write(image_data)
                    input_path = input_file.name
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_file:
                    output_path = output_file.name
                
                # Try to decode with current codec
                cmd = [
                    'ffmpeg',
                    '-f', 'rawvideo',
                    '-vcodec', codec,
                    '-s', f'{width}x{height}',
                    '-i', input_path,
                    '-f', 'image2',
                    '-vcodec', 'png',
                    '-y',  # Overwrite output file
                    output_path
                ]
                
                # Run FFmpeg
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                
                if result.returncode == 0:
                    # Successfully decoded, load the image
                    try:
                        image = cv2.imread(output_path)
                        if image is not None:
                            logger.info(f"Successfully decoded with codec: {codec}")
                            # Clean up temp files
                            try:
                                os.unlink(input_path)
                                os.unlink(output_path)
                            except:
                                pass
                            return image
                    except Exception as e:
                        logger.warning(f"Failed to load decoded image for codec {codec}: {e}")
                
                # Clean up temp files
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except:
                    pass
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"FFmpeg timeout for codec: {codec}")
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except:
                    pass
            except Exception as e:
                logger.warning(f"FFmpeg failed for codec {codec}: {e}")
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except:
                    pass
        
        # Try alternative approach: decode as compressed video stream
        try:
            logger.info("Trying FFmpeg as compressed video stream...")
            
            with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as input_file:
                input_file.write(image_data)
                input_path = input_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_file:
                output_path = output_file.name
            
            # Try to auto-detect format and decode
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-f', 'image2',
                '-vcodec', 'png',
                '-vframes', '1',  # Extract only first frame
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                try:
                    image = cv2.imread(output_path)
                    if image is not None:
                        logger.info("Successfully decoded with FFmpeg auto-detection")
                        # Clean up temp files
                        try:
                            os.unlink(input_path)
                            os.unlink(output_path)
                        except:
                            pass
                        return image
                except Exception as e:
                    logger.warning(f"Failed to load auto-detected image: {e}")
            
            # Clean up temp files
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass
                
        except Exception as e:
            logger.warning(f"FFmpeg auto-detection failed: {e}")
        
        logger.warning("All FFmpeg decoding attempts failed")
        return None

    def decode_image_from_kafka(self, message_value):
        """Decode image from Kafka message"""
        try:
            # Handle the case where message_value is already a dict
            if isinstance(message_value, dict):
                message_json = message_value
                logger.info("Message is already a dictionary")
            else:
                # Try to decode as JSON first
                try:
                    if isinstance(message_value, bytes):
                        message_json = json.loads(message_value.decode('utf-8'))
                    else:
                        message_json = json.loads(message_value)
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                    # If not JSON, try direct base64 decode
                    try:
                        if isinstance(message_value, str):
                            image_data = base64.b64decode(message_value)
                        else:
                            image_data = message_value
                        
                        # Convert bytes to OpenCV image
                        nparr = np.frombuffer(image_data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        return image
                    except Exception as inner_e:
                        logger.error(f"Failed to decode as base64 or raw bytes: {inner_e}")
                        return None
            
            # Now we have a dictionary, extract image data and metadata
            logger.info(f"Message keys: {list(message_json.keys())}")
            
            # Get resolution from message
            resolution = message_json.get('resolution', {})
            width = resolution.get('width', 1920)
            height = resolution.get('height', 1080)
            format_type = message_json.get('format', 'unknown')
            
            logger.info(f"Image format: {format_type}, resolution: {width}x{height}")
            
            # Check various possible keys for image data
            image_data = None
            possible_keys = ['frame', 'image', 'data', 'img', 'picture', 'photo']
            
            for key in possible_keys:
                if key in message_json:
                    logger.info(f"Found image data in key: {key}")
                    raw_data = message_json[key]
                    
                    # Handle different data types
                    if isinstance(raw_data, str):
                        # NEW: Try hex decoding first (like the working code)
                        try:
                            logger.info("Attempting hex decoding...")
                            image_data = bytes.fromhex(raw_data)
                            logger.info(f"Successfully decoded hex from key {key}")
                            break
                        except ValueError as hex_e:
                            logger.warning(f"Hex decode failed: {hex_e}")
                            
                            # Fall back to base64 decoding approaches
                            try:
                                # First try direct base64 decode
                                image_data = base64.b64decode(raw_data)
                                logger.info(f"Successfully decoded base64 from key {key}")
                                break
                            except Exception as e1:
                                logger.warning(f"Direct base64 decode failed: {e1}")
                                try:
                                    # Try adding padding
                                    missing_padding = len(raw_data) % 4
                                    if missing_padding:
                                        raw_data += '=' * (4 - missing_padding)
                                    image_data = base64.b64decode(raw_data)
                                    logger.info(f"Successfully decoded base64 with padding from key {key}")
                                    break
                                except Exception as e2:
                                    logger.warning(f"Base64 decode with padding failed: {e2}")
                                    try:
                                        # Try URL-safe base64
                                        image_data = base64.urlsafe_b64decode(raw_data)
                                        logger.info(f"Successfully decoded URL-safe base64 from key {key}")
                                        break
                                    except Exception as e3:
                                        logger.warning(f"URL-safe base64 decode failed: {e3}")
                                        try:
                                            # Try to decode as latin-1 and then base64
                                            if isinstance(raw_data, str):
                                                raw_bytes = raw_data.encode('latin-1')
                                                image_data = raw_bytes
                                                logger.info(f"Encoded string as latin-1 from key {key}")
                                                break
                                        except Exception as e4:
                                            logger.warning(f"Latin-1 encoding failed: {e4}")
                                            continue
                    elif isinstance(raw_data, (bytes, bytearray)):
                        # Raw bytes
                        image_data = raw_data
                        logger.info(f"Using raw bytes from key {key}")
                        break
                    elif isinstance(raw_data, list):
                        # Array of bytes
                        try:
                            image_data = bytes(raw_data)
                            logger.info(f"Converted list to bytes from key {key}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to convert list to bytes from key {key}: {e}")
                            continue
            
            if image_data is None:
                logger.error(f"Could not find image data in message. Available keys: {list(message_json.keys())}")
                return None
            
            # Try multiple decoding approaches
            image = None
            decoding_method = "unknown"
            
            # Method 1: Standard OpenCV decode (JPEG/PNG) - should work with hex-decoded frames
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    decoding_method = "cv2_standard"
                    logger.info(f"Successfully decoded using cv2.imdecode with shape: {image.shape}")
                    return self.save_decoded_image(image, message_json, decoding_method)
            except Exception as e:
                logger.warning(f"cv2.imdecode failed: {e}")
            
            # Method 2: Try with PIL (handles more formats)
            try:
                from PIL import Image as PILImage
                pil_image = PILImage.open(io.BytesIO(image_data))
                # Convert PIL image to OpenCV format
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                decoding_method = "pil_standard"
                logger.info(f"Successfully decoded using PIL with shape: {image.shape}")
                return self.save_decoded_image(image, message_json, decoding_method)
            except Exception as e:
                logger.warning(f"PIL decoding failed: {e}")
            
            # Method 3: Try FFmpeg-based decoding
            try:
                logger.info("Attempting FFmpeg decoding...")
                image = self.decode_with_ffmpeg(image_data, width, height)
                if image is not None:
                    decoding_method = "ffmpeg"
                    logger.info(f"Successfully decoded using FFmpeg with shape: {image.shape}")
                    return self.save_decoded_image(image, message_json, decoding_method)
            except Exception as e:
                logger.warning(f"FFmpeg decoding failed: {e}")
            
            # Method 4: Try as raw YUV420 data
            try:
                logger.info("Attempting YUV420 decoding...")
                expected_size = width * height * 3 // 2  # YUV420 format
                if len(image_data) >= expected_size:
                    # Reshape as YUV420
                    yuv_data = np.frombuffer(image_data[:expected_size], dtype=np.uint8)
                    yuv_image = yuv_data.reshape((height * 3 // 2, width))
                    
                    # Convert YUV420 to BGR
                    image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
                    decoding_method = "yuv420"
                    logger.info(f"Successfully decoded YUV420 with shape: {image.shape}")
                    return self.save_decoded_image(image, message_json, decoding_method)
            except Exception as e:
                logger.warning(f"YUV420 decoding failed: {e}")
            
            # Method 5: Try as raw RGB data
            try:
                logger.info("Attempting raw RGB decoding...")
                expected_size = width * height * 3  # RGB format
                if len(image_data) >= expected_size:
                    rgb_data = np.frombuffer(image_data[:expected_size], dtype=np.uint8)
                    image = rgb_data.reshape((height, width, 3))
                    # Convert RGB to BGR for OpenCV
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    decoding_method = "raw_rgb"
                    logger.info(f"Successfully decoded raw RGB with shape: {image.shape}")
                    return self.save_decoded_image(image, message_json, decoding_method)
            except Exception as e:
                logger.warning(f"Raw RGB decoding failed: {e}")
            
            # Method 6: Try as raw BGR data
            try:
                logger.info("Attempting raw BGR decoding...")
                expected_size = width * height * 3  # BGR format
                if len(image_data) >= expected_size:
                    bgr_data = np.frombuffer(image_data[:expected_size], dtype=np.uint8)
                    image = bgr_data.reshape((height, width, 3))
                    decoding_method = "raw_bgr"
                    logger.info(f"Successfully decoded raw BGR with shape: {image.shape}")
                    return self.save_decoded_image(image, message_json, decoding_method)
            except Exception as e:
                logger.warning(f"Raw BGR decoding failed: {e}")
            
            # Method 7: Try as grayscale and convert to BGR
            try:
                logger.info("Attempting grayscale decoding...")
                expected_size = width * height  # Grayscale format
                if len(image_data) >= expected_size:
                    gray_data = np.frombuffer(image_data[:expected_size], dtype=np.uint8)
                    gray_image = gray_data.reshape((height, width))
                    image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                    decoding_method = "grayscale"
                    logger.info(f"Successfully decoded grayscale with shape: {image.shape}")
                    return self.save_decoded_image(image, message_json, decoding_method)
            except Exception as e:
                logger.warning(f"Grayscale decoding failed: {e}")
            
            # Method 8: Try interpreting the data as a compressed format and create a dummy image
            try:
                logger.info("Creating visualization of the raw data...")
                # Create a visualization by treating the data as pixel values
                data_length = len(image_data)
                
                # Calculate dimensions for visualization (aim for roughly square)
                viz_width = int(np.sqrt(data_length))
                viz_height = data_length // viz_width
                
                if viz_width * viz_height > 0:
                    # Take only the data we can visualize
                    viz_data = image_data[:viz_width * viz_height]
                    viz_image = np.frombuffer(viz_data, dtype=np.uint8).reshape((viz_height, viz_width))
                    
                    # Convert to BGR and resize to a reasonable size
                    viz_image_bgr = cv2.cvtColor(viz_image, cv2.COLOR_GRAY2BGR)
                    viz_image_resized = cv2.resize(viz_image_bgr, (800, 600))
                    
                    # Add text overlay showing this is raw data visualization
                    cv2.putText(viz_image_resized, f"RAW DATA VISUALIZATION", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(viz_image_resized, f"Data size: {data_length} bytes", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(viz_image_resized, f"Original dims: {viz_width}x{viz_height}", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    decoding_method = "raw_visualization"
                    logger.info(f"Created raw data visualization with shape: {viz_image_resized.shape}")
                    return self.save_decoded_image(viz_image_resized, message_json, decoding_method)
                    
            except Exception as e:
                logger.warning(f"Raw data visualization failed: {e}")
            
            # If all methods fail, log debug info and save raw data
            logger.error("All decoding methods failed")
            logger.info(f"Image data length: {len(image_data)} bytes")
            logger.info(f"Expected sizes - YUV420: {width * height * 3 // 2}, RGB/BGR: {width * height * 3}, Gray: {width * height}")
            logger.info(f"First 20 bytes: {image_data[:20]}")
            logger.info(f"Last 20 bytes: {image_data[-20:]}")
            
            # Save raw data for debugging with better file names
            self.save_debug_data(image_data, message_json)
            
            return None
            
        except Exception as e:
            logger.error(f"Error decoding image from Kafka: {e}")
            logger.error(f"Message type: {type(message_value)}")
            return None

    def save_decoded_image(self, image, message_json, decoding_method):
        """Save successfully decoded image in viewable format"""
        try:
            self.frame_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_seq = message_json.get('frame_seq', 0)
            
            # Save as PNG (lossless) and JPEG (smaller file)
            base_filename = f"frame_{self.frame_counter:06d}_seq{frame_seq}_{decoding_method}_{timestamp}"
            
            png_path = os.path.join(self.output_dir, f"{base_filename}.png")
            jpg_path = os.path.join(self.output_dir, f"{base_filename}.jpg")
            
            # Save both formats
            cv2.imwrite(png_path, image)
            cv2.imwrite(jpg_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            logger.info(f"Successfully saved decoded image:")
            logger.info(f"  PNG: {png_path}")
            logger.info(f"  JPEG: {jpg_path}")
            logger.info(f"  Method: {decoding_method}")
            logger.info(f"  Shape: {image.shape}")
            
            # Also save metadata
            metadata_path = os.path.join(self.output_dir, f"{base_filename}_metadata.json")
            metadata = {
                'frame_number': self.frame_counter,
                'frame_seq': frame_seq,
                'timestamp': timestamp,
                'decoding_method': decoding_method,
                'image_shape': list(image.shape),
                'resolution': message_json.get('resolution', {}),
                'format': message_json.get('format', 'unknown'),
                'message_timestamp': message_json.get('timestamp', 0),
                'files': {
                    'png': os.path.basename(png_path),
                    'jpg': os.path.basename(jpg_path)
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to save decoded image: {e}")
            return image

    def save_debug_data(self, image_data, message_json):
        """Save raw data and metadata for debugging"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_seq = message_json.get('frame_seq', 0)
            
            # Save raw data
            debug_file = os.path.join(self.output_dir, f"debug_frame_seq{frame_seq}_{timestamp}.raw")
            with open(debug_file, 'wb') as f:
                f.write(image_data)
            
            # Save metadata
            metadata_file = debug_file.replace('.raw', '_metadata.json')
            metadata = {
                'data_length': len(image_data),
                'frame_seq': frame_seq,
                'timestamp': timestamp,
                'resolution': message_json.get('resolution', {}),
                'format': message_json.get('format', 'unknown'),
                'message_timestamp': message_json.get('timestamp', 0),
                'first_20_bytes': list(image_data[:20]),
                'last_20_bytes': list(image_data[-20:]),
                'message_keys': list(message_json.keys()),
                'raw_file': os.path.basename(debug_file)
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved debug data: {debug_file}")
            logger.info(f"Saved debug metadata: {metadata_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug data: {e}")

    def calculate_overlap(self, bbox, green_box):
        """
        Calculate the overlap area between a detection bounding box and the green box.
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

    def process_image(self, image):
        """
        Process a single image and return results with bounding boxes
        """
        if image is None:
            logger.error("Received None image")
            return None
            
        original_height, original_width = image.shape[:2]
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Step 1: Run segmentation
        transform = self.aug.get_transform(image)
        transformed_image = transform.apply_image(image)
        transformed_image = torch.as_tensor(transformed_image.astype("float32").transpose(2, 0, 1))
        
        batch_input = {
            "image": transformed_image.to(self.device),
            "height": original_height,
            "width": original_width,
        }
        
        with torch.no_grad():
            outputs = self.seg_model([batch_input])
        
        instances = outputs[0]["instances"]
        person_indices = (instances.pred_classes == 0).nonzero().flatten()
        
        if len(person_indices) == 0:
            logger.info("No persons detected in image")
            return {
                "detections": [], 
                "features": [], 
                "vis_image": vis_image,
                "image_dimensions": [int(original_width), int(original_height)],
                "store_area": [0, int(original_height * 0.10), int(original_width), int(original_height)],
                "total_detections": 0,
                "valid_detections": 0
            }
        
        # Get person boxes, scores, and masks - USE ORIGINAL BOXES ONLY
        person_boxes = instances.pred_boxes.tensor[person_indices].cpu().numpy()
        person_scores = instances.scores[person_indices].cpu().numpy()
        person_masks = instances.pred_masks[person_indices].cpu().numpy()
        
        # NO SCALING - Use the original bounding boxes directly
        logger.info(f"Using original bounding boxes without any scaling/transformation")
        
        # Filter duplicate detections using original boxes
        filtered_boxes, filtered_scores, filtered_masks = filter_duplicate_detections(
            person_boxes, person_scores, person_masks, image, iou_threshold=0.9
        )
        
        logger.info(f"After duplicate filtering: {len(filtered_boxes)} detections remain")
        
        # Define green box (store entrance area)
        height, width = image.shape[:2]
        shrink_percentage_top = 0.10
        line_y = int(height * shrink_percentage_top)
        green_box = [0, line_y, width, height]
        
        # Draw green box on visualization
        cv2.rectangle(vis_image, (0, line_y), (width, height), (0, 255, 0), 3)
        cv2.putText(vis_image, "Store Area", (10, line_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw only the original bounding boxes (the ones that work correctly)
        logger.info(f"=== DRAWING ORIGINAL BOXES (CORRECTLY POSITIONED) ===")
        
        for i, bbox in enumerate(filtered_boxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # RED for original
            cv2.putText(vis_image, f"DETECT {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            logger.info(f"Drew detection {i+1} in RED: [{x1}, {y1}, {x2}, {y2}]")
        
        # Process valid detections
        valid_detections = []
        detection_data = []
        
        logger.info(f"=== PROCESSING DETECTIONS FOR OVERLAP ===")
        
        for j, bbox in enumerate(filtered_boxes):
            score = filtered_scores[j]
            mask = filtered_masks[j]
            
            # Calculate overlap with store area
            overlap_area, bbox_area = self.calculate_overlap(bbox, green_box)
            overlap_ratio = overlap_area / bbox_area if bbox_area > 0 else 0
            
            logger.info(f"Detection {j+1}: bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}], overlap_ratio={overlap_ratio:.3f}")
            
            # Only keep detections that significantly overlap with store area
            if overlap_ratio > 0.7:
                logger.info(f"Detection {j+1} ACCEPTED (overlap > 0.7)")
                
                # Convert bbox format
                x1, y1, x2, y2 = map(int, bbox)
                bbox_center_x = int((x1 + x2) / 2)
                bbox_center_y = int((y1 + y2) / 2)
                bbox_y_80 = int(y1 + 0.8 * (y2 - y1))  # 80% down from top of bbox
                
                detection_info = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(score),
                    "center": [int(bbox_center_x), int(bbox_center_y)],
                    "bottom_80": [int(bbox_center_x), int(bbox_y_80)],
                    "overlap_ratio": float(overlap_ratio)
                }
                
                detection_data.append(detection_info)
                
                # Draw final accepted detection in YELLOW
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # YELLOW
                
                # Draw center point
                cv2.circle(vis_image, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)
                
                # Draw 80% point
                cv2.circle(vis_image, (bbox_center_x, bbox_y_80), 5, (255, 255, 0), -1)
                
                # Add text with detection info
                accepted_detection_num = len(detection_data)
                label = f"FINAL {accepted_detection_num}"
                score_text = f"Score: {score:.2f}"
                coords_text = f"({bbox_center_x}, {bbox_center_y})"
                
                # Draw text background
                text_lines = [label, score_text, coords_text]
                text_y = y1 - 60
                for line in reversed(text_lines):
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(vis_image, (x1, text_y - text_size[1] - 5), 
                                (x1 + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
                    cv2.putText(vis_image, line, (x1 + 2, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    text_y -= (text_size[1] + 8)
                
                # Skip feature extraction for now - just focus on coordinates
                detection_info["has_features"] = False
                valid_detections.append(detection_info)
            else:
                logger.info(f"Detection {j+1} REJECTED (overlap {overlap_ratio:.3f} <= 0.7)")
        
        return {
            "detections": valid_detections,
            "features": [],
            "vis_image": vis_image,
            "image_dimensions": [int(original_width), int(original_height)],
            "store_area": [int(x) for x in green_box],
            "total_detections": int(len(filtered_boxes)),
            "valid_detections": int(len(valid_detections))
        }

    def save_processed_frame(self, result, timestamp, camera_id=None):
        """Save the processed frame and detection data"""
        if result is None:
            return
            
        self.frame_counter += 1
        
        # Save visualization image
        output_image_path = os.path.join(self.output_dir, f"frame_{self.frame_counter:06d}_{timestamp}.jpg")
        cv2.imwrite(output_image_path, result['vis_image'])
        
        # Save detection data as JSON
        output_json_path = os.path.join(self.output_dir, f"frame_{self.frame_counter:06d}_{timestamp}.json")
        result_data = {
            "camera_id": camera_id,
            "frame_number": self.frame_counter,
            "timestamp": timestamp,
            "image_dimensions": result.get('image_dimensions', [800, 600]),  # Default dimensions if not available
            "store_area": result.get('store_area', [0, 0, 800, 600]),  # Default store area
            "total_detections": result.get('total_detections', 0),
            "valid_detections": result.get('valid_detections', 0),
            "detection_data": result.get('detections', []),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Processed frame {self.frame_counter} saved to: {output_image_path}")
        logger.info(f"Detection data saved to: {output_json_path}")
        logger.info(f"Found {len(result.get('detections', []))} valid person detections")

    def run(self):
        """Main processing loop for Kafka messages"""
        logger.info("Starting Kafka image processing...")
        logger.info(f"Listening to topic: {self.kafka_topic}")
        logger.info(f"Bootstrap servers: {self.kafka_bootstrap_servers}")
        logger.info(f"Output directory: {self.output_dir}")
        
        try:
            while True:
                try:
                    # Poll for messages
                    message_batch = self.consumer.poll(timeout_ms=1000)
                    
                    if not message_batch:
                        logger.debug("No messages received, continuing...")
                        continue
                    
                    for topic_partition, messages in message_batch.items():
                        for message in messages:
                            logger.info(f"Received message from {topic_partition.topic}:{topic_partition.partition} offset {message.offset}")
                            
                            # Decode image from Kafka message
                            camera_id = message.key.decode('utf-8')
                            logger.info(f"camera id is {camera_id}")
                            image = self.decode_image_from_kafka(message.value)
                            
                            if image is not None:
                                # Process the image
                                start_time = time.time()
                                result = self.process_image(image)
                                processing_time = time.time() - start_time
                                
                                # Generate timestamp
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                
                                # Save results
                                self.save_processed_frame(result, timestamp, camera_id)
                                
                                logger.info(f"Frame processed in {processing_time:.2f}s")
                            else:
                                logger.warning("Failed to decode image from Kafka message - saved raw data for debugging")
                    
                except KafkaError as e:
                    logger.error(f"Kafka error: {e}")
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        finally:
            if self.consumer:
                self.consumer.close()
                logger.info("Kafka consumer closed")

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'consumer') and self.consumer:
            self.consumer.close()


class SimpleImageProcessor:
    """Simplified processor for single image analysis (original functionality)"""
    
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize the segmentation model (Detectron2)
        self.seg_predictor = setup_predictor()
        self.seg_model = self.seg_predictor.model
        self.seg_model.eval()
        self.aug = self.seg_predictor.aug
        logger.info("Segmentation model initialized")
        
        # Initialize TransReID model
        self.model = make_model(cfg, num_class=1041, camera_num=0, view_num=0).to(self.device)
        self.model.load_param(cfg.TEST.WEIGHT)
        self.model.eval()
        self.transform = build_transforms(cfg, is_train=False)
        self.extract_features = extract_features
        logger.info("TransReID model initialized")

    def calculate_overlap(self, bbox, green_box):
        """
        Calculate the overlap area between a detection bounding box and the green box.
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

    def process_image(self, image_path, output_dir):
        """
        Process a single image and save results with bounding boxes
        """
        logger.info(f"Processing image: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
            
        original_height, original_width = image.shape[:2]
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Step 1: Run segmentation
        transform = self.aug.get_transform(image)
        transformed_image = transform.apply_image(image)
        transformed_image = torch.as_tensor(transformed_image.astype("float32").transpose(2, 0, 1))
        
        batch_input = {
            "image": transformed_image.to(self.device),
            "height": original_height,
            "width": original_width,
        }
        
        with torch.no_grad():
            outputs = self.seg_model([batch_input])
        
        instances = outputs[0]["instances"]
        person_indices = (instances.pred_classes == 0).nonzero().flatten()
        
        if len(person_indices) == 0:
            logger.info("No persons detected in image")
            return {"detections": [], "features": []}
        
        # Get person boxes, scores, and masks - USE ORIGINAL BOXES ONLY
        person_boxes = instances.pred_boxes.tensor[person_indices].cpu().numpy()
        person_scores = instances.scores[person_indices].cpu().numpy()
        person_masks = instances.pred_masks[person_indices].cpu().numpy()
        
        # NO SCALING - Use the original bounding boxes directly
        logger.info(f"Using original bounding boxes without any scaling/transformation")
        
        # Filter duplicate detections using original boxes
        filtered_boxes, filtered_scores, filtered_masks = filter_duplicate_detections(
            person_boxes, person_scores, person_masks, image, iou_threshold=0.9
        )
        
        logger.info(f"After duplicate filtering: {len(filtered_boxes)} detections remain")
        
        # Define green box (store entrance area)
        height, width = image.shape[:2]
        shrink_percentage_top = 0.10
        line_y = int(height * shrink_percentage_top)
        green_box = [0, line_y, width, height]
        
        # Draw green box on visualization
        cv2.rectangle(vis_image, (0, line_y), (width, height), (0, 255, 0), 3)
        cv2.putText(vis_image, "Store Area", (10, line_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw only the original bounding boxes (the ones that work correctly)
        logger.info(f"=== DRAWING ORIGINAL BOXES (CORRECTLY POSITIONED) ===")
        
        for i, bbox in enumerate(filtered_boxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # RED for original
            cv2.putText(vis_image, f"DETECT {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            logger.info(f"Drew detection {i+1} in RED: [{x1}, {y1}, {x2}, {y2}]")
        
        # Process valid detections
        valid_detections = []
        detection_data = []
        
        logger.info(f"=== PROCESSING DETECTIONS FOR OVERLAP ===")
        
        for j, bbox in enumerate(filtered_boxes):
            score = filtered_scores[j]
            mask = filtered_masks[j]
            
            # Calculate overlap with store area
            overlap_area, bbox_area = self.calculate_overlap(bbox, green_box)
            overlap_ratio = overlap_area / bbox_area if bbox_area > 0 else 0
            
            logger.info(f"Detection {j+1}: bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}], overlap_ratio={overlap_ratio:.3f}")
            
            # Only keep detections that significantly overlap with store area
            if overlap_ratio > 0.7:
                logger.info(f"Detection {j+1} ACCEPTED (overlap > 0.7)")
                
                # Convert bbox format
                x1, y1, x2, y2 = map(int, bbox)
                bbox_center_x = int((x1 + x2) / 2)
                bbox_center_y = int((y1 + y2) / 2)
                bbox_y_80 = int(y1 + 0.8 * (y2 - y1))  # 80% down from top of bbox
                
                detection_info = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(score),
                    "center": [int(bbox_center_x), int(bbox_center_y)],
                    "bottom_80": [int(bbox_center_x), int(bbox_y_80)],
                    "overlap_ratio": float(overlap_ratio)
                }
                
                detection_data.append(detection_info)
                
                # Draw final accepted detection in YELLOW
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # YELLOW
                
                # Draw center point
                cv2.circle(vis_image, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)
                
                # Draw 80% point
                cv2.circle(vis_image, (bbox_center_x, bbox_y_80), 5, (255, 255, 0), -1)
                
                # Add text with detection info
                accepted_detection_num = len(detection_data)
                label = f"FINAL {accepted_detection_num}"
                score_text = f"Score: {score:.2f}"
                coords_text = f"({bbox_center_x}, {bbox_center_y})"
                
                # Draw text background
                text_lines = [label, score_text, coords_text]
                text_y = y1 - 60
                for line in reversed(text_lines):
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(vis_image, (x1, text_y - text_size[1] - 5), 
                                (x1 + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
                    cv2.putText(vis_image, line, (x1 + 2, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    text_y -= (text_size[1] + 8)
                
                # Skip feature extraction for now - just focus on coordinates
                detection_info["has_features"] = False
                valid_detections.append(detection_info)
            else:
                logger.info(f"Detection {j+1} REJECTED (overlap {overlap_ratio:.3f} <= 0.7)")
        
        # Save visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_image_path = os.path.join(output_dir, f"{base_name}_processed.jpg")
        cv2.imwrite(output_image_path, vis_image)
        
        # Save detection data as JSON
        output_json_path = os.path.join(output_dir, f"{base_name}_detections.json")
        result_data = {
            "image_path": image_path,
            "image_dimensions": [int(original_width), int(original_height)],
            "store_area": [int(x) for x in green_box],
            "total_detections": int(len(filtered_boxes)),
            "valid_detections": int(len(valid_detections)),
            "detection_data": valid_detections,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Processed image saved to: {output_image_path}")
        logger.info(f"Detection data saved to: {output_json_path}")
        logger.info(f"Found {len(valid_detections)} valid person detections")
        
        return result_data


def process_folder(input_folder, output_folder, device=None):
    """
    Process all images in a folder
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize processor
    processor = SimpleImageProcessor(device=device)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = []
    for file in os.listdir(input_folder):
        if os.path.splitext(file.lower())[1] in image_extensions:
            image_files.append(os.path.join(input_folder, file))
    
    if not image_files:
        logger.error(f"No image files found in {input_folder}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    total_detections = 0
    successful_images = 0
    
    for i, image_path in enumerate(image_files):
        try:
            logger.info(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            start_time = time.time()
            
            result = processor.process_image(image_path, output_folder)
            
            if result:
                total_detections += result['valid_detections']
                successful_images += 1
                
            processing_time = time.time() - start_time
            logger.info(f"Image processed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}", exc_info=True)
    
    # Summary
    logger.info(f"\nProcessing Summary:")
    logger.info(f"Total images processed: {successful_images}/{len(image_files)}")
    logger.info(f"Total person detections: {total_detections}")
    logger.info(f"Average detections per image: {total_detections/max(1,successful_images):.2f}")
    logger.info(f"Results saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Kafka Image Processor for Human Detection')
    parser.add_argument('--mode', choices=['kafka', 'folder', 'single'], default='kafka',
                       help='Processing mode: kafka (stream from Kafka), folder (process folder), or single (process single image)')
    parser.add_argument('--kafka-servers', default='35.181.243.135:29092',
                       help='Kafka bootstrap servers (default: 35.181.243.135:29092)')
    parser.add_argument('--kafka-topic', default='store-109',
                       help='Kafka topic to consume from (default: store-109)')
    parser.add_argument('--input', '-i', help='Input folder containing images (for folder mode)')
    parser.add_argument('--output', '-o', required=True, help='Output folder for processed images and data')
    parser.add_argument('--device', default=None, help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--single-image', help='Process a single image instead of a folder (for single mode)')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'folder' and not args.input:
        parser.error("--input is required for folder mode")
    
    if args.mode == 'single' and not args.single_image:
        parser.error("--single-image is required for single mode")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Mode: {args.mode}")
    
    try:
        if args.mode == 'kafka':
            # Kafka streaming mode
            logger.info("Starting Kafka streaming mode...")
            processor = KafkaImageProcessor(
                kafka_bootstrap_servers=args.kafka_servers,
                kafka_topic=args.kafka_topic,
                device=device,
                output_dir=args.output
            )
            processor.run()
            
        elif args.mode == 'single':
            # Process single image
            if not os.path.exists(args.single_image):
                logger.error(f"Image file not found: {args.single_image}")
                return
            
            os.makedirs(args.output, exist_ok=True)
            processor = SimpleImageProcessor(device=device)
            result = processor.process_image(args.single_image, args.output)
            
            if result:
                logger.info(f"Successfully processed single image with {result['valid_detections']} detections")
                
        elif args.mode == 'folder':
            # Process folder
            if not os.path.exists(args.input):
                logger.error(f"Input folder not found: {args.input}")
                return
            
            process_folder(args.input, args.output, device=device)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)


if __name__ == "__main__":
    main()