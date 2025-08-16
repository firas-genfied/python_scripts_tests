#!/usr/bin/env python3
"""
Dummy Kafka Producer for Testing RTSP Stream Processor

This producer simulates camera feeds by streaming test images to Kafka topics
in the format expected by the rtsp_stream_processor_multiple_cameras.py
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from kafka import KafkaProducer
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("dummy_kafka_producer")

class DummyKafkaProducer:
    """
    Produces dummy camera frames to Kafka topics for testing the RTSP stream processor
    """
    
    def __init__(self, 
                 bootstrap_servers: str = "localhost:9092",
                 test_images_dir: str = "test_images",
                 store_id: int = 5,
                 camera_id: int = 1,
                 fps: float = 5.0,
                 loop_images: bool = True):
        """
        Initialize the dummy Kafka producer
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            test_images_dir: Directory containing test images
            store_id: Store ID for the camera feed
            camera_id: Camera ID for the feed
            fps: Frames per second to stream
            loop_images: Whether to loop through images continuously
        """
        self.bootstrap_servers = bootstrap_servers
        self.test_images_dir = Path(test_images_dir)
        self.store_id = store_id
        self.camera_id = camera_id
        self.fps = fps
        self.loop_images = loop_images
        self.frame_interval = 1.0 / fps
        
        # Kafka configuration
        self.topic_name = f"store-{store_id}"  # Format: store-5 (matches ^store-([0-9]+)$ pattern)
        self.camera_key = f"camera-{camera_id}"  # Format: camera-1
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            key_serializer=lambda k: k.encode('utf-8'),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            batch_size=16384,
            linger_ms=10,
            buffer_memory=33554432,
            max_request_size=10485760  # 10MB for large images
        )
        
        self.running = False
        self.frame_count = 0
        self.total_frames_sent = 0
        self.start_time = None
        
        logger.info(f"Initialized DummyKafkaProducer:")
        logger.info(f"  Topic: {self.topic_name}")
        logger.info(f"  Camera Key: {self.camera_key}")
        logger.info(f"  Store ID: {store_id}")
        logger.info(f"  Camera ID: {camera_id}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Test Images Dir: {self.test_images_dir}")
    
    def load_test_images(self) -> List[np.ndarray]:
        """
        Load test images from the specified directory
        
        Returns:
            List of loaded images as numpy arrays
        """
        if not self.test_images_dir.exists():
            logger.error(f"Test images directory does not exist: {self.test_images_dir}")
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in self.test_images_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.error(f"No image files found in {self.test_images_dir}")
            return []
        
        # Sort files for consistent ordering
        image_files.sort()
        
        images = []
        for image_file in image_files:
            try:
                img = cv2.imread(str(image_file))
                if img is not None:
                    images.append(img)
                    logger.info(f"Loaded image: {image_file.name} ({img.shape})")
                else:
                    logger.warning(f"Failed to load image: {image_file}")
            except Exception as e:
                logger.error(f"Error loading image {image_file}: {e}")
        
        logger.info(f"Successfully loaded {len(images)} test images")
        return images
    
    def create_dummy_images(self, count: int = 10) -> List[np.ndarray]:
        """
        Create dummy images if no test images are available
        
        Args:
            count: Number of dummy images to create
            
        Returns:
            List of generated dummy images
        """
        logger.info(f"Creating {count} dummy images")
        images = []
        
        for i in range(count):
            # Create a colorful dummy image with text
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some structure to make it more realistic
            cv2.rectangle(img, (50, 50), (590, 430), (255, 255, 255), 2)
            cv2.rectangle(img, (100, 100), (540, 380), (0, 255, 0), 2)
            
            # Add text
            cv2.putText(img, f"Dummy Frame {i+1}", (150, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Store {self.store_id} - Camera {self.camera_id}", 
                       (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            images.append(img)
        
        return images
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """
        Encode frame as hex string (matching the expected format)
        
        Args:
            frame: OpenCV image as numpy array
            
        Returns:
            Hex-encoded string of the JPEG-compressed frame
        """
        try:
            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.error("Failed to encode frame as JPEG")
                return ""
            
            # Convert to hex string
            hex_string = buffer.tobytes().hex()
            return hex_string
            
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return ""
    
    def create_frame_message(self, frame: np.ndarray, frame_id: str) -> dict:
        """
        Create a Kafka message for a frame
        
        Args:
            frame: OpenCV image as numpy array
            frame_id: Unique frame identifier
            
        Returns:
            Dictionary containing the frame message
        """
        encoded_frame = self.encode_frame(frame)
        if not encoded_frame:
            return {}
        
        message = {
            "frame_id": frame_id,
            "frame": encoded_frame,
            "timestamp": datetime.utcnow().isoformat(),
            "camera_id": self.camera_id,
            "store_id": self.store_id,
            "frame_count": self.frame_count,
            "metadata": {
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2],
                "encoding": "jpeg"
            }
        }
        
        return message
    
    def create_message_headers(self) -> List[tuple]:
        """
        Create Kafka message headers
        
        Returns:
            List of header tuples
        """
        headers = [
            ("timestamp", datetime.utcnow().isoformat().encode('utf-8')),
            ("store_id", str(self.store_id).encode('utf-8')),
            ("camera_id", str(self.camera_id).encode('utf-8')),
            ("producer", "dummy_kafka_producer".encode('utf-8'))
        ]
        return headers
    
    async def send_frame(self, frame: np.ndarray) -> bool:
        """
        Send a single frame to Kafka
        
        Args:
            frame: OpenCV image as numpy array
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate unique frame ID
            frame_id = f"{self.store_id}-{self.camera_id}-{self.frame_count}-{uuid.uuid4().hex[:8]}"
            
            # Create message
            message = self.create_frame_message(frame, frame_id)
            if not message:
                logger.error("Failed to create frame message")
                return False
            
            # Create headers
            headers = self.create_message_headers()
            
            # Send to Kafka
            future = self.producer.send(
                topic=self.topic_name,
                key=self.camera_key,
                value=message,
                headers=headers
            )
            
            # Wait for the message to be sent (with timeout)
            record_metadata = future.get(timeout=1)
            
            self.frame_count += 1
            self.total_frames_sent += 1
            
            if self.total_frames_sent % 50 == 0:  # Log every 50 frames
                elapsed = time.time() - self.start_time if self.start_time else 0
                actual_fps = self.total_frames_sent / elapsed if elapsed > 0 else 0
                logger.info(f"Sent frame {self.total_frames_sent} to {record_metadata.topic}:"
                           f"{record_metadata.partition}:{record_metadata.offset} "
                           f"(actual FPS: {actual_fps:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending frame: {e}")
            return False
    
    async def stream_images(self, duration_seconds: Optional[int] = None):
        """
        Stream test images to Kafka
        
        Args:
            duration_seconds: How long to stream (None for infinite)
        """
        logger.info("Starting image streaming...")
        
        # Load test images
        images = self.load_test_images()
        if not images:
            logger.warning("No test images found, creating dummy images")
            images = self.create_dummy_images(10)
        
        if not images:
            logger.error("No images available for streaming")
            return
        
        self.running = True
        self.start_time = time.time()
        image_index = 0
        last_frame_time = time.time()
        
        logger.info(f"Starting to stream {len(images)} images at {self.fps} FPS")
        if duration_seconds:
            logger.info(f"Will stream for {duration_seconds} seconds")
        else:
            logger.info("Will stream indefinitely (Ctrl+C to stop)")
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check if we should stop based on duration
                if duration_seconds and (current_time - self.start_time) >= duration_seconds:
                    logger.info(f"Streaming duration of {duration_seconds} seconds completed")
                    break
                
                # Check if it's time to send the next frame
                if current_time - last_frame_time >= self.frame_interval:
                    # Get current image
                    current_image = images[image_index]
                    
                    # Send frame
                    success = await self.send_frame(current_image)
                    if not success:
                        logger.warning("Failed to send frame, continuing...")
                    
                    # Update timing
                    last_frame_time = current_time
                    
                    # Move to next image
                    image_index += 1
                    if image_index >= len(images):
                        if self.loop_images:
                            image_index = 0  # Loop back to first image
                            logger.debug("Looped back to first image")
                        else:
                            logger.info("Reached end of images, stopping")
                            break
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
        finally:
            self.running = False
            
            # Final statistics
            elapsed = time.time() - self.start_time
            actual_fps = self.total_frames_sent / elapsed if elapsed > 0 else 0
            
            logger.info("Streaming completed:")
            logger.info(f"  Total frames sent: {self.total_frames_sent}")
            logger.info(f"  Duration: {elapsed:.2f} seconds")
            logger.info(f"  Target FPS: {self.fps}")
            logger.info(f"  Actual FPS: {actual_fps:.2f}")
            logger.info(f"  Topic: {self.topic_name}")
    
    def stop(self):
        """Stop the streaming"""
        logger.info("Stopping streaming...")
        self.running = False
    
    def close(self):
        """Close the Kafka producer"""
        try:
            self.producer.flush(timeout=5)
            self.producer.close(timeout=5)
            logger.info("Kafka producer closed")
        except Exception as e:
            logger.error(f"Error closing Kafka producer: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Dummy Kafka Producer for testing RTSP Stream Processor')
    
    # Kafka settings
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    
    # Camera settings  
    parser.add_argument('--store-id', type=int, default=5,
                       help='Store ID for the camera feed')
    parser.add_argument('--camera-id', type=int, default=1,
                       help='Camera ID for the feed')
    
    # Streaming settings
    parser.add_argument('--fps', type=float, default=5.0,
                       help='Frames per second to stream')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration to stream in seconds (infinite if not specified)')
    parser.add_argument('--test-images-dir', default='test_images',
                       help='Directory containing test images')
    parser.add_argument('--no-loop', action='store_true',
                       help='Do not loop through images (stop after one pass)')
    
    args = parser.parse_args()
    
    # Validate test images directory
    test_images_path = Path(args.test_images_dir)
    if not test_images_path.exists():
        logger.warning(f"Test images directory does not exist: {test_images_path}")
        logger.info("Creating directory and will use dummy images")
        test_images_path.mkdir(parents=True, exist_ok=True)
    
    # Create producer
    producer = DummyKafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        test_images_dir=args.test_images_dir,
        store_id=args.store_id,
        camera_id=args.camera_id,
        fps=args.fps,
        loop_images=not args.no_loop
    )
    
    try:
        logger.info("=" * 60)
        logger.info("DUMMY KAFKA PRODUCER STARTED")
        logger.info("=" * 60)
        logger.info(f"Kafka Servers: {args.bootstrap_servers}")
        logger.info(f"Topic: store-{args.store_id:03d}-frames")
        logger.info(f"Camera Key: camera-{args.camera_id}")
        logger.info(f"FPS: {args.fps}")
        logger.info(f"Duration: {'Infinite' if args.duration is None else f'{args.duration}s'}")
        logger.info(f"Loop Images: {not args.no_loop}")
        logger.info("=" * 60)
        
        # Start streaming
        await producer.stream_images(duration_seconds=args.duration)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        producer.stop()
        producer.close()
        logger.info("Dummy Kafka Producer stopped")

if __name__ == "__main__":
    asyncio.run(main())