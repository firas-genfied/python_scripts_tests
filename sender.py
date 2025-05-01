# sender.py

import asyncio
import logging
import sys
from typing import Dict, Any, List
import os
import json
import aiohttp
from detection_postprocessing import process_video_entries
from config import config

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),        # Log to stdout
        logging.FileHandler("sender.log")         # Log to a file named sender.log
    ]
)

logger = logging.getLogger(__name__)

# Configuration Variables
DETECTION_ENDPOINT = config.get("detection_endpoint")

if not DETECTION_ENDPOINT:
    logger.error("Detection endpoint not found in configuration.")
    sys.exit(1)

# Number of retries for failed requests
MAX_RETRIES = 3
# Delay between retries (in seconds)
RETRY_DELAY = 1


def format_detection_data(detection_data: List[Dict]) -> List[Dict]:
    """
    Process detection data directly without reading from a file.
    Args:
        detection_data (List[Dict]): List containing detection results.
    Returns:
        List[Dict]: Formatted data ready to be sent to the endpoint as a list.
    """
    if not detection_data or not isinstance(detection_data, list):
        raise ValueError("Invalid detection data format")
    
    formatted_results = []
    
    for frame_data in detection_data:
        # Format according to API requirements
        result = {
            "camera_id": frame_data.get("camera_id", ""),
            "image_url": frame_data.get("image_url", ""),
            "is_organised": frame_data.get("is_organised", True),
            "no_of_people": frame_data.get("no_of_people", 0),
            "date_time": frame_data.get("date_time", ""),
            "persons": frame_data.get("persons", [])
        }
        formatted_results.append(result)
    
    return formatted_results

async def send_detection_data(detection_data: List[Dict]) -> bool:
    """
    Processes the detection data and sends it directly to the API endpoint.
    
    Args:
        detection_data (List[Dict]): List containing detection results.
        
    Returns:
        bool: True if the data was sent successfully, False otherwise.
    """
    try:
        # Process the detection data directly
        formatted_data = format_detection_data(detection_data)
        logger.info(f"Processed detection data for camera {detection_data[0].get('camera_id')}, frame {detection_data[0].get('frame_id')}, time {detection_data[0].get('date_time')}")
    except Exception as e:
        logger.error(f"Failed to process detection data: {e}", exc_info=True)
        return False
    
    # Implement retry logic
    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(DETECTION_ENDPOINT, json=formatted_data, timeout=10) as response:
                    if response.status == 200:
                        logger.info(f"Data sent successfully for frame {detection_data[0].get('frame_id')} time {detection_data[0].get('date_time')}")
                        response_text = await response.text()
                        logger.debug(f"Response Text: {response_text}")
                        return True
                    else:
                        logger.error(f"Failed to send data. Status: {response.status}")
                        response_text = await response.text()
                        logger.error(f"Response Text: {response_text}")
                        
                        # If this was the last attempt, return False
                        if attempt == MAX_RETRIES - 1:
                            return False
                            
                        # Otherwise wait before retrying
                        logger.info(f"Retrying in {RETRY_DELAY} seconds (attempt {attempt+1}/{MAX_RETRIES})...")
                        await asyncio.sleep(RETRY_DELAY)
        except aiohttp.ClientError as client_error:
            logger.error(f"HTTP Client error occurred: {client_error}", exc_info=True)
            if attempt == MAX_RETRIES - 1:
                return False
            logger.info(f"Retrying in {RETRY_DELAY} seconds (attempt {attempt+1}/{MAX_RETRIES})...")
            await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending data: {e}", exc_info=True)
            if attempt == MAX_RETRIES - 1:
                return False
            logger.info(f"Retrying in {RETRY_DELAY} seconds (attempt {attempt+1}/{MAX_RETRIES})...")
            await asyncio.sleep(RETRY_DELAY)
    
    # If all retries failed
    return False

async def send_detection_data_batches(detections_jsonfile_path: str):
    """
    Processes the raw JSON file and sends the formatted data to the detection API endpoint.
    
    Args:
        detections_jsonfile_path (str): The path to the raw JSON file containing detections.
    
    Raises:
        SystemExit: If sending data fails after retries.
    """
    try:
        # Process the raw JSON file to get formatted data
        formatted_data: Dict[str, Any] = process_video_entries(detections_jsonfile_path)
        logger.info(f"Processed data from '{detections_jsonfile_path}' successfully.")
        
    except Exception as e:
        logger.error(f"Failed to process video entries: {e}", exc_info=True)
        sys.exit(1)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(DETECTION_ENDPOINT, json=formatted_data) as response:
                if response.status == 200:
                    logger.info("Data sent successfully.")
                    response_text = await response.text()
                    logger.info(f"Response Text: {response_text}")
                else:
                    logger.error(f"Failed to send data. Status: {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response Text: {response_text}")
                    # Optionally, implement retry logic or other error handling here
    except aiohttp.ClientError as client_error:
        logger.error(f"HTTP Client error occurred: {client_error}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending data: {e}", exc_info=True)
        sys.exit(1)



async def main():
    """
    The main entry point for the script.
    """
    # Define the path to your raw JSON file
    # detections_jsonfile_path = "demo_video_snippet.json"  # Update this path as needed
    
    # Optionally, you can make the JSON file path configurable via environment variables or command-line arguments
    
    # await send_detection_data(detections_jsonfile_path)
    test_detection = [{
        "store_id": "test-store",
        "camera_id": "test-camera",
        "frame_id": 1,
        "timestamp": "2023-01-01T00:00:00Z",
        "processed_timestamp": "2023-01-01T00:00:01Z",
        "entity_coordinates": [],
        "singles": 1,
        "couples": 0,
        "groups": 0,
        "total_people": 1
    }]
    
    # Send the test detection data
    success = await send_detection_data(test_detection)
    
    if success:
        logger.info("Test data sent successfully.")
    else:
        logger.error("Failed to send test data after multiple attempts.")
        sys.exit(1)


if __name__ == "__main__":
    """
    When the script is run directly, execute the main coroutine.
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user.")
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)
