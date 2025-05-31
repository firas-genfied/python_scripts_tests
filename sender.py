# sender.py
import asyncio
import logging
import sys
from typing import Dict, Any, List
import os
import json
import aiohttp
from detection_postprocessing import process_video_entries
from auth_manager import AuthManager
from config import config

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Changed to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),        # Log to stdout
        logging.FileHandler("sender.log")         # Log to a file named sender.log
    ]
)

logger = logging.getLogger(__name__)

# Configuration Variables
DETECTION_ENDPOINT = config.get("detection_endpoint")
AUTH_ENDPOINT = config.get("auth_endpoint", os.environ.get("AUTH_ENDPOINT"))
AUTH_USERNAME = config.get("auth_username", os.environ.get("AUTH_USERNAME"))
AUTH_PASSWORD = config.get("auth_password", os.environ.get("AUTH_PASSWORD"))
AUTH_REFRESH_INTERVAL = int(config.get("auth_refresh_interval", os.environ.get("AUTH_REFRESH_INTERVAL")))  # 30 minutes default

if not DETECTION_ENDPOINT:
    logger.error("Detection endpoint not found in configuration.")
    sys.exit(1)
else:
    logger.debug(f"DETECTION ENDPOINT IS {DETECTION_ENDPOINT}")

if not AUTH_USERNAME or not AUTH_PASSWORD:
    logger.error("Authentication credentials not found in configuration or environment variables.")
    sys.exit(1)

# Number of retries for failed requests
MAX_RETRIES = 3
# Delay between retries (in seconds)
RETRY_DELAY = 1

auth_manager = None

async def initialize_auth():
    """Initialize the authentication manager"""
    global auth_manager
    auth_manager = AuthManager(
        auth_url=AUTH_ENDPOINT,
        username=AUTH_USERNAME,
        password=AUTH_PASSWORD,
        refresh_interval=AUTH_REFRESH_INTERVAL
    )
    await auth_manager.start()
    logger.debug("Authentication initialized")

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
        if frame_data.get("no_of_people", 0) == 0:
            logger.debug(f"Skipping frame with no people detected: camera_id={frame_data.get('camera_id')}, frame_id={frame_data.get('frame_id')}")
            continue
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
    Processes the detection data and sends it directly to the API endpoint with authentication.
    Skips sending if no people were detected in any frames.    

    Args:
        detection_data (List[Dict]): List containing detection results.
        
    Returns:
        bool: True if the data was sent successfully, False otherwise.
    """
    global auth_manager
    if auth_manager is None:
        logger.debug("Auth manager not initialized, initializing now...")
        await initialize_auth()

    # Check for required config variables
    if not DETECTION_ENDPOINT:
        logger.error("DETECTION_ENDPOINT is not set or empty")
        return False
    
    logger.debug(f"Will send data to endpoint: {DETECTION_ENDPOINT}")

    try:
        # Process the detection data directly
        formatted_data = format_detection_data(detection_data)
        if not formatted_data:
            logger.debug("No frames with people detected, skipping API call")
            return True
        logger.debug(f"Sending {len(formatted_data)} frames with people detected")
        first_frame = formatted_data[0]
        logger.debug(f"Processed detection data for camera {first_frame.get('camera_id')} with {first_frame.get('no_of_people')} people, frame {first_frame.get('frame_id')}, time {first_frame.get('date_time')}")
    except Exception as e:
        logger.error(f"Failed to process detection data: {e}", exc_info=True)
        return False
    
    # Implement retry logic
    for attempt in range(MAX_RETRIES):
        try:
            auth_headers = await auth_manager.get_auth_header()
            logger.debug(f"Using auth headers: {auth_headers}")
            logger.debug(f"Full payload for detection data: {json.dumps(formatted_data, indent=2)}")
            
            async with aiohttp.ClientSession() as session:
                logger.debug("Creating POST request...")
                async with session.post(
                    DETECTION_ENDPOINT, 
                    json=formatted_data, 
                    headers=auth_headers, 
                    timeout=10
                ) as response:
                    logger.debug(f"Received response with status code: {response.status}")
                    
                    # Read and log the full response text
                    response_text = await response.text()
                    logger.debug(f"Response Text: {response_text}")
                    
                    # Try to parse response as JSON if possible
                    try:
                        response_json = await response.json()
                        logger.debug(f"Response JSON: {json.dumps(response_json, indent=2)}")
                    except Exception:
                        logger.debug("Response was not a valid JSON")
                    
                    if response.status == 200:
                        logger.debug(f"Data sent successfully for frame {detection_data[0].get('frame_id')} time {detection_data[0].get('date_time')}")
                        return True
                    elif response.status == 401 or response.status == 403:
                        logger.error(f"Authentication error (status {response.status}). Refreshing token and retrying...")
                        await auth_manager.refresh_token()
                        continue
                    else:
                        logger.error(f"Failed to send data. Status: {response.status}")
                        
                        # If this was the last attempt, return False
                        if attempt == MAX_RETRIES - 1:
                            return False
                            
                        # Otherwise wait before retrying
                        logger.debug(f"Retrying in {RETRY_DELAY} seconds (attempt {attempt+1}/{MAX_RETRIES})...")
                        await asyncio.sleep(RETRY_DELAY)
        except aiohttp.ClientError as client_error:
            logger.error(f"HTTP Client error occurred: {client_error}", exc_info=True)
            if attempt == MAX_RETRIES - 1:
                return False
            logger.debug(f"Retrying in {RETRY_DELAY} seconds (attempt {attempt+1}/{MAX_RETRIES})...")
            await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending data: {e}", exc_info=True)
            if attempt == MAX_RETRIES - 1:
                return False
            logger.debug(f"Retrying in {RETRY_DELAY} seconds (attempt {attempt+1}/{MAX_RETRIES})...")
            await asyncio.sleep(RETRY_DELAY)
    
    # If all retries failed
    return False

async def main():
    """
    The main entry point for the script.
    """
    try:
        # Initialize authentication
        await initialize_auth()
        
        # Create test detection data that matches the specified schema
        test_detection = [{
            "camera_id": 1,
            "image_url": "https://example.com/detection1.jpg",
            "is_organised": True,
            "no_of_people": 3,
            "date_time": "2025-03-28T21:00:53.555Z",
            "persons": [
                {
                    "person_id": "person123",
                    "coords": {
                        "x": "0.5",
                        "y": "0.7"
                    },
                    "type": "customer",
                    "group_id": "group1"
                },
                {
                    "person_id": "person456",
                    "coords": {
                        "x": "0.3",
                        "y": "0.4"
                    },
                    "type": "employee",
                    "group_id": "group2"
                },
                {
                    "person_id": "person789",
                    "coords": {
                        "x": "0.7",
                        "y": "0.2"
                    },
                    "type": "customer",
                    "group_id": "group1"
                }
            ]
        }]
        
        # Send the test detection data
        success = await send_detection_data(test_detection)
        
        if success:
            logger.debug("Test data sent successfully.")
        else:
            logger.error("Failed to send test data after multiple attempts.")
            sys.exit(1)
            
    finally:
        # Ensure we stop the auth manager when done
        if auth_manager:
            await auth_manager.stop()


if __name__ == "__main__":
    """
    When the script is run directly, execute the main coroutine.
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.debug("Script interrupted by user.")
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)