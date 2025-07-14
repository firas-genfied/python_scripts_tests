# sender.py - Fixed version for API compatibility
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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sender.log")
    ]
)

logger = logging.getLogger(__name__)

# Configuration Variables
DETECTION_ENDPOINT = config.get("detection_endpoint")
AUTH_ENDPOINT = config.get("auth_endpoint", os.environ.get("AUTH_ENDPOINT"))
AUTH_USERNAME = config.get("auth_username", os.environ.get("AUTH_USERNAME"))
AUTH_PASSWORD = config.get("auth_password", os.environ.get("AUTH_PASSWORD"))
AUTH_REFRESH_INTERVAL = int(config.get("auth_refresh_interval", os.environ.get("AUTH_REFRESH_INTERVAL", "1800")))

if not DETECTION_ENDPOINT:
    logger.error("Detection endpoint not found in configuration.")
    sys.exit(1)
else:
    logger.debug(f"DETECTION ENDPOINT IS {DETECTION_ENDPOINT}")

if not AUTH_USERNAME or not AUTH_PASSWORD:
    logger.error("Authentication credentials not found in configuration or environment variables.")
    sys.exit(1)

MAX_RETRIES = 3
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
    logger.info("Authentication initialized")

def format_detection_data(detection_data: List[Dict]) -> List[Dict]:
    """
    Process detection data and ensure it matches the exact API requirements.
    
    Args:
        detection_data (List[Dict]): List containing detection results.
    Returns:
        List[Dict]: Formatted data ready to be sent to the endpoint.
    """
    if not detection_data or not isinstance(detection_data, list):
        raise ValueError("Invalid detection data format")
    
    formatted_results = []
    
    for frame_data in detection_data:
        # Skip frames with no people
        if frame_data.get("no_of_people", 0) == 0:
            logger.debug(f"Skipping frame with no people: camera_id={frame_data.get('camera_id')}, frame_id={frame_data.get('frame_id')}")
            continue
        
        # Ensure camera_id is an integer
        camera_id = frame_data.get("camera_id")
        if isinstance(camera_id, str):
            try:
                camera_id = int(camera_id)
            except (ValueError, TypeError):
                logger.warning(f"Invalid camera_id format: {camera_id}, defaulting to 0")
                camera_id = 0
        elif not isinstance(camera_id, int):
            camera_id = 0
        
        # Process persons array
        persons = []
        for person in frame_data.get("persons", []):
            # Ensure coordinates are strings
            coords = person.get("coords", {})
            formatted_coords = {
                "x": str(coords.get("x", "0")),
                "y": str(coords.get("y", "0"))
            }
            
            # Build person object with required fields
            formatted_person = {
                "person_id": str(person.get("person_id", "")),
                "coords": formatted_coords,
                "type": str(person.get("type", "customer"))  # Default to customer if not specified
            }
            
            # Add group_id only if it exists and is not empty
            group_id = person.get("group_id", "")
            if group_id and str(group_id).strip():
                formatted_person["group_id"] = str(group_id)
            
            persons.append(formatted_person)
        
        # Build the result object matching API specification exactly
        result = {
            "camera_id": camera_id,
            "image_url": str(frame_data.get("image_url", "")),
            "is_organised": bool(frame_data.get("is_organised", True)),
            "no_of_people": int(frame_data.get("no_of_people", len(persons))),
            "date_time": str(frame_data.get("date_time", "")),
            "persons": persons
        }
        
        # Validate the result before adding
        if validate_detection_result(result):
            formatted_results.append(result)
        else:
            logger.warning(f"Skipping invalid detection result: {result}")
    
    return formatted_results

def validate_detection_result(result: Dict) -> bool:
    """
    Validate a single detection result against API requirements.
    
    Args:
        result (Dict): Detection result to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check required fields
        required_fields = ["camera_id", "image_url", "is_organised", "no_of_people", "date_time", "persons"]
        for field in required_fields:
            if field not in result:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check data types
        if not isinstance(result["camera_id"], int):
            logger.error(f"camera_id must be integer, got: {type(result['camera_id'])}")
            return False
        
        if not isinstance(result["image_url"], str):
            logger.error(f"image_url must be string, got: {type(result['image_url'])}")
            return False
        
        if not isinstance(result["is_organised"], bool):
            logger.error(f"is_organised must be boolean, got: {type(result['is_organised'])}")
            return False
        
        if not isinstance(result["no_of_people"], int):
            logger.error(f"no_of_people must be integer, got: {type(result['no_of_people'])}")
            return False
        
        if not isinstance(result["date_time"], str):
            logger.error(f"date_time must be string, got: {type(result['date_time'])}")
            return False
        
        if not isinstance(result["persons"], list):
            logger.error(f"persons must be list, got: {type(result['persons'])}")
            return False
        
        # Validate persons array
        for i, person in enumerate(result["persons"]):
            if not isinstance(person, dict):
                logger.error(f"persons[{i}] must be dict, got: {type(person)}")
                return False
            
            # Check required person fields
            person_required = ["person_id", "coords", "type"]
            for field in person_required:
                if field not in person:
                    logger.error(f"persons[{i}] missing required field: {field}")
                    return False
            
            # Check person field types
            if not isinstance(person["person_id"], str):
                logger.error(f"persons[{i}].person_id must be string")
                return False
            
            if not isinstance(person["coords"], dict):
                logger.error(f"persons[{i}].coords must be dict")
                return False
            
            if "x" not in person["coords"] or "y" not in person["coords"]:
                logger.error(f"persons[{i}].coords missing x or y")
                return False
            
            if not isinstance(person["coords"]["x"], str) or not isinstance(person["coords"]["y"], str):
                logger.error(f"persons[{i}].coords.x and y must be strings")
                return False
            
            if not isinstance(person["type"], str):
                logger.error(f"persons[{i}].type must be string")
                return False
            
            # group_id is optional, but if present must be string
            if "group_id" in person and not isinstance(person["group_id"], str):
                logger.error(f"persons[{i}].group_id must be string if present")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

async def send_detection_data(detection_data: List[Dict]) -> bool:
    """
    Processes the detection data and sends it to the API endpoint with proper formatting.
    
    Args:
        detection_data (List[Dict]): List containing detection results.
        
    Returns:
        bool: True if the data was sent successfully, False otherwise.
    """
    global auth_manager
    if auth_manager is None:
        logger.debug("Auth manager not initialized, initializing now...")
        await initialize_auth()

    if not DETECTION_ENDPOINT:
        logger.error("DETECTION_ENDPOINT is not set or empty")
        return False
    
    logger.debug(f"Will send data to endpoint: {DETECTION_ENDPOINT}")

    try:
        # Process and validate the detection data
        formatted_data = format_detection_data(detection_data)
        if not formatted_data:
            logger.debug("No valid frames with people detected, skipping API call")
            return True
        
        logger.info(f"Sending {len(formatted_data)} frames with people detected")
        
        # Log first frame for debugging
        if formatted_data:
            first_frame = formatted_data[0]
            logger.debug(f"First frame: camera_id={first_frame.get('camera_id')} ({type(first_frame.get('camera_id'))}), "
                        f"people={first_frame.get('no_of_people')} ({type(first_frame.get('no_of_people'))}), "
                        f"is_organised={first_frame.get('is_organised')} ({type(first_frame.get('is_organised'))})")
        
    except Exception as e:
        logger.error(f"Failed to process detection data: {e}", exc_info=True)
        return False
    
    # Implement retry logic
    for attempt in range(MAX_RETRIES):
        try:
            auth_headers = await auth_manager.get_auth_header()
            
            # Add explicit content-type header
            headers = {
                **auth_headers,
                "Content-Type": "application/json"
            }
            
            # Log the payload for debugging (only on first attempt)
            if attempt == 0:
                logger.debug(f"Sending payload: {json.dumps(formatted_data, indent=2)}")
            
            async with aiohttp.ClientSession() as session:
                logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES}: Creating POST request...")
                async with session.post(
                    DETECTION_ENDPOINT, 
                    json=formatted_data, 
                    headers=headers, 
                    timeout=30  # Increased timeout
                ) as response:
                    logger.debug(f"Received response with status code: {response.status}")
                    
                    # Read response text
                    response_text = await response.text()
                    
                    if response.status == 200:
                        logger.info(f"Data sent successfully! Response: {response_text}")
                        return True
                    elif response.status == 400:
                        logger.error(f"Bad Request (400). Response: {response_text}")
                        
                        # For 400 errors, don't retry as it's likely a data format issue
                        try:
                            # Try to parse error details
                            error_json = json.loads(response_text)
                            logger.error(f"API Error Details: {json.dumps(error_json, indent=2)}")
                        except:
                            logger.error(f"Raw error response: {response_text}")
                        
                        return False
                    elif response.status == 401 or response.status == 403:
                        logger.error(f"Authentication error (status {response.status}). Refreshing token and retrying...")
                        await auth_manager.refresh_token()
                        continue
                    else:
                        logger.error(f"Failed to send data. Status: {response.status}, Response: {response_text}")
                        
                        if attempt == MAX_RETRIES - 1:
                            return False
                            
                        logger.debug(f"Retrying in {RETRY_DELAY} seconds...")
                        await asyncio.sleep(RETRY_DELAY)
                        
        except aiohttp.ClientError as client_error:
            logger.error(f"HTTP Client error: {client_error}", exc_info=True)
            if attempt == MAX_RETRIES - 1:
                return False
            logger.debug(f"Retrying in {RETRY_DELAY} seconds...")
            await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            if attempt == MAX_RETRIES - 1:
                return False
            logger.debug(f"Retrying in {RETRY_DELAY} seconds...")
            await asyncio.sleep(RETRY_DELAY)
    
    return False

async def main():
    """Test the sender with properly formatted data"""
    try:
        await initialize_auth()
        
        # Test with properly formatted data matching the API spec exactly
        test_detection = [{
            "camera_id": "1",  # This will be converted to int
            "image_url": "https://example.com/detection1.jpg",
            "is_organised": True,
            "no_of_people": 2,
            "date_time": "2025-01-15T10:30:00.000Z",
            "persons": [
                {
                    "person_id": "123",
                    "coords": {
                        "x": "0.5",
                        "y": "0.7"
                    },
                    "type": "customer",
                    "group_id": "group1"
                },
                {
                    "person_id": "456", 
                    "coords": {
                        "x": "0.3",
                        "y": "0.4"
                    },
                    "type": "customer"
                    # No group_id for this person
                }
            ]
        }]
        
        success = await send_detection_data(test_detection)
        
        if success:
            logger.info("Test data sent successfully.")
        else:
            logger.error("Failed to send test data.")
            sys.exit(1)
            
    finally:
        if auth_manager:
            await auth_manager.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)