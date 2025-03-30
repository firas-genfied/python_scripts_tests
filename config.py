# config.py
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Configuration dictionary
config = {
    "azure_connection_string": os.getenv("AZURE_CONNECTION_STRING", ""),
    "azure_container_name": os.getenv("AZURE_CONTAINER_NAME", ""),
    "detection_endpoint": os.getenv("DETECTION_ENDPOINT", "http://genfied-api.xperie.nz:8000/api/v1/detections/"),
    "database_url": os.getenv("DATABASE_URL", "sqlite:///./processed_videos.db"),
}

# Azure Storage configuration
AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME')

# Other configuration settings can go here