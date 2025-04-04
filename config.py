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

def get_milvus_host_port(store_id: int) -> tuple[str, str]:
    """
    Return host and port based on store_id. Defaults to local if not mapped.
    """
    # Determine shard index by store_id
    shard_index = (store_id - 1) // 10 + 1  # e.g. store_id 7 → shard 1

    host = os.getenv(f"MILVUS_SHARD_{shard_index}_HOST", os.getenv("MILVUS_LOCAL_HOST", "localhost"))
    port = os.getenv(f"MILVUS_SHARD_{shard_index}_PORT", os.getenv("MILVUS_LOCAL_PORT", "19530"))

    return host, port