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
    Return host and port based on store_id with failover capability.
    """
    # Determine shard index by store_id
    shard_index = (store_id - 1) // 10 + 1  # e.g., store_id 7 → shard 1
    
    # Primary endpoint
    primary_host = os.getenv(f"MILVUS_SHARD_{shard_index}_HOST", os.getenv("MILVUS_LOCAL_HOST", "localhost"))
    primary_port = os.getenv(f"MILVUS_SHARD_{shard_index}_PORT", os.getenv("MILVUS_LOCAL_PORT", "19530"))
    
    # Failover endpoint (if configured)
    backup_host = os.getenv(f"MILVUS_SHARD_{shard_index}_BACKUP_HOST", "")
    backup_port = os.getenv(f"MILVUS_SHARD_{shard_index}_BACKUP_PORT", "")
    
    # Check if primary endpoint is reachable
    import socket
    if primary_host and primary_port:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((primary_host, int(primary_port)))
        sock.close()
        
        if result == 0:  # Connection successful
            return primary_host, primary_port
    
    # Try backup if available
    if backup_host and backup_port:
        return backup_host, backup_port
        
    # Fall back to default
    return os.getenv("MILVUS_DEFAULT_HOST", "localhost"), os.getenv("MILVUS_DEFAULT_PORT", "19530")