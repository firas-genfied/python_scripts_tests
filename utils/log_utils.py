import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)
import torch

def log_total_memory(gpu_processors):
    total_allocated = 0
    total_reserved = 0
    for idx, processor in enumerate(gpu_processors):
        device = processor.device  # Ensure each processor has a 'device' attribute.
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        logger.info(
            f"Processor {idx} on {device}: allocated = {allocated/1024**2:.2f} MB, "
            f"reserved = {reserved/1024**2:.2f} MB"
        )
        total_allocated += allocated
        total_reserved += reserved
        
    logger.info(
        f"Total across all processors: allocated = {total_allocated/1024**2:.2f} MB, "
        f"reserved = {total_reserved/1024**2:.2f} MB"
    )
