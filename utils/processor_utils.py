import torch
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def set_memory_limit(fraction=0.9):
    """Limit GPU memory usage to a fraction of available memory"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_properties = torch.cuda.get_device_properties(device)
        total_memory = device_properties.total_memory
        
        # Set a limit on reserved memory
        max_memory = int(total_memory * fraction)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        
        logger.info(f"Set GPU memory limit to {fraction * 100:.0f}% of total ({max_memory / (1024**3):.2f} GB)")

async def initialize_processors(gpu_processors):
    """Initialize processors in sequence to avoid memory spikes"""
    for i, processor in enumerate(gpu_processors):
        logger.info(f"Initializing processor {i}...")
        
        # Force initialization of models if not already done
        if not hasattr(processor, 'model') or processor.model is None:
            # Initialize model here or call a method that does
            pass
            
        await asyncio.sleep(0.5)  # Brief pause between initializations
        
        # Log memory after each initialization
        if processor.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(processor.device) / (1024**2)
            reserved = torch.cuda.memory_reserved(processor.device) / (1024**2)
            logger.info(f"After initializing processor {i}: allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")
