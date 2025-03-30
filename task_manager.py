from collections import defaultdict
import asyncio
import logging
import os

# Configure logger at the top of your module (or in a separate config module)
LOG_FILENAME = "tracker.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self):
        # Set to store all active tasks
        self.active_tasks = set()
        # Optional: categorize tasks by type
        self.task_categories = defaultdict(set)
        
    def create_task(self, coroutine, category=None):
        """Create and track an asyncio task."""
        task = asyncio.create_task(coroutine)
        
        # Add task to the active set
        self.active_tasks.add(task)
        
        # Categorize if needed
        if category:
            self.task_categories[category].add(task)
            
        # Set up completion callback
        task.add_done_callback(self._task_completed)
        
        return task
        
    def _task_completed(self, task):
        """Handle task completion."""
        # Remove from active tasks
        self.active_tasks.discard(task)
        
        # Remove from categories
        for category_tasks in self.task_categories.values():
            category_tasks.discard(task)
            
        # Handle exceptions
        if not task.cancelled() and task.exception():
            logging.error(f"Task failed with error: {task.exception()}")
            # Could also reraise or use a custom error handler
            
    async def wait_for_all(self, timeout=None):
        """Wait for all active tasks to complete."""
        if not self.active_tasks:
            return
            
        if timeout:
            await asyncio.wait(self.active_tasks, timeout=timeout)
        else:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
