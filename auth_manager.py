# auth_manager.py
import asyncio
import logging
import time
import aiohttp
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class AuthManager:
    """Manages authentication tokens with automatic refreshing"""
    
    def __init__(self, auth_url: str, username: str, password: str, refresh_interval: int = 1800):
        """
        Initialize the authentication manager
        
        Args:
            auth_url: URL for authentication endpoint
            username: Username for authentication
            password: Password for authentication
            refresh_interval: Token refresh interval in seconds (default: 30 minutes)
        """
        self.auth_url = auth_url
        self.username = username
        self.password = password
        self.refresh_interval = refresh_interval
        self.access_token = None
        self.token_type = None
        self.token_expiry = 0
        self.lock = asyncio.Lock()
        self.refresh_task = None
    
    async def start(self):
        """Start the token manager and fetch initial token"""
        await self.refresh_token()
        # Start background refresh task
        self.refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("Authentication manager started")
    
    async def stop(self):
        """Stop the token refresh background task"""
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
            self.refresh_task = None
            logger.info("Authentication manager stopped")
    
    async def _refresh_loop(self):
        """Background loop to refresh token before expiry"""
        while True:
            # Wait until token is close to expiry (90% of refresh interval)
            sleep_time = max(1, self.refresh_interval * 0.9)
            await asyncio.sleep(sleep_time)
            try:
                await self.refresh_token()
                logger.info("Token refreshed successfully in background task")
            except Exception as e:
                logger.error(f"Error refreshing token in background task: {e}")
    
    async def refresh_token(self):
        """Fetch a new authentication token"""
        async with self.lock:
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {"username": self.username, "password": self.password}
                    async with session.post(self.auth_url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.access_token = data.get("access_token")
                            self.token_type = data.get("token_type", "Bearer")
                            self.token_expiry = time.time() + self.refresh_interval
                            logger.info("Authentication token refreshed successfully")
                            return True
                        else:
                            error_text = await response.text()
                            logger.error(f"Failed to refresh token. Status: {response.status}, Response: {error_text}")
                            return False
            except Exception as e:
                logger.error(f"Error obtaining authentication token: {e}")
                raise
    
    async def get_auth_header(self) -> Dict[str, str]:
        """
        Get authentication header with valid token
        
        Returns:
            Dict containing the Authorization header
        """
        async with self.lock:
            # If token is expired or close to expiry, refresh it
            if not self.access_token or time.time() > self.token_expiry - 60:
                await self.refresh_token()
            
            if not self.access_token:
                raise ValueError("Failed to obtain valid authentication token")
                
            return {"Authorization": f"{self.token_type} {self.access_token}"}
