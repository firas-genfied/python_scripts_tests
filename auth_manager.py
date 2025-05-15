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
        logger.info("Starting authentication manager")
        try:
            await self.refresh_token()
            # Start background refresh task
            self.refresh_task = asyncio.create_task(self._refresh_loop())
            logger.info("Authentication manager started successfully")
        except Exception as e:
            logger.error(f"Failed to start authentication manager: {e}")
            raise
    
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
                # Prepare payload with grant_type for OAuth2
                payload = {
                    "grant_type": "password",
                    "username": self.username,
                    "password": self.password
                }
                
                # Prepare headers
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json'
                }
                
                # Log authentication attempt details
                logger.info(f"Authentication attempt details:")
                logger.info(f"Auth URL: {self.auth_url}")
                logger.info(f"Username provided: {self.username}")
                logger.info(f"Password length: {len(self.password)}")
                logger.info(f"Payload keys: {list(payload.keys())}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.auth_url, 
                        data=payload,  # Use data for form-urlencoded
                        headers=headers
                    ) as response:
                        # Log response details
                        logger.info(f"Response status: {response.status}")
                        
                        # Read response text for logging
                        response_text = await response.text()
                        logger.info(f"Raw response text: {response_text}")
                        
                        # Successful authentication
                        if response.status == 200:
                            try:
                                data = await response.json()
                                self.access_token = data.get("access_token")
                                self.token_type = data.get("token_type", "Bearer").lower()
                                
                                # Set token expiry (use refresh interval or token's expiration if available)
                                self.token_expiry = time.time() + self.refresh_interval
                                
                                logger.info("Authentication successful")
                                logger.info(f"Token type: {self.token_type}")
                                logger.info(f"Token present: {bool(self.access_token)}")
                                
                                return True
                            except Exception as json_error:
                                logger.error(f"JSON parsing error: {json_error}")
                                return False
                        
                        # Handle rate limiting or authentication errors
                        elif response.status == 429:
                            logger.error("Rate limited: Too many authentication attempts")
                            return False
                        elif response.status == 401 or response.status == 403:
                            logger.error("Authentication failed: Invalid credentials")
                            return False
                        else:
                            logger.error(f"Authentication failed. Status: {response.status}")
                            logger.error(f"Response text: {response_text}")
                            return False
            
            except Exception as e:
                logger.error(f"Comprehensive authentication error: {e}", exc_info=True)
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