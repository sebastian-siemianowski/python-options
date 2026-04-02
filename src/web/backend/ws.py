"""
WebSocket connection manager for real-time updates.

Broadcasts:
  - task_progress: {task_id, type, status, progress, message}
  - data_ready:    {section, timestamp}
  - error:         {message}
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WS connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WS disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        payload = json.dumps(message, default=str)
        disconnected = []
        async with self._lock:
            for ws in self.active_connections:
                try:
                    await ws.send_text(payload)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                self.active_connections.remove(ws)

    async def send_task_progress(
        self,
        task_id: str,
        task_type: str,
        status: str,
        progress: float = 0.0,
        message: str = "",
        result: Optional[dict] = None,
    ):
        """Broadcast task progress update."""
        await self.broadcast({
            "type": "task_progress",
            "task_id": task_id,
            "task_type": task_type,
            "status": status,
            "progress": progress,
            "message": message,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def send_data_ready(self, section: str):
        """Notify clients that a section's data has been updated."""
        await self.broadcast({
            "type": "data_ready",
            "section": section,
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def send_error(self, message: str, task_id: Optional[str] = None):
        """Broadcast an error message."""
        await self.broadcast({
            "type": "error",
            "message": message,
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat(),
        })


# Singleton instance
manager = ConnectionManager()
