"""WebSocket 連接管理器"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_data: Dict[WebSocket, Dict] = {}
        
    async def connect(self, websocket: WebSocket, client_info: Dict = None):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_data[websocket] = client_info or {}
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            if websocket in self.connection_data:
                del self.connection_data[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
            
    async def broadcast(self, message: str):
        if not self.active_connections:
            return
            
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.add(connection)
                
        # 清理斷開的連接
        for connection in disconnected:
            self.disconnect(connection)
            
    async def broadcast_json(self, data: Dict):
        message = json.dumps(data, default=str)
        await self.broadcast(message)
        
    def get_connection_count(self) -> int:
        return len(self.active_connections)


# 全局 WebSocket 管理器實例
websocket_manager = WebSocketManager()
