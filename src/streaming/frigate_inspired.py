# -*- coding: utf-8 -*-
"""
Frigate 啟發的串流處理模組
參考 Frigate 的 go2rtc 架構設計
"""

import asyncio
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StreamFormat(Enum):
    """串流格式枚舉"""
    WEBRTC = "webrtc"
    MJPEG = "mjpeg"
    HLS = "hls"
    RTSP = "rtsp"

@dataclass
class StreamConfig:
    """串流配置"""
    name: str
    source: str
    format: StreamFormat
    width: int = 1280
    height: int = 720
    fps: int = 15
    audio: bool = False

class StreamConverter:
    """串流轉換器 - 類似 Frigate 的 go2rtc"""
    
    def __init__(self):
        self.streams: Dict[str, StreamConfig] = {}
        self.active_streams: Dict[str, cv2.VideoCapture] = {}
        self.local_proxy_port = 8554
        self.is_running = False
        
    def add_stream(self, config: StreamConfig) -> bool:
        """添加串流配置"""
        try:
            self.streams[config.name] = config
            logger.info(f"添加串流: {config.name} -> {config.source}")
            return True
        except Exception as e:
            logger.error(f"添加串流失敗: {e}")
            return False
    
    def start_stream(self, stream_name: str) -> bool:
        """啟動串流"""
        if stream_name not in self.streams:
            logger.error(f"串流不存在: {stream_name}")
            return False
            
        config = self.streams[stream_name]
        
        try:
            # 創建 VideoCapture
            cap = cv2.VideoCapture(config.source)
            
            if not cap.isOpened():
                logger.error(f"無法打開串流: {config.source}")
                return False
                
            # 設置參數
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            cap.set(cv2.CAP_PROP_FPS, config.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.active_streams[stream_name] = cap
            logger.info(f"串流啟動成功: {stream_name}")
            return True
            
        except Exception as e:
            logger.error(f"啟動串流失敗: {e}")
            return False
    
    def get_frame(self, stream_name: str) -> Optional[np.ndarray]:
        """獲取串流幀"""
        if stream_name not in self.active_streams:
            return None
            
        cap = self.active_streams[stream_name]
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"無法讀取幀: {stream_name}")
            return None
            
        return frame
    
    def stop_stream(self, stream_name: str) -> bool:
        """停止串流"""
        if stream_name in self.active_streams:
            self.active_streams[stream_name].release()
            del self.active_streams[stream_name]
            logger.info(f"串流已停止: {stream_name}")
            return True
        return False
    
    def cleanup(self):
        """清理資源"""
        for stream_name in list(self.active_streams.keys()):
            self.stop_stream(stream_name)
        self.is_running = False

class WebRTCStreamer:
    """WebRTC 串流器"""
    
    def __init__(self, converter: StreamConverter):
        self.converter = converter
        self.connections: Dict[str, asyncio.Queue] = {}
        
    async def add_connection(self, stream_name: str, connection_id: str):
        """添加 WebRTC 連接"""
        if stream_name not in self.converter.active_streams:
            return False
            
        self.connections[connection_id] = asyncio.Queue(maxsize=10)
        logger.info(f"WebRTC 連接已添加: {connection_id}")
        return True
    
    async def remove_connection(self, connection_id: str):
        """移除 WebRTC 連接"""
        if connection_id in self.connections:
            del self.connections[connection_id]
            logger.info(f"WebRTC 連接已移除: {connection_id}")
    
    async def stream_to_connections(self, stream_name: str):
        """向所有連接串流數據"""
        while stream_name in self.converter.active_streams:
            frame = self.converter.get_frame(stream_name)
            if frame is None:
                await asyncio.sleep(0.1)
                continue
                
            # 轉換為 JPEG
            _, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # 發送到所有連接
            for connection_id, queue in self.connections.items():
                try:
                    await queue.put(jpeg_data.tobytes())
                except asyncio.QueueFull:
                    # 隊列滿了，跳過這一幀
                    pass
                    
            await asyncio.sleep(1/30)  # 30 FPS

class MJPEGStreamer:
    """MJPEG 串流器 - 備用方案"""
    
    def __init__(self, converter: StreamConverter):
        self.converter = converter
        
    def generate_mjpeg_stream(self, stream_name: str):
        """生成 MJPEG 串流"""
        while stream_name in self.converter.active_streams:
            frame = self.converter.get_frame(stream_name)
            if frame is None:
                time.sleep(0.1)
                continue
                
            # 轉換為 JPEG
            _, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # 生成 MJPEG 格式
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   jpeg_data.tobytes() + b'\r\n')
            
            time.sleep(1/15)  # 15 FPS

class FrigateInspiredStreamManager:
    """Frigate 啟發的串流管理器"""
    
    def __init__(self):
        self.converter = StreamConverter()
        self.webrtc_streamer = WebRTCStreamer(self.converter)
        self.mjpeg_streamer = MJPEGStreamer(self.converter)
        self.is_running = False
        
    def add_camera(self, name: str, rtsp_url: str, **kwargs) -> bool:
        """添加攝影機"""
        config = StreamConfig(
            name=name,
            source=rtsp_url,
            format=StreamFormat.RTSP,
            **kwargs
        )
        return self.converter.add_stream(config)
    
    def start_camera(self, name: str) -> bool:
        """啟動攝影機串流"""
        return self.converter.start_stream(name)
    
    def stop_camera(self, name: str) -> bool:
        """停止攝影機串流"""
        return self.converter.stop_stream(name)
    
    def get_frame(self, name: str) -> Optional[np.ndarray]:
        """獲取攝影機幀"""
        return self.converter.get_frame(name)
    
    async def start_webrtc_stream(self, stream_name: str, connection_id: str):
        """啟動 WebRTC 串流"""
        await self.webrtc_streamer.add_connection(stream_name, connection_id)
        await self.webrtc_streamer.stream_to_connections(stream_name)
    
    def get_mjpeg_stream(self, stream_name: str):
        """獲取 MJPEG 串流"""
        return self.mjpeg_streamer.generate_mjpeg_stream(stream_name)
    
    def cleanup(self):
        """清理所有資源"""
        self.converter.cleanup()
        self.is_running = False

# 使用示例
def create_frigate_like_system():
    """創建類似 Frigate 的系統"""
    manager = FrigateInspiredStreamManager()
    
    # 添加攝影機
    manager.add_camera(
        name="ip_camera",
        rtsp_url="rtsp://camera.example.invalid/stream",
        width=1280,
        height=720,
        fps=15
    )
    
    # 啟動攝影機
    manager.start_camera("ip_camera")
    
    return manager

if __name__ == "__main__":
    # 測試代碼
    manager = create_frigate_like_system()
    
    try:
        # 模擬運行
        for i in range(100):
            frame = manager.get_frame("ip_camera")
            if frame is not None:
                print(f"獲取到幀 {i}: {frame.shape}")
            time.sleep(0.1)
    finally:
        manager.cleanup()
