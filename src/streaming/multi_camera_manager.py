# -*- coding: utf-8 -*-
"""
多攝影機管理模組
支援同時管理多台攝影機的串流
"""

import asyncio
import cv2
import numpy as np
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import queue
import json

logger = logging.getLogger(__name__)

class CameraStatus(Enum):
    """攝影機狀態"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECORDING = "recording"

@dataclass
class CameraConfig:
    """攝影機配置"""
    name: str
    rtsp_url: str
    width: int = 1280
    height: int = 720
    fps: int = 15
    enabled: bool = True
    detection_enabled: bool = True
    recording_enabled: bool = False
    audio_enabled: bool = False

@dataclass
class CameraFrame:
    """攝影機幀數據"""
    camera_name: str
    frame: np.ndarray
    timestamp: float
    frame_number: int
    detection_data: Optional[dict] = None

class MultiCameraManager:
    """多攝影機管理器"""
    
    def __init__(self, max_cameras: int = 8):
        self.max_cameras = max_cameras
        self.cameras: Dict[str, CameraConfig] = {}
        self.camera_captures: Dict[str, cv2.VideoCapture] = {}
        self.camera_status: Dict[str, CameraStatus] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.camera_threads: Dict[str, threading.Thread] = {}
        self.is_running = False
        self.frame_callbacks: List[Callable[[CameraFrame], None]] = []
        
    def add_camera(self, config: CameraConfig) -> bool:
        """添加攝影機"""
        if len(self.cameras) >= self.max_cameras:
            logger.error(f"已達到最大攝影機數量限制: {self.max_cameras}")
            return False
            
        if config.name in self.cameras:
            logger.warning(f"攝影機已存在: {config.name}")
            return False
            
        self.cameras[config.name] = config
        self.camera_status[config.name] = CameraStatus.DISCONNECTED
        self.frame_queues[config.name] = queue.Queue(maxsize=10)
        
        logger.info(f"攝影機已添加: {config.name} -> {config.rtsp_url}")
        return True
    
    def remove_camera(self, camera_name: str) -> bool:
        """移除攝影機"""
        if camera_name not in self.cameras:
            return False
            
        # 停止攝影機
        self.stop_camera(camera_name)
        
        # 清理資源
        if camera_name in self.camera_captures:
            self.camera_captures[camera_name].release()
            del self.camera_captures[camera_name]
            
        if camera_name in self.frame_queues:
            del self.frame_queues[camera_name]
            
        del self.cameras[camera_name]
        del self.camera_status[camera_name]
        
        logger.info(f"攝影機已移除: {camera_name}")
        return True
    
    def start_camera(self, camera_name: str) -> bool:
        """啟動攝影機"""
        if camera_name not in self.cameras:
            logger.error(f"攝影機不存在: {camera_name}")
            return False
            
        if camera_name in self.camera_captures:
            logger.warning(f"攝影機已在運行: {camera_name}")
            return True
            
        config = self.cameras[camera_name]
        self.camera_status[camera_name] = CameraStatus.CONNECTING
        
        try:
            # 創建 VideoCapture
            cap = cv2.VideoCapture(config.rtsp_url)
            
            if not cap.isOpened():
                logger.error(f"無法打開攝影機: {camera_name} -> {config.rtsp_url}")
                self.camera_status[camera_name] = CameraStatus.ERROR
                return False
                
            # 設置參數
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            cap.set(cv2.CAP_PROP_FPS, config.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.camera_captures[camera_name] = cap
            self.camera_status[camera_name] = CameraStatus.CONNECTED
            
            # 啟動讀取線程
            thread = threading.Thread(
                target=self._camera_reader_thread,
                args=(camera_name,),
                daemon=True
            )
            thread.start()
            self.camera_threads[camera_name] = thread
            
            logger.info(f"攝影機啟動成功: {camera_name}")
            return True
            
        except Exception as e:
            logger.error(f"啟動攝影機失敗: {camera_name} - {e}")
            self.camera_status[camera_name] = CameraStatus.ERROR
            return False
    
    def stop_camera(self, camera_name: str) -> bool:
        """停止攝影機"""
        if camera_name not in self.camera_captures:
            return False
            
        # 停止線程
        if camera_name in self.camera_threads:
            # 線程會自動停止，因為 cap.read() 會返回 False
            pass
            
        # 釋放資源
        self.camera_captures[camera_name].release()
        del self.camera_captures[camera_name]
        
        if camera_name in self.camera_threads:
            del self.camera_threads[camera_name]
            
        self.camera_status[camera_name] = CameraStatus.DISCONNECTED
        
        logger.info(f"攝影機已停止: {camera_name}")
        return True
    
    def _camera_reader_thread(self, camera_name: str):
        """攝影機讀取線程"""
        cap = self.camera_captures[camera_name]
        config = self.cameras[camera_name]
        frame_queue = self.frame_queues[camera_name]
        frame_number = 0
        
        logger.info(f"攝影機讀取線程啟動: {camera_name}")
        
        while camera_name in self.camera_captures:
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"無法讀取幀: {camera_name}")
                    time.sleep(0.1)
                    continue
                    
                # 調整幀大小
                if frame.shape[:2] != (config.height, config.width):
                    frame = cv2.resize(frame, (config.width, config.height))
                
                # 創建幀數據
                camera_frame = CameraFrame(
                    camera_name=camera_name,
                    frame=frame,
                    timestamp=time.time(),
                    frame_number=frame_number
                )
                
                # 添加到隊列
                try:
                    frame_queue.put_nowait(camera_frame)
                except queue.Full:
                    # 隊列滿了，丟棄舊幀
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait(camera_frame)
                    except queue.Empty:
                        pass
                
                # 觸發回調
                for callback in self.frame_callbacks:
                    try:
                        callback(camera_frame)
                    except Exception as e:
                        logger.error(f"幀回調錯誤: {e}")
                
                frame_number += 1
                time.sleep(1 / config.fps)
                
            except Exception as e:
                logger.error(f"攝影機讀取錯誤: {camera_name} - {e}")
                time.sleep(1)
        
        logger.info(f"攝影機讀取線程結束: {camera_name}")
    
    def get_latest_frame(self, camera_name: str) -> Optional[CameraFrame]:
        """獲取最新幀"""
        if camera_name not in self.frame_queues:
            return None
            
        frame_queue = self.frame_queues[camera_name]
        try:
            return frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_latest_frames(self) -> Dict[str, CameraFrame]:
        """獲取所有攝影機的最新幀"""
        frames = {}
        for camera_name in self.cameras.keys():
            frame = self.get_latest_frame(camera_name)
            if frame is not None:
                frames[camera_name] = frame
        return frames
    
    def add_frame_callback(self, callback: Callable[[CameraFrame], None]):
        """添加幀回調"""
        self.frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable[[CameraFrame], None]):
        """移除幀回調"""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
    
    def get_camera_status(self, camera_name: str) -> Optional[CameraStatus]:
        """獲取攝影機狀態"""
        return self.camera_status.get(camera_name)
    
    def get_all_camera_status(self) -> Dict[str, CameraStatus]:
        """獲取所有攝影機狀態"""
        return self.camera_status.copy()
    
    def get_camera_info(self, camera_name: str) -> Optional[dict]:
        """獲取攝影機信息"""
        if camera_name not in self.cameras:
            return None
            
        config = self.cameras[camera_name]
        status = self.camera_status[camera_name]
        
        return {
            "name": camera_name,
            "rtsp_url": config.rtsp_url,
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "enabled": config.enabled,
            "status": status.value,
            "is_recording": status == CameraStatus.RECORDING
        }
    
    def get_all_camera_info(self) -> List[dict]:
        """獲取所有攝影機信息"""
        return [self.get_camera_info(name) for name in self.cameras.keys()]
    
    def start_all_cameras(self) -> Dict[str, bool]:
        """啟動所有攝影機"""
        results = {}
        for camera_name in self.cameras.keys():
            results[camera_name] = self.start_camera(camera_name)
        return results
    
    def stop_all_cameras(self) -> Dict[str, bool]:
        """停止所有攝影機"""
        results = {}
        for camera_name in list(self.camera_captures.keys()):
            results[camera_name] = self.stop_camera(camera_name)
        return results
    
    def cleanup(self):
        """清理所有資源"""
        logger.info("清理多攝影機管理器...")
        self.stop_all_cameras()
        self.cameras.clear()
        self.camera_captures.clear()
        self.camera_status.clear()
        self.frame_queues.clear()
        self.camera_threads.clear()
        self.frame_callbacks.clear()
        self.is_running = False

# 使用示例
def create_multi_camera_system():
    """創建多攝影機系統"""
    manager = MultiCameraManager(max_cameras=4)
    
    # 添加攝影機
    cameras = [
        CameraConfig(
            name="camera_1",
            rtsp_url="rtsp://camera.example.invalid/stream",
            width=1280,
            height=720,
            fps=15
        ),
        CameraConfig(
            name="camera_2", 
            rtsp_url="rtsp://camera.example.invalid/stream",
            width=1280,
            height=720,
            fps=15
        ),
        CameraConfig(
            name="camera_3",
            rtsp_url="rtsp://camera.example.invalid/stream", 
            width=1280,
            height=720,
            fps=15
        )
    ]
    
    for camera in cameras:
        manager.add_camera(camera)
    
    return manager

if __name__ == "__main__":
    # 測試代碼
    manager = create_multi_camera_system()
    
    try:
        # 啟動所有攝影機
        print("啟動所有攝影機...")
        results = manager.start_all_cameras()
        print(f"啟動結果: {results}")
        
        # 運行一段時間
        for i in range(100):
            frames = manager.get_all_latest_frames()
            print(f"獲取到 {len(frames)} 個攝影機的幀")
            time.sleep(1)
            
    finally:
        manager.cleanup()
