#!/usr/bin/env python3
"""
多攝像頭支持系統
提供同步多攝像頭檢測、統一管理
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import queue
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class CameraStatus(Enum):
    """攝像頭狀態"""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class CameraConfig:
    """攝像頭配置"""
    camera_id: str
    name: str
    source: str  # RTSP URL, 文件路徑, 或攝像頭索引
    enabled: bool = True
    resolution: tuple = (1920, 1080)
    fps: int = 30
    quality: int = 80
    roi: Optional[List[List[int]]] = None  # ROI 區域
    calibration_data: Optional[Dict[str, Any]] = None
    detection_config: Optional[Dict[str, Any]] = None


@dataclass
class CameraFrame:
    """攝像頭幀數據"""
    camera_id: str
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    metadata: Dict[str, Any] = None


@dataclass
class DetectionResult:
    """檢測結果"""
    camera_id: str
    detections: List[Dict[str, Any]]
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    processing_time: float


class CameraManager:
    """攝像頭管理器"""
    
    def __init__(self, max_workers: int = 4):
        self.cameras: Dict[str, CameraConfig] = {}
        self.camera_captures: Dict[str, cv2.VideoCapture] = {}
        self.camera_status: Dict[str, CameraStatus] = {}
        self.camera_threads: Dict[str, threading.Thread] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.detection_callbacks: List[Callable[[DetectionResult], None]] = []
        
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.lock = threading.Lock()
        
        # 統計數據
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "camera_errors": 0,
            "processing_errors": 0
        }
    
    def add_camera(self, config: CameraConfig) -> bool:
        """添加攝像頭"""
        try:
            with self.lock:
                self.cameras[config.camera_id] = config
                self.camera_status[config.camera_id] = CameraStatus.IDLE
                self.frame_queues[config.camera_id] = queue.Queue(maxsize=10)
                logger.info(f"攝像頭已添加: {config.name} ({config.camera_id})")
                return True
        except Exception as e:
            logger.error(f"添加攝像頭失敗: {e}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """移除攝像頭"""
        try:
            with self.lock:
                if camera_id in self.cameras:
                    # 停止攝像頭
                    self.stop_camera(camera_id)
                    
                    # 清理資源
                    del self.cameras[camera_id]
                    del self.camera_status[camera_id]
                    if camera_id in self.frame_queues:
                        del self.frame_queues[camera_id]
                    
                    logger.info(f"攝像頭已移除: {camera_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"移除攝像頭失敗: {e}")
            return False
    
    def start_camera(self, camera_id: str) -> bool:
        """啟動攝像頭"""
        try:
            if camera_id not in self.cameras:
                logger.error(f"攝像頭不存在: {camera_id}")
                return False
            
            config = self.cameras[camera_id]
            if not config.enabled:
                logger.warning(f"攝像頭已禁用: {camera_id}")
                return False
            
            with self.lock:
                self.camera_status[camera_id] = CameraStatus.CONNECTING
            
            # 創建攝像頭捕獲對象
            cap = cv2.VideoCapture(config.source)
            if not cap.isOpened():
                logger.error(f"無法打開攝像頭: {camera_id}")
                self.camera_status[camera_id] = CameraStatus.ERROR
                return False
            
            # 設置攝像頭參數
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, config.fps)
            
            self.camera_captures[camera_id] = cap
            self.camera_status[camera_id] = CameraStatus.CONNECTED
            
            # 啟動攝像頭線程
            thread = threading.Thread(
                target=self._camera_worker,
                args=(camera_id,),
                daemon=True
            )
            self.camera_threads[camera_id] = thread
            thread.start()
            
            logger.info(f"攝像頭已啟動: {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"啟動攝像頭失敗: {e}")
            self.camera_status[camera_id] = CameraStatus.ERROR
            return False
    
    def stop_camera(self, camera_id: str) -> bool:
        """停止攝像頭"""
        try:
            with self.lock:
                if camera_id in self.camera_captures:
                    self.camera_captures[camera_id].release()
                    del self.camera_captures[camera_id]
                
                if camera_id in self.camera_threads:
                    # 線程會自動結束
                    del self.camera_threads[camera_id]
                
                self.camera_status[camera_id] = CameraStatus.DISCONNECTED
                logger.info(f"攝像頭已停止: {camera_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"停止攝像頭失敗: {e}")
            return False
    
    def start_all_cameras(self) -> Dict[str, bool]:
        """啟動所有攝像頭"""
        results = {}
        for camera_id in self.cameras:
            results[camera_id] = self.start_camera(camera_id)
        return results
    
    def stop_all_cameras(self) -> Dict[str, bool]:
        """停止所有攝像頭"""
        results = {}
        for camera_id in list(self.camera_captures.keys()):
            results[camera_id] = self.stop_camera(camera_id)
        return results
    
    def _camera_worker(self, camera_id: str):
        """攝像頭工作線程"""
        config = self.cameras[camera_id]
        cap = self.camera_captures[camera_id]
        frame_queue = self.frame_queues[camera_id]
        
        frame_number = 0
        
        try:
            self.camera_status[camera_id] = CameraStatus.STREAMING
            
            while self.is_running and camera_id in self.camera_captures:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"攝像頭讀取失敗: {camera_id}")
                    time.sleep(0.1)
                    continue
                
                # 應用 ROI
                if config.roi:
                    frame = self._apply_roi(frame, config.roi)
                
                # 創建幀數據
                camera_frame = CameraFrame(
                    camera_id=camera_id,
                    frame=frame,
                    timestamp=datetime.now(),
                    frame_number=frame_number,
                    metadata={
                        "resolution": frame.shape[:2],
                        "fps": config.fps,
                        "quality": config.quality
                    }
                )
                
                # 添加到隊列
                try:
                    frame_queue.put_nowait(camera_frame)
                    frame_number += 1
                    self.stats["total_frames"] += 1
                except queue.Full:
                    # 隊列滿了，丟棄舊幀
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait(camera_frame)
                    except queue.Empty:
                        pass
                
                # 控制幀率
                time.sleep(1.0 / config.fps)
                
        except Exception as e:
            logger.error(f"攝像頭工作線程錯誤: {e}")
            self.camera_status[camera_id] = CameraStatus.ERROR
            self.stats["camera_errors"] += 1
        finally:
            self.camera_status[camera_id] = CameraStatus.DISCONNECTED
    
    def _apply_roi(self, frame: np.ndarray, roi: List[List[int]]) -> np.ndarray:
        """應用 ROI 區域"""
        if not roi or len(roi) < 3:
            return frame
        
        # 創建遮罩
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        roi_points = np.array(roi, dtype=np.int32)
        cv2.fillPoly(mask, [roi_points], 255)
        
        # 應用遮罩
        result = frame.copy()
        result[mask == 0] = 0
        
        return result
    
    def get_camera_status(self, camera_id: str) -> Optional[CameraStatus]:
        """獲取攝像頭狀態"""
        return self.camera_status.get(camera_id)
    
    def get_all_camera_status(self) -> Dict[str, CameraStatus]:
        """獲取所有攝像頭狀態"""
        return self.camera_status.copy()
    
    def get_camera_frame(self, camera_id: str, timeout: float = 1.0) -> Optional[CameraFrame]:
        """獲取攝像頭幀"""
        if camera_id not in self.frame_queues:
            return None
        
        try:
            return self.frame_queues[camera_id].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def add_detection_callback(self, callback: Callable[[DetectionResult], None]):
        """添加檢測回調函數"""
        self.detection_callbacks.append(callback)
    
    def remove_detection_callback(self, callback: Callable[[DetectionResult], None]):
        """移除檢測回調函數"""
        if callback in self.detection_callbacks:
            self.detection_callbacks.remove(callback)
    
    def _notify_detection_callbacks(self, result: DetectionResult):
        """通知檢測回調函數"""
        for callback in self.detection_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"檢測回調函數錯誤: {e}")
                self.stats["processing_errors"] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取統計數據"""
        with self.lock:
            return {
                "total_cameras": len(self.cameras),
                "active_cameras": len([s for s in self.camera_status.values() if s == CameraStatus.STREAMING]),
                "total_frames": self.stats["total_frames"],
                "total_detections": self.stats["total_detections"],
                "camera_errors": self.stats["camera_errors"],
                "processing_errors": self.stats["processing_errors"],
                "camera_status": {k: v.value for k, v in self.camera_status.items()}
            }
    
    def start(self):
        """啟動多攝像頭系統"""
        self.is_running = True
        logger.info("多攝像頭系統已啟動")
    
    def stop(self):
        """停止多攝像頭系統"""
        self.is_running = False
        self.stop_all_cameras()
        logger.info("多攝像頭系統已停止")


class MultiCameraDetector:
    """多攝像頭檢測器"""
    
    def __init__(self, camera_manager: CameraManager, detection_model=None):
        self.camera_manager = camera_manager
        self.detection_model = detection_model
        self.is_running = False
        self.detection_threads = {}
        
        # 添加檢測回調
        self.camera_manager.add_detection_callback(self._on_detection_result)
    
    def start_detection(self):
        """開始檢測"""
        self.is_running = True
        
        # 為每個攝像頭啟動檢測線程
        for camera_id in self.camera_manager.cameras:
            if self.camera_manager.get_camera_status(camera_id) == CameraStatus.STREAMING:
                thread = threading.Thread(
                    target=self._detection_worker,
                    args=(camera_id,),
                    daemon=True
                )
                self.detection_threads[camera_id] = thread
                thread.start()
        
        logger.info("多攝像頭檢測已啟動")
    
    def stop_detection(self):
        """停止檢測"""
        self.is_running = False
        self.detection_threads.clear()
        logger.info("多攝像頭檢測已停止")
    
    def _detection_worker(self, camera_id: str):
        """檢測工作線程"""
        while self.is_running and camera_id in self.detection_threads:
            try:
                # 獲取幀
                camera_frame = self.camera_manager.get_camera_frame(camera_id, timeout=1.0)
                if camera_frame is None:
                    continue
                
                # 執行檢測
                start_time = time.time()
                detections = self._run_detection(camera_frame.frame, camera_id)
                processing_time = time.time() - start_time
                
                # 創建檢測結果
                result = DetectionResult(
                    camera_id=camera_id,
                    detections=detections,
                    frame=camera_frame.frame,
                    timestamp=camera_frame.timestamp,
                    frame_number=camera_frame.frame_number,
                    processing_time=processing_time
                )
                
                # 通知回調函數
                self._notify_detection_callbacks(result)
                
            except Exception as e:
                logger.error(f"檢測工作線程錯誤: {e}")
                time.sleep(0.1)
    
    def _run_detection(self, frame: np.ndarray, camera_id: str) -> List[Dict[str, Any]]:
        """執行檢測"""
        if self.detection_model is None:
            return []
        
        try:
            # 這裡應該調用實際的檢測模型
            # 暫時返回空結果
            return []
        except Exception as e:
            logger.error(f"檢測執行錯誤: {e}")
            return []
    
    def _on_detection_result(self, result: DetectionResult):
        """處理檢測結果"""
        # 更新統計
        self.camera_manager.stats["total_detections"] += len(result.detections)
        
        # 這裡可以添加更多處理邏輯
        logger.debug(f"檢測結果: {result.camera_id} - {len(result.detections)} 個目標")
    
    def _notify_detection_callbacks(self, result: DetectionResult):
        """通知檢測回調函數"""
        self.camera_manager._notify_detection_callbacks(result)


# 全局多攝像頭管理器實例
_multi_camera_manager = None


def get_multi_camera_manager() -> CameraManager:
    """獲取多攝像頭管理器實例"""
    global _multi_camera_manager
    if _multi_camera_manager is None:
        _multi_camera_manager = CameraManager()
    return _multi_camera_manager


def init_multi_camera_system():
    """初始化多攝像頭系統"""
    global _multi_camera_manager
    _multi_camera_manager = CameraManager()
    logger.info("多攝像頭系統已初始化")
