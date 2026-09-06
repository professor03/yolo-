"""通用工具函數整合模組"""
import os
import sys
import time
import logging
import threading
import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import psutil
import cv2
import numpy as np
from ultralytics import YOLO

# 配置日誌
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, 
                 console: bool = True) -> logging.Logger:
    """設置日誌系統"""
    logger = logging.getLogger("yolo_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除現有處理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台處理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件處理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def ensure_model_weights(model_path: str) -> str:
    """確保模型權重文件存在，不存在則下載"""
    if not os.path.exists(model_path):
        logger = logging.getLogger("yolo_detection")
        logger.info(f"模型文件不存在，開始下載: {model_path}")
        try:
            # 使用Ultralytics下載模型
            model = YOLO(model_path)
            logger.info(f"模型下載完成: {model_path}")
        except Exception as e:
            logger.error(f"模型下載失敗: {e}")
            raise
    return model_path

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """載入YAML配置文件並支援環境變數覆蓋"""
    import yaml
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # 環境變數覆蓋
        config = apply_env_overrides(config)
        return config
    except FileNotFoundError:
        logging.getLogger("yolo_detection").warning(f"配置文件不存在: {config_path}")
        return {}
    except Exception as e:
        logging.getLogger("yolo_detection").error(f"載入配置文件失敗: {e}")
        return {}

def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """應用環境變數覆蓋配置"""
    # 視頻源配置
    if 'RTSP_URL' in os.environ:
        config.setdefault('video', {})['source'] = os.environ['RTSP_URL']
    
    # 模型配置
    if 'MODEL_PATH' in os.environ:
        config.setdefault('model', {})['weights'] = os.environ['MODEL_PATH']
    if 'CONFIDENCE' in os.environ:
        config.setdefault('model', {})['conf'] = float(os.environ['CONFIDENCE'])
    if 'DEVICE' in os.environ:
        config.setdefault('model', {})['device'] = os.environ['DEVICE']
    
    # 性能配置
    if 'USE_FP16' in os.environ:
        config.setdefault('model', {})['use_fp16'] = os.environ['USE_FP16'].lower() == 'true'
    if 'QUEUE_SIZE' in os.environ:
        config.setdefault('processing', {})['queue_size'] = int(os.environ['QUEUE_SIZE'])
    if 'DROP_FRAME_POLICY' in os.environ:
        config.setdefault('processing', {})['drop_frame_policy'] = os.environ['DROP_FRAME_POLICY']
    
    # API配置
    if 'API_HOST' in os.environ:
        config.setdefault('server', {})['host'] = os.environ['API_HOST']
    if 'API_PORT' in os.environ:
        config.setdefault('server', {})['port'] = int(os.environ['API_PORT'])
    if 'LOG_LEVEL' in os.environ:
        config.setdefault('logging', {})['level'] = os.environ['LOG_LEVEL']
    if 'CORS_ORIGINS' in os.environ:
        origins = os.environ['CORS_ORIGINS'].split(',')
        config.setdefault('server', {})['cors_origins'] = origins
    
    # 安全配置
    if 'AUTH_TOKEN' in os.environ:
        config.setdefault('server', {})['auth_token'] = os.environ['AUTH_TOKEN']
    
    # 日誌配置
    if 'LOG_FILE' in os.environ:
        config.setdefault('logging', {})['file'] = os.environ['LOG_FILE']
    if 'LOG_FORMAT' in os.environ:
        config.setdefault('logging', {})['format'] = os.environ['LOG_FORMAT']
    
    # 重連配置
    if 'RECONNECT_ATTEMPTS' in os.environ:
        config.setdefault('video', {})['reconnect_attempts'] = int(os.environ['RECONNECT_ATTEMPTS'])
    if 'RECONNECT_DELAY' in os.environ:
        config.setdefault('video', {})['reconnect_delay'] = float(os.environ['RECONNECT_DELAY'])
    if 'RECONNECT_BACKOFF' in os.environ:
        config.setdefault('video', {})['reconnect_backoff'] = float(os.environ['RECONNECT_BACKOFF'])
    
    return config

def merge_configs(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """合併配置，用戶配置優先"""
    merged = default_config.copy()
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

class ResourceMonitor:
    """資源監控器"""
    
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(interval=None)  # 初始化CPU監控
        
    def start(self, stats_callback=None):
        """啟動監控"""
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._monitor_loop, args=(stats_callback,))
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """停止監控"""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
    
    def _monitor_loop(self, stats_callback=None):
        """監控循環"""
        while not self.stop_event.wait(self.interval):
            try:
                mem_info = self.process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                cpu_percent = self.process.cpu_percent(interval=None)
                
                stats = {
                    "rss_mb": round(rss_mb, 2),
                    "cpu_percent": round(cpu_percent, 1),
                    "timestamp": time.time()
                }
                
                if stats_callback:
                    stats_callback(stats)
                    
            except Exception as e:
                logging.getLogger("yolo_detection").warning(f"資源監控錯誤: {e}")

class DroppingQueue:
    """丟棄式佇列，滿時丟棄最舊的項目"""
    
    def __init__(self, maxsize: int = 10, drop_policy: str = "drop_oldest", max_wait: float = 0.1):
        self.maxsize = maxsize
        self.drop_policy = drop_policy
        self.max_wait = max_wait
        self.queue = []
        self.lock = threading.Lock()
        self.dropped_count = 0
    
    def put(self, item, timeout=None):
        """添加項目到佇列"""
        with self.lock:
            if len(self.queue) >= self.maxsize:
                if self.drop_policy == "drop_oldest":
                    self.queue.pop(0)  # 丟棄最舊的項目
                    self.dropped_count += 1
                elif self.drop_policy == "drop_newest":
                    self.dropped_count += 1
                    return  # 丟棄新項目
            self.queue.append(item)
    
    def get(self, timeout=None):
        """從佇列獲取項目"""
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
            return None
    
    def empty(self):
        """檢查佇列是否為空"""
        with self.lock:
            return len(self.queue) == 0
    
    def qsize(self):
        """獲取佇列大小"""
        with self.lock:
            return len(self.queue)

def open_video_capture(source, max_attempts: int = 5, delay: float = 2.0) -> Optional[cv2.VideoCapture]:
    """開啟視頻捕獲，支援重試"""
    attempt = 0
    current_delay = delay
    
    while attempt < max_attempts:
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # 測試讀取一幀
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                else:
                    cap.release()
        except Exception:
            pass
        
        attempt += 1
        if attempt < max_attempts:
            time.sleep(current_delay)
            current_delay *= 1.5
    
    return None

def cleanup_resources(*resources):
    """清理資源"""
    for resource in resources:
        if resource is not None:
            try:
                if hasattr(resource, 'release'):
                    resource.release()
                elif hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'stop'):
                    resource.stop()
            except Exception as e:
                logging.getLogger("yolo_detection").warning(f"清理資源失敗: {e}")

def save_events_csv(events, output_file: str, source_id: str):
    """保存事件到CSV檔案"""
    if not events:
        return
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'timestamp', 'source_id', 'event_type', 'line_id', 'zone_id', 
            'track_id', 'x_px', 'y_px', 'x_m', 'y_m'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for event in events:
            writer.writerow({
                'timestamp': getattr(event, 'timestamp', ''),
                'source_id': source_id,
                'event_type': getattr(event, 'event_type', ''),
                'line_id': getattr(event, 'line_id', ''),
                'zone_id': getattr(event, 'zone_id', ''),
                'track_id': getattr(event, 'track_id', ''),
                'x_px': getattr(event, 'x_px', ''),
                'y_px': getattr(event, 'y_px', ''),
                'x_m': getattr(event, 'x_m', ''),
                'y_m': getattr(event, 'y_m', '')
            })
