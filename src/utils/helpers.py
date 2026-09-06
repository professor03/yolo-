#!/usr/bin/env python3
"""
通用工具函數模組
提供系統中常用的工具函數和輔助方法
"""

import os
import sys
import time
import json
import yaml
import logging
import hashlib
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class FileUtils:
    """檔案工具類"""
    
    @staticmethod
    def ensure_dir(directory: Union[str, Path]) -> Path:
        """確保目錄存在"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """獲取檔案大小"""
        return os.path.getsize(file_path)
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
        """計算檔案雜湊值"""
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    @staticmethod
    def safe_remove(file_path: Union[str, Path]) -> bool:
        """安全刪除檔案"""
        try:
            os.remove(file_path)
            return True
        except (OSError, FileNotFoundError):
            return False
    
    @staticmethod
    def backup_file(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """備份檔案"""
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        if backup_dir is None:
            backup_dir = file_path.parent / "backups"
        
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            logger.error(f"備份檔案失敗: {e}")
            return None

class TimeUtils:
    """時間工具類"""
    
    @staticmethod
    def get_timestamp() -> float:
        """獲取當前時間戳"""
        return time.time()
    
    @staticmethod
    def format_timestamp(timestamp: float, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """格式化時間戳"""
        return datetime.fromtimestamp(timestamp).strftime(format_str)
    
    @staticmethod
    def get_iso_timestamp() -> str:
        """獲取 ISO 格式時間戳"""
        return datetime.now().isoformat()
    
    @staticmethod
    def parse_iso_timestamp(iso_str: str) -> datetime:
        """解析 ISO 格式時間戳"""
        return datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
    
    @staticmethod
    def time_since(timestamp: float) -> float:
        """計算自指定時間戳以來的時間"""
        return time.time() - timestamp
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """格式化持續時間"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分鐘"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小時"

class MathUtils:
    """數學工具類"""
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """計算兩點間距離"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """計算兩點間角度"""
        return np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """正規化角度到 [0, 2π]"""
        return angle % (2 * np.pi)
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """限制數值範圍"""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """線性插值"""
        return a + (b - a) * t
    
    @staticmethod
    def smooth_step(edge0: float, edge1: float, x: float) -> float:
        """平滑步進函數"""
        t = MathUtils.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

class ImageUtils:
    """影像工具類"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    keep_aspect_ratio: bool = True) -> np.ndarray:
        """調整影像大小"""
        if keep_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # 計算縮放比例
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 調整大小
            resized = cv2.resize(image, (new_w, new_h))
            
            # 創建目標大小的畫布
            canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
            
            # 計算居中位置
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            return canvas
        else:
            return cv2.resize(image, target_size)
    
    @staticmethod
    def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """裁剪影像"""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def draw_text(image: np.ndarray, text: str, position: Tuple[int, int],
                 font_scale: float = 0.5, color: Tuple[int, int, int] = (255, 255, 255),
                 thickness: int = 1) -> np.ndarray:
        """在影像上繪製文字"""
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
        return image
    
    @staticmethod
    def draw_rectangle(image: np.ndarray, bbox: Tuple[int, int, int, int],
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
        """在影像上繪製矩形"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image
    
    @staticmethod
    def draw_line(image: np.ndarray, point1: Tuple[int, int], point2: Tuple[int, int],
                 color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2) -> np.ndarray:
        """在影像上繪製線段"""
        cv2.line(image, point1, point2, color, thickness)
        return image

class SystemUtils:
    """系統工具類"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """獲取系統資訊"""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """獲取記憶體使用情況"""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
    
    @staticmethod
    def get_cpu_usage() -> float:
        """獲取 CPU 使用率"""
        return psutil.cpu_percent(interval=1)
    
    @staticmethod
    def is_gpu_available() -> bool:
        """檢查 GPU 是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @staticmethod
    def get_gpu_info() -> Optional[Dict[str, Any]]:
        """獲取 GPU 資訊"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_allocated": torch.cuda.memory_allocated(0),
                    "memory_reserved": torch.cuda.memory_reserved(0)
                }
        except ImportError:
            pass
        return None

class ThreadUtils:
    """執行緒工具類"""
    
    @staticmethod
    def run_in_thread(func, *args, **kwargs):
        """在執行緒中運行函數"""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread
    
    @staticmethod
    @contextmanager
    def thread_lock(lock: threading.Lock):
        """執行緒鎖上下文管理器"""
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

class ValidationUtils:
    """驗證工具類"""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """驗證 URL 格式"""
        import re
        pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return pattern.match(url) is not None
    
    @staticmethod
    def is_valid_file_path(file_path: Union[str, Path]) -> bool:
        """驗證檔案路徑"""
        try:
            Path(file_path).resolve()
            return True
        except (OSError, ValueError):
            return False
    
    @staticmethod
    def is_valid_image_file(file_path: Union[str, Path]) -> bool:
        """驗證是否為有效的影像檔案"""
        try:
            image = cv2.imread(str(file_path))
            return image is not None
        except Exception:
            return False
    
    @staticmethod
    def validate_bbox(bbox: Tuple[float, float, float, float]) -> bool:
        """驗證邊界框格式"""
        if len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        return x1 < x2 and y1 < y2 and all(coord >= 0 for coord in bbox)

class ConfigUtils:
    """配置工具類"""
    
    
    @staticmethod
    def save_yaml_config(config: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """儲存 YAML 配置"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            logger.error(f"儲存 YAML 配置失敗: {e}")
            return False
    
    @staticmethod
    def load_json_config(file_path: Union[str, Path]) -> Dict[str, Any]:
        """載入 JSON 配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"載入 JSON 配置失敗: {e}")
            return {}
    
    @staticmethod
    def save_json_config(config: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """儲存 JSON 配置"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"儲存 JSON 配置失敗: {e}")
            return False

class PerformanceUtils:
    """性能工具類"""
    
    @staticmethod
    def measure_time(func):
        """測量函數執行時間的裝飾器"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"{func.__name__} 執行時間: {end_time - start_time:.4f} 秒")
            return result
        return wrapper
    
    @staticmethod
    def calculate_fps(frame_count: int, start_time: float) -> float:
        """計算 FPS"""
        elapsed_time = time.time() - start_time
        return frame_count / elapsed_time if elapsed_time > 0 else 0.0
    
    @staticmethod
    def get_memory_usage_mb() -> float:
        """獲取當前記憶體使用量 (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

# 全域工具函數
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，避免除零錯誤"""
    return a / b if b != 0 else default

def safe_int(value: Any, default: int = 0) -> int:
    """安全轉換為整數"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value: Any, default: float = 0.0) -> float:
    """安全轉換為浮點數"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_str(value: Any, default: str = "") -> str:
    """安全轉換為字串"""
    try:
        return str(value)
    except Exception:
        return default
