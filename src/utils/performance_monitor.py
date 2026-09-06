#!/usr/bin/env python3
"""
性能監控模組
提供 GPU/CPU FPS 和記憶體使用率監控
"""

import time
import psutil
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指標數據類"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    fps_processing: Optional[float] = None
    fps_source: Optional[float] = None
    fps_inference: Optional[float] = None
    latency_ms: Optional[float] = None
    queue_size: Optional[int] = None
    dropped_frames: Optional[int] = None


class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = []
        self.max_history = 1000
        self.is_monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # 性能配置
        self.enable_gpu_monitoring = config.get("performance", {}).get("enable_gpu_monitoring", True)
        self.enable_cpu_monitoring = config.get("performance", {}).get("enable_cpu_monitoring", True)
        self.enable_memory_monitoring = config.get("performance", {}).get("enable_memory_monitoring", True)
        self.metrics_interval = config.get("performance", {}).get("metrics_interval", 1.0)
        
        # FPS 計算
        self.frame_times = []
        self.max_frame_times = 100
        self.last_frame_time = time.time()
        
        # 統計數據
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        self.start_time = time.time()
    
    def start_monitoring(self):
        """開始性能監控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("性能監控已啟動")
    
    def stop_monitoring(self):
        """停止性能監控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("性能監控已停止")
    
    def _monitor_loop(self):
        """監控循環"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                with self.lock:
                    self.metrics_history.append(metrics)
                    if len(self.metrics_history) > self.max_history:
                        self.metrics_history.pop(0)
                
                time.sleep(self.metrics_interval)
            except Exception as e:
                logger.error(f"性能監控錯誤: {e}")
                time.sleep(1.0)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指標"""
        timestamp = datetime.now()
        
        # CPU 和記憶體監控
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_available_mb = 0.0
        
        if self.enable_cpu_monitoring:
            cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if self.enable_memory_monitoring:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
        
        # GPU 監控
        gpu_utilization = None
        gpu_memory_used_gb = None
        gpu_memory_total_gb = None
        gpu_temperature = None
        
        if self.enable_gpu_monitoring and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                # 使用 PyTorch 獲取 GPU 信息
                gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                gpu_utilization = (gpu_memory_used_gb / gpu_memory_total_gb) * 100
                
                # 嘗試獲取 GPU 溫度
                if GPU_UTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_temperature = gpus[0].temperature
                    except:
                        pass
            except Exception as e:
                logger.warning(f"GPU 監控錯誤: {e}")
        
        # FPS 計算
        current_time = time.time()
        fps_processing = self._calculate_fps()
        
        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_mb=memory_available_mb,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            gpu_temperature=gpu_temperature,
            fps_processing=fps_processing,
            fps_source=None,  # 由外部設置
            fps_inference=None,  # 由外部設置
            latency_ms=None,  # 由外部設置
            queue_size=None,  # 由外部設置
            dropped_frames=None  # 由外部設置
        )
    
    def _calculate_fps(self) -> float:
        """計算處理 FPS"""
        current_time = time.time()
        if not self.frame_times:
            return 0.0
        
        # 計算最近一秒內的幀數
        recent_times = [t for t in self.frame_times if current_time - t < 1.0]
        return len(recent_times)
    
    def record_frame_processed(self):
        """記錄處理的幀"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # 保持最近的幀時間
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        
        self.total_frames_processed += 1
        self.last_frame_time = current_time
    
    def record_frame_dropped(self):
        """記錄丟棄的幀"""
        self.total_frames_dropped += 1
    
    def update_external_metrics(self, **kwargs):
        """更新外部指標"""
        with self.lock:
            if self.metrics_history:
                latest = self.metrics_history[-1]
                for key, value in kwargs.items():
                    if hasattr(latest, key):
                        setattr(latest, key, value)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """獲取當前性能指標"""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, limit: int = 100) -> list:
        """獲取性能指標歷史"""
        with self.lock:
            return self.metrics_history[-limit:] if self.metrics_history else []
    
    def get_average_metrics(self, duration_seconds: int = 60) -> Optional[Dict[str, float]]:
        """獲取平均性能指標"""
        with self.lock:
            if not self.metrics_history:
                return None
            
            current_time = time.time()
            recent_metrics = [
                m for m in self.metrics_history
                if (current_time - m.timestamp.timestamp()) <= duration_seconds
            ]
            
            if not recent_metrics:
                return None
            
            # 計算平均值
            avg_metrics = {
                "cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                "memory_percent": sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                "memory_available_mb": sum(m.memory_available_mb for m in recent_metrics) / len(recent_metrics),
            }
            
            # GPU 指標（如果可用）
            gpu_metrics = [m for m in recent_metrics if m.gpu_utilization is not None]
            if gpu_metrics:
                avg_metrics["gpu_utilization"] = sum(m.gpu_utilization for m in gpu_metrics) / len(gpu_metrics)
                avg_metrics["gpu_memory_used_gb"] = sum(m.gpu_memory_used_gb for m in gpu_metrics) / len(gpu_metrics)
                avg_metrics["gpu_temperature"] = sum(m.gpu_temperature for m in gpu_metrics if m.gpu_temperature is not None) / len([m for m in gpu_metrics if m.gpu_temperature is not None])
            
            # FPS 指標
            fps_metrics = [m for m in recent_metrics if m.fps_processing is not None]
            if fps_metrics:
                avg_metrics["fps_processing"] = sum(m.fps_processing for m in fps_metrics) / len(fps_metrics)
            
            return avg_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """獲取性能摘要"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "total_frames_processed": self.total_frames_processed,
            "total_frames_dropped": self.total_frames_dropped,
            "drop_rate": self.total_frames_dropped / max(self.total_frames_processed + self.total_frames_dropped, 1),
            "average_fps": self.total_frames_processed / max(uptime, 1),
            "current_metrics": self.get_current_metrics(),
            "average_metrics_1min": self.get_average_metrics(60),
            "average_metrics_5min": self.get_average_metrics(300)
        }


# 全局性能監控器實例
_performance_monitor = None


def get_performance_monitor(config: Optional[Dict[str, Any]] = None) -> PerformanceMonitor:
    """獲取性能監控器實例"""
    global _performance_monitor
    if _performance_monitor is None and config:
        _performance_monitor = PerformanceMonitor(config)
    return _performance_monitor


def start_performance_monitoring(config: Dict[str, Any]):
    """開始性能監控"""
    monitor = get_performance_monitor(config)
    if monitor:
        monitor.start_monitoring()


def stop_performance_monitoring():
    """停止性能監控"""
    global _performance_monitor
    if _performance_monitor:
        _performance_monitor.stop_monitoring()
        _performance_monitor = None
