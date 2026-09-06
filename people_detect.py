#!/usr/bin/env python3
"""YOLO 人流偵測系統 - 優化版本
整合A+B+C所有功能，減少冗餘代碼
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO

from src.logic.anomaly_detection import AnomalyDetector
from src.utils.common import (
    setup_logging, ensure_model_weights, load_yaml_config,
    ResourceMonitor, DroppingQueue, open_video_capture, cleanup_resources, save_events_csv
)
from src.logic.counter import (
    LineConfig, LineEvent, MultiLineCounter, ZoneConfig, ZoneOccupancy, TrackInfo as CounterTrackInfo
)
from src.logic.calibration import HomographyCalibrator, CalibrationConfig
from src.api.data_store import data_store
from src.database.models import get_database_manager
from src.database.services import (
    DetectionEventService, LineCountService, ZoneOccupancyService,
    SystemMetricsService, AlertService, SessionService
)
from src.utils.performance_monitor import get_performance_monitor, start_performance_monitoring, stop_performance_monitoring
from src.video.frame_buffer import frame_buffer

# 全局日誌器
logger = setup_logging()

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FRAME_DIR = PROJECT_ROOT / "data" / "frames"
FRAME_DIR = Path(os.getenv("FRAME_OUTPUT_DIR", str(DEFAULT_FRAME_DIR))).resolve()
FRAME_DIR.mkdir(parents=True, exist_ok=True)

def save_frame_to_disk(stream_id: str, image_bytes: bytes) -> None:
    target = FRAME_DIR / f"{stream_id}.jpg"
    tmp = target.with_suffix(".jpg.tmp")
    with tmp.open("wb") as f:
        f.write(image_bytes)
    tmp.replace(target)


class PerformanceOptimizer:
    """性能優化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 從新的性能配置結構中獲取設置
        performance_config = config.get("performance", {})
        self.use_fp16 = performance_config.get("use_fp16", False)
        self.batch_size = performance_config.get("batch_size", 1)
        self.num_workers = performance_config.get("num_workers", 4)
        self.pin_memory = performance_config.get("pin_memory", True)
        
        # 佇列配置
        self.queue_size = performance_config.get("queue_size", 10)
        self.drop_policy = performance_config.get("drop_frame_policy", "drop_oldest")
        self.max_queue_wait = performance_config.get("max_queue_wait", 0.1)
        self.enable_queue_monitoring = performance_config.get("enable_queue_monitoring", True)
        
        # 性能監控配置
        self.enable_gpu_monitoring = performance_config.get("enable_gpu_monitoring", True)
        self.enable_cpu_monitoring = performance_config.get("enable_cpu_monitoring", True)
        self.enable_memory_monitoring = performance_config.get("enable_memory_monitoring", True)
        self.metrics_interval = performance_config.get("metrics_interval", 1.0)
        
        # 推論優化配置
        inference_config = performance_config.get("inference_optimization", {})
        self.enable_tensorrt = inference_config.get("enable_tensorrt", False)
        self.enable_onnx = inference_config.get("enable_onnx", False)
        self.enable_openvino = inference_config.get("enable_openvino", False)
        self.precision = inference_config.get("precision", "fp32")
        
        # 記憶體管理配置
        memory_config = performance_config.get("memory_management", {})
        self.max_memory_usage = memory_config.get("max_memory_usage", 0.8)
        self.gc_threshold = memory_config.get("gc_threshold", 1000)
        self.enable_memory_pool = memory_config.get("enable_memory_pool", True)
        
        # 性能統計
        self.fps_stats = {
            "inference_fps": 0.0,
            "processing_fps": 0.0,
            "gpu_utilization": 0.0,
            "memory_usage": 0.0,
            "queue_size": 0,
            "dropped_frames": 0
        }
        
        self.frame_times = []
        self.inference_times = []
        
    def setup_model_optimization(self, model: YOLO) -> YOLO:
        """設置模型優化"""
        try:
            # 設置設備
            device = self.config.get("device", "")
            if device == "":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 設置 FP16 優化
            if self.use_fp16 and device != "cpu":
                model.model.half()  # 轉換為半精度
                logger.info("啟用 FP16 半精度推論")
            
            # 移動模型到設備
            model.to(device)
            
            # FP16優化
            if self.use_fp16 and device == "cuda":
                logger.info("啟用FP16半精度推理")
                model.model.half()
            
            # 推論優化
            if self.enable_tensorrt and device == "cuda":
                logger.info("啟用TensorRT優化")
                # TensorRT 優化需要額外的實現
                pass
            
            if self.enable_onnx:
                logger.info("啟用ONNX優化")
                # ONNX 優化需要額外的實現
                pass
            
            if self.enable_openvino:
                logger.info("啟用OpenVINO優化")
                # OpenVINO 優化需要額外的實現
                pass
            
            # 記憶體管理
            if self.enable_memory_pool and device == "cuda":
                logger.info("啟用記憶體池")
                # 記憶體池優化
                pass
                torch.backends.cudnn.benchmark = True
            else:
                logger.info(f"使用FP32精度推理 (設備: {device})")
            
            # 設置推理模式
            model.model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"模型優化設置失敗: {e}")
            return model
    
    def create_optimized_queue(self) -> DroppingQueue:
        """創建優化的幀佇列"""
        return DroppingQueue(
            maxsize=self.queue_size,
            drop_policy=self.drop_policy,
            max_wait=self.max_queue_wait
        )
    
    def measure_inference_time(self, func):
        """測量推理時間裝飾器"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            inference_time = end_time - start_time
            self.inference_times.append(inference_time)
            
            # 保持最近100次記錄
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return result
        return wrapper
    
    def update_fps_stats(self, frame_time: float):
        """更新FPS統計"""
        self.frame_times.append(frame_time)
        
        # 保持最近60秒記錄
        if len(self.frame_times) > 60:
            self.frame_times = self.frame_times[-60:]
        
        # 計算FPS
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps_stats["processing_fps"] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        # 計算推理FPS
        if len(self.inference_times) > 1:
            avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            self.fps_stats["inference_fps"] = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        # GPU使用率
        if torch.cuda.is_available():
            try:
                self.fps_stats["gpu_utilization"] = torch.cuda.utilization()
                self.fps_stats["memory_usage"] = torch.cuda.memory_allocated() / 1024**3  # GB
            except:
                self.fps_stats["gpu_utilization"] = 0
                self.fps_stats["memory_usage"] = 0
        
        # 內存使用
        process = psutil.Process()
        self.fps_stats["memory_usage"] = process.memory_info().rss / 1024**3  # GB
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """獲取性能指標"""
        return {
            "fps": {
                "processing": round(self.fps_stats["processing_fps"], 2),
                "inference": round(self.fps_stats["inference_fps"], 2)
            },
            "gpu": {
                "utilization": round(self.fps_stats["gpu_utilization"], 1),
                "memory_gb": round(self.fps_stats["memory_usage"], 2)
            },
            "queue": {
                "size": self.fps_stats["queue_size"],
                "dropped_frames": self.fps_stats["dropped_frames"]
            },
            "optimization": {
                "fp16_enabled": self.use_fp16,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers
            }
        }

def create_line_configs(config: Dict[str, Any]) -> List[LineConfig]:
    """創建線段配置"""
    line_configs = []
    for line_data in config.get("lines", []):
        line_config = LineConfig(
            id=line_data["id"],
            points=line_data["points"],
            direction=line_data.get("direction", "both"),
            uturn_cooldown=line_data.get("uturn_cooldown", 30),
            jitter_frames=line_data.get("jitter_frames", 2)
        )
        line_configs.append(line_config)
    return line_configs

def create_zone_configs(config: Dict[str, Any]) -> List[ZoneConfig]:
    """創建區域配置"""
    zone_configs = []
    
    # 正常區域
    for zone_data in config.get("zones", []):
        zone_config = ZoneConfig(
            id=zone_data["id"],
            polygon=zone_data["polygon"],
            ignore=zone_data.get("ignore", False)
        )
        zone_configs.append(zone_config)
    
    # 忽略區域
    for ignore_data in config.get("ignore_zones", []):
        zone_config = ZoneConfig(
            id=ignore_data["id"],
            polygon=ignore_data["polygon"],
            ignore=True
        )
        zone_configs.append(zone_config)
    
    return zone_configs

def create_calibrator(config: Dict[str, Any]) -> HomographyCalibrator:
    """創建標定器"""
    calibration_config = CalibrationConfig(
        enabled=config.get("calibration", {}).get("enabled", False),
        src_points=config.get("calibration", {}).get("src_points", []),
        dst_points=config.get("calibration", {}).get("dst_points", [])
    )
    return HomographyCalibrator(calibration_config)

def process_detection_results(
    results,
    frame_time: float,
    event_service=None,
    session_id=None,
    frame_number=None,
    allowed_classes: Optional[List[int]] = None,
) -> List[CounterTrackInfo]:
    """處理偵測結果，轉換為追蹤信息"""
    tracks = []
    all_detections = []  # 儲存所有檢測到的物體

    if results and len(results) > 0:
        result = results[0]
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                # 獲取類別ID和信心度
                class_id = int(box.cls.cpu().numpy()[0]) if hasattr(box, 'cls') else -1
                confidence = float(box.conf.cpu().numpy()[0]) if hasattr(box, 'conf') else 0.0

                if allowed_classes is not None and class_id not in allowed_classes:
                    continue

                # 計算邊框座標
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # 記錄所有檢測到的物體到資料庫
                detection_info = {
                    'class_id': class_id,
                    'class_name': get_class_name(class_id),
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'timestamp': frame_time
                }
                all_detections.append(detection_info)
                
                # 寫入資料庫事件
                if event_service and session_id is not None:
                    try:
                        event_data = {
                            'timestamp': datetime.fromtimestamp(frame_time),
                            'source_id': 'detection',
                            'event_type': 'object_detected',
                            'x_px': center_x,
                            'y_px': center_y,
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': get_class_name(class_id),
                            'frame_number': frame_number,
                            'session_id': session_id,
                            'metadata_json': json.dumps({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'detection_id': f'det_{int(frame_time * 1000)}_{class_id}'
                            })
                        }
                        event_service.create_event(event_data)
                    except Exception as e:
                        print(f"Warning: Failed to write detection event to database: {e}")
                
                # 只對人物類別（class_id = 0）創建追蹤信息
                if class_id == 0 and hasattr(box, 'id') and box.id is not None:
                    track = CounterTrackInfo(
                        id=int(box.id.cpu().numpy()[0]),
                        center=(center_x, center_y),
                        bbox=(x1, y1, x2, y2),
                        score=confidence,
                        timestamp=frame_time
                    )
                    tracks.append(track)
    
    # 將所有檢測結果存儲到資料庫
    if all_detections:
        data_store.add_detection_batch(all_detections)
    
    return tracks

def get_class_name(class_id: int) -> str:
    """根據類別ID獲取類別名稱"""
    class_names = {
        0: 'person',
        1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
        21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
        26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
        31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
        41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
        66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
        71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
        76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
    return class_names.get(class_id, f'unknown_{class_id}')

def draw_annotations(frame, line_configs: List[LineConfig], zone_configs: List[ZoneConfig],
                    line_counts: Dict[str, Dict[str, int]], tracks: List[CounterTrackInfo] = None, include_overlays: bool = True):
    """繪製偵測框及可選的統計資訊。"""
    if tracks:
        for track in tracks:
            if hasattr(track, "bbox") and track.bbox:
                x1, y1, x2, y2 = track.bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                if include_overlays and hasattr(track, "track_id") and track.track_id:
                    label = f"ID:{track.track_id}"
                    if hasattr(track, "confidence") and track.confidence:
                        label += f" ({track.confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (int(x1), int(y1) - 25), (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if not include_overlays:
        return

    for line_config in line_configs:
        if len(line_config.points) >= 2:
            start_point = (int(line_config.points[0][0]), int(line_config.points[0][1]))
            end_point = (int(line_config.points[1][0]), int(line_config.points[1][1]))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(frame, line_config.id, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for zone_config in zone_configs:
        if len(zone_config.polygon) >= 3:
            pts = np.array(zone_config.polygon, np.int32)
            color = (255, 0, 0) if zone_config.ignore else (0, 0, 255)
            cv2.polylines(frame, [pts], True, color, 2)
            cv2.putText(frame, zone_config.id, (int(zone_config.polygon[0][0]), int(zone_config.polygon[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    y_offset = 30
    for line_id, counts in line_counts.items():
        text = f"{line_id}: In={counts['in']}, Out={counts['out']}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25

def run_detection(
    source: Union[str, int],
    config: Optional[Dict[str, Any]] = None,
    show: bool = True,
    save: bool = False,
    outdir: str = "runs/people_detect",
    duration: Optional[float] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """運行人流偵測 - 優化版本"""
    
    # 載入配置
    if config is None:
        config = load_yaml_config("configs/sample.yaml")
    
    # 創建獨立會話資料夾
    if session_id is None:
        # 自動生成會話ID (格式: YYYYMMDDHHMMSS)
        session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 創建會話專用資料夾
    session_dir = os.path.join(outdir, session_id)
    os.makedirs(session_dir, exist_ok=True)
    stream_id = config.get("source_id", str(source))

    raw_allowed_classes = config.get("allowed_classes", [0])
    allowed_classes: Optional[List[int]]
    if raw_allowed_classes is None:
        allowed_classes = None
    elif isinstance(raw_allowed_classes, (int, float, str)):
        try:
            allowed_classes = [int(raw_allowed_classes)]
        except (TypeError, ValueError):
            allowed_classes = None
    else:
        allowed_classes = []
        for cls in raw_allowed_classes:
            try:
                allowed_classes.append(int(cls))
            except (TypeError, ValueError):
                continue
        if not allowed_classes:
            allowed_classes = None


    # 設置日誌
    log_level = config.get("log_level", "INFO")
    log_file = os.path.join(session_dir, "detection.log")
    no_log_console = config.get("no_log_console", False)
    
    logger = setup_logging(log_level, log_file, not no_log_console)
    logger.info(f"Created session directory: {session_dir}")
    logger.info("Allowed detection classes: %s", allowed_classes if allowed_classes is not None else "all")

    # 啟動性能監控
    start_performance_monitoring(config)
    performance_monitor = get_performance_monitor(config)
    
    # 初始化資料庫
    db_manager = get_database_manager()
    db_manager.create_tables()  # 確保表存在
    db_session = db_manager.get_session()
    
    # 創建資料庫服務
    event_service = DetectionEventService(db_session)
    line_service = LineCountService(db_session)
    zone_service = ZoneOccupancyService(db_session)
    metrics_service = SystemMetricsService(db_session)
    alert_service = AlertService(db_session)
    session_service = SessionService(db_session)
    
    # 創建會話記錄
    session_data = {
        "session_id": session_id,
        "start_time": datetime.now(),
        "source_id": str(source),
        "source_type": "file" if isinstance(source, str) else "camera",
        "source_url": str(source),
        "is_active": True,
        "config_json": json.dumps(config)
    }
    db_session_record = session_service.create_session(session_data)
    
    logger.info("detection_start", extra={
        "source": str(source),
        "model": config.get("model", "yolov8m.pt"),
        "track": config.get("track", False),
        "session_id": session_id
    })
    
    # 確保模型存在
    model_path = ensure_model_weights(config.get("model", "yolov8m.pt"))
    model = YOLO(model_path)
    
    # 開啟視頻捕獲
    cap = open_video_capture(
        source, 
        config.get("reconnect_attempts", 5),
        config.get("reconnect_delay", 2.0)
    )
    if cap is None:
        logger.error("failed_to_open_capture", extra={"source": str(source)})
        return {"error": "Failed to open video source"}
    
    # 獲取視頻屬性
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 初始化追蹤器
    tracker = model.track if config.get("track", False) else None
    
    # 創建配置對象
    line_configs = create_line_configs(config)
    zone_configs = create_zone_configs(config)
    calibrator = create_calibrator(config)
    
    # 初始化計數器
    counter_config = config.get("counter", {})
    line_counter = MultiLineCounter(
        line_configs,
        cooldown_frames=counter_config.get("cooldown_frames", 30),
        max_gap_frames=counter_config.get("max_gap_frames", 15),
        max_relink_dist=counter_config.get("max_relink_dist", 40.0),
        history_size=counter_config.get("history_size", 30)
    )
    
    zone_occupancy = ZoneOccupancy(zone_configs)
    
    # 初始化異常檢測器
    anomaly_detector = AnomalyDetector(
        stay_threshold=config.get("stay_threshold", 10.0),
        crowd_threshold=config.get("crowd_threshold", 5)
    )
    
    # 初始化視頻寫入器
    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 使用會話資料夾和簡潔的檔案名
        output_path = os.path.join(session_dir, f"{session_id}.mp4")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Video will be saved to: {output_path}")
    
    # 初始化性能優化器
    optimizer = PerformanceOptimizer(config)
    
    # 優化模型
    model = optimizer.setup_model_optimization(model)
    
    # 初始化優化幀佇列
    frame_queue = optimizer.create_optimized_queue()
    
    # 統計變數
    stats = {
        "frames": 0,
        "source_frames": 0,
        "start_time": time.perf_counter(),
        "people_count": 0,
        "line_counts": {},
        "line_events": []
    }
    
    # 啟動資源監控
    monitor = ResourceMonitor(config.get("monitor_interval", 5.0))
    monitor.start(lambda stats_data: logger.info("resource_usage", extra=stats_data))

    predict_kwargs = {
        "conf": config.get("conf", 0.25),
        "iou": config.get("iou", 0.7),
        "imgsz": config.get("imgsz", 640),
        "verbose": False,
    }
    if allowed_classes is not None:
        predict_kwargs["classes"] = allowed_classes

    # 主循環
    start_time = time.perf_counter()
    last_frame_time = start_time
    
    try:
        while True:
            # 檢查持續時間
            if duration and (time.perf_counter() - start_time) >= duration:
                break
            
            # 讀取幀
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.perf_counter()
            frame_latency = (current_time - last_frame_time) * 1000
            last_frame_time = current_time
            
            stats["source_frames"] += 1
            
            # 將幀放入佇列
            try:
                frame_queue.put((frame, current_time), timeout=0.1)
            except:
                continue
            
            # 從佇列取幀
            try:
                frame, frame_time = frame_queue.get(timeout=0.1)
            except:
                continue
            
            # 執行偵測 (使用性能優化)
            @optimizer.measure_inference_time
            def run_inference():
                if tracker:
                    return tracker(frame, persist=True, tracker=config.get("tracker", "bytetrack.yaml"), **predict_kwargs)
                else:
                    return model.predict(frame, **predict_kwargs)
            
            results = run_inference()
            
            # 記錄性能指標
            if performance_monitor:
                performance_monitor.record_frame_processed()
                performance_monitor.update_external_metrics(
                    fps_source=stats["source_frames"] / max(current_time - start_time, 1e-6),
                    fps_inference=optimizer.fps_stats.get("inference_fps", 0),
                    latency_ms=frame_latency,
                    queue_size=frame_queue.qsize(),
                    dropped_frames=frame_queue.dropped_count
                )
            
            # 處理結果
            tracks = process_detection_results(
                results,
                frame_time,
                event_service,
                session_id,
                stats["frames"],
                allowed_classes=allowed_classes,
            )
            
            # 更新計數器
            line_events = line_counter.update(tracks, stats["frames"], calibrator)
            zone_events = zone_occupancy.update(tracks, calibrator)
            
            # 寫入線計數事件到資料庫
            if line_service and line_events:
                for event in line_events:
                    try:
                        line_count_data = {
                            'timestamp': datetime.fromtimestamp(frame_time),
                            'line_id': event.line_id,
                            'direction': event.direction,
                            'count': 1,
                            'session_id': session_id
                        }
                        line_service.create_count(line_count_data)
                    except Exception as e:
                        print(f"Warning: Failed to write line count to database: {e}")
            
            # 寫入區域佔用事件到資料庫
            if zone_service and zone_events:
                for event in zone_events:
                    try:
                        zone_occupancy_data = {
                            'timestamp': datetime.fromtimestamp(frame_time),
                            'zone_id': event.zone_id,
                            'occupancy_count': 1 if event.event_type == 'entry' else -1,
                            'session_id': session_id
                        }
                        zone_service.create_occupancy(zone_occupancy_data)
                    except Exception as e:
                        print(f"Warning: Failed to write zone occupancy to database: {e}")
            
            # 更新統計
            stats["people_count"] = len(tracks)
            stats["line_counts"] = line_counter.get_counts()
            stats["line_events"].extend(line_events)
            
            # 更新性能統計
            optimizer.update_fps_stats(frame_latency / 1000)
            optimizer.fps_stats["queue_size"] = frame_queue.qsize()
            optimizer.fps_stats["dropped_frames"] = frame_queue.dropped_count
            
            
            # 更新 API 指標
            performance_metrics = optimizer.get_performance_metrics()
            cpu_usage = psutil.cpu_percent()
            memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)

            data_store.update_metrics(
                stream_id=stream_id,
                fps_processing=performance_metrics["fps"].get("processing", 0.0),
                fps_source=stats["source_frames"] / max(current_time - start_time, 1e-6),
                people_count=len(tracks),
                latency_ms=frame_latency,
                cpu_percent=cpu_usage,
                memory_mb=memory_usage_mb
            )
            data_store.update_source(
                stream_id,
                str(source),
                "online",
                performance_metrics["fps"].get("processing", 0.0),
                len(tracks),
                fps_source=stats["source_frames"] / max(current_time - start_time, 1e-6),
                latency_ms=frame_latency,
                cpu_percent=cpu_usage,
                memory_mb=memory_usage_mb
            )
            if metrics_service and stats["frames"] % 30 == 0:  # 每30幀寫入一次
                try:
                    metrics_data = {
                        'timestamp': datetime.fromtimestamp(frame_time),
                        'fps_processing': performance_metrics["fps"]["processing"],
                        'fps_source': stats["source_frames"] / max(current_time - start_time, 1e-6),
                        'fps_inference': performance_metrics["fps"].get("inference", 0),
                        'latency_ms': frame_latency,
                        'cpu_percent': cpu_usage,
                        'memory_mb': memory_usage_mb,
                        'people_count': len(tracks),
                        'total_detections': stats.get("total_detections", 0),
                        'queue_size': frame_queue.qsize(),
                        'dropped_frames': frame_queue.dropped_count,
                        'session_id': session_id,
                        'source_id': str(source)
                    }
                    metrics_service.create_metrics(metrics_data)
                except Exception as e:
                    print(f"Warning: Failed to write metrics to database: {e}")
            
            # 安全處理line_counts
            line_counts = stats.get("line_counts", {})
            total_events = 0
            if isinstance(line_counts, dict):
                for value in line_counts.values():
                    if isinstance(value, dict):
                        total_events += sum(
                            v for v in value.values() if isinstance(v, (int, float))
                        )
                    elif isinstance(value, (int, float)):
                        total_events += value
            data_store.update_line_counts(line_counts, total_events)
            # 安全處理zone_occupancy
            try:
                data_store.update_zone_occupancy(zone_occupancy.get_occupancy())
            except AttributeError:
                pass  # 如果方法不存在就跳過
            
            # 異常檢測
            if tracks:
                anomaly_events = anomaly_detector.update(tracks, current_time, len(tracks))
                for event in anomaly_events:
                    if isinstance(event, dict):
                        data_store.add_alert(
                            level=event.get("level", "info"),
                            message=event.get("message", "檢測到異常行為"),
                            source_id="anomaly_detector"
                        )
            
            # 記錄幀指標
            logger.info("frame_metrics", extra={
                "frame": stats["frames"],
                "people_count": len(tracks),
                "fps_processing": 1.0 / max(frame_latency / 1000, 1e-6),
                "fps_source": stats["source_frames"] / max(current_time - start_time, 1e-6),
                "latency_ms": frame_latency
            })
            
            stats["frames"] += 1

            annotated_frame = frame.copy()
            draw_annotations(annotated_frame, line_configs, zone_configs, stats["line_counts"], tracks, include_overlays=False)

            success, encoded_frame = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                frame_bytes = encoded_frame.tobytes()
                frame_buffer.update(stream_id, frame_bytes)
                save_frame_to_disk(stream_id, frame_bytes)

            # 顯示結果
            if show:
                cv2.imshow("YOLO People Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 保存影片
            if writer is not None:
                writer.write(annotated_frame)
    
    except KeyboardInterrupt:
        logger.info("detection_interrupted")
    except Exception as exc:
        logger.error("detection_error", extra={"error": str(exc)})
        raise
    finally:
        # 清理資源
        cleanup_resources(cap, writer, monitor)
        
        # 記憶體清理
        try:
            if 'tracks' in locals():
                del tracks
            if 'detections' in locals():
                del detections
            if 'frame' in locals():
                del frame
            if 'results' in locals():
                del results
            if 'optimizer' in locals():
                del optimizer
        except NameError:
            pass  # 變數可能不存在
        
        # 強制垃圾回收
        import gc
        gc.collect()
        
        # 清理 CUDA 快取
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if show:
            cv2.destroyAllWindows()
        
        # 保存事件CSV到會話資料夾
        source_id = config.get("source_id", "camera_01")
        csv_file = os.path.join(session_dir, f"{session_id}_events.csv")
        all_events = stats["line_events"] + zone_occupancy.get_events()
        save_events_csv(all_events, csv_file, source_id)
        logger.info(f"Events CSV saved to: {csv_file}")
        
        # 停止性能監控
        try:
            stop_performance_monitoring()
            logger.info("Performance monitoring stopped")
        except Exception as e:
            logger.warning(f"Failed to stop performance monitoring: {e}")
        
        # 更新API狀態
        data_store.update_source(
            stream_id,
            str(source),
            "offline",
            0.0,
            0,
            fps_source=0.0,
            latency_ms=None,
            cpu_percent=0.0,
            memory_mb=0.0)
        
        # 結束資料庫會話
        try:
            if session_service and session_id:
                session_service.end_session(session_id)
                logger.info(f"Database session ended: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to end database session: {e}")
        
        # 關閉資料庫連接
        try:
            if db_session:
                db_session.close()
            if db_manager:
                db_manager.close()
        except Exception as e:
            logger.warning(f"Failed to close database connection: {e}")
    
    # 返回統計結果
    total_time = time.perf_counter() - start_time
    return {
        "frames": stats["frames"],
        "source_frames": stats["source_frames"],
        "people_count": stats["people_count"],
        "line_counts": stats["line_counts"],
        "line_events": len(stats["line_events"]),
        "total_time": total_time,
        "fps_avg": stats["frames"] / max(total_time, 1e-6),
        "source_fps_avg": stats["source_frames"] / max(total_time, 1e-6)
    }

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="YOLO People Detection - Optimized")
    parser.add_argument("--source", help="Video source (path/URL or camera index)")
    parser.add_argument("--config", default="configs/sample.yaml", help="Configuration file")
    parser.add_argument("--show", action="store_true", help="Show video window")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--outdir", default="runs/people_detect", help="Output directory")
    parser.add_argument("--duration", type=float, help="Duration in seconds")
    parser.add_argument("--session-id", help="Session ID for file organization (e.g., 114092801)")
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_yaml_config(args.config)
    source = args.source or config.get("default_source") or os.getenv("VIDEO_SOURCE") or "videos/sample.avi"
    
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    
    # 運行偵測
    result = run_detection(
        source=source,
        config=config,
        show=args.show,
        save=args.save,
        outdir=args.outdir,
        duration=args.duration,
        session_id=args.session_id
    )
    
    print(f"Detection completed: {result}")

if __name__ == "__main__":
    main()
