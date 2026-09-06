#!/usr/bin/env python3
"""
整合版即時串流人流偵測
整合現有的 people_detect.py 功能到即時串流中
"""

import argparse
from pathlib import Path
import cv2
import time
import threading
import queue
import numpy as np
from typing import Optional, Union, List, Dict, Any
import os
import sys
import psutil

# 添加專案路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.common import load_yaml_config, setup_logging
from src.logic.counter import MultiLineCounter, LineConfig, TrackInfo as CounterTrackInfo
from src.logic.calibration import HomographyCalibrator, CalibrationConfig
from src.api.data_store import data_store
from src.video.frame_buffer import frame_buffer
from scripts.database.anomaly_detection import AnomalyDetector
FRAME_DIR = Path(os.getenv("FRAME_OUTPUT_DIR", "data/frames"))
FRAME_DIR.mkdir(parents=True, exist_ok=True)

def save_frame_to_disk(stream_id: str, image_bytes: bytes) -> None:
    target = FRAME_DIR / f"{stream_id}.jpg"
    tmp = target.with_suffix(".jpg.tmp")
    with tmp.open("wb") as f:
        f.write(image_bytes)
    tmp.replace(target)

from ultralytics import YOLO

class RealtimeDetectionSystem:
    """即時人流偵測系統"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_yaml_config(config_path)
        self.logger = setup_logging()

        # 初始化模型
        self.model = None
        self.tracker = None
        self.allowed_classes = self._normalize_classes(self.config.get("allowed_classes", [0]))
        
        # 初始化計數器
        self.line_counter = None
        self.calibrator = None
        self.anomaly_detector = None
        
        # 串流狀態
        self.is_running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.stream_id = self.config.get("source_id", "camera_01")
        
        # 統計資訊
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_count = 0
        
        # 初始化系統
        self._initialize_system()

    @staticmethod
    def _normalize_classes(raw_value):
        if raw_value is None:
            return None
        if isinstance(raw_value, (int, float, str)):
            try:
                return [int(raw_value)]
            except (TypeError, ValueError):
                return None
        normalized = []
        for item in raw_value:
            try:
                normalized.append(int(item))
            except (TypeError, ValueError):
                continue
        return normalized or None

    def _initialize_system(self):
        """初始化系統組件"""
        try:
            # 載入 YOLO 模型
            model_path = self.config.get("model", {}).get("weights", "yolov8n.pt")
            self.model = YOLO(model_path)

            # 設定設備
            device = self.config.get("model", {}).get("device", "")
            if device == "":
                device = "cuda" if hasattr(self.model, 'cuda') and self.model.cuda else "cpu"
            
            self.model.to(device)
            self.logger.info(f"YOLO 模型載入成功，使用設備: {device}")
            self.logger.info("允許的偵測類別: %s", self.allowed_classes if self.allowed_classes is not None else "all")
            
            # 初始化軌跡追蹤
            if self.config.get("track", True):
                self.tracker = self.model.track
                self.logger.info("軌跡追蹤已啟用")
            
            # 初始化計數器
            self._setup_counting_system()
            
            # 初始化校準系統
            self._setup_calibration_system()
            
            # 初始化異常檢測
            self.anomaly_detector = AnomalyDetector()
            
            self.logger.info("系統初始化完成")
            
        except Exception as e:
            self.logger.error(f"系統初始化失敗: {e}")
            raise
    
    def _setup_counting_system(self):
        """設定計數系統"""
        try:
            # 從配置載入線段設定
            lines_config = self.config.get("lines", [])
            if lines_config:
                line_configs = []
                for line_data in lines_config:
                    line_config = LineConfig(
                        id=line_data["id"],
                        points=line_data["points"],
                        direction=line_data.get("direction", "both"),
                        uturn_cooldown=line_data.get("uturn_cooldown", 30),
                        jitter_frames=line_data.get("jitter_frames", 2)
                    )
                    line_configs.append(line_config)
                
                self.line_counter = MultiLineCounter(line_configs)
                self.logger.info(f"計數系統已設定，共 {len(line_configs)} 條線段")
            else:
                self.logger.warning("未找到線段配置，計數功能將被禁用")
                
        except Exception as e:
            self.logger.error(f"計數系統設定失敗: {e}")
    
    def _setup_calibration_system(self):
        """設定校準系統"""
        try:
            calib_config = self.config.get("calibration", {})
            if calib_config.get("enabled", False):
                calib_points = calib_config.get("points", [])
                if len(calib_points) >= 4:
                    calib_config_obj = CalibrationConfig(
                        points=calib_points,
                        pixel_size=calib_config.get("pixel_size", 1.0),
                        world_size=calib_config.get("world_size", 1.0)
                    )
                    self.calibrator = HomographyCalibrator(calib_config_obj)
                    self.logger.info("校準系統已設定")
                else:
                    self.logger.warning("校準點不足，校準功能將被禁用")
            else:
                self.logger.info("校準功能已禁用")
                
        except Exception as e:
            self.logger.error(f"校準系統設定失敗: {e}")
    
    def setup_camera(self, source: Union[str, int]) -> bool:
        """設定攝影機來源"""
        try:
            self.logger.info(f"嘗試連接攝影機: {source}")
            self.cap = cv2.VideoCapture(source)
            
            if self.cap.isOpened():
                # 設定攝影機參數
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 減少延遲
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                # 獲取攝影機資訊
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                self.logger.info(f"攝影機連接成功: {width}x{height} @ {fps:.1f}fps")
                return True
            else:
                self.logger.error(f"無法連接攝影機: {source}")
                return False
                
        except Exception as e:
            self.logger.error(f"攝影機設定失敗: {e}")
            return False
    
    def frame_capture_thread(self):
        """幀捕獲執行緒"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # 計算 FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                # 將幀放入佇列
                try:
                    self.frame_queue.put_nowait((frame.copy(), current_time, self.frame_count))
                    self.frame_count += 1
                except queue.Full:
                    # 佇列滿了，丟棄舊幀
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((frame.copy(), current_time, self.frame_count))
                        self.frame_count += 1
                    except queue.Empty:
                        pass
            else:
                self.logger.warning("無法讀取攝影機幀")
                time.sleep(0.1)
    
    def detection_thread_func(self):
        """檢測執行緒"""
        while self.is_running:
            try:
                frame, timestamp, frame_number = self.frame_queue.get(timeout=1.0)
                
                # 執行檢測
                self._process_frame(frame, timestamp, frame_number)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"檢測執行緒錯誤: {e}")
    
    def _process_frame(self, frame: np.ndarray, timestamp: float, frame_number: int):
        """處理單一幀"""
        try:
            # 檢查幀是否有效
            if frame is None or frame.size == 0:
                self.logger.warning("收到無效幀，跳過處理")
                return
            
            # 執行 YOLO 檢測
            if self.allowed_classes is not None:
                results = self.model(frame, verbose=False, classes=self.allowed_classes)
            else:
                results = self.model(frame, verbose=False)
            
            # 處理檢測結果
            detections = []
            tracks = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    target_classes = self.allowed_classes or [0]
                    match_mask = np.isin(class_ids, target_classes)
                    person_boxes = boxes[match_mask]
                    person_confs = confidences[match_mask]
                    person_class_ids = class_ids[match_mask]

                    # 建立檢測結果
                    for box, conf, class_id in zip(person_boxes, person_confs, person_class_ids):
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        detection = {
                            'bbox': box,
                            'confidence': conf,
                            'class_id': int(class_id),
                            'center': (center_x, center_y)
                        }
                        detections.append(detection)
                
                # 執行軌跡追蹤
                if self.tracker and len(detections) > 0:
                    track_results = self.tracker(frame, persist=True, verbose=False)
                    if track_results and len(track_results) > 0:
                        track_result = track_results[0]
                        if hasattr(track_result, 'boxes') and track_result.boxes is not None:
                            if hasattr(track_result.boxes, 'id') and track_result.boxes.id is not None:
                                track_ids = track_result.boxes.id.cpu().numpy()
                                track_boxes = track_result.boxes.xyxy.cpu().numpy()
                                
                                for i, (track_id, box) in enumerate(zip(track_ids, track_boxes)):
                                    x1, y1, x2, y2 = box
                                    center_x = (x1 + x2) / 2
                                    center_y = (y1 + y2) / 2
                                    
                                    track_info = CounterTrackInfo(
                                        id=int(track_id),
                                        center=(center_x, center_y),
                                        bbox=box,
                                        score=1.0,
                                        timestamp=timestamp
                                    )
                                    tracks.append(track_info)
            
            # 執行計數
            if self.line_counter and tracks:
                events = self.line_counter.update(tracks, frame_number)
                if events:
                    for event in events:
                        self.logger.info(f"計數事件: {event}")
            
            # 執行異常檢測
            if self.anomaly_detector:
                anomaly_events = self.anomaly_detector.update(tracks, timestamp, len(tracks))
                if anomaly_events:
                    for event in anomaly_events:
                        self.logger.info(f"異常事件: {event}")
            
            # 更新資料存儲
            # 更新資料彙總
            data_store.add_detection_batch(detections)
            cpu_percent = psutil.cpu_percent()
            memory_mb = psutil.virtual_memory().used / 1024 / 1024
            data_store.update_metrics(
                stream_id=self.stream_id,
                fps_processing=self.current_fps,
                fps_source=self.current_fps,  # 假設來源和處理 FPS 相同
                people_count=len(detections),
                latency_ms=1000.0 / self.current_fps if self.current_fps > 0 else 0,  # 估計延遲
                cpu_percent=cpu_percent,
                memory_mb=memory_mb
            )
            data_store.update_source(
                self.stream_id,
                str(getattr(self, "source", self.config.get("default_source", self.stream_id))),
                "online",
                self.current_fps,
                len(detections),
                fps_source=self.current_fps,
                latency_ms=1000.0 / self.current_fps if self.current_fps > 0 else 0,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb
            )
            # 顯示結果
            self._display_frame(frame, detections, tracks)
            
        except Exception as e:
            self.logger.error(f"幀處理錯誤: {e}")
    
    def _display_frame(self, frame: np.ndarray, detections: List[Dict], tracks: List[CounterTrackInfo]):
        """顯示處理後的影像 (僅保留偵測框)."""
        try:
            for detection in detections:
                bbox = detection.get('bbox')
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            success, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                frame_bytes = encoded_frame.tobytes()
                frame_buffer.update(self.stream_id, frame_bytes)
                save_frame_to_disk(self.stream_id, frame_bytes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False

        except Exception as e:
            self.logger.error(f'顯示影像時發生錯誤: {e}')

    def start_detection(self, source: Union[str, int]):
        """開始即時偵測"""
        if not self.setup_camera(source):
            self.logger.error("攝影機設定失敗")
            return False
        
        self.source = source
        self.stream_id = self.config.get("source_id", str(source))
        self.is_running = True
        
        # 啟動幀捕獲執行緒
        capture_thread = threading.Thread(target=self.frame_capture_thread)
        capture_thread.daemon = True
        capture_thread.start()
        
        # 啟動檢測執行緒
        detection_thread = threading.Thread(target=self.detection_thread_func)
        detection_thread.daemon = True
        detection_thread.start()
        
        self.logger.info("即時人流偵測已啟動")
        
        try:
            # 主執行緒等待
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("收到中斷信號，正在停止...")
        finally:
            self.stop_detection()
    
    def stop_detection(self):
        """停止即時偵測"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.logger.info("即時人流偵測已停止")

def main():
    parser = argparse.ArgumentParser(description="即時人流偵測系統")
    parser.add_argument("--source", required=True, 
                       help="輸入來源 (RTSP URL, USB 設備 ID, 或影片檔案)")
    parser.add_argument("--config", default="config.yaml", help="配置檔案路徑")
    
    args = parser.parse_args()
    
    # 判斷輸入來源類型
    source = args.source
    if source.isdigit():
        source = int(source)  # USB 攝影機
    elif source.startswith(('rtsp://', 'http://', 'https://')):
        source = source  # 網路串流
    else:
        source = source  # 檔案路徑
    
    # 建立偵測系統
    detector = RealtimeDetectionSystem(args.config)
    
    # 開始偵測
    detector.start_detection(source)

if __name__ == "__main__":
    main()
