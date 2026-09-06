#!/usr/bin/env python3
"""
非同步處理模組
提供非同步的影像處理和檢測功能
"""

import asyncio
import cv2
import numpy as np
from typing import Optional, Callable, Any, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class AsyncFrameProcessor:
    """非同步幀處理器"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = asyncio.Queue(maxsize=10)
        self.is_running = False
        
    async def start_processing(self, 
                             frame_source: Callable[[], np.ndarray],
                             process_func: Callable[[np.ndarray], Any],
                             display_func: Optional[Callable[[np.ndarray, Any], None]] = None):
        """開始非同步處理"""
        self.is_running = True
        
        # 啟動幀捕獲任務
        capture_task = asyncio.create_task(self._capture_frames(frame_source))
        
        # 啟動處理任務
        process_task = asyncio.create_task(self._process_frames(process_func, display_func))
        
        try:
            await asyncio.gather(capture_task, process_task)
        except Exception as e:
            logger.error(f"非同步處理錯誤: {e}")
        finally:
            self.is_running = False
    
    async def _capture_frames(self, frame_source: Callable[[], np.ndarray]):
        """非同步幀捕獲"""
        while self.is_running:
            try:
                frame = frame_source()
                if frame is not None:
                    await self.processing_queue.put((frame, time.time()))
                await asyncio.sleep(0.01)  # 避免過度佔用 CPU
            except Exception as e:
                logger.error(f"幀捕獲錯誤: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_frames(self, 
                            process_func: Callable[[np.ndarray], Any],
                            display_func: Optional[Callable[[np.ndarray, Any], None]] = None):
        """非同步幀處理"""
        while self.is_running:
            try:
                frame, timestamp = await asyncio.wait_for(
                    self.processing_queue.get(), timeout=1.0
                )
                
                # 在執行緒池中執行處理函數
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, process_func, frame
                )
                
                # 顯示結果
                if display_func:
                    await loop.run_in_executor(
                        self.executor, display_func, frame, result
                    )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"幀處理錯誤: {e}")
    
    def stop(self):
        """停止處理"""
        self.is_running = False
        self.executor.shutdown(wait=True)

class AsyncDetectionPipeline:
    """非同步檢測管道"""
    
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.processor = AsyncFrameProcessor(
            max_workers=config.get("max_workers", 4)
        )
        
    async def process_stream(self, 
                           frame_source: Callable[[], np.ndarray],
                           on_detection: Optional[Callable[[Any], None]] = None):
        """處理串流"""
        
        def detection_func(frame: np.ndarray):
            """檢測函數"""
            try:
                results = self.model(frame, verbose=False)
                return self._process_detection_results(results)
            except Exception as e:
                logger.error(f"檢測錯誤: {e}")
                return None
        
        def display_func(frame: np.ndarray, result: Any):
            """顯示函數"""
            try:
                if result:
                    self._draw_detections(frame, result)
                    if on_detection:
                        on_detection(result)
                
                cv2.imshow("Async Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.processor.stop()
            except Exception as e:
                logger.error(f"顯示錯誤: {e}")
        
        await self.processor.start_processing(
            frame_source, detection_func, display_func
        )
    
    def _process_detection_results(self, results):
        """處理檢測結果"""
        if not results or len(results) == 0:
            return None
        
        result = results[0]
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                if class_id == 0:  # person
                    detections.append({
                        'bbox': box,
                        'confidence': conf,
                        'class_id': int(class_id)
                    })
        
        return {
            'detections': detections,
            'timestamp': time.time()
        }
    
    def _draw_detections(self, frame: np.ndarray, result: Dict):
        """繪製檢測結果"""
        for detection in result['detections']:
            x1, y1, x2, y2 = detection['bbox'].astype(int)
            conf = detection['confidence']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {conf:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
