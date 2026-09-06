"""視頻流管理器 - 簡化版本"""
import cv2
import threading
import time
import numpy as np
from typing import Optional
from queue import Queue

# 嘗試導入 YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLO 模型載入中...")
    model = YOLO('yolov8n.pt')
    print("YOLO 模型載入成功!")
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: 無法載入 YOLO 模型，將只顯示原始畫面")

def detect_people(frame):
    """使用 YOLO 偵測人物"""
    if not YOLO_AVAILABLE:
        return frame
    
    try:
        results = model(frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) == 0:  # person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Person {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (0, 255, 0), -1)
                            cv2.putText(frame, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
        
    except Exception as e:
        print(f"YOLO 偵測錯誤: {e}")
        return frame

class StreamManager:
    """視頻流管理器 - 簡化版本"""
    
    def __init__(self):
        self.cameras = {
            "rtsp_camera": {
                "name": "RTSP 攝影機",
                "url": "rtsp://camera.example.invalid/stream",
                "type": "rtsp",
                "cap": None,
                "connected": False,
                "frame": None
            },
            "mjpg_camera": {
                "name": "MJPEG 攝影機", 
                "url": "rtsp://camera.example.invalid/stream",
                "type": "mjpeg",
                "cap": None,
                "connected": False,
                "frame": None
            }
        }
        
        self.current_camera = "rtsp_camera"
        self.frame_queue = Queue(maxsize=5)
        self.is_running = False
        self.capture_thread = None
        self.latest_frame = None
        self._initialized = False
        
        print("StreamManager 初始化完成")
    
    def start_camera(self, camera_id: str) -> bool:
        """啟動攝影機"""
        if camera_id not in self.cameras:
            return False
        
        camera = self.cameras[camera_id]
        if camera['connected']:
            return True
        
        print(f"正在啟動攝影機: {camera['name']}")
        
        try:
            cap = cv2.VideoCapture(camera['url'])
            
            if camera['type'] == 'rtsp':
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 10)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            elif camera['type'] == 'mjpeg':
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)
            
            if not cap.isOpened():
                print(f"無法開啟 {camera['name']}")
                return False
            
            # 測試讀取第一幀
            ret, frame = cap.read()
            if not ret:
                print(f"無法讀取 {camera['name']} 的畫面")
                cap.release()
                return False
            
            camera['cap'] = cap
            camera['connected'] = True
            camera['frame'] = frame.copy()
            
            print(f"✅ {camera['name']} 連接成功!")
            return True
            
        except Exception as e:
            print(f"啟動 {camera['name']} 時發生錯誤: {e}")
            return False
    
    def switch_camera(self, camera_id: str) -> bool:
        """切換到指定攝影機"""
        if camera_id not in self.cameras:
            return False
        
        if not self.cameras[camera_id]['connected']:
            if not self.start_camera(camera_id):
                return False
        
        self.current_camera = camera_id
        print(f"切換到攝影機: {self.cameras[camera_id]['name']}")
        return True
    
    def fast_switch_camera(self, camera_id: str) -> bool:
        """快速切換攝影機"""
        return self.switch_camera(camera_id)
    
    def start_stream(self, camera_id: Optional[str] = None) -> bool:
        """開始視頻流"""
        if camera_id and not self.switch_camera(camera_id):
            return False
        
        if self.is_running:
            return True
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print(f"開始視頻流: {self.cameras[self.current_camera]['name']}")
        return True
    
    def stop_stream(self):
        """停止視頻流"""
        self.is_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        print("視頻流已停止")
    
    def _capture_loop(self):
        """視頻捕獲循環"""
        print("視頻捕獲循環開始")
        frame_count = 0
        
        while self.is_running:
            try:
                camera = self.cameras[self.current_camera]
                if not camera['connected'] or camera['cap'] is None:
                    time.sleep(0.1)
                    continue
                
                ret, frame = camera['cap'].read()
                if ret:
                    frame_count += 1
                    
                    # 執行人物偵測
                    detected_frame = detect_people(frame.copy())
                    
                    # 更新最新幀
                    self.latest_frame = detected_frame
                    camera['frame'] = detected_frame
                    
                    # 放入隊列
                    try:
                        self.frame_queue.put_nowait(detected_frame)
                    except:
                        pass  # 隊列滿了，跳過
                    
                    if frame_count % 50 == 0:
                        print(f"{camera['name']}: 已處理 {frame_count} 幀")
                else:
                    print(f"{camera['name']} 讀取失敗")
                
                time.sleep(0.05)  # 20 FPS
                
            except Exception as e:
                print(f"捕獲循環錯誤: {e}")
                time.sleep(1)
        
        print("視頻捕獲循環結束")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """獲取最新的視頻幀"""
        # 如果還沒有初始化，先初始化攝影機
        if not self._initialized:
            self._initialize_cameras()
        
        return self.latest_frame
    
    def _initialize_cameras(self):
        """初始化攝影機（延遲初始化）"""
        if self._initialized:
            return
        
        print("開始初始化攝影機...")
        # 在背景線程中初始化，避免阻塞
        threading.Thread(target=self._init_cameras_thread, daemon=True).start()
        self._initialized = True
    
    def _init_cameras_thread(self):
        """在背景線程中初始化攝影機"""
        try:
            # 嘗試連接攝影機
            for camera_id in self.cameras:
                self.start_camera(camera_id)
        except Exception as e:
            print(f"攝影機初始化錯誤: {e}")
    
    def get_latest_frame_bytes(self) -> Optional[bytes]:
        """獲取最新的視頻幀（編碼的JPEG bytes）"""
        frame = self.latest_frame
        if frame is not None:
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    return buffer.tobytes()
            except:
                pass
        return None
    
    def get_camera_list(self) -> dict:
        """獲取攝影機列表"""
        return {
            camera_id: {
                "name": camera['name'],
                "type": camera['type'],
                "connected": camera['connected'],
                "url": camera['url']
            }
            for camera_id, camera in self.cameras.items()
        }
    
    def get_status(self) -> dict:
        """獲取系統狀態"""
        return {
            "is_running": self.is_running,
            "current_camera": self.current_camera,
            "cameras": self.get_camera_list(),
            "frame_queue_size": self.frame_queue.qsize()
        }
    
    def test_camera_connection(self, camera_id: str) -> dict:
        """測試攝影機連接"""
        if camera_id not in self.cameras:
            return {"success": False, "error": "攝影機不存在"}
        
        camera = self.cameras[camera_id]
        
        try:
            cap = cv2.VideoCapture(camera['url'])
            
            if not cap.isOpened():
                return {"success": False, "error": f"無法開啟攝影機 - URL: {camera['url']}"}
            
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                return {
                    "success": True,
                    "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                    "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                    "url": camera['url']
                }
            else:
                return {"success": False, "error": "無法讀取畫面"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

# 創建全局實例
stream_manager = StreamManager()