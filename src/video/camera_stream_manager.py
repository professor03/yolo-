import os
import time
import threading
from pathlib import Path
from typing import Dict
from threading import Lock

import cv2
import psutil

from src.video.frame_buffer import frame_buffer
from src.api.data_store import data_store


class CameraStreamWorker(threading.Thread):
    def __init__(
        self,
        stream_id: str,
        source: str,
        frame_dir: Path,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
        consecutive_failures: int = 3,
    ) -> None:
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.source = source
        self.frame_dir = frame_dir
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.consecutive_failures = consecutive_failures
        self._stop_event = threading.Event()
        self._process = psutil.Process(os.getpid())
        self._last_frame_ts: float | None = None
        self._reconnect_count = 0
        self._failure_count = 0
        self._last_successful_frame = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        print(f"CameraStreamWorker: {self.stream_id} worker started")
        while not self._stop_event.is_set():
            # 檢查是否超過最大重連次數
            if self._reconnect_count >= self.max_reconnect_attempts:
                print(f"Camera {self.stream_id}: 達到最大重連次數 ({self.max_reconnect_attempts})，停止重連")
                self._update_offline_status("max_reconnect_exceeded")
                break
            
            cap = self._open_capture()
            if not cap or not cap.isOpened():
                self._failure_count += 1
                self._reconnect_count += 1
                print(f"Camera {self.stream_id}: 連接失敗 ({self._failure_count}/{self.consecutive_failures})")
                self._update_offline_status("connection_failed", reconnect_attempts=self._reconnect_count)
                self._wait_reconnect()
                continue

            try:
                # Reset failure counter when the stream resumes
                self._failure_count = 0
                # If we were in a reconnect loop, log and reset
                if self._reconnect_count > 0:
                    print(f"[CameraStreamWorker] {self.stream_id}: connection re-established, resetting reconnect counter")
                    if isinstance(self.source, str):
                        if self.source.startswith("rtsp://"):
                            print(f"[CameraStreamWorker] {self.stream_id}: RTSP reconnect succeeded")
                        elif self.source.startswith("http://") or self.source.startswith("https://"):
                            print(f"[CameraStreamWorker] {self.stream_id}: MJPEG reconnect succeeded")
                    self._reconnect_count = 0

                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        self._failure_count += 1
                        print(f"Camera {self.stream_id}: 讀取幀失敗 ({self._failure_count}/{self.consecutive_failures})")
                        if self._failure_count >= self.consecutive_failures:
                            self._reconnect_count += 1
                            print(f"Camera {self.stream_id}: 連續失敗次數達到 {self.consecutive_failures}，重新連接")
                            self._update_offline_status("consecutive_failures", reconnect_attempts=self._reconnect_count)
                            break
                        time.sleep(0.1)  # 短暫等待後重試
                        continue

                    # 重置失敗計數，因為成功讀取到幀
                    self._failure_count = 0
                    
                    now = time.time()
                    fps = 0.0
                    if self._last_frame_ts:
                        delta = now - self._last_frame_ts
                        if delta > 0:
                            fps = 1.0 / delta
                    self._last_frame_ts = now

                    success, buffer = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    if not success:
                        continue

                    image_bytes = buffer.tobytes()
                    frame_buffer.update(self.stream_id, image_bytes)
                    self._persist_frame(image_bytes)
                    
                    # 保存最後成功的幀，用於離線時顯示
                    self._last_successful_frame = image_bytes

                    cpu = psutil.cpu_percent(interval=None)
                    memory = self._process.memory_info().rss / (1024 * 1024)

                    data_store.update_metrics(
                        stream_id=self.stream_id,
                        fps_processing=fps,
                        fps_source=fps,
                        people_count=0,
                        latency_ms=0.0,
                        cpu_percent=cpu,
                        memory_mb=memory,
                    )
                    data_store.update_source(
                        self.stream_id,
                        self.source,
                        "online",
                        fps,
                        0,
                        fps_source=fps,
                        latency_ms=0.0,
                        cpu_percent=cpu,
                        memory_mb=memory,
                        reconnect_attempts=self._reconnect_count,  # 保持重連次數
                        last_error=None,  # 清除錯誤
                        connection_quality="excellent" if fps > 10 else "good" if fps > 5 else "poor",
                    )

            finally:
                cap.release()
                self._wait_reconnect()

    def _wait_reconnect(self) -> None:
        for _ in range(int(self.reconnect_delay * 10)):
            if self._stop_event.is_set():
                break
            time.sleep(0.1)
    
    def _update_offline_status(self, reason: str, reconnect_attempts: int = None) -> None:
        """更新離線狀態並生成黑屏幀"""
        data_store.update_source(
            self.stream_id,
            self.source,
            "offline",
            0.0,
            0,
            fps_source=0.0,
            latency_ms=None,
            cpu_percent=0.0,
            memory_mb=0.0,
            reconnect_attempts=reconnect_attempts,
            last_error=reason,
            connection_quality="poor",
        )
        
        # 生成黑屏幀或使用最後成功的幀
        if self._last_successful_frame:
            # 使用最後成功的幀
            frame_buffer.update(self.stream_id, self._last_successful_frame)
        else:
            # 生成黑屏幀
            self._generate_offline_frame(reason)
    
    def _generate_offline_frame(self, reason: str) -> None:
        """生成離線狀態的黑屏幀"""
        import numpy as np
        
        # 創建黑屏幀 (480x360)
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        
        # 添加離線訊息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)  # 白色文字
        thickness = 2
        
        # 計算文字位置（居中）
        text = f"Camera Offline"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (480 - text_size[0]) // 2
        text_y = (360 + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # 添加原因
        reason_text = f"Reason: {reason}"
        reason_size = cv2.getTextSize(reason_text, font, 0.5, 1)[0]
        reason_x = (480 - reason_size[0]) // 2
        reason_y = text_y + 30
        
        cv2.putText(frame, reason_text, (reason_x, reason_y), font, 0.5, color, 1)
        
        # 編碼為 JPEG
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
                    image_bytes = buffer.tobytes()
                    frame_buffer.update(self.stream_id, image_bytes)
                    try:
                        self._persist_frame(image_bytes)
                    except Exception as exc:
                        print(
                            f"[CameraStreamWorker] {self.stream_id}: "
                            f"failed to persist frame ({exc})"
                        )

    def _persist_frame(self, image_bytes: bytes) -> None:
        target = self.frame_dir / f"{self.stream_id}.jpg"
        tmp = target.with_suffix(".jpg.tmp")
        try:
            with tmp.open("wb") as fp:
                fp.write(image_bytes)
        except OSError as exc:
            print(
                f"[CameraStreamWorker] {self.stream_id}: "
                f"failed to write temp frame file ({exc})"
            )
            return

        replace_error = None
        for attempt in range(10):
            try:
                os.replace(tmp, target)
                return
            except PermissionError as exc:
                replace_error = exc
                time.sleep(min(0.1 * (attempt + 1), 0.5))
            except OSError as exc:
                replace_error = exc
                break

        # Fallback to direct truncation when replace keeps failing
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            fd = os.open(str(target), flags)
            with os.fdopen(fd, "wb") as fp:
                fp.write(image_bytes)
            if replace_error:
                print(
                    f"[CameraStreamWorker] {self.stream_id}: persisted frame via fallback "
                    f"after replace failure ({replace_error})"
                )
        except OSError as exc:
            print(
                f"[CameraStreamWorker] {self.stream_id}: fallback persist failed ({exc})"
            )
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except TypeError:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass

    def _open_capture(self):
        try:
            params = []
            if self.source.startswith("rtsp://") or self.source.startswith("http"):
                params = [cv2.CAP_FFMPEG]
            cap = cv2.VideoCapture(self.source, *params)
            if not cap.isOpened():
                cap.release()
                return None
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        except Exception:
            return None


class CameraStreamManager:
    def __init__(
        self,
        camera_sources: Dict[str, str],
        frame_dir: Path,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
        consecutive_failures: int = 3,
    ) -> None:
        self._workers: Dict[str, CameraStreamWorker] = {}
        self.frame_dir = frame_dir
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_attempts = max_reconnect_attempts
        self._consecutive_failures = consecutive_failures
        self._lock = Lock()
        self._running = False
        self._camera_sources: Dict[str, str] = {}
        # 先設置 sources，但不啟動 worker
        self._camera_sources = {k: v for k, v in camera_sources.items() if v}
        self._create_workers(camera_sources)

    def _create_workers(self, camera_sources: Dict[str, str]) -> None:
        """創建 worker 但不啟動"""
        camera_sources = {k: v for k, v in camera_sources.items() if v}
        with self._lock:
            for stream_id, source in camera_sources.items():
                self._workers[stream_id] = CameraStreamWorker(
                    stream_id=stream_id,
                    source=source,
                    frame_dir=self.frame_dir,
                    reconnect_delay=self._reconnect_delay,
                    max_reconnect_attempts=self._max_reconnect_attempts,
                    consecutive_failures=self._consecutive_failures,
                )

    def start_all(self) -> None:
        with self._lock:
            self._running = True
            print(f"CameraStreamManager: Starting {len(self._workers)} workers")
            for stream_id, worker in self._workers.items():
                if not worker.is_alive():
                    worker.start()
                    print(f"CameraStreamManager: Started worker for {stream_id}")
                else:
                    print(f"CameraStreamManager: Worker for {stream_id} already running")

    def stop_all(self) -> None:
        with self._lock:
            self._running = False
            for worker in self._workers.values():
                worker.stop()
            for worker in self._workers.values():
                if worker.is_alive():
                    worker.join(timeout=2.0)

    def get_sources(self) -> Dict[str, str]:
        return {sid: worker.source for sid, worker in self._workers.items()}

    def reload_sources(self, camera_sources: Dict[str, str]) -> None:
        camera_sources = {k: v for k, v in camera_sources.items() if v}
        with self._lock:
            current_ids = set(self._workers.keys())
            new_ids = set(camera_sources.keys())

            # Stop removed cameras
            for stream_id in current_ids - new_ids:
                worker = self._workers.pop(stream_id)
                worker.stop()
                if worker.is_alive():
                    worker.join(timeout=2.0)
                data_store.update_source(
                    stream_id,
                    camera_sources.get(stream_id, worker.source),
                    "offline",
                    0.0,
                    0,
                    fps_source=0.0,
                    latency_ms=None,
                    cpu_percent=0.0,
                    memory_mb=0.0,
                )

            # Update existing or add new workers
            for stream_id, source in camera_sources.items():
                if stream_id in self._workers:
                    worker = self._workers[stream_id]
                    if worker.source == source:
                        continue
                    worker.stop()
                    if worker.is_alive():
                        worker.join(timeout=2.0)
                    self._workers[stream_id] = CameraStreamWorker(
                        stream_id=stream_id,
                        source=source,
                        frame_dir=self.frame_dir,
                        reconnect_delay=self._reconnect_delay,
                        max_reconnect_attempts=self._max_reconnect_attempts,
                        consecutive_failures=self._consecutive_failures,
                    )
                else:
                    self._workers[stream_id] = CameraStreamWorker(
                        stream_id=stream_id,
                        source=source,
                        frame_dir=self.frame_dir,
                        reconnect_delay=self._reconnect_delay,
                        max_reconnect_attempts=self._max_reconnect_attempts,
                        consecutive_failures=self._consecutive_failures,
                    )

                if self._running:
                    worker = self._workers[stream_id]
                    if not worker.is_alive():
                        worker.start()

            self._camera_sources = camera_sources.copy()
