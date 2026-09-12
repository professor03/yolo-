"""優化的FastAPI應用 - 整合所有API功能"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import os, json, asyncio, logging, cv2, time, numpy as np
import csv
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from pathlib import Path
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy.orm import Session
from pydantic import BaseModel, constr, Field

# ===== 匯入 =====
from src.api.data_store import data_store
from src.api.models import (
    SourceStatus, Metrics, LineCount, Alert, WebSocketMessage,
    MetricsResponse, CountsResponse, AlertsResponse, SourcesResponse
)
# Phase B 新增的 API 模組
from src.api.admin_routes import router as admin_router
from src.api.export_routes import router as export_router
from src.learnsight.routes import router as learnsight_router
# 導入現有的認證系統
from src.auth.security import SecurityManager, get_security_manager
from src.server.health import get_health_status, get_detailed_health_status
from src.utils.logging_config import setup_default_logging, api_logger
from src.database.models import get_database_manager
from src.database.dependencies import get_db_session
from src.video.frame_buffer import frame_buffer
from src.database.services import (
    DetectionEventService, LineCountService, ZoneOccupancyService,
    SystemMetricsService, AlertService, SessionService, StatisticsService
)
from src.video.camera_stream_manager import CameraStreamManager
from src.database.user_repository import UserRepository
from src.database.camera_repository import CameraRepository

logger = logging.getLogger(__name__)

# ===== Prometheus metrics =====
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
PEOPLE_COUNT = Gauge('people_count_total', 'Current number of people detected')
FPS_PROCESSING = Gauge('fps_processing', 'Processing FPS')
FPS_SOURCE = Gauge('fps_source', 'Source FPS')
FPS_INFERENCE = Gauge('fps_inference', 'Inference FPS')
LATENCY_MS = Gauge('latency_ms', 'Processing latency in milliseconds')
CPU_PERCENT = Gauge('cpu_percent', 'CPU usage percentage')
MEMORY_MB = Gauge('memory_mb', 'Memory usage in MB')
GPU_UTILIZATION = Gauge('gpu_utilization', 'GPU utilization percentage')
GPU_MEMORY_GB = Gauge('gpu_memory_gb', 'GPU memory usage in GB')
QUEUE_SIZE = Gauge('queue_size', 'Frame queue size')
DROPPED_FRAMES = Counter('dropped_frames_total', 'Total dropped frames')
LINE_COUNTS = Gauge('line_counts', 'Line crossing counts', ['line_id', 'direction'])
ZONE_OCCUPANCY = Gauge('zone_occupancy', 'Zone occupancy', ['zone_id'])

# 模板已移除 - API 專用服務

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAME_DIR = PROJECT_ROOT / "data" / "frames"
FRAME_DIR = Path(os.getenv("FRAME_OUTPUT_DIR", str(DEFAULT_FRAME_DIR))).resolve()
FRAME_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HLS_DIR = PROJECT_ROOT / "hls"
HLS_DIR = Path(os.getenv("HLS_OUTPUT_DIR", str(DEFAULT_HLS_DIR))).resolve()
HLS_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_SOURCES = {
    "rtsp_camera": os.getenv(
        "RTSP_CAMERA_URL",
        "",
    ),
    "mjpg_camera": os.getenv(
        "MJPG_CAMERA_URL",
        "",
    ),
}
# ==================================================
# 輔助函數
# ==================================================
def _sync_camera_sources_to_db(camera_repository):
    """同步 CAMERA_SOURCES 配置到資料庫"""
    try:
        for slug, source_url in CAMERA_SOURCES.items():
            if not source_url:
                continue
                
            # 檢查攝影機是否已存在
            existing_camera = camera_repository.get_camera(slug)
            if not existing_camera:
                # 判斷協議類型
                protocol = "rtsp" if source_url.startswith("rtsp://") else "mjpeg"
                name = f"{'RTSP' if protocol == 'rtsp' else 'MJPEG'} Camera"
                
                # 創建新攝影機
                camera_repository.create_camera(
                    slug=slug,
                    name=name,
                    protocol=protocol,
                    source_url=source_url,
                    enabled=True,
                    description=f"Auto-synced {protocol.upper()} camera",
                    owner_id=None
                )
                logger.info("Synced camera to database: %s", slug)
            else:
                # 更新現有攝影機的 source_url（如果不同）
                if existing_camera.get('source_url') != source_url:
                    camera_repository.update_camera(
                        slug=slug,
                        source_url=source_url
                    )
                    logger.info("Updated camera source for: %s", slug)
    except Exception as exc:
        logger.warning(f"Failed to sync camera sources to database: {exc}")

# ==================================================
# 建立 App
# ==================================================
def create_app() -> FastAPI:
    # 使用現有的 SecurityManager
    security_manager = get_security_manager()
    db_manager = get_database_manager()
    user_repository = UserRepository(db_manager)
    camera_repository = CameraRepository(db_manager)
    disable_camera_streams = os.getenv("DISABLE_CAMERA_STREAMS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    try:
        db_manager.create_tables()
    except Exception as exc:
        logger.warning("Database initialization failed: %s", exc)

    # 同步 CAMERA_SOURCES 到資料庫
    if not disable_camera_streams:
        _sync_camera_sources_to_db(camera_repository)

    initial_sources = {} if disable_camera_streams else camera_repository.get_enabled_camera_sources()
    if not initial_sources and not disable_camera_streams:
        initial_sources = {k: v for k, v in CAMERA_SOURCES.items() if v}

    camera_manager = CameraStreamManager(
        initial_sources,
        FRAME_DIR,
        reconnect_delay=float(os.getenv("CAMERA_RECONNECT_DELAY", "5")),
    )

    def _reload_camera_streams() -> None:
        if disable_camera_streams:
            return
        sources = camera_repository.get_enabled_camera_sources()
        camera_manager.reload_sources(sources)

    # 定義 lifespan 事件處理器
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 啟動時執行
        if disable_camera_streams:
            print("Camera streams disabled by configuration.")
        else:
            print("Starting camera manager...")
            camera_manager.start_all()
        yield
        # 關閉時執行
        if not disable_camera_streams:
            print("Stopping camera manager...")
            camera_manager.stop_all()
    
    # 創建 FastAPI 實例並設置 lifespan
    app = FastAPI(
        title="YOLO People Detection API",
        description="人流偵測系統API - 優化版本",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # 設置 camera_manager
    app.state.camera_manager = camera_manager
    # ===== CORS =====
    cors_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8080,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8787,http://127.0.0.1:8787",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # ===== API 路由優先定義 =====
    @app.get("/api")
    async def api_info():
        """API 服務資訊"""
        return {
            "service": "YOLO People Detection API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/healthz",
                "docs": "/docs",
                "metrics": "/api/v1/metrics",
                "cameras": "/api/v1/cameras",
                "auth": "/auth/login",
                "video_feed": "/video_feed?camera=<camera_slug>",
                "login_page": "/login.html",
                "dashboard": "/index.html"
            }
        }
    # ===== 認證依賴 =====
    def get_current_user(authorization: str = Header(None)):
        if not authorization:
            raise HTTPException(status_code=401, detail="未提供認證令牌")
        
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="認證令牌格式錯誤")
        
        token = authorization.split(" ")[1]
        if not token:
            raise HTTPException(status_code=401, detail="認證令牌為空")

        payload = security_manager.verify_token(token, "access")
        if not payload:
            raise HTTPException(status_code=401, detail="無效的認證令牌")

        return payload

    def _user_has_permission(user_payload: Dict[str, Any], permission: str) -> bool:
        if not user_payload or not permission:
            return False
        user_permissions = user_payload.get("permissions") or []
        if isinstance(user_permissions, list) and permission in user_permissions:
            return True
        role = user_payload.get("role")
        return security_manager.check_permission(role, permission)

    def _ensure_detection_permission(user: dict):
        """確保用戶有偵測控制權限"""
        if not _user_has_permission(user, "detection_control"):
            raise HTTPException(
                status_code=403,
                detail="沒有權限執行偵測控制操作"
            )

    def _collect_camera_payload(include_sensitive: bool = False) -> Dict[str, Any]:
        configs = camera_repository.list_cameras(include_source_url=include_sensitive)
        source_metrics = data_store.get_sources()
        metrics_map: Dict[str, Dict[str, Any]] = {}
        for item in source_metrics:
            if not isinstance(item, dict):
                continue
            stream_id = str(item.get("source_id") or item.get("slug") or "").strip()
            if stream_id:
                metrics_map[stream_id] = item

        cameras: List[Dict[str, Any]] = []
        last_seen_dt: Optional[datetime] = None
        online_count = 0

        for config in configs:
            slug = config.get("slug")
            metrics = metrics_map.get(slug, {})
            status_str = str(metrics.get("status", "")).lower()
            online = status_str == "online"
            if online:
                online_count += 1

            last_seen_raw = metrics.get("last_seen")
            last_seen_value: Optional[datetime] = None
            if isinstance(last_seen_raw, datetime):
                last_seen_value = last_seen_raw
            elif isinstance(last_seen_raw, str):
                try:
                    last_seen_value = datetime.fromisoformat(last_seen_raw)
                except ValueError:
                    last_seen_value = None
            if last_seen_value and (last_seen_dt is None or last_seen_value > last_seen_dt):
                last_seen_dt = last_seen_value

            hls_relative_path = None
            hls_available = False
            if slug:
                hls_relative_path = f"/hls/{slug}/stream.m3u8"
                if (HLS_DIR / slug / "stream.m3u8").exists():
                    hls_available = True

            camera_entry = {
                "slug": slug,
                "name": config.get("name"),
                "protocol": config.get("protocol"),
                "enabled": config.get("enabled", True),
                "description": config.get("description"),
                "created_at": _serialize_datetime(config.get("created_at")),
                "updated_at": _serialize_datetime(config.get("updated_at")),
                "last_tested_at": _serialize_datetime(config.get("last_tested_at")),
                "last_test_status": config.get("last_test_status"),
                "last_test_message": config.get("last_test_message"),
                "status": {
                    "online": online,
                    "status": status_str or ("online" if online else "offline"),
                    "last_seen": _serialize_datetime(last_seen_value) if last_seen_value else last_seen_raw,
                    "fps_processing": metrics.get("fps_processing"),
                    "fps_source": metrics.get("fps_source"),
                    "latency_ms": metrics.get("latency_ms"),
                    "cpu_percent": metrics.get("cpu_percent"),
                    "memory_mb": metrics.get("memory_mb"),
                    "people_count": metrics.get("people_count"),
                    "connection_quality": metrics.get("connection_quality"),
                    "reconnect_attempts": metrics.get("reconnect_attempts"),
                    "last_error": metrics.get("last_error"),
                    "first_seen": _serialize_datetime(metrics.get("first_seen")),
                    "last_online": _serialize_datetime(metrics.get("last_online")),
                    "total_uptime": metrics.get("total_uptime"),
                },
                "hls_url": hls_relative_path,
                "hls_available": hls_available,
            }

            if include_sensitive:
                camera_entry["source_url"] = config.get("source_url")

            cameras.append(camera_entry)

        status_payload = {
            "is_running": any(cam["status"]["online"] for cam in cameras),
            "available_cameras": len([cam for cam in cameras if cam.get("enabled")]),
            "online_cameras": online_count,
        }
        if last_seen_dt:
            status_payload["last_update"] = last_seen_dt.isoformat()

        return {"cameras": cameras, "status": status_payload}

    def _collect_metrics_history(minutes: int = 60) -> List[Dict[str, Any]]:
        history = data_store.get_metrics_history(minutes=minutes) or []
        if not history:
            current = data_store.get_current_metrics()
            if current:
                history = [current]
        return history
    # 網頁端點已移除 - 只提供 API 服務
    # ===== 健康檢查 =====
    @app.get("/healthz")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    @app.get("/health")
    async def detailed_health():
        return get_detailed_health_status()
    # ===== Demo Video Feed =====
    def _gen_cam_frames(source: str):
        cap = None
        frame_count = 0
        consecutive_failures = 0
        max_failures = 5
        
        while True:
            try:
                # 如果沒有連接或連接失敗，嘗試重新連接
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    
                    # 根據攝影機類型使用不同的設定
                    if 'rtsp://' in source:
                        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # RTSP 需要更長時間
                        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 減少緩衝區
                    elif 'mjpg' in source or 'mjpeg' in source:
                        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # MJPEG 特定設定
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                    else:
                        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                    
                    if not cap.isOpened():
                        consecutive_failures += 1
                        # 連接失敗，顯示錯誤畫面
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, f"無法連接攝影機", (50, 200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, f"來源: {source[:50]}...", (50, 250), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"重試中... ({consecutive_failures})", (50, 300), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        _, buffer = cv2.imencode('.jpg', frame)
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        time.sleep(min(2 + consecutive_failures, 10))  # 漸進式重試間隔
                        continue
                    else:
                        consecutive_failures = 0  # 重置失敗計數
                
                # 嘗試讀取畫面
                success, frame = cap.read()
                if not success or frame is None:
                    consecutive_failures += 1
                    # 讀取失敗，顯示錯誤畫面
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"無法讀取畫面", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"重試中... ({consecutive_failures})", (50, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    # 如果連續失敗太多次，重新建立連接
                    if consecutive_failures >= max_failures:
                        print(f"連續失敗 {consecutive_failures} 次，重新建立連接...")
                        if cap is not None:
                            cap.release()
                            cap = None
                        consecutive_failures = 0
                        time.sleep(2)
                    else:
                        time.sleep(0.5)
                    continue
                
                # 成功讀取畫面，重置失敗計數
                consecutive_failures = 0
                
                # 檢查畫面是否為全黑（MJPEG 常見問題）
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if np.mean(gray) < 10:  # 如果平均亮度太低，可能是黑畫面
                    print(f"檢測到黑畫面，跳過此幀...")
                    time.sleep(0.1)
                    continue
                
                # 添加時間戳和攝影機資訊
                camera_type = "RTSP" if 'rtsp://' in source else "MJPEG" if 'mjpg' in source else "CAM"
                cv2.putText(frame, f"時間: {time.strftime('%H:%M:%S')}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"幀數: {frame_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"類型: {camera_type}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 根據攝影機類型調整品質和幀率
                if 'mjpg' in source or 'mjpeg' in source:
                    # MJPEG 攝影機使用較低品質以保持穩定性
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    time.sleep(0.2)  # 5 FPS for MJPEG
                else:
                    # RTSP 攝影機使用較高品質
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    time.sleep(0.1)  # 10 FPS for RTSP
                
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                frame_count += 1
                
            except Exception as e:
                print(f"視頻流錯誤: {e}")
                consecutive_failures += 1
                # 顯示錯誤畫面
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"錯誤: {str(e)[:50]}", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"重試中... ({consecutive_failures})", (50, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(min(1 + consecutive_failures, 5))
    @app.get("/video_feed")
    def video_feed(request: Request):
        if disable_camera_streams:
            raise HTTPException(status_code=503, detail="Camera streams are disabled.")
        camera = request.query_params.get("camera", "default")
        stream_id = camera or "default"

        source = CAMERA_SOURCES.get(stream_id)

        if not source:
            # 預設產生測試畫面供前端顯示
            def generate_frames():
                frame_count = 0
                while True:
                    try:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "YOLO 人流偵測系統", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"時間: {time.strftime('%H:%M:%S')}", (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        center_x = int(320 + 200 * np.sin(frame_count * 0.1))
                        center_y = int(240 + 100 * np.cos(frame_count * 0.1))
                        cv2.circle(frame, (center_x, center_y), 30, (0, 0, 255), -1)
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        frame_count += 1
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"視頻流錯誤: {e}")
                        time.sleep(1)
            return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

        def generate():
            fallback_iter = None
            frame_path = FRAME_DIR / f"{stream_id}.jpg"
            last_buffer_timestamp = 0.0

            while True:
                buffer_frame = frame_buffer.get(stream_id)
                if buffer_frame and buffer_frame.timestamp > last_buffer_timestamp:
                    last_buffer_timestamp = buffer_frame.timestamp
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer_frame.image + b'\r\n')
                    time.sleep(0.03)
                    continue

                if frame_path.exists():
                    try:
                        image = frame_path.read_bytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                        time.sleep(0.05)
                        continue
                    except Exception as exc:
                        logger.warning("Failed to read frame for %s: %s", stream_id, exc)

                if fallback_iter is None:
                    fallback_iter = _gen_cam_frames(source)
                try:
                    yield next(fallback_iter)
                except StopIteration:
                    fallback_iter = None
                    time.sleep(0.1)
                except Exception as exc:
                    logger.warning("Stream generator error for %s: %s", stream_id, exc)
                    fallback_iter = None
                    time.sleep(0.5)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
    # ===== 認證端點 =====
    @app.post("/auth/login")
    async def login(request: dict):
        """用戶登入"""
        username = request.get("username")
        password = request.get("password")

        user_data = security_manager.authenticate_user(username, password)
        if not user_data:
            raise HTTPException(status_code=401, detail="用戶名或密碼錯誤")

        access_token = security_manager.create_access_token(user_data)
        refresh_token = security_manager.create_refresh_token(user_data)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": security_manager.access_token_expire_minutes * 60,
            "user": user_data,
        }

    @app.post("/auth/refresh")
    async def refresh_tokens(request: dict):
        refresh_token = request.get("refresh_token")
        if not refresh_token:
            raise HTTPException(status_code=400, detail="缺少 refresh token")

        refreshed = security_manager.refresh_access_token(refresh_token)
        if not refreshed:
            raise HTTPException(status_code=401, detail="refresh token 無效或已過期")

        return {
            "access_token": refreshed["access_token"],
            "refresh_token": refreshed["refresh_token"],
            "token_type": "bearer",
            "expires_in": refreshed["expires_in"],
            "user": refreshed["user"],
        }

    @app.post("/auth/logout")
    async def logout_endpoint(user: dict = Depends(get_current_user)):
        """用戶登出（前端負責清除 token）"""
        return {
            "message": "Logout successful",
            "username": user.get("username"),
            "status": "success"
        }
    @app.get("/auth/verify")
    async def verify_token(user: dict = Depends(get_current_user)):
        """驗證令牌"""
        return user
    @app.get("/auth/me")
    async def get_current_user_info(user: dict = Depends(get_current_user)):
        """獲取當前用戶資訊"""
        return user
    # ===== API 端點 =====
    def _to_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    def _to_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default
    def _serialize_datetime(value):
        if isinstance(value, datetime):
            return value.isoformat()
        return value
    def _json_default(value):
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value
    def _default_metrics():
        return {
            "people_count": 0,
            "fps_processing": 0.0,
            "fps_source": 0.0,
            "latency_ms": 0.0,
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
            "stream_count": 0,
            "streams": {},
        }
    @app.get("/api/v1/metrics")
    async def get_metrics(user: dict = Depends(get_current_user)):
        """即時性能數據（若無真實資料則提供假資料）。"""
        metrics = data_store.get_current_metrics()
        if metrics and (metrics.get("is_mock") or metrics.get("mock")):
            metrics = None

        if not metrics:
            return _default_metrics()

        response = _default_metrics()
        response.update(
            {
                "people_count": _to_int(metrics.get("people_count")),
                "fps_processing": _to_float(metrics.get("fps_processing")),
                "fps_source": _to_float(metrics.get("fps_source")),
                "latency_ms": _to_float(metrics.get("latency_ms")),
                "cpu_percent": _to_float(metrics.get("cpu_percent")),
                "memory_mb": _to_float(metrics.get("memory_mb")),
            }
        )
        timestamp = metrics.get("timestamp")
        if timestamp:
            response["timestamp"] = _serialize_datetime(timestamp)
        streams = metrics.get("streams")
        if streams:
            response["streams"] = streams
            response["stream_count"] = metrics.get("stream_count", len(streams))
        return response
    @app.websocket("/ws/stream")
    async def stream_updates(websocket: WebSocket):
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=1008)
            return
        payload = security_manager.verify_token(token, "access")
        if not payload:
            await websocket.close(code=1008)
            return

        await websocket.accept()
        try:
            while True:
                metrics = data_store.get_current_metrics() or _default_metrics()
                sources = data_store.get_sources()
                camera_status = {
                    "is_running": any(
                        isinstance(src, dict)
                        and str(src.get("status", "")).lower() == "online"
                        for src in (sources or [])
                    ),
                    "available_cameras": len(sources or []),
                }
                message = {
                    "metrics": metrics,
                    "status": camera_status,
                }
                await websocket.send_text(json.dumps(message, default=_json_default))
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("WebSocket stream error: %s", exc)
        finally:
            try:
                await websocket.close()
            except Exception:
                pass
    @app.get("/api/v1/counts")
    async def get_counts(user: dict = Depends(get_current_user)):
        """偵測計數數據"""
        snapshot = data_store.get_dashboard_data()
        metrics = snapshot.get("current_metrics") or {}
        line_counts = snapshot.get("line_counts", [])
        zone_occupancy = snapshot.get("zone_occupancy") or data_store.get_zone_occupancy()
        line_counts_payload = []
        for line in line_counts:
            if not isinstance(line, dict):
                continue
            total = _to_int(line.get("total", line.get("count", 0)))
            in_count = _to_int(line.get("in"))
            out_count = _to_int(line.get("out"))
            line_counts_payload.append(
                {
                    "line_id": line.get("line_id"),
                    "count": total,
                    "in": in_count,
                    "out": out_count,
                    "last_updated": _serialize_datetime(line.get("last_updated")),
                }
            )
        return {
            "total_events": _to_int(snapshot.get("total_events")),
            "current_count": _to_int(metrics.get("people_count")),
            "line_counts": line_counts_payload,
            "zone_occupancy": zone_occupancy,
        }
    @app.get("/api/v1/alerts")
    async def get_alerts(user: dict = Depends(get_current_user)):
        """偵測警報資訊"""
        alerts = data_store.get_alerts(limit=100)
        serialized_alerts = []
        for idx, alert in enumerate(alerts):
            if not isinstance(alert, dict):
                continue
            serialized_alerts.append(
                {
                    "id": alert.get("id") or f"alert_{idx}",
                    "timestamp": _serialize_datetime(alert.get("timestamp")),
                    "type": alert.get("type", "system"),
                    "message": alert.get("message", ""),
                    "severity": alert.get("level", "info"),
                    "source_id": alert.get("source_id", "system"),
                }
            )
        return {"alerts": serialized_alerts, "total": len(serialized_alerts)}
    @app.get("/api/v1/cameras")
    async def get_cameras_list(user: dict = Depends(get_current_user)):
        """當前攝影機狀態"""
        include_sensitive = _user_has_permission(user, "camera_management")
        payload = _collect_camera_payload(include_sensitive=include_sensitive)
        if not include_sensitive:
            for camera in payload.get("cameras", []):
                camera.pop("source_url", None)
        return payload
    @app.post("/api/v1/cameras/{camera_id}/instant-switch")
    async def instant_switch_camera(camera_id: str, user: dict = Depends(get_current_user)):
        """切換攝影機來源"""
        return {
            "success": True,
            "message": f"已切換到攝影機 {camera_id}",
            "camera_id": camera_id,
        }
    @app.post("/api/v1/cameras/stop")
    async def stop_all_cameras(user: dict = Depends(get_current_user)):
        """停止所有攝影機"""
        return {
            "success": True,
            "message": "所有攝影機已停止",
        }
    # ===== 攝影機管理 API =====

    class CameraCreateRequest(BaseModel):
        slug: constr(strip_whitespace=True, min_length=2, max_length=100)
        name: constr(strip_whitespace=True, min_length=2, max_length=200)
        source_url: constr(strip_whitespace=True, min_length=4, max_length=1024)
        protocol: constr(strip_whitespace=True, min_length=2, max_length=20) = "rtsp"
        description: Optional[str] = None
        enabled: bool = True


    class CameraUpdateRequest(BaseModel):
        name: Optional[constr(strip_whitespace=True, min_length=2, max_length=200)] = None
        source_url: Optional[constr(strip_whitespace=True, min_length=4, max_length=1024)] = None
        protocol: Optional[constr(strip_whitespace=True, min_length=2, max_length=20)] = None
        description: Optional[str] = None
        enabled: Optional[bool] = None


    class CameraTestRequest(BaseModel):
        source_url: Optional[str] = None
        protocol: Optional[str] = None


    def _ensure_camera_permission(user_payload: dict) -> None:
        if not _user_has_permission(user_payload, "camera_management"):
            raise HTTPException(status_code=403, detail="需要攝影機管理權限")


    def _ensure_export_permission(user_payload: dict) -> None:
        if not _user_has_permission(user_payload, "data_export"):
            raise HTTPException(status_code=403, detail="需要資料匯出權限")


    # 管理員攝影機路由已移至 admin_routes.py


    # 管理員攝影機創建路由已移至 admin_routes.py


    # 管理員攝影機更新路由已移至 admin_routes.py


    # 管理員攝影機刪除和測試路由已移至 admin_routes.py


    # ===== 用戶管理 API =====

    class UserCreateRequest(BaseModel):
        username: constr(strip_whitespace=True, min_length=3, max_length=80)
        password: constr(min_length=6, max_length=128)
        role: constr(strip_whitespace=True, min_length=3, max_length=50) = "viewer"
        permissions: Optional[List[str]] = Field(default=None)
        is_active: bool = True


    class UserUpdateRequest(BaseModel):
        new_username: Optional[constr(strip_whitespace=True, min_length=3, max_length=80)] = None
        password: Optional[constr(min_length=6, max_length=128)] = None
        role: Optional[constr(strip_whitespace=True, min_length=3, max_length=50)] = None
        permissions: Optional[List[str]] = None
        is_active: Optional[bool] = None


    def _ensure_admin(user_payload: dict) -> None:
        if user_payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="需要管理員權限")


    @app.get("/api/v1/admin/users")
    async def get_users(user: dict = Depends(get_current_user)):
        """獲取用戶列表（僅管理員）"""
        _ensure_admin(user)

        users = user_repository.list_users()
        return {"users": users}


    @app.post("/api/v1/admin/users", status_code=201)
    async def create_user(
        payload: UserCreateRequest, user: dict = Depends(get_current_user)
    ):
        """建立新用戶"""
        _ensure_admin(user)

        try:
            created = user_repository.create_user(
                username=payload.username,
                password=payload.password,
                role=payload.role,
                permissions=payload.permissions,
                is_active=payload.is_active,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return {"user": created}


    @app.put("/api/v1/admin/users/{username}")
    async def update_user(
        username: str,
        payload: UserUpdateRequest,
        user: dict = Depends(get_current_user),
    ):
        """更新用戶資訊"""
        _ensure_admin(user)

        try:
            updated = user_repository.update_user(
                username,
                new_username=payload.new_username,
                password=payload.password,
                role=payload.role,
                permissions=payload.permissions,
                is_active=payload.is_active,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if not updated:
            raise HTTPException(status_code=404, detail="用戶不存在")

        return {"user": updated}


    @app.delete("/api/v1/admin/users/{username}", status_code=204)
    async def delete_user(username: str, user: dict = Depends(get_current_user)):
        """刪除用戶"""
        _ensure_admin(user)

        deleted = user_repository.delete_user(username)
        if not deleted:
            raise HTTPException(status_code=404, detail="用戶不存在")

        return Response(status_code=204)


    # ===== 匯出 API =====

    @app.get("/export/metrics.csv")
    async def export_metrics_csv(user: dict = Depends(get_current_user)):
        _ensure_export_permission(user)
        history = _collect_metrics_history(minutes=240)
        fieldnames = [
            "timestamp",
            "stream_id",
            "people_count",
            "fps_processing",
            "fps_source",
            "latency_ms",
            "cpu_percent",
            "memory_mb",
        ]

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow({
                "timestamp": item.get("timestamp"),
                "stream_id": item.get("stream_id"),
                "people_count": item.get("people_count"),
                "fps_processing": item.get("fps_processing"),
                "fps_source": item.get("fps_source"),
                "latency_ms": item.get("latency_ms"),
                "cpu_percent": item.get("cpu_percent"),
                "memory_mb": item.get("memory_mb"),
            })

        csv_bytes = buffer.getvalue().encode("utf-8-sig")
        output = BytesIO(csv_bytes)
        output.seek(0)
        filename = f"metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        headers = {"Content-Disposition": f"attachment; filename={filename}"}
        return StreamingResponse(output, media_type="text/csv", headers=headers)


    @app.get("/export/metrics.xlsx")
    async def export_metrics_excel(user: dict = Depends(get_current_user)):
        _ensure_export_permission(user)
        history = _collect_metrics_history(minutes=240)
        try:
            from openpyxl import Workbook
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="Excel 匯出需要 openpyxl，請先安裝對應套件",
            ) from exc

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Metrics"

        headers_row = [
            "timestamp",
            "stream_id",
            "people_count",
            "fps_processing",
            "fps_source",
            "latency_ms",
            "cpu_percent",
            "memory_mb",
        ]
        sheet.append(headers_row)
        for item in history:
            sheet.append(
                [
                    item.get("timestamp"),
                    item.get("stream_id"),
                    item.get("people_count"),
                    item.get("fps_processing"),
                    item.get("fps_source"),
                    item.get("latency_ms"),
                    item.get("cpu_percent"),
                    item.get("memory_mb"),
                ]
            )

        output = BytesIO()
        workbook.save(output)
        output.seek(0)
        filename = f"metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
        headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )

    # ===== 偵測控制 API =====
    @app.post("/api/v1/detection/start")
    async def start_detection_jobs(user: dict = Depends(get_current_user)):
        """啟動偵測作業"""
        _ensure_detection_permission(user)
        try:
            import subprocess
            import os
            
            # 檢查是否已經在運行
            if hasattr(app.state, 'detection_process') and app.state.detection_process and app.state.detection_process.poll() is None:
                return {"status": "already_running", "message": "偵測作業已在運行中"}
            
            # 啟動 people_detect.py
            process = subprocess.Popen([
                "python", "people_detect.py",
                "--config", "configs/sample.yaml"
            ], cwd=os.getcwd())
            
            app.state.detection_process = process
            return {"status": "success", "message": "偵測作業已啟動", "pid": process.pid}
            
        except Exception as e:
            return {"status": "error", "message": f"啟動偵測失敗: {str(e)}"}

    @app.post("/api/v1/detection/stop")
    async def stop_detection_jobs(user: dict = Depends(get_current_user)):
        """停止偵測作業"""
        _ensure_detection_permission(user)
        try:
            if hasattr(app.state, 'detection_process') and app.state.detection_process:
                if app.state.detection_process.poll() is None:
                    app.state.detection_process.terminate()
                    app.state.detection_process.wait(timeout=5)
                    return {"status": "success", "message": "偵測作業已停止"}
                else:
                    return {"status": "not_running", "message": "偵測作業未在運行"}
            else:
                return {"status": "not_running", "message": "偵測作業未在運行"}
                
        except Exception as e:
            return {"status": "error", "message": f"停止偵測失敗: {str(e)}"}

    @app.get("/api/v1/detection/status")
    async def get_detection_status(user: dict = Depends(get_current_user)):
        """獲取偵測狀態"""
        try:
            if hasattr(app.state, 'detection_process') and app.state.detection_process:
                is_running = app.state.detection_process.poll() is None
                return {
                    "status": "running" if is_running else "stopped",
                    "pid": app.state.detection_process.pid if is_running else None
                }
            else:
                return {"status": "stopped", "pid": None}
        except Exception as e:
            return {"status": "error", "message": f"獲取狀態失敗: {str(e)}"}

    # ===== Phase B 新增路由註冊 =====
    app.include_router(admin_router)
    app.include_router(export_router)
    app.include_router(learnsight_router)

    # ===== 靜態文件服務 (掛載到 /static 路徑) =====
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # ===== 根路徑重定向到登入頁面 =====
    @app.get("/")
    async def root():
        return FileResponse("static/login.html")
    
    @app.get("/login.html")
    async def login_page():
        return FileResponse("static/login.html")
    
    @app.get("/index.html")
    async def index_page():
        return FileResponse("static/index.html")

    @app.get("/learnsight.html")
    async def learnsight_page():
        return FileResponse("static/learnsight.html")
    
    @app.get("/admin.html")
    async def admin_page():
        return FileResponse("static/admin.html")
    
    @app.get("/simple-test.html")
    async def simple_test_page():
        return FileResponse("static/simple-test.html")
    
    @app.get("/step-test.html")
    async def step_test_page():
        return FileResponse("static/step-test.html")
    
    return app

# ==================================================
# Prometheus 更新函數
# ==================================================
def update_prometheus_metrics():
    try:
        current_metrics = data_store.get_current_metrics()
        line_counts = data_store.get_line_counts()
        zone_occupancy = data_store.get_zone_occupancy()
        if current_metrics:
            PEOPLE_COUNT.set(current_metrics.get('people_count', 0))
            FPS_PROCESSING.set(current_metrics.get('fps_processing', 0))
            FPS_SOURCE.set(current_metrics.get('fps_source', 0))
            LATENCY_MS.set(current_metrics.get('latency_ms', 0))
            CPU_PERCENT.set(current_metrics.get('cpu_percent', 0))
            MEMORY_MB.set(current_metrics.get('memory_mb', 0))
    except Exception as e:
        print(f"Error updating Prometheus metrics: {e}")

# ==================================================
# 建立應用
# ==================================================
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


