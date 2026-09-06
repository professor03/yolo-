"""
增強版 FastAPI 應用
整合所有優化功能
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from typing import Dict, List, Optional
import os
import json
import asyncio
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import logging

from ..api.data_store import data_store
from ..api.models import (
    SourceStatus, Metrics, LineCount, Alert, WebSocketMessage,
    MetricsResponse, CountsResponse, AlertsResponse, SourcesResponse
)
from ..auth.routes import router as auth_router
from ..auth.dependencies import get_current_user, get_current_user_optional, require_admin, require_operator_or_admin
from .health import get_health_status, get_detailed_health_status
from .enhanced_routes import router as enhanced_router
from ..utils.logging_config import setup_default_logging, api_logger

# 設置日誌
logger = logging.getLogger(__name__)

# Prometheus metrics
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

def create_enhanced_app() -> FastAPI:
    """創建增強版 FastAPI 應用"""
    
    app = FastAPI(
        title="YOLO 人流偵測系統 - 增強版",
        description="基於 YOLO 的即時人流偵測與分析系統",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # 添加中間件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    
    # 包含路由
    app.include_router(auth_router, prefix="/auth", tags=["認證"])
    app.include_router(enhanced_router, tags=["增強功能"])
    
    # 靜態檔案
    app.mount("/static", StaticFiles(directory="src/server/static"), name="static")
    
    # 基本路由
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """根路徑 - 重定向到增強版儀表板"""
        return FileResponse("src/server/static/enhanced_dashboard.html")
    
    @app.get("/classic", response_class=HTMLResponse)
    async def classic_dashboard():
        """經典版儀表板"""
        return FileResponse("src/server/static/index.html")
    
    @app.get("/health")
    async def health_check():
        """健康檢查"""
        return get_detailed_health_status()
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics"""
        update_prometheus_metrics()
        metrics_data = generate_latest()
        return Response(metrics_data, media_type=CONTENT_TYPE_LATEST)
    
    # API 端點
    @app.get("/api/v1/sources", response_model=SourcesResponse)
    async def get_sources(current_user: dict = Depends(get_current_user_optional)):
        """獲取來源狀態"""
        sources = data_store.get_sources()
        return SourcesResponse(sources=sources)
    
    @app.get("/api/v1/metrics", response_model=MetricsResponse)
    async def get_metrics(current_user: dict = Depends(get_current_user_optional)):
        """獲取指標數據"""
        current = data_store.get_current_metrics()
        history = data_store.get_metrics_history(minutes=60)
        
        if current is None:
            current = {
                "timestamp": datetime.now().isoformat(),
                "fps_processing": 0.0,
                "fps_source": 0.0,
                "people_count": 0,
                "latency_ms": 0.0,
                "cpu_percent": 0.0,
                "memory_mb": 0.0
            }
        
        return MetricsResponse(current=current, history=history)
    
    @app.get("/api/v1/counts", response_model=CountsResponse)
    async def get_counts(current_user: dict = Depends(get_current_user_optional)):
        """獲取計數數據"""
        line_counts = data_store.get_line_counts()
        zone_occupancy = data_store.get_zone_occupancy()
        
        return CountsResponse(
            line_counts=line_counts or {},
            zone_occupancy=zone_occupancy or {}
        )
    
    @app.get("/api/v1/alerts", response_model=AlertsResponse)
    async def get_alerts(
        limit: int = 50,
        current_user: dict = Depends(get_current_user_optional)
    ):
        """獲取警報列表"""
        alerts = data_store.get_alerts(limit=limit)
        return AlertsResponse(alerts=alerts or [])
    
    # WebSocket 端點
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket 連接"""
        await websocket.accept()
        try:
            while True:
                # 獲取即時數據
                sources = data_store.get_sources()
                metrics = data_store.get_current_metrics()
                line_counts = data_store.get_line_counts()
                alerts = data_store.get_alerts(limit=10)
                
                message = WebSocketMessage(
                    type="update",
                    timestamp=datetime.now().isoformat(),
                    data={
                        "sources": sources,
                        "metrics": metrics,
                        "line_counts": line_counts,
                        "alerts": alerts
                    }
                )
                
                await websocket.send_text(message.model_dump_json())
                await asyncio.sleep(1.0)
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    # 視頻流端點
    @app.get("/video_feed")
    async def video_feed():
        """視頻流端點 (需要實際實現)"""
        # 這裡需要實現實際的視頻流
        return {"message": "Video feed endpoint - needs implementation"}
    
    return app

def update_prometheus_metrics():
    """更新 Prometheus 指標"""
    try:
        # 獲取當前指標
        current_metrics = data_store.get_current_metrics()
        if current_metrics:
            PEOPLE_COUNT.set(current_metrics.get('people_count', 0))
            FPS_PROCESSING.set(current_metrics.get('fps_processing', 0))
            FPS_SOURCE.set(current_metrics.get('fps_source', 0))
            LATENCY_MS.set(current_metrics.get('latency_ms', 0))
            CPU_PERCENT.set(current_metrics.get('cpu_percent', 0))
            MEMORY_MB.set(current_metrics.get('memory_mb', 0))
        
        # 更新線段計數
        line_counts = data_store.get_line_counts()
        if line_counts:
            for line_id, counts in line_counts.items():
                if isinstance(counts, dict):
                    LINE_COUNTS.labels(line_id=line_id, direction='in').set(counts.get('in', 0))
                    LINE_COUNTS.labels(line_id=line_id, direction='out').set(counts.get('out', 0))
        
        # 更新區域佔用
        zone_occupancy = data_store.get_zone_occupancy()
        if zone_occupancy:
            for zone_id, occupancy in zone_occupancy.items():
                ZONE_OCCUPANCY.labels(zone_id=zone_id).set(occupancy)
        
        # 更新性能監控指標
        try:
            from ..utils.performance_monitor import get_performance_monitor
            performance_monitor = get_performance_monitor()
            if performance_monitor:
                current_metrics = performance_monitor.get_current_metrics()
                if current_metrics:
                    if current_metrics.gpu_utilization is not None:
                        GPU_UTILIZATION.set(current_metrics.gpu_utilization)
                    if current_metrics.gpu_memory_used_gb is not None:
                        GPU_MEMORY_GB.set(current_metrics.gpu_memory_used_gb)
        except Exception as e:
            logger.warning(f"Failed to update performance metrics: {e}")
            
    except Exception as e:
        logger.error(f"Error updating Prometheus metrics: {e}")

# 創建應用實例
app = create_enhanced_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
