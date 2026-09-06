"""FastAPI 路由定義"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
import logging

from .models import (
    HealthResponse, SourcesResponse, MetricsResponse, 
    CountsResponse, AlertsResponse, WebSocketMessage
)
from .data_store import data_store
from .websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    health_data = data_store.get_health()
    return HealthResponse(**health_data)


@router.get("/sources", response_model=SourcesResponse)
async def get_sources():
    """獲取來源狀態"""
    sources = data_store.get_sources()
    return SourcesResponse(sources=sources)


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """獲取性能指標"""
    from .models import MetricData
    
    current = data_store.get_current_metrics()
    history = data_store.get_metrics_history(minutes=60)
    
    # 如果沒有當前指標，創建一個默認值
    if current is None:
        current = MetricData(
            timestamp=datetime.now(),
            fps_processing=0.0,
            fps_source=0.0,
            people_count=0,
            latency_ms=None,
            cpu_percent=0.0,
            memory_mb=0.0
        )
    else:
        current = MetricData(**current)
    
    # 轉換歷史數據
    history_models = [MetricData(**h) for h in history]
    
    return MetricsResponse(
        current=current,
        history=history_models
    )


@router.get("/counts", response_model=CountsResponse)
async def get_counts():
    """獲取計數統計"""
    from .models import CountData
    
    line_counts = data_store.get_line_counts()
    total_events = data_store.total_events
    
    # 轉換為CountData模型
    count_models = [CountData(**c) for c in line_counts]
    
    return CountsResponse(
        line_counts=count_models,
        total_events=total_events
    )


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(limit: int = 100):
    """獲取警報信息"""
    from .models import AlertData
    
    alerts = data_store.get_alerts(limit=limit)
    
    # 轉換為AlertData模型
    alert_models = [AlertData(**a) for a in alerts]
    
    return AlertsResponse(
        alerts=alert_models,
        total=len(alert_models)
    )


@router.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 實時數據流"""
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # 發送當前儀表板數據
            dashboard_data = data_store.get_dashboard_data()
            
            message = WebSocketMessage(
                type="dashboard_update",
                data=dashboard_data,
                timestamp=datetime.now()
            )
            
            await websocket_manager.send_personal_message(
                message.model_dump_json(), 
                websocket
            )
            
            # 等待一段時間再發送下一次更新
            import asyncio
            await asyncio.sleep(1.0)  # 每秒更新一次
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)
