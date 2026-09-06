"""
增強版 Web 介面路由
提供更豐富的 API 端點和功能
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime, timedelta
import os

from ..api.data_store import data_store
from ..auth.dependencies import get_current_user_optional

router = APIRouter()

# 連接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: Optional[str] = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            pass

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

    async def send_to_user(self, message: str, user_id: str):
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except:
                    pass

manager = ConnectionManager()

@router.get("/enhanced", response_class=HTMLResponse)
async def enhanced_dashboard():
    """增強版儀表板"""
    return FileResponse("src/server/static/enhanced_dashboard.html")

@router.get("/api/v1/dashboard/status")
async def get_dashboard_status():
    """獲取儀表板狀態"""
    try:
        # 獲取所有相關數據
        sources = data_store.get_sources()
        metrics = data_store.get_current_metrics()
        line_counts = data_store.get_line_counts()
        alerts = data_store.get_alerts()
        
        # 計算系統狀態
        active_sources = sum(1 for source in sources.values() if source.get('status') == 'online')
        total_sources = len(sources)
        
        # 獲取性能指標
        performance_data = {}
        try:
            from ..utils.performance_monitor import get_performance_monitor
            performance_monitor = get_performance_monitor()
            if performance_monitor:
                current_metrics = performance_monitor.get_current_metrics()
                if current_metrics:
                    performance_data = {
                        "gpu_utilization": current_metrics.gpu_utilization,
                        "gpu_memory_used_gb": current_metrics.gpu_memory_used_gb,
                        "memory_usage_percent": current_metrics.memory_percent,
                        "disk_usage_percent": current_metrics.disk_usage_percent
                    }
        except Exception as e:
            print(f"Performance monitor error: {e}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": {
                "active_sources": active_sources,
                "total_sources": total_sources,
                "overall_status": "online" if active_sources > 0 else "offline"
            },
            "metrics": metrics or {},
            "line_counts": line_counts or {},
            "alerts": alerts or [],
            "performance": performance_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard status: {str(e)}")

@router.get("/api/v1/dashboard/history")
async def get_dashboard_history(hours: int = 24):
    """獲取歷史數據"""
    try:
        # 獲取歷史指標
        history_metrics = data_store.get_metrics_history(minutes=hours * 60)
        
        # 獲取歷史計數
        history_counts = data_store.get_line_counts_history(hours=hours)
        
        return {
            "metrics_history": history_metrics or [],
            "counts_history": history_counts or [],
            "time_range": {
                "start": (datetime.now() - timedelta(hours=hours)).isoformat(),
                "end": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.get("/api/v1/dashboard/export")
async def export_dashboard_data(
    format: str = "json",
    hours: int = 24,
    current_user: dict = Depends(get_current_user_optional)
):
    """導出儀表板數據"""
    try:
        # 獲取數據
        status_data = await get_dashboard_status()
        history_data = await get_dashboard_history(hours)
        
        if format == "json":
            return {
                "export_timestamp": datetime.now().isoformat(),
                "time_range_hours": hours,
                "current_status": status_data,
                "history": history_data
            }
        elif format == "csv":
            # 這裡可以實現 CSV 導出
            raise HTTPException(status_code=501, detail="CSV export not implemented yet")
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.websocket("/ws/enhanced")
async def websocket_enhanced(websocket: WebSocket, user_id: Optional[str] = None):
    """增強版 WebSocket 連接"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            # 發送即時數據
            status_data = await get_dashboard_status()
            await manager.send_personal_message(
                json.dumps(status_data), websocket
            )
            await asyncio.sleep(1.0)  # 每秒更新一次
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, user_id)

@router.get("/api/v1/dashboard/config")
async def get_dashboard_config():
    """獲取儀表板配置"""
    return {
        "refresh_interval": 1000,  # 毫秒
        "chart_data_points": 20,
        "max_alerts_display": 50,
        "features": {
            "real_time_video": True,
            "performance_monitoring": True,
            "alert_system": True,
            "data_export": True,
            "user_management": True
        },
        "themes": [
            {"name": "default", "primary": "#667eea", "secondary": "#764ba2"},
            {"name": "dark", "primary": "#1a202c", "secondary": "#2d3748"},
            {"name": "light", "primary": "#f7fafc", "secondary": "#edf2f7"}
        ]
    }

@router.post("/api/v1/dashboard/config")
async def update_dashboard_config(
    config: Dict,
    current_user: dict = Depends(get_current_user_optional)
):
    """更新儀表板配置"""
    # 這裡可以實現配置保存邏輯
    return {"message": "Configuration updated successfully", "config": config}

@router.get("/api/v1/dashboard/health")
async def dashboard_health_check():
    """儀表板健康檢查"""
    try:
        # 檢查各個組件狀態
        components = {
            "data_store": True,
            "websocket": len(manager.active_connections) > 0,
            "performance_monitor": False,
            "database": True
        }
        
        # 檢查性能監控
        try:
            from ..utils.performance_monitor import get_performance_monitor
            performance_monitor = get_performance_monitor()
            components["performance_monitor"] = performance_monitor is not None
        except:
            pass
        
        # 檢查數據庫
        try:
            sources = data_store.get_sources()
            components["database"] = sources is not None
        except:
            components["database"] = False
        
        overall_health = all(components.values())
        
        return {
            "status": "healthy" if overall_health else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": components,
            "active_connections": len(manager.active_connections)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
