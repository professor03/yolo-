"""健康檢查模組 - P3企業級"""
import psutil
import time
from typing import Dict, Any
from datetime import datetime, timedelta
from ..api.data_store import data_store


class HealthChecker:
    """系統健康檢查器"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.last_check = None
        self.check_interval = 30  # 秒
    
    def get_system_health(self) -> Dict[str, Any]:
        """獲取系統健康狀態"""
        now = datetime.now()
        
        # 基本系統信息
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 應用程序健康狀態
        app_health = self._check_application_health()
        
        # 服務運行時間
        uptime = (now - self.start_time).total_seconds()
        
        # 整體健康狀態
        overall_status = "healthy"
        if cpu_percent > 90:
            overall_status = "warning"
        if memory.percent > 90 or disk.percent > 90:
            overall_status = "critical"
        if not app_health["data_store_healthy"]:
            overall_status = "critical"
        
        return {
            "status": overall_status,
            "timestamp": now.isoformat(),
            "uptime_seconds": uptime,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available // (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free // (1024 * 1024 * 1024)
            },
            "application": app_health,
            "checks": {
                "last_check": self.last_check.isoformat() if self.last_check else None,
                "check_interval": self.check_interval
            }
        }
    
    def _check_application_health(self) -> Dict[str, Any]:
        """檢查應用程序健康狀態"""
        try:
            # 檢查數據存儲
            data_store_healthy = data_store is not None
            
            # 檢查數據存儲方法
            try:
                dashboard_data = data_store.get_dashboard_data()
                data_store_responsive = True
            except Exception:
                data_store_responsive = False
                data_store_healthy = False
            
            # 檢查最近活動
            recent_activity = self._check_recent_activity()
            
            return {
                "data_store_healthy": data_store_healthy,
                "data_store_responsive": data_store_responsive,
                "recent_activity": recent_activity,
                "sources_count": len(data_store.sources) if data_store else 0,
                "last_update": data_store.last_update.isoformat() if data_store and data_store.last_update else None
            }
            
        except Exception as e:
            return {
                "data_store_healthy": False,
                "data_store_responsive": False,
                "recent_activity": False,
                "error": str(e)
            }
    
    def _check_recent_activity(self) -> bool:
        """檢查最近是否有活動"""
        if not data_store or not data_store.last_update:
            return False
        
        # 檢查最近5分鐘內是否有更新
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        return data_store.last_update > five_minutes_ago
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """獲取詳細健康狀態"""
        basic_health = self.get_system_health()
        
        # 添加更多詳細信息
        detailed_health = basic_health.copy()
        detailed_health.update({
            "process": {
                "pid": psutil.Process().pid,
                "create_time": psutil.Process().create_time(),
                "num_threads": psutil.Process().num_threads(),
                "memory_info": psutil.Process().memory_info()._asdict()
            },
            "network": {
                "connections": len(psutil.net_connections()),
                "io_counters": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None
            }
        })
        
        return detailed_health


# 全局健康檢查器實例
health_checker = HealthChecker()


def get_health_status() -> Dict[str, Any]:
    """獲取健康狀態"""
    return health_checker.get_system_health()


def get_detailed_health_status() -> Dict[str, Any]:
    """獲取詳細健康狀態"""
    return health_checker.get_detailed_health()
