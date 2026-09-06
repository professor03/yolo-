"""數據存儲和狀態管理 (跨進程共享)"""
import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import logging
import tempfile

import numpy as np

logger = logging.getLogger(__name__)


class DataStore:
    """採用檔案持久化的資料儲存，以便多個進程共享狀態。"""

    def __init__(self, storage_path: Optional[str] = None, max_history_size: int = 3600):
        self.max_history_size = max_history_size
        self.lock = threading.Lock()
        default_path = os.getenv("DATASTORE_PATH", "data/datastore.json")
        self.storage_path = Path(storage_path or default_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = {
            "start_time": datetime.utcnow().isoformat(),
            "sources": {},
            "current_metrics": None,
            "stream_metrics": {},
            "metrics_history": [],
            "line_counts": {},
            "total_events": 0,
            "alerts": [],
            "recent_detections": [],
            "object_counts": {},
        }
        self._load_state()

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._state.update(data)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to load datastore JSON: %s", exc)

    def _save_state(self) -> None:
        payload = self._prepare_for_json(self._state)
        tmp_dir = self.storage_path.parent
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=tmp_dir, suffix=".tmp") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
        except OSError as exc:
            logger.warning("Failed to write datastore temp file: %s", exc)
            return

        if tmp_path is None:
            return

        for attempt in range(5):
            try:
                tmp_path.replace(self.storage_path)
                break
            except FileNotFoundError as exc:
                logger.warning("Datastore temp file missing during replace: %s", exc)
                break
            except PermissionError as exc:
                if attempt == 4:
                    logger.warning("Failed to persist datastore (attempts exhausted): %s", exc)
                    raise
                time.sleep(0.1 * (attempt + 1))
            except OSError as exc:
                logger.warning("Unexpected error replacing datastore: %s", exc)
                break

    def _append_history(self, metric_data: Dict[str, Any]) -> None:
        history: List[Dict[str, Any]] = self._state.get("metrics_history", [])
        history.append(metric_data)
        if len(history) > self.max_history_size:
            history = history[-self.max_history_size :]
        self._state["metrics_history"] = history

    # ------------------------------------------------------------------
    # JSON 序列化輔助
    # ------------------------------------------------------------------
    def _prepare_for_json(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {k: self._prepare_for_json(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self._prepare_for_json(v) for v in payload]
        if isinstance(payload, tuple):
            return [self._prepare_for_json(v) for v in payload]
        if isinstance(payload, np.ndarray):
            return [self._prepare_for_json(v) for v in payload.tolist()]
        if isinstance(payload, np.generic):
            return payload.item()
        return payload

    def _normalise_line_counts(self, line_counts: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """確保線計數資料格式一致。"""
        normalised: Dict[str, Dict[str, int]] = {}
        for line_id, value in (line_counts or {}).items():
            if isinstance(value, dict):
                normalised[line_id] = {
                    "in": int(value.get("in", 0)),
                    "out": int(value.get("out", 0)),
                }
            else:
                normalised[line_id] = {
                    "in": int(value),
                    "out": 0,
                }
        return normalised

    def _aggregate_stream_metrics(self) -> Optional[Dict[str, Any]]:
        stream_metrics: Dict[str, Dict[str, Any]] = self._state.get("stream_metrics", {})
        if not stream_metrics:
            return None

        metrics_list = list(stream_metrics.values())
        if not metrics_list:
            return None

        def _avg(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        aggregated = {
            "people_count": sum(m.get("people_count", 0) for m in metrics_list),
            "fps_processing": _avg([m.get("fps_processing", 0.0) for m in metrics_list]),
            "fps_source": _avg([m.get("fps_source", 0.0) for m in metrics_list]),
            "latency_ms": _avg([m.get("latency_ms", 0.0) for m in metrics_list]),
            "cpu_percent": _avg([m.get("cpu_percent", 0.0) for m in metrics_list]),
            "memory_mb": _avg([m.get("memory_mb", 0.0) for m in metrics_list]),
            "timestamp": datetime.utcnow().isoformat(),
            "stream_count": len(metrics_list),
            "streams": stream_metrics,
        }
        return aggregated

    # ------------------------------------------------------------------
    # 更新方法（由偵測流程呼叫）
    # ------------------------------------------------------------------
    def update_source(
        self,
        source_id: str,
        source: str,
        status: str,
        fps_processing: float = 0.0,
        people_count: int = 0,
        *,
        fps_source: Optional[float] = None,
        latency_ms: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None,
        reconnect_attempts: Optional[int] = None,
        last_error: Optional[str] = None,
        connection_quality: Optional[str] = None,
    ) -> None:
        with self.lock:
            self._load_state()
            now = datetime.utcnow()
            
            # 獲取現有數據以保持某些字段
            existing = self._state.setdefault("sources", {}).get(source_id, {})
            
            self._state["sources"][source_id] = {
                "source_id": source_id,
                "source": source,
                "status": status,
                "last_seen": now.isoformat(),
                "fps_processing": float(fps_processing),
                "fps_source": float(fps_source) if fps_source is not None else None,
                "people_count": int(people_count),
                "latency_ms": float(latency_ms) if latency_ms is not None else None,
                "cpu_percent": float(cpu_percent) if cpu_percent is not None else None,
                "memory_mb": float(memory_mb) if memory_mb is not None else None,
                # 新增的狀態追蹤字段
                "reconnect_attempts": int(reconnect_attempts) if reconnect_attempts is not None else existing.get("reconnect_attempts", 0),
                "last_error": last_error if last_error is not None else existing.get("last_error"),
                "connection_quality": connection_quality if connection_quality is not None else existing.get("connection_quality", "unknown"),
                "first_seen": existing.get("first_seen", now.isoformat()),
                "total_uptime": existing.get("total_uptime", 0.0),
                "last_online": existing.get("last_online") if status != "online" else now.isoformat(),
            }
            
            # 更新總運行時間
            if status == "online" and existing.get("last_seen"):
                try:
                    last_seen = datetime.fromisoformat(existing["last_seen"])
                    uptime_delta = (now - last_seen).total_seconds()
                    if uptime_delta > 0 and uptime_delta < 60:  # 避免異常大的時間差
                        self._state["sources"][source_id]["total_uptime"] = existing.get("total_uptime", 0.0) + uptime_delta
                except (ValueError, TypeError):
                    pass
            
            self._save_state()

    def update_metrics(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        *,
        stream_id: Optional[str] = None,
        fps_processing: float = 0.0,
        fps_source: float = 0.0,
        people_count: int = 0,
        latency_ms: Optional[float] = None,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
    ) -> None:
        with self.lock:
            self._load_state()
            if metrics is not None:
                metric_data = metrics.copy()
                stream_id = metric_data.get("stream_id", stream_id)
                metric_data.setdefault("stream_id", stream_id)
                metric_data.setdefault("timestamp", datetime.utcnow().isoformat())
                metric_data.setdefault("fps_processing", 0.0)
                metric_data.setdefault("fps_source", 0.0)
                metric_data.setdefault("people_count", 0)
                metric_data.setdefault("latency_ms", None)
                metric_data.setdefault("cpu_percent", 0.0)
                metric_data.setdefault("memory_mb", 0.0)
            else:
                metric_data = {
                    "stream_id": stream_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "fps_processing": fps_processing,
                    "fps_source": fps_source,
                    "people_count": people_count,
                    "latency_ms": latency_ms,
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                }

            if stream_id:
                streams = self._state.setdefault("stream_metrics", {})
                streams[stream_id] = metric_data
            aggregated = self._aggregate_stream_metrics() or metric_data
            self._state["current_metrics"] = aggregated
            self._append_history({k: v for k, v in aggregated.items() if k != "streams"})
            self._save_state()

    def update_line_counts(self, line_counts: Dict[str, int], total_events: int) -> None:
        with self.lock:
            self._load_state()
            self._state["line_counts"] = self._normalise_line_counts(line_counts)
            self._state["total_events"] = int(total_events)
            self._save_state()

    def add_alert(self, level: str, message: str, source_id: str = "system") -> None:
        with self.lock:
            self._load_state()
            alerts: List[Dict[str, Any]] = self._state.get("alerts", [])
            alerts.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": level,
                    "message": message,
                    "source_id": source_id,
                }
            )
            max_alerts = 1000
            if len(alerts) > max_alerts:
                alerts = alerts[-max_alerts:]
            self._state["alerts"] = alerts
            self._save_state()

    def add_detection_batch(self, detections: List[Dict]) -> None:
        with self.lock:
            self._load_state()
            recent = self._state.get("recent_detections", [])
            object_counts = self._state.get("object_counts", {})
            for detection in detections:
                detection = detection.copy()
                detection["timestamp"] = datetime.utcnow().isoformat()
                recent.append(detection)
                class_name = detection.get("class_name", "unknown")
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
            if len(recent) > self.max_history_size:
                recent = recent[-self.max_history_size :]
            self._state["recent_detections"] = recent
            self._state["object_counts"] = object_counts
            self._save_state()

    # ------------------------------------------------------------------
    # 讀取方法（由 API 呼叫）
    # ------------------------------------------------------------------
    def get_health(self) -> Dict[str, Any]:
        self._load_state()
        start_iso = self._state.get("start_time")
        try:
            start_time = datetime.fromisoformat(start_iso) if start_iso else datetime.utcnow()
        except ValueError:
            start_time = datetime.utcnow()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": (datetime.utcnow() - start_time).total_seconds(),
        }

    def get_sources(self) -> List[Dict]:
        with self.lock:
            self._load_state()
            return list(self._state.get("sources", {}).values())
    
    def get_connection_health(self) -> Dict[str, Any]:
        """獲取所有攝影機的連接健康狀態"""
        with self.lock:
            self._load_state()
            sources = self._state.get("sources", {})
            now = datetime.utcnow()
            
            health_data = {}
            for source_id, source_data in sources.items():
                try:
                    last_seen = datetime.fromisoformat(source_data.get("last_seen", ""))
                    time_since_last_seen = (now - last_seen).total_seconds()
                except (ValueError, TypeError):
                    time_since_last_seen = float('inf')
                
                # 判斷連接健康狀態
                if source_data.get("status") == "online":
                    if time_since_last_seen < 5:  # 5秒內有更新
                        health_status = "excellent"
                    elif time_since_last_seen < 15:  # 15秒內有更新
                        health_status = "good"
                    else:
                        health_status = "poor"
                else:
                    health_status = "offline"
                
                health_data[source_id] = {
                    "status": source_data.get("status", "unknown"),
                    "health": health_status,
                    "last_seen_seconds_ago": time_since_last_seen,
                    "reconnect_attempts": source_data.get("reconnect_attempts", 0),
                    "last_error": source_data.get("last_error"),
                    "connection_quality": source_data.get("connection_quality", "unknown"),
                    "total_uptime": source_data.get("total_uptime", 0.0),
                    "fps_processing": source_data.get("fps_processing", 0.0),
                }
            
            return health_data

    def get_current_metrics(self) -> Optional[Dict]:
        with self.lock:
            self._load_state()
            return self._state.get("current_metrics")

    def get_metrics_history(self, minutes: int = 60) -> List[Dict]:
        with self.lock:
            self._load_state()
            cutoff = datetime.utcnow() - timedelta(minutes=minutes)
            history = self._state.get("metrics_history", [])
            result = []
            for item in history:
                ts = item.get("timestamp")
                try:
                    when = datetime.fromisoformat(ts) if ts else None
                except ValueError:
                    when = None
                if when and when >= cutoff:
                    result.append(item)
            return result

    def get_line_counts(self) -> List[Dict]:
        with self.lock:
            self._load_state()
            counts = self._state.get("line_counts", {})
            result = []
            for line_id, value in counts.items():
                if isinstance(value, dict):
                    in_count = int(value.get("in", 0))
                    out_count = int(value.get("out", 0))
                    total = in_count + out_count
                else:
                    in_count = int(value)
                    out_count = 0
                    total = in_count
                result.append(
                    {
                        "line_id": line_id,
                        "in": in_count,
                        "out": out_count,
                        "total": total,
                        "last_updated": datetime.utcnow().isoformat(),
                    }
                )
            return result

    def get_alerts(self, limit: int = 100) -> List[Dict]:
        with self.lock:
            self._load_state()
            alerts = self._state.get("alerts", [])
            return alerts[-limit:] if alerts else []

    def get_detections_history(self, minutes: int = 60) -> List[Dict]:
        with self.lock:
            self._load_state()
            cutoff = datetime.utcnow() - timedelta(minutes=minutes)
            result = []
            for detection in self._state.get("recent_detections", []):
                ts = detection.get("timestamp")
                try:
                    when = datetime.fromisoformat(ts) if ts else None
                except ValueError:
                    when = None
                if when and when >= cutoff:
                    result.append(detection)
            return result

    def get_object_counts(self) -> Dict[str, int]:
        with self.lock:
            self._load_state()
            return dict(self._state.get("object_counts", {}))

    def get_detections_by_class(self, class_name: str, minutes: int = 60) -> List[Dict]:
        return [d for d in self.get_detections_history(minutes) if d.get("class_name") == class_name]

    def get_dashboard_data(self) -> Dict[str, Any]:
        with self.lock:
            self._load_state()
            return {
                "sources": list(self._state.get("sources", {}).values()),
                "current_metrics": self._state.get("current_metrics"),
                "line_counts": self.get_line_counts(),
                "total_events": self._state.get("total_events", 0),
                "recent_alerts": self.get_alerts(limit=10),
                "object_counts": self._state.get("object_counts", {}),
                "recent_detections": self.get_detections_history(minutes=10),
            }

    def get_zone_occupancy(self) -> Dict[str, int]:
        # 尚未整合實際區域佔用計算
        return {}


# 全局數據存儲實例
_data_store_instance = DataStore()


def get_data_store() -> DataStore:
    return _data_store_instance


data_store = get_data_store()
