"""API 數據模型"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    uptime: float


class SourceStatus(BaseModel):
    source_id: str
    source: str
    status: str
    last_seen: Optional[datetime]
    fps: float
    people_count: int

# 向後相容
SourceInfo = SourceStatus


class SourcesResponse(BaseModel):
    sources: List[SourceInfo]


class MetricData(BaseModel):
    timestamp: datetime
    fps_processing: float
    fps_source: float
    people_count: int
    latency_ms: Optional[float]
    cpu_percent: float
    memory_mb: float

# 向後相容
Metrics = MetricData


class MetricsResponse(BaseModel):
    current: MetricData
    history: List[MetricData]


class CountData(BaseModel):
    line_id: str
    count: int
    last_updated: datetime

# 向後相容
LineCount = CountData


class CountsResponse(BaseModel):
    line_counts: List[CountData]
    total_events: int


class AlertData(BaseModel):
    timestamp: datetime
    level: str
    message: str
    source_id: str

# 向後相容
Alert = AlertData


class AlertsResponse(BaseModel):
    alerts: List[AlertData]
    total: int


class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime
