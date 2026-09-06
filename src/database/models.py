#!/usr/bin/env python3
"""
資料庫模型定義
定義 SQLite 資料庫的表結構和關聯
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
import os
import json
from pathlib import Path

Base = declarative_base()


class User(Base):
    """用戶表"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="viewer")
    is_active = Column(Boolean, default=True, index=True)
    permissions_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 關聯關係
    cameras = relationship("Camera", back_populates="owner", cascade="all, delete-orphan")

    def get_permissions(self) -> List[str]:
        if not self.permissions_json:
            return []
        try:
            data = json.loads(self.permissions_json)
            if isinstance(data, list):
                return [str(item) for item in data]
        except json.JSONDecodeError:
            pass
        return []

    def set_permissions(self, permissions: Optional[List[str]]) -> None:
        if permissions:
            self.permissions_json = json.dumps(permissions)
        else:
            self.permissions_json = None


class Camera(Base):
    """攝影機表"""

    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    source_url = Column(String(1024), nullable=False)
    protocol = Column(String(20), nullable=False, default="rtsp")
    description = Column(Text, nullable=True)
    enabled = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_tested_at = Column(DateTime, nullable=True)
    last_test_status = Column(String(20), nullable=True)
    last_test_message = Column(Text, nullable=True)
    
    # 用戶關聯
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    owner = relationship("User", back_populates="cameras")
    
    # 連接狀態
    is_online = Column(Boolean, default=False, index=True)
    last_seen_at = Column(DateTime, nullable=True)
    reconnect_attempts = Column(Integer, default=0)
    max_reconnect_attempts = Column(Integer, default=5)

    __table_args__ = (
        Index("idx_camera_slug", "slug"),
        Index("idx_camera_enabled", "enabled"),
    )


class DetectionEvent(Base):
    """檢測事件表"""
    __tablename__ = "detection_events"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source_id = Column(String(100), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # 'person_detected', 'line_crossing', 'zone_entry', etc.
    
    # 位置信息
    x_px = Column(Float, nullable=True)
    y_px = Column(Float, nullable=True)
    x_m = Column(Float, nullable=True)
    y_m = Column(Float, nullable=True)
    
    # 檢測信息
    confidence = Column(Float, nullable=True)
    track_id = Column(Integer, nullable=True, index=True)
    class_id = Column(Integer, nullable=True)
    class_name = Column(String(50), nullable=True)
    
    # 關聯信息
    line_id = Column(String(50), nullable=True, index=True)
    zone_id = Column(String(50), nullable=True, index=True)
    direction = Column(String(20), nullable=True)  # 'in', 'out', 'left', 'right'
    
    # 元數據
    frame_number = Column(Integer, nullable=True)
    session_id = Column(String(100), nullable=True, index=True)
    metadata_json = Column(Text, nullable=True)  # JSON 格式的額外數據
    
    # 索引
    __table_args__ = (
        Index('idx_timestamp_source', 'timestamp', 'source_id'),
        Index('idx_event_type_timestamp', 'event_type', 'timestamp'),
        Index('idx_track_id_timestamp', 'track_id', 'timestamp'),
    )


class LineCount(Base):
    """線計數表"""
    __tablename__ = "line_counts"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    line_id = Column(String(50), nullable=False, index=True)
    direction = Column(String(20), nullable=False)  # 'in', 'out'
    count = Column(Integer, default=0)
    session_id = Column(String(100), nullable=True, index=True)
    
    # 索引
    __table_args__ = (
        Index('idx_line_timestamp', 'line_id', 'timestamp'),
        Index('idx_session_line', 'session_id', 'line_id'),
    )


class ZoneOccupancy(Base):
    """區域佔用表"""
    __tablename__ = "zone_occupancy"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    zone_id = Column(String(50), nullable=False, index=True)
    occupancy_count = Column(Integer, default=0)
    session_id = Column(String(100), nullable=True, index=True)
    
    # 索引
    __table_args__ = (
        Index('idx_zone_timestamp', 'zone_id', 'timestamp'),
        Index('idx_session_zone', 'session_id', 'zone_id'),
    )


class SystemMetrics(Base):
    """系統指標表"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # 性能指標
    fps_processing = Column(Float, nullable=True)
    fps_source = Column(Float, nullable=True)
    fps_inference = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    
    # 系統資源
    cpu_percent = Column(Float, nullable=True)
    memory_mb = Column(Float, nullable=True)
    gpu_utilization = Column(Float, nullable=True)
    gpu_memory_gb = Column(Float, nullable=True)
    
    # 檢測統計
    people_count = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    queue_size = Column(Integer, default=0)
    dropped_frames = Column(Integer, default=0)
    
    # 會話信息
    session_id = Column(String(100), nullable=True, index=True)
    source_id = Column(String(100), nullable=True, index=True)
    
    # 索引
    __table_args__ = (
        Index('idx_timestamp_session', 'timestamp', 'session_id'),
        Index('idx_source_timestamp', 'source_id', 'timestamp'),
    )


class Alert(Base):
    """警報表"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # 警報信息
    level = Column(String(20), nullable=False, index=True)  # 'info', 'warning', 'error', 'critical'
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # 關聯信息
    source_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    
    # 狀態
    is_resolved = Column(Boolean, default=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # 元數據
    metadata_json = Column(Text, nullable=True)
    
    # 索引
    __table_args__ = (
        Index('idx_level_timestamp', 'level', 'timestamp'),
        Index('idx_resolved_timestamp', 'is_resolved', 'timestamp'),
    )


class Session(Base):
    """會話表"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    start_time = Column(DateTime, default=datetime.utcnow, index=True)
    end_time = Column(DateTime, nullable=True, index=True)
    
    # 會話信息
    source_id = Column(String(100), nullable=True, index=True)
    source_type = Column(String(50), nullable=True)  # 'rtsp', 'file', 'camera'
    source_url = Column(String(500), nullable=True)
    
    # 統計信息
    total_frames = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    total_people_count = Column(Integer, default=0)
    total_line_crossings = Column(Integer, default=0)
    total_zone_entries = Column(Integer, default=0)
    
    # 配置信息
    config_json = Column(Text, nullable=True)  # 會話使用的配置
    
    # 狀態
    is_active = Column(Boolean, default=True, index=True)
    
    # 索引
    __table_args__ = (
        Index('idx_start_time', 'start_time'),
        Index('idx_source_start', 'source_id', 'start_time'),
    )


# 資料庫連接和會話管理
class DatabaseManager:
    """資料庫管理器"""

    def __init__(self, database_url: str = None):
        env_url = os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URL")
        if database_url is None:
            database_url = env_url

        if database_url is None:
            # 默認使用 SQLite
            default_path = Path(os.getcwd()) / "data" / "yolo_detection.db"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{default_path}" 

        self.database_url = database_url
        url = make_url(database_url)

        connect_args = {}
        if url.drivername.startswith("sqlite"):
            db_path = url.database
            if db_path and db_path != ":memory:":
                db_path = Path(db_path)
                if not db_path.is_absolute():
                    # 若提供相對路徑，轉換為專案根目錄下的實際路徑
                    project_root = Path(__file__).resolve().parents[2]
                    db_path = (project_root / db_path).resolve()
                db_path.parent.mkdir(parents=True, exist_ok=True)
                self.database_url = url._replace(database=str(db_path)).render_as_string(hide_password=False)
            connect_args.update({"check_same_thread": False})

        engine_kwargs = {
            "echo": False,  # 設為 True 可以看到 SQL 語句
            "pool_pre_ping": True,
        }
        if connect_args:
            engine_kwargs["connect_args"] = connect_args

        self.engine = create_engine(self.database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """創建所有表"""
        Base.metadata.create_all(bind=self.engine)
        try:
            from .user_repository import UserRepository
            from .camera_repository import CameraRepository

            UserRepository(self).ensure_default_users()
            CameraRepository(self).ensure_default_cameras()
        except Exception as exc:
            # 避免初始化失敗阻礙主流程，錯誤交由上層處理或日誌觀察
            print(f"[DatabaseManager] seed defaults warning: {exc}")

    def get_session(self):
        """獲取資料庫會話"""
        return self.SessionLocal()
    
    def close(self):
        """關閉資料庫連接"""
        self.engine.dispose()


# 全局資料庫管理器實例
db_manager = DatabaseManager()


def get_database_manager() -> DatabaseManager:
    """獲取資料庫管理器實例"""
    return db_manager


def get_db_session():
    """獲取資料庫會話（用於依賴注入）"""
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()
