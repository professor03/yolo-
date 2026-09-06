#!/usr/bin/env python3
"""
資料庫服務層
提供資料庫操作的業務邏輯
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, and_, or_, func
from .models import (
    DetectionEvent, LineCount, ZoneOccupancy, SystemMetrics, 
    Alert, Session as DBSession, get_database_manager
)
import json


class DetectionEventService:
    """檢測事件服務"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_event(self, event_data: Dict[str, Any]) -> DetectionEvent:
        """創建檢測事件"""
        try:
            event = DetectionEvent(**event_data)
            self.db.add(event)
            self.db.commit()
            self.db.refresh(event)
            return event
        except Exception as e:
            self.db.rollback()
            raise e
    
    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        source_id: Optional[str] = None,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DetectionEvent]:
        """查詢檢測事件"""
        query = self.db.query(DetectionEvent)
        
        if start_time:
            query = query.filter(DetectionEvent.timestamp >= start_time)
        if end_time:
            query = query.filter(DetectionEvent.timestamp <= end_time)
        if source_id:
            query = query.filter(DetectionEvent.source_id == source_id)
        if event_type:
            query = query.filter(DetectionEvent.event_type == event_type)
        if session_id:
            query = query.filter(DetectionEvent.session_id == session_id)
        
        return query.order_by(desc(DetectionEvent.timestamp)).offset(offset).limit(limit).all()
    
    def get_events_count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        source_id: Optional[str] = None,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        """獲取事件總數"""
        query = self.db.query(DetectionEvent)
        
        if start_time:
            query = query.filter(DetectionEvent.timestamp >= start_time)
        if end_time:
            query = query.filter(DetectionEvent.timestamp <= end_time)
        if source_id:
            query = query.filter(DetectionEvent.source_id == source_id)
        if event_type:
            query = query.filter(DetectionEvent.event_type == event_type)
        if session_id:
            query = query.filter(DetectionEvent.session_id == session_id)
        
        return query.count()
    
    def get_events_by_track(self, track_id: int, limit: int = 50) -> List[DetectionEvent]:
        """根據軌跡ID查詢事件"""
        return (
            self.db.query(DetectionEvent)
            .filter(DetectionEvent.track_id == track_id)
            .order_by(desc(DetectionEvent.timestamp))
            .limit(limit)
            .all()
        )
    
    def get_recent_events(self, hours: int = 1, limit: int = 100) -> List[DetectionEvent]:
        """獲取最近的事件"""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        return self.get_events(start_time=start_time, limit=limit)


class LineCountService:
    """線計數服務"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_count(self, count_data: Dict[str, Any]) -> LineCount:
        """創建線計數記錄"""
        count = LineCount(**count_data)
        self.db.add(count)
        self.db.commit()
        self.db.refresh(count)
        return count
    
    def get_counts(
        self,
        line_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[LineCount]:
        """查詢線計數"""
        query = self.db.query(LineCount)
        
        if line_id:
            query = query.filter(LineCount.line_id == line_id)
        if start_time:
            query = query.filter(LineCount.timestamp >= start_time)
        if end_time:
            query = query.filter(LineCount.timestamp <= end_time)
        if session_id:
            query = query.filter(LineCount.session_id == session_id)
        
        return query.order_by(desc(LineCount.timestamp)).limit(limit).all()
    
    def get_total_counts(
        self,
        line_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, int]:
        """獲取總計數"""
        query = self.db.query(LineCount)
        
        if line_id:
            query = query.filter(LineCount.line_id == line_id)
        if start_time:
            query = query.filter(LineCount.timestamp >= start_time)
        if end_time:
            query = query.filter(LineCount.timestamp <= end_time)
        if session_id:
            query = query.filter(LineCount.session_id == session_id)
        
        # 按方向分組統計
        results = (
            query.with_entities(LineCount.direction, func.sum(LineCount.count))
            .group_by(LineCount.direction)
            .all()
        )
        
        return {direction: total for direction, total in results}


class ZoneOccupancyService:
    """區域佔用服務"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_occupancy(self, occupancy_data: Dict[str, Any]) -> ZoneOccupancy:
        """創建區域佔用記錄"""
        occupancy = ZoneOccupancy(**occupancy_data)
        self.db.add(occupancy)
        self.db.commit()
        self.db.refresh(occupancy)
        return occupancy
    
    def get_occupancy(
        self,
        zone_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ZoneOccupancy]:
        """查詢區域佔用"""
        query = self.db.query(ZoneOccupancy)
        
        if zone_id:
            query = query.filter(ZoneOccupancy.zone_id == zone_id)
        if start_time:
            query = query.filter(ZoneOccupancy.timestamp >= start_time)
        if end_time:
            query = query.filter(ZoneOccupancy.timestamp <= end_time)
        if session_id:
            query = query.filter(ZoneOccupancy.session_id == session_id)
        
        return query.order_by(desc(ZoneOccupancy.timestamp)).limit(limit).all()
    
    def get_current_occupancy(self, session_id: Optional[str] = None) -> Dict[str, int]:
        """獲取當前區域佔用情況"""
        query = self.db.query(ZoneOccupancy)
        
        if session_id:
            query = query.filter(ZoneOccupancy.session_id == session_id)
        
        # 獲取每個區域的最新佔用情況
        subquery = (
            query.with_entities(
                ZoneOccupancy.zone_id,
                func.max(ZoneOccupancy.timestamp).label('max_timestamp')
            )
            .group_by(ZoneOccupancy.zone_id)
            .subquery()
        )
        
        results = (
            self.db.query(ZoneOccupancy)
            .join(subquery, and_(
                ZoneOccupancy.zone_id == subquery.c.zone_id,
                ZoneOccupancy.timestamp == subquery.c.max_timestamp
            ))
            .all()
        )
        
        return {occupancy.zone_id: occupancy.occupancy_count for occupancy in results}


class SystemMetricsService:
    """系統指標服務"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_metrics(self, metrics_data: Dict[str, Any]) -> SystemMetrics:
        """創建系統指標記錄"""
        metrics = SystemMetrics(**metrics_data)
        self.db.add(metrics)
        self.db.commit()
        self.db.refresh(metrics)
        return metrics
    
    def get_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None,
        source_id: Optional[str] = None,
        limit: int = 100
    ) -> List[SystemMetrics]:
        """查詢系統指標"""
        query = self.db.query(SystemMetrics)
        
        if start_time:
            query = query.filter(SystemMetrics.timestamp >= start_time)
        if end_time:
            query = query.filter(SystemMetrics.timestamp <= end_time)
        if session_id:
            query = query.filter(SystemMetrics.session_id == session_id)
        if source_id:
            query = query.filter(SystemMetrics.source_id == source_id)
        
        return query.order_by(desc(SystemMetrics.timestamp)).limit(limit).all()
    
    def get_latest_metrics(self, session_id: Optional[str] = None) -> Optional[SystemMetrics]:
        """獲取最新指標"""
        query = self.db.query(SystemMetrics)
        
        if session_id:
            query = query.filter(SystemMetrics.session_id == session_id)
        
        return query.order_by(desc(SystemMetrics.timestamp)).first()


class AlertService:
    """警報服務"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """創建警報"""
        alert = Alert(**alert_data)
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        return alert
    
    def get_alerts(
        self,
        level: Optional[str] = None,
        is_resolved: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Alert]:
        """查詢警報"""
        query = self.db.query(Alert)
        
        if level:
            query = query.filter(Alert.level == level)
        if is_resolved is not None:
            query = query.filter(Alert.is_resolved == is_resolved)
        if start_time:
            query = query.filter(Alert.timestamp >= start_time)
        if end_time:
            query = query.filter(Alert.timestamp <= end_time)
        if session_id:
            query = query.filter(Alert.session_id == session_id)
        
        return query.order_by(desc(Alert.timestamp)).limit(limit).all()
    
    def resolve_alert(self, alert_id: int) -> bool:
        """解決警報"""
        alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.is_resolved = True
            alert.resolved_at = datetime.utcnow()
            self.db.commit()
            return True
        return False


class SessionService:
    """會話服務"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_session(self, session_data: Dict[str, Any]) -> DBSession:
        """創建會話"""
        session = DBSession(**session_data)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[DBSession]:
        """獲取會話"""
        return self.db.query(DBSession).filter(DBSession.session_id == session_id).first()
    
    def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """更新會話"""
        session = self.get_session(session_id)
        if session:
            for key, value in update_data.items():
                setattr(session, key, value)
            self.db.commit()
            return True
        return False
    
    def end_session(self, session_id: str) -> bool:
        """結束會話"""
        return self.update_session(session_id, {
            'end_time': datetime.utcnow(),
            'is_active': False
        })
    
    def get_active_sessions(self) -> List[DBSession]:
        """獲取活躍會話"""
        return self.db.query(DBSession).filter(DBSession.is_active == True).all()
    
    def get_sessions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        source_id: Optional[str] = None,
        limit: int = 100
    ) -> List[DBSession]:
        """查詢會話"""
        query = self.db.query(DBSession)
        
        if start_time:
            query = query.filter(DBSession.start_time >= start_time)
        if end_time:
            query = query.filter(DBSession.end_time <= end_time)
        if source_id:
            query = query.filter(DBSession.source_id == source_id)
        
        return query.order_by(desc(DBSession.start_time)).limit(limit).all()


class StatisticsService:
    """統計服務"""
    
    def __init__(self, db: Session):
        self.db = db
        self.event_service = DetectionEventService(db)
        self.line_service = LineCountService(db)
        self.zone_service = ZoneOccupancyService(db)
        self.metrics_service = SystemMetricsService(db)
        self.alert_service = AlertService(db)
        self.session_service = SessionService(db)
    
    def get_dashboard_stats(
        self,
        session_id: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """獲取儀表板統計數據"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # 基本統計
        total_events = self.event_service.get_events_count(
            start_time=start_time, end_time=end_time, session_id=session_id
        )
        
        # 按事件類型統計
        event_types = self.db.query(
            DetectionEvent.event_type,
            func.count(DetectionEvent.id).label('count')
        ).filter(
            DetectionEvent.timestamp >= start_time,
            DetectionEvent.timestamp <= end_time
        )
        if session_id:
            event_types = event_types.filter(DetectionEvent.session_id == session_id)
        
        event_type_stats = {
            row.event_type: row.count 
            for row in event_types.group_by(DetectionEvent.event_type).all()
        }
        
        # 線計數統計
        line_counts = self.line_service.get_total_counts(
            start_time=start_time, end_time=end_time, session_id=session_id
        )
        
        # 區域佔用統計
        zone_occupancy = self.zone_service.get_current_occupancy(session_id)
        
        # 警報統計
        total_alerts = self.db.query(Alert).filter(
            Alert.timestamp >= start_time,
            Alert.timestamp <= end_time
        )
        if session_id:
            total_alerts = total_alerts.filter(Alert.session_id == session_id)
        
        alert_stats = {
            'total': total_alerts.count(),
            'unresolved': total_alerts.filter(Alert.is_resolved == False).count(),
            'by_level': {}
        }
        
        # 按級別統計警報
        for level in ['info', 'warning', 'error', 'critical']:
            alert_stats['by_level'][level] = total_alerts.filter(Alert.level == level).count()
        
        # 最新指標
        latest_metrics = self.metrics_service.get_latest_metrics(session_id)
        
        return {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'hours': hours
            },
            'events': {
                'total': total_events,
                'by_type': event_type_stats
            },
            'line_counts': line_counts,
            'zone_occupancy': zone_occupancy,
            'alerts': alert_stats,
            'latest_metrics': {
                'fps_processing': latest_metrics.fps_processing if latest_metrics else None,
                'fps_source': latest_metrics.fps_source if latest_metrics else None,
                'people_count': latest_metrics.people_count if latest_metrics else 0,
                'cpu_percent': latest_metrics.cpu_percent if latest_metrics else None,
                'memory_mb': latest_metrics.memory_mb if latest_metrics else None
            } if latest_metrics else None
        }
    
    def get_historical_stats(
        self,
        start_time: datetime,
        end_time: datetime,
        session_id: Optional[str] = None,
        group_by: str = 'hour'  # 'minute', 'hour', 'day'
    ) -> Dict[str, Any]:
        """獲取歷史統計數據"""
        # 根據分組方式調整時間格式
        if group_by == 'minute':
            time_format = func.strftime('%Y-%m-%d %H:%M', DetectionEvent.timestamp)
        elif group_by == 'hour':
            time_format = func.strftime('%Y-%m-%d %H:00', DetectionEvent.timestamp)
        elif group_by == 'day':
            time_format = func.strftime('%Y-%m-%d', DetectionEvent.timestamp)
        else:
            time_format = func.strftime('%Y-%m-%d %H:00', DetectionEvent.timestamp)
        
        # 事件統計
        event_stats = (
            self.db.query(
                time_format.label('time_period'),
                DetectionEvent.event_type,
                func.count(DetectionEvent.id).label('count')
            )
            .filter(
                DetectionEvent.timestamp >= start_time,
                DetectionEvent.timestamp <= end_time
            )
        )
        
        if session_id:
            event_stats = event_stats.filter(DetectionEvent.session_id == session_id)
        
        event_stats = event_stats.group_by('time_period', DetectionEvent.event_type).all()
        
        # 線計數統計
        line_stats = (
            self.db.query(
                time_format.label('time_period'),
                LineCount.line_id,
                LineCount.direction,
                func.sum(LineCount.count).label('total_count')
            )
            .filter(
                LineCount.timestamp >= start_time,
                LineCount.timestamp <= end_time
            )
        )
        
        if session_id:
            line_stats = line_stats.filter(LineCount.session_id == session_id)
        
        line_stats = line_stats.group_by('time_period', LineCount.line_id, LineCount.direction).all()
        
        return {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'group_by': group_by
            },
            'events': [
                {
                    'time_period': row.time_period,
                    'event_type': row.event_type,
                    'count': row.count
                }
                for row in event_stats
            ],
            'line_counts': [
                {
                    'time_period': row.time_period,
                    'line_id': row.line_id,
                    'direction': row.direction,
                    'total_count': row.total_count
                }
                for row in line_stats
            ]
        }
