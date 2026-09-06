#!/usr/bin/env python3
"""
增強版異常檢測模組
提供完整的異常檢測功能，包括人群密度、停留檢測、速度異常等
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import math
import statistics


class TrackInfo:
    """軌跡信息類"""
    def __init__(self, track_id: int, x: float, y: float, timestamp: float, 
                 bbox: Optional[tuple] = None, score: Optional[float] = None):
        self.track_id = track_id
        self.x = x
        self.y = y
        self.timestamp = timestamp
        self.bbox = bbox
        self.score = score


class AnomalyDetector:
    """增強版異常檢測器"""
    
    def __init__(self, 
                 stay_threshold: float = 10.0, 
                 crowd_threshold: int = 5,
                 speed_threshold: float = 50.0,
                 history_size: int = 100):
        self.stay_threshold = stay_threshold
        self.crowd_threshold = crowd_threshold
        self.speed_threshold = speed_threshold
        self.history_size = history_size
        
        # 軌跡歷史記錄
        self.track_history = {}  # track_id -> [(timestamp, x, y, bbox), ...]
        self.track_velocities = {}  # track_id -> [velocity, ...]
        
        # 統計數據
        self.total_tracks = 0
        self.anomaly_count = 0
        
    def _calculate_velocity(self, track_id: int, current_pos: tuple, current_time: float) -> Optional[float]:
        """計算軌跡速度"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return None
            
        # 獲取最近的兩個位置
        last_pos = self.track_history[track_id][-1]
        prev_pos = self.track_history[track_id][-2]
        
        # 計算距離和時間差
        distance = math.sqrt((current_pos[0] - last_pos[1])**2 + (current_pos[1] - last_pos[2])**2)
        time_diff = current_time - last_pos[0]
        
        if time_diff > 0:
            velocity = distance / time_diff
            return velocity
        return None
    
    def _detect_crowd_density(self, track_count: int) -> List[Dict[str, Any]]:
        """檢測人群密度異常"""
        events = []
        
        if track_count > self.crowd_threshold:
            severity = "warning" if track_count <= self.crowd_threshold * 2 else "critical"
            events.append({
                "level": severity,
                "title": "人群密度異常",
                "message": f"檢測到 {track_count} 人，超過閾值 {self.crowd_threshold}",
                "timestamp": datetime.now().isoformat(),
                "metric": "crowd_density",
                "value": track_count,
                "threshold": self.crowd_threshold
            })
        
        return events
    
    def _detect_stationary_objects(self, tracks: List[Any], current_time: float) -> List[Dict[str, Any]]:
        """檢測停留物件"""
        events = []
        
        for track in tracks:
            if not hasattr(track, 'track_id'):
                continue
                
            track_id = track.track_id
            current_pos = (getattr(track, 'x', 0), getattr(track, 'y', 0))
            
            # 更新軌跡歷史
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append((current_time, current_pos[0], current_pos[1], getattr(track, 'bbox', None)))
            
            # 保持歷史記錄大小
            if len(self.track_history[track_id]) > self.history_size:
                self.track_history[track_id] = self.track_history[track_id][-self.history_size:]
            
            # 檢查停留時間
            if len(self.track_history[track_id]) >= 30:  # 至少30幀
                # 計算停留時間
                first_time = self.track_history[track_id][0][0]
                stay_duration = current_time - first_time
                
                if stay_duration > self.stay_threshold:
                    events.append({
                        "level": "info",
                        "title": "停留檢測",
                        "message": f"軌跡 {track_id} 停留 {stay_duration:.1f} 秒",
                        "timestamp": datetime.now().isoformat(),
                        "metric": "stationary_time",
                        "value": stay_duration,
                        "track_id": track_id
                    })
        
        return events
    
    def _detect_speed_anomalies(self, tracks: List[Any], current_time: float) -> List[Dict[str, Any]]:
        """檢測速度異常"""
        events = []
        
        for track in tracks:
            if not hasattr(track, 'track_id'):
                continue
                
            track_id = track.track_id
            current_pos = (getattr(track, 'x', 0), getattr(track, 'y', 0))
            
            # 計算速度
            velocity = self._calculate_velocity(track_id, current_pos, current_time)
            
            if velocity is not None:
                # 更新速度歷史
                if track_id not in self.track_velocities:
                    self.track_velocities[track_id] = []
                
                self.track_velocities[track_id].append(velocity)
                
                # 保持速度歷史大小
                if len(self.track_velocities[track_id]) > 20:
                    self.track_velocities[track_id] = self.track_velocities[track_id][-20:]
                
                # 檢查速度異常
                if velocity > self.speed_threshold:
                    events.append({
                        "level": "warning",
                        "title": "速度異常",
                        "message": f"軌跡 {track_id} 速度過快: {velocity:.1f} px/s",
                        "timestamp": datetime.now().isoformat(),
                        "metric": "speed",
                        "value": velocity,
                        "track_id": track_id
                    })
        
        return events
    
    def _detect_tracking_quality(self, tracks: List[Any]) -> List[Dict[str, Any]]:
        """檢測追蹤品質"""
        events = []
        
        # 檢查追蹤ID的連續性
        active_tracks = set(track.track_id for track in tracks if hasattr(track, 'track_id'))
        
        # 檢查是否有新的追蹤ID出現
        new_tracks = active_tracks - set(self.track_history.keys())
        if len(new_tracks) > 5:  # 如果同時出現太多新追蹤ID
            events.append({
                "level": "warning",
                "title": "追蹤品質異常",
                "message": f"同時出現 {len(new_tracks)} 個新追蹤ID，可能影響追蹤品質",
                "timestamp": datetime.now().isoformat(),
                "metric": "tracking_quality",
                "value": len(new_tracks)
            })
        
        return events
    
    def update(self, tracks: List[Any], current_time: float, track_count: int) -> List[Dict[str, Any]]:
        """
        更新異常檢測狀態
        
        Args:
            tracks: 當前軌跡列表
            current_time: 當前時間
            track_count: 軌跡數量
            
        Returns:
            異常事件列表
        """
        events = []
        
        # 更新統計
        self.total_tracks = max(self.total_tracks, track_count)
        
        # 1. 檢測人群密度異常
        events.extend(self._detect_crowd_density(track_count))
        
        # 2. 檢測停留物件
        events.extend(self._detect_stationary_objects(tracks, current_time))
        
        # 3. 檢測速度異常
        events.extend(self._detect_speed_anomalies(tracks, current_time))
        
        # 4. 檢測追蹤品質
        events.extend(self._detect_tracking_quality(tracks))
        
        # 更新異常計數
        if events:
            self.anomaly_count += len(events)
        
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取異常檢測統計信息"""
        return {
            "total_tracks": self.total_tracks,
            "anomaly_count": self.anomaly_count,
            "active_tracks": len(self.track_history),
            "avg_velocity": {
                track_id: statistics.mean(velocities) if velocities else 0
                for track_id, velocities in self.track_velocities.items()
            }
        }
