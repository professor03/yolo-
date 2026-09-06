from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import time
import math

from .geometry import Point, line_side, point_in_polygon, uturn_detected, distance_between_points


@dataclass
class TrackInfo:
    id: int
    center: Point
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: Optional[float] = None
    timestamp: Optional[float] = None


@dataclass
class LineConfig:
    id: str
    points: List[Point]  # 線段端點列表
    direction: str = "both"  # "in", "out", "both"
    uturn_cooldown: int = 30
    jitter_frames: int = 2


@dataclass
class ZoneConfig:
    id: str
    polygon: Sequence[Point]
    ignore: bool = False


@dataclass
class TrackState:
    last_position: Optional[Point] = None
    last_update_frame: int = 0
    last_side: Optional[int] = None
    last_cross_frame: int = 0
    history: List[Point] = field(default_factory=list)
    jitter_count: int = 0
    uturn_cooldown: int = 0
    cooldown: int = 0
    last_direction: Optional[str] = None


@dataclass
class LineEvent:
    event_type: str  # "in", "out"
    line_id: str
    track_id: int
    timestamp: float
    x_px: float
    y_px: float
    direction: str  # "in", "out"
    x_m: Optional[float] = None
    y_m: Optional[float] = None


@dataclass
class ZoneEvent:
    event_type: str  # "occupancy", "entry", "exit"
    zone_id: str
    track_id: int
    timestamp: float
    x_px: float
    y_px: float
    x_m: Optional[float] = None
    y_m: Optional[float] = None


class MultiLineCounter:
    """多線計數器，支援方向過濾、U-turn去重、重入冷卻等"""
    
    def __init__(self, lines: List[LineConfig], cooldown_frames: int = 30, 
                 max_gap_frames: int = 15, max_relink_dist: float = 40.0,
                 history_size: int = 30):
        self.lines = {line.id: line for line in lines}
        self.cooldown_frames = cooldown_frames
        self.max_gap_frames = max_gap_frames
        self.max_relink_dist = max_relink_dist
        self.history_size = history_size
        
        self.track_states: Dict[int, TrackState] = {}
        self.line_counts: Dict[str, Dict[str, int]] = {
            line_id: {"in": 0, "out": 0} for line_id in self.lines.keys()
        }
        self.events: List[LineEvent] = []
        
    def update(self, tracks: List[TrackInfo], frame_number: int, 
               calibrator: Optional[object] = None) -> List[LineEvent]:
        """更新計數器狀態並返回新事件"""
        new_events = []
        current_time = time.time()
        
        # 更新軌跡狀態
        for track in tracks:
            track_id = track.id
            center = track.center
            
            if track_id not in self.track_states:
                self.track_states[track_id] = TrackState()
            
            state = self.track_states[track_id]
            
            # 檢查是否需要重連ID
            if state.last_position is not None:
                distance = distance_between_points(state.last_position, center)
                if distance > self.max_relink_dist:
                    # 可能是新軌跡，重置狀態
                    state = TrackState()
                    self.track_states[track_id] = state
            
            # 更新軌跡歷史
            state.history.append(center)
            if len(state.history) > self.history_size:
                state.history.pop(0)
            
            # 檢查每條線的穿越
            for line_id, line_config in self.lines.items():
                if len(line_config.points) < 2:
                    continue
                
                # 檢查穿越
                event = self._check_line_crossing(
                    track, line_config, state, frame_number, current_time, calibrator
                )
                if event:
                    new_events.append(event)
                    self.events.append(event)
            
            # 更新狀態
            state.last_position = center
            state.last_update_frame = frame_number
        
        # 清理過期軌跡
        self._cleanup_expired_tracks(frame_number)
        
        return new_events
    
    def _check_line_crossing(self, track: TrackInfo, line_config: LineConfig, 
                           state: TrackState, frame_number: int, 
                           current_time: float, calibrator: Optional[object]) -> Optional[LineEvent]:
        """檢查線段穿越"""
        if len(line_config.points) < 2:
            return None
        
        center = track.center
        line_start, line_end = line_config.points[0], line_config.points[1]
        
        # 計算當前側
        current_side = line_side(center, (line_start, line_end))
        
        # 如果沒有歷史位置，記錄當前側
        if state.last_position is None:
            state.last_side = int(current_side > 0)
            return None
        
        # 檢查是否穿越
        if state.last_side is not None:
            last_side = state.last_side
            current_side_int = int(current_side > 0)
            
            # 方向抖動過濾
            if current_side_int != last_side:
                state.jitter_count += 1
                if state.jitter_count < line_config.jitter_frames:
                    return None
            else:
                state.jitter_count = 0
            
            # 檢查穿越
            if last_side != current_side_int and state.jitter_count >= line_config.jitter_frames:
                # 檢查冷卻期
                if frame_number - state.last_cross_frame < self.cooldown_frames:
                    return None
                
                # 檢查U-turn
                if len(state.history) >= 3 and uturn_detected(state.history[-3:], 90.0):
                    if state.uturn_cooldown > 0:
                        return None
                    state.uturn_cooldown = line_config.uturn_cooldown
                
                # 確定穿越方向
                direction = "in" if current_side_int == 1 else "out"
                
                # 檢查方向過濾
                if line_config.direction != "both" and line_config.direction != direction:
                    return None
                
                # 更新計數
                self.line_counts[line_config.id][direction] += 1
                state.last_cross_frame = frame_number
                state.last_direction = direction
                
                # 計算世界座標
                x_m, y_m = None, None
                if calibrator:
                    try:
                        world_pos = calibrator.pixel_to_world(center)
                        x_m, y_m = world_pos
                    except:
                        pass
                
                return LineEvent(
                    event_type=direction,
                    line_id=line_config.id,
                    track_id=track.id,
                    timestamp=current_time,
                    x_px=center[0],
                    y_px=center[1],
                    direction=direction,
                    x_m=x_m,
                    y_m=y_m
                )
        
        state.last_side = int(current_side > 0)
        return None
    
    def _cleanup_expired_tracks(self, frame_number: int):
        """清理過期的軌跡狀態"""
        expired_tracks = []
        for track_id, state in self.track_states.items():
            if frame_number - state.last_update_frame > self.max_gap_frames:
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.track_states[track_id]
    
    def get_counts(self) -> Dict[str, Dict[str, int]]:
        """獲取當前計數"""
        return self.line_counts.copy()
    
    def get_events(self) -> List[LineEvent]:
        """獲取所有事件"""
        return self.events.copy()


class ZoneOccupancy:
    """區域佔用檢測器"""
    
    def __init__(self, zones: List[ZoneConfig]):
        self.zones = {zone.id: zone for zone in zones}
        self.zone_occupancy: Dict[str, set] = {zone_id: set() for zone_id in self.zones.keys()}
        self.events: List[ZoneEvent] = []
        self.track_states: Dict[int, Dict[str, bool]] = {}  # track_id -> {zone_id: is_inside}
    
    def update(self, tracks: List[TrackInfo], calibrator: Optional[object] = None) -> List[ZoneEvent]:
        """更新區域佔用狀態並返回新事件"""
        new_events = []
        current_time = time.time()
        
        # 初始化軌跡狀態
        for track in tracks:
            if track.id not in self.track_states:
                self.track_states[track.id] = {zone_id: False for zone_id in self.zones.keys()}
        
        # 檢查每個軌跡的區域狀態
        for track in tracks:
            center = track.center
            track_states = self.track_states[track.id]
            
            for zone_id, zone_config in self.zones.items():
                is_inside = point_in_polygon(center, zone_config.polygon)
                was_inside = track_states.get(zone_id, False)
                
                # 狀態變化檢測
                if is_inside != was_inside:
                    if is_inside:
                        # 進入區域
                        event_type = "entry"
                        self.zone_occupancy[zone_id].add(track.id)
                    else:
                        # 離開區域
                        event_type = "exit"
                        self.zone_occupancy[zone_id].discard(track.id)
                    
                    # 計算世界座標
                    x_m, y_m = None, None
                    if calibrator:
                        try:
                            world_pos = calibrator.pixel_to_world(center)
                            x_m, y_m = world_pos
                        except:
                            pass
                    
                    event = ZoneEvent(
                        event_type=event_type,
                        zone_id=zone_id,
                        track_id=track.id,
                        timestamp=current_time,
                        x_px=center[0],
                        y_px=center[1],
                        x_m=x_m,
                        y_m=y_m
                    )
                    new_events.append(event)
                    self.events.append(event)
                
                track_states[zone_id] = is_inside
        
        # 生成佔用事件
        for zone_id, zone_config in self.zones.items():
            if not zone_config.ignore:
                for track_id in self.zone_occupancy[zone_id]:
                    # 計算世界座標
                    x_m, y_m = None, None
                    if calibrator:
                        try:
                            # 找到軌跡中心點
                            track_center = None
                            for track in tracks:
                                if track.id == track_id:
                                    track_center = track.center
                                    break
                            if track_center:
                                world_pos = calibrator.pixel_to_world(track_center)
                                x_m, y_m = world_pos
                        except:
                            pass
                    
                    event = ZoneEvent(
                        event_type="occupancy",
                        zone_id=zone_id,
                        track_id=track_id,
                        timestamp=current_time,
                        x_px=center[0] if 'center' in locals() else 0,
                        y_px=center[1] if 'center' in locals() else 0,
                        x_m=x_m,
                        y_m=y_m
                    )
                    new_events.append(event)
        
        return new_events
    
    def get_occupancy(self) -> Dict[str, int]:
        """獲取各區域佔用人數"""
        return {zone_id: len(track_ids) for zone_id, track_ids in self.zone_occupancy.items()}
    
    def get_events(self) -> List[ZoneEvent]:
        """獲取所有事件"""
        return self.events.copy()