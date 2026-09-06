#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行為模式分析器
分析人流行為模式和異常行為檢測
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import json

logger = logging.getLogger(__name__)


@dataclass
class BehaviorPattern:
    """行為模式"""
    pattern_id: str
    pattern_type: str  # normal, suspicious, abnormal
    description: str
    frequency: float
    confidence: float
    features: Dict[str, float]
    first_seen: datetime
    last_seen: datetime


@dataclass
class BehaviorEvent:
    """行為事件"""
    event_id: str
    behavior_type: str
    severity: str  # low, medium, high, critical
    description: str
    features: Dict[str, float]
    timestamp: datetime
    pattern_id: Optional[str] = None


class BehaviorAnalyzer:
    """行為模式分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history_size = config.get("history_size", 5000)
        self.pattern_threshold = config.get("pattern_threshold", 0.7)
        self.anomaly_threshold = config.get("anomaly_threshold", 0.8)
        
        # 數據存儲
        self.behavior_history: deque = deque(maxlen=self.history_size)
        self.patterns: Dict[str, BehaviorPattern] = {}
        self.behavior_events: List[BehaviorEvent] = []
        
        # 行為類型定義
        self.behavior_types = {
            "normal_flow": "正常流動",
            "crowd_gathering": "人群聚集",
            "rapid_movement": "快速移動",
            "stationary_behavior": "靜止行為",
            "loitering": "徘徊行為",
            "suspicious_loitering": "可疑徘徊",
            "abnormal_density": "異常密度",
            "direction_reversal": "方向逆轉",
            "group_formation": "群組形成",
            "scattered_movement": "分散移動"
        }
        
        # 線程安全
        self.lock = threading.Lock()
        
        logger.info("行為模式分析器已初始化")
    
    def analyze_behavior(self, detection_data: List[Dict[str, Any]]) -> List[BehaviorEvent]:
        """分析行為模式"""
        if not detection_data:
            return []
        
        # 提取行為特徵
        features = self._extract_behavior_features(detection_data)
        
        # 檢測行為模式
        behavior_events = []
        
        # 正常流動檢測
        normal_flow = self._detect_normal_flow(features)
        if normal_flow:
            behavior_events.append(normal_flow)
        
        # 人群聚集檢測
        crowd_gathering = self._detect_crowd_gathering(features)
        if crowd_gathering:
            behavior_events.append(crowd_gathering)
        
        # 快速移動檢測
        rapid_movement = self._detect_rapid_movement(features)
        if rapid_movement:
            behavior_events.append(rapid_movement)
        
        # 靜止行為檢測
        stationary_behavior = self._detect_stationary_behavior(features)
        if stationary_behavior:
            behavior_events.append(stationary_behavior)
        
        # 徘徊行為檢測
        loitering = self._detect_loitering(features)
        if loitering:
            behavior_events.append(loitering)
        
        # 異常密度檢測
        abnormal_density = self._detect_abnormal_density(features)
        if abnormal_density:
            behavior_events.append(abnormal_density)
        
        # 方向逆轉檢測
        direction_reversal = self._detect_direction_reversal(features)
        if direction_reversal:
            behavior_events.append(direction_reversal)
        
        # 群組形成檢測
        group_formation = self._detect_group_formation(features)
        if group_formation:
            behavior_events.append(group_formation)
        
        # 更新歷史數據
        with self.lock:
            self.behavior_history.append({
                "timestamp": datetime.now(),
                "features": features,
                "detection_count": len(detection_data),
                "events": behavior_events
            })
            
            # 添加事件到歷史
            for event in behavior_events:
                self.behavior_events.append(event)
                
                # 保持事件數量限制
                if len(self.behavior_events) > self.history_size:
                    self.behavior_events.pop(0)
        
        return behavior_events
    
    def _extract_behavior_features(self, detection_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """提取行為特徵"""
        if not detection_data:
            return {}
        
        # 基本統計
        count = len(detection_data)
        
        # 位置特徵
        positions = [(d.get("x", 0), d.get("y", 0)) for d in detection_data]
        if positions:
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            
            # 密度特徵
            if len(positions) > 1:
                # 計算位置分散度
                x_std = np.std(x_coords) if len(x_coords) > 1 else 0
                y_std = np.std(y_coords) if len(y_coords) > 1 else 0
                spread = np.sqrt(x_std**2 + y_std**2)
                
                # 計算中心點
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                
                # 計算到中心的平均距離
                distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in positions]
                avg_distance = np.mean(distances) if distances else 0
            else:
                spread = 0
                avg_distance = 0
        else:
            spread = 0
            avg_distance = 0
        
        # 速度特徵（如果有歷史數據）
        velocity_features = self._calculate_velocity_features(detection_data)
        
        # 方向特徵
        direction_features = self._calculate_direction_features(detection_data)
        
        return {
            "count": count,
            "spread": spread,
            "avg_distance": avg_distance,
            "density": count / max(spread, 1),  # 密度 = 數量 / 分散度
            **velocity_features,
            **direction_features
        }
    
    def _calculate_velocity_features(self, detection_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """計算速度特徵"""
        if len(self.behavior_history) < 2:
            return {"avg_velocity": 0.0, "velocity_std": 0.0, "max_velocity": 0.0}
        
        # 獲取最近的歷史數據
        recent_history = list(self.behavior_history)[-5:]  # 最近5個時間點
        
        velocities = []
        for i in range(1, len(recent_history)):
            prev_data = recent_history[i-1]
            curr_data = recent_history[i]
            
            # 計算平均移動距離
            if prev_data["detection_count"] > 0 and curr_data["detection_count"] > 0:
                # 簡化計算：基於檢測數量的變化
                count_change = abs(curr_data["detection_count"] - prev_data["detection_count"])
                time_diff = (curr_data["timestamp"] - prev_data["timestamp"]).total_seconds()
                if time_diff > 0:
                    velocity = count_change / time_diff
                    velocities.append(velocity)
        
        if velocities:
            return {
                "avg_velocity": np.mean(velocities),
                "velocity_std": np.std(velocities),
                "max_velocity": np.max(velocities)
            }
        else:
            return {"avg_velocity": 0.0, "velocity_std": 0.0, "max_velocity": 0.0}
    
    def _calculate_direction_features(self, detection_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """計算方向特徵"""
        if len(self.behavior_history) < 3:
            return {"direction_consistency": 0.0, "direction_change": 0.0}
        
        # 簡化實現：基於檢測數量的變化趨勢
        recent_counts = [h["detection_count"] for h in list(self.behavior_history)[-5:]]
        recent_counts.append(len(detection_data))
        
        if len(recent_counts) < 3:
            return {"direction_consistency": 0.0, "direction_change": 0.0}
        
        # 計算趨勢一致性
        trends = []
        for i in range(1, len(recent_counts)):
            if recent_counts[i] > recent_counts[i-1]:
                trends.append(1)  # 上升
            elif recent_counts[i] < recent_counts[i-1]:
                trends.append(-1)  # 下降
            else:
                trends.append(0)  # 持平
        
        if trends:
            direction_consistency = 1.0 - (np.std(trends) / 2.0)  # 標準差越小越一致
            direction_change = abs(trends[-1] - trends[0]) if len(trends) > 1 else 0
        else:
            direction_consistency = 0.0
            direction_change = 0.0
        
        return {
            "direction_consistency": max(0, direction_consistency),
            "direction_change": direction_change
        }
    
    def _detect_normal_flow(self, features: Dict[str, float]) -> Optional[BehaviorEvent]:
        """檢測正常流動"""
        count = features.get("count", 0)
        density = features.get("density", 0)
        direction_consistency = features.get("direction_consistency", 0)
        
        # 正常流動的特徵：適中的密度，方向一致
        if 5 <= count <= 50 and density > 0.1 and direction_consistency > 0.6:
            return BehaviorEvent(
                event_id=f"normal_flow_{int(datetime.now().timestamp())}",
                behavior_type="normal_flow",
                severity="low",
                description=f"正常流動檢測: 人數={count}, 密度={density:.2f}",
                features=features,
                timestamp=datetime.now()
            )
        return None
    
    def _detect_crowd_gathering(self, features: Dict[str, float]) -> Optional[BehaviorEvent]:
        """檢測人群聚集"""
        count = features.get("count", 0)
        density = features.get("density", 0)
        spread = features.get("spread", 0)
        
        # 人群聚集的特徵：高密度，小分散度
        if count > 20 and density > 1.0 and spread < 100:
            severity = "high" if count > 50 else "medium"
            return BehaviorEvent(
                event_id=f"crowd_gathering_{int(datetime.now().timestamp())}",
                behavior_type="crowd_gathering",
                severity=severity,
                description=f"人群聚集檢測: 人數={count}, 密度={density:.2f}",
                features=features,
                timestamp=datetime.now()
            )
        return None
    
    def _detect_rapid_movement(self, features: Dict[str, float]) -> Optional[BehaviorEvent]:
        """檢測快速移動"""
        avg_velocity = features.get("avg_velocity", 0)
        velocity_std = features.get("velocity_std", 0)
        
        # 快速移動的特徵：高速度，高變異性
        if avg_velocity > 2.0 or velocity_std > 1.5:
            return BehaviorEvent(
                event_id=f"rapid_movement_{int(datetime.now().timestamp())}",
                behavior_type="rapid_movement",
                severity="medium",
                description=f"快速移動檢測: 平均速度={avg_velocity:.2f}",
                features=features,
                timestamp=datetime.now()
            )
        return None
    
    def _detect_stationary_behavior(self, features: Dict[str, float]) -> Optional[BehaviorEvent]:
        """檢測靜止行為"""
        avg_velocity = features.get("avg_velocity", 0)
        count = features.get("count", 0)
        
        # 靜止行為的特徵：低速度，持續存在
        if avg_velocity < 0.1 and count > 0:
            return BehaviorEvent(
                event_id=f"stationary_behavior_{int(datetime.now().timestamp())}",
                behavior_type="stationary_behavior",
                severity="low",
                description=f"靜止行為檢測: 人數={count}, 速度={avg_velocity:.2f}",
                features=features,
                timestamp=datetime.now()
            )
        return None
    
    def _detect_loitering(self, features: Dict[str, float]) -> Optional[BehaviorEvent]:
        """檢測徘徊行為"""
        # 需要更長時間的數據來檢測徘徊
        if len(self.behavior_history) < 10:
            return None
        
        # 檢查最近是否有持續的靜止或緩慢移動
        recent_velocities = []
        for h in list(self.behavior_history)[-10:]:
            if "features" in h and "avg_velocity" in h["features"]:
                recent_velocities.append(h["features"]["avg_velocity"])
        
        if recent_velocities:
            avg_recent_velocity = np.mean(recent_velocities)
            if avg_recent_velocity < 0.5:  # 持續低速度
                return BehaviorEvent(
                    event_id=f"loitering_{int(datetime.now().timestamp())}",
                    behavior_type="loitering",
                    severity="medium",
                    description=f"徘徊行為檢測: 平均速度={avg_recent_velocity:.2f}",
                    features=features,
                    timestamp=datetime.now()
                )
        return None
    
    def _detect_abnormal_density(self, features: Dict[str, float]) -> Optional[BehaviorEvent]:
        """檢測異常密度"""
        density = features.get("density", 0)
        count = features.get("count", 0)
        
        # 異常密度的特徵：極高或極低密度
        if density > 5.0 or (density < 0.01 and count > 10):
            severity = "high" if density > 10.0 else "medium"
            return BehaviorEvent(
                event_id=f"abnormal_density_{int(datetime.now().timestamp())}",
                behavior_type="abnormal_density",
                severity=severity,
                description=f"異常密度檢測: 密度={density:.2f}, 人數={count}",
                features=features,
                timestamp=datetime.now()
            )
        return None
    
    def _detect_direction_reversal(self, features: Dict[str, float]) -> Optional[BehaviorEvent]:
        """檢測方向逆轉"""
        direction_change = features.get("direction_change", 0)
        direction_consistency = features.get("direction_consistency", 0)
        
        # 方向逆轉的特徵：方向變化大，一致性低
        if direction_change > 1.5 and direction_consistency < 0.3:
            return BehaviorEvent(
                event_id=f"direction_reversal_{int(datetime.now().timestamp())}",
                behavior_type="direction_reversal",
                severity="medium",
                description=f"方向逆轉檢測: 變化={direction_change:.2f}",
                features=features,
                timestamp=datetime.now()
            )
        return None
    
    def _detect_group_formation(self, features: Dict[str, float]) -> Optional[BehaviorEvent]:
        """檢測群組形成"""
        count = features.get("count", 0)
        spread = features.get("spread", 0)
        avg_distance = features.get("avg_distance", 0)
        
        # 群組形成的特徵：適中人數，小分散度，小平均距離
        if 3 <= count <= 15 and spread < 50 and avg_distance < 30:
            return BehaviorEvent(
                event_id=f"group_formation_{int(datetime.now().timestamp())}",
                behavior_type="group_formation",
                severity="low",
                description=f"群組形成檢測: 人數={count}, 分散度={spread:.2f}",
                features=features,
                timestamp=datetime.now()
            )
        return None
    
    def get_behavior_statistics(self) -> Dict[str, Any]:
        """獲取行為統計"""
        with self.lock:
            total_events = len(self.behavior_events)
            
            # 按類型統計
            type_stats = {}
            for behavior_type in self.behavior_types.keys():
                type_stats[behavior_type] = len([
                    e for e in self.behavior_events 
                    if e.behavior_type == behavior_type
                ])
            
            # 按嚴重程度統計
            severity_stats = {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0
            }
            for event in self.behavior_events:
                severity_stats[event.severity] += 1
            
            # 最近活躍的行為類型
            recent_events = [e for e in self.behavior_events 
                           if (datetime.now() - e.timestamp).total_seconds() < 3600]  # 最近1小時
            
            recent_types = {}
            for event in recent_events:
                if event.behavior_type not in recent_types:
                    recent_types[event.behavior_type] = 0
                recent_types[event.behavior_type] += 1
            
            return {
                "total_events": total_events,
                "recent_events": len(recent_events),
                "type_statistics": type_stats,
                "severity_statistics": severity_stats,
                "recent_activity": recent_types,
                "patterns_count": len(self.patterns),
                "history_size": len(self.behavior_history)
            }
    
    def get_recent_events(self, limit: int = 50) -> List[BehaviorEvent]:
        """獲取最近的行為事件"""
        with self.lock:
            return list(self.behavior_events)[-limit:]


# 全局行為分析器實例
_behavior_analyzer = None


def get_behavior_analyzer(config: Optional[Dict[str, Any]] = None) -> BehaviorAnalyzer:
    """獲取行為分析器實例"""
    global _behavior_analyzer
    if _behavior_analyzer is None and config:
        _behavior_analyzer = BehaviorAnalyzer(config)
    return _behavior_analyzer


def init_behavior_analyzer(config: Dict[str, Any]):
    """初始化行為分析器"""
    global _behavior_analyzer
    _behavior_analyzer = BehaviorAnalyzer(config)
    logger.info("行為分析器已初始化")
