#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能異常檢測器
基於機器學習的異常檢測系統
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import threading
import json

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    """異常分數"""
    score: float
    confidence: float
    anomaly_type: str
    description: str
    timestamp: datetime
    features: Dict[str, float]


@dataclass
class AnomalyEvent:
    """異常事件"""
    event_id: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    score: float
    confidence: float
    description: str
    features: Dict[str, float]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class AnomalyDetector:
    """智能異常檢測器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history_size = config.get("history_size", 1000)
        self.anomaly_threshold = config.get("anomaly_threshold", 0.7)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        
        # 數據存儲
        self.feature_history: deque = deque(maxlen=self.history_size)
        self.anomaly_events: List[AnomalyEvent] = []
        
        # 統計數據
        self.baseline_stats = {
            "fps_mean": 0.0,
            "fps_std": 0.0,
            "latency_mean": 0.0,
            "latency_std": 0.0,
            "cpu_mean": 0.0,
            "cpu_std": 0.0,
            "memory_mean": 0.0,
            "memory_std": 0.0
        }
        
        # 線程安全
        self.lock = threading.Lock()
        
        # 異常類型定義
        self.anomaly_types = {
            "performance_degradation": "性能下降",
            "unusual_traffic_pattern": "異常流量模式",
            "system_overload": "系統過載",
            "detection_failure": "檢測失敗",
            "resource_exhaustion": "資源耗盡",
            "network_anomaly": "網絡異常"
        }
        
        logger.info("智能異常檢測器已初始化")
    
    def update_baseline(self, features: Dict[str, float]):
        """更新基線統計數據"""
        with self.lock:
            self.feature_history.append(features)
            
            if len(self.feature_history) < 10:
                return  # 需要足夠的數據來計算基線
            
            # 計算基線統計
            fps_values = [f.get("fps_processing", 0) for f in self.feature_history]
            latency_values = [f.get("latency_ms", 0) for f in self.feature_history]
            cpu_values = [f.get("cpu_percent", 0) for f in self.feature_history]
            memory_values = [f.get("memory_percent", 0) for f in self.feature_history]
            
            self.baseline_stats.update({
                "fps_mean": np.mean(fps_values),
                "fps_std": np.std(fps_values),
                "latency_mean": np.mean(latency_values),
                "latency_std": np.std(latency_values),
                "cpu_mean": np.mean(cpu_values),
                "cpu_std": np.std(cpu_values),
                "memory_mean": np.mean(memory_values),
                "memory_std": np.std(memory_values)
            })
    
    def detect_anomalies(self, features: Dict[str, float]) -> List[AnomalyScore]:
        """檢測異常"""
        anomalies = []
        
        with self.lock:
            if len(self.feature_history) < 10:
                return anomalies  # 需要足夠的歷史數據
            
            # 性能下降檢測
            fps_anomaly = self._detect_performance_degradation(features)
            if fps_anomaly:
                anomalies.append(fps_anomaly)
            
            # 系統過載檢測
            overload_anomaly = self._detect_system_overload(features)
            if overload_anomaly:
                anomalies.append(overload_anomaly)
            
            # 異常流量模式檢測
            traffic_anomaly = self._detect_traffic_anomaly(features)
            if traffic_anomaly:
                anomalies.append(traffic_anomaly)
            
            # 檢測失敗檢測
            detection_anomaly = self._detect_detection_failure(features)
            if detection_anomaly:
                anomalies.append(detection_anomaly)
        
        return anomalies
    
    def _detect_performance_degradation(self, features: Dict[str, float]) -> Optional[AnomalyScore]:
        """檢測性能下降"""
        fps = features.get("fps_processing", 0)
        latency = features.get("latency_ms", 0)
        
        # FPS 異常檢測
        fps_score = 0.0
        if self.baseline_stats["fps_std"] > 0:
            fps_z_score = abs(fps - self.baseline_stats["fps_mean"]) / self.baseline_stats["fps_std"]
            if fps_z_score > 2.0:  # 2個標準差
                fps_score = min(fps_z_score / 3.0, 1.0)
        
        # 延遲異常檢測
        latency_score = 0.0
        if self.baseline_stats["latency_std"] > 0:
            latency_z_score = abs(latency - self.baseline_stats["latency_mean"]) / self.baseline_stats["latency_std"]
            if latency_z_score > 2.0:
                latency_score = min(latency_z_score / 3.0, 1.0)
        
        # 綜合評分
        overall_score = max(fps_score, latency_score)
        
        if overall_score > self.anomaly_threshold:
            confidence = min(overall_score * 1.2, 1.0)
            return AnomalyScore(
                score=overall_score,
                confidence=confidence,
                anomaly_type="performance_degradation",
                description=f"性能下降檢測: FPS={fps:.1f}, 延遲={latency:.1f}ms",
                timestamp=datetime.now(),
                features=features
            )
        
        return None
    
    def _detect_system_overload(self, features: Dict[str, float]) -> Optional[AnomalyScore]:
        """檢測系統過載"""
        cpu = features.get("cpu_percent", 0)
        memory = features.get("memory_percent", 0)
        
        # CPU 過載檢測
        cpu_score = 0.0
        if cpu > 90:
            cpu_score = (cpu - 90) / 10.0  # 90%以上開始計分
        
        # 記憶體過載檢測
        memory_score = 0.0
        if memory > 85:
            memory_score = (memory - 85) / 15.0  # 85%以上開始計分
        
        overall_score = max(cpu_score, memory_score)
        
        if overall_score > self.anomaly_threshold:
            confidence = min(overall_score * 1.1, 1.0)
            return AnomalyScore(
                score=overall_score,
                confidence=confidence,
                anomaly_type="system_overload",
                description=f"系統過載檢測: CPU={cpu:.1f}%, 記憶體={memory:.1f}%",
                timestamp=datetime.now(),
                features=features
            )
        
        return None
    
    def _detect_traffic_anomaly(self, features: Dict[str, float]) -> Optional[AnomalyScore]:
        """檢測異常流量模式"""
        # 這裡可以實現更複雜的流量模式分析
        # 例如：突然的流量激增、異常的流量分布等
        
        # 簡化實現：檢測突然的流量變化
        if len(self.feature_history) < 5:
            return None
        
        recent_fps = [f.get("fps_processing", 0) for f in list(self.feature_history)[-5:]]
        current_fps = features.get("fps_processing", 0)
        
        if len(recent_fps) > 0:
            avg_recent_fps = np.mean(recent_fps)
            if avg_recent_fps > 0:
                change_ratio = abs(current_fps - avg_recent_fps) / avg_recent_fps
                if change_ratio > 0.5:  # 50%以上的變化
                    score = min(change_ratio, 1.0)
                    confidence = min(score * 0.8, 1.0)
                    
                    return AnomalyScore(
                        score=score,
                        confidence=confidence,
                        anomaly_type="unusual_traffic_pattern",
                        description=f"異常流量模式: 當前FPS={current_fps:.1f}, 平均FPS={avg_recent_fps:.1f}",
                        timestamp=datetime.now(),
                        features=features
                    )
        
        return None
    
    def _detect_detection_failure(self, features: Dict[str, float]) -> Optional[AnomalyScore]:
        """檢測檢測失敗"""
        detection_count = features.get("detection_count", 0)
        fps = features.get("fps_processing", 0)
        
        # 如果處理FPS正常但檢測數量為0，可能是檢測失敗
        if fps > 5 and detection_count == 0:
            score = 0.8
            confidence = 0.9
            
            return AnomalyScore(
                score=score,
                confidence=confidence,
                anomaly_type="detection_failure",
                description="檢測失敗: 系統運行正常但無檢測結果",
                timestamp=datetime.now(),
                features=features
            )
        
        return None
    
    def create_anomaly_event(self, anomaly_score: AnomalyScore) -> AnomalyEvent:
        """創建異常事件"""
        event_id = f"anomaly_{int(datetime.now().timestamp())}"
        
        # 根據分數確定嚴重程度
        if anomaly_score.score >= 0.9:
            severity = "critical"
        elif anomaly_score.score >= 0.7:
            severity = "high"
        elif anomaly_score.score >= 0.5:
            severity = "medium"
        else:
            severity = "low"
        
        event = AnomalyEvent(
            event_id=event_id,
            anomaly_type=anomaly_score.anomaly_type,
            severity=severity,
            score=anomaly_score.score,
            confidence=anomaly_score.confidence,
            description=anomaly_score.description,
            features=anomaly_score.features,
            timestamp=anomaly_score.timestamp
        )
        
        with self.lock:
            self.anomaly_events.append(event)
            
            # 保持事件數量限制
            if len(self.anomaly_events) > self.history_size:
                self.anomaly_events.pop(0)
        
        return event
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """獲取異常統計"""
        with self.lock:
            total_anomalies = len(self.anomaly_events)
            unresolved = len([e for e in self.anomaly_events if not e.resolved])
            
            # 按類型統計
            type_stats = {}
            for anomaly_type in self.anomaly_types.keys():
                type_stats[anomaly_type] = len([
                    e for e in self.anomaly_events 
                    if e.anomaly_type == anomaly_type
                ])
            
            # 按嚴重程度統計
            severity_stats = {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0
            }
            for event in self.anomaly_events:
                severity_stats[event.severity] += 1
            
            return {
                "total_anomalies": total_anomalies,
                "unresolved_anomalies": unresolved,
                "resolved_anomalies": total_anomalies - unresolved,
                "type_statistics": type_stats,
                "severity_statistics": severity_stats,
                "baseline_stats": self.baseline_stats,
                "history_size": len(self.feature_history)
            }
    
    def resolve_anomaly(self, event_id: str) -> bool:
        """解決異常事件"""
        with self.lock:
            for event in self.anomaly_events:
                if event.event_id == event_id and not event.resolved:
                    event.resolved = True
                    event.resolved_at = datetime.now()
                    logger.info(f"異常事件已解決: {event_id}")
                    return True
        return False
    
    def get_recent_anomalies(self, limit: int = 50) -> List[AnomalyEvent]:
        """獲取最近的異常事件"""
        with self.lock:
            return list(self.anomaly_events)[-limit:]
    
    def clear_old_anomalies(self, days: int = 7):
        """清理舊的異常事件"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with self.lock:
            self.anomaly_events = [
                event for event in self.anomaly_events
                if event.timestamp > cutoff_time
            ]
        
        logger.info(f"已清理 {days} 天前的異常事件")


# 全局異常檢測器實例
_anomaly_detector = None


def get_anomaly_detector(config: Optional[Dict[str, Any]] = None) -> AnomalyDetector:
    """獲取異常檢測器實例"""
    global _anomaly_detector
    if _anomaly_detector is None and config:
        _anomaly_detector = AnomalyDetector(config)
    return _anomaly_detector


def init_anomaly_detector(config: Dict[str, Any]):
    """初始化異常檢測器"""
    global _anomaly_detector
    _anomaly_detector = AnomalyDetector(config)
    logger.info("異常檢測器已初始化")
