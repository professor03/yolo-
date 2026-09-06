#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人流預測模型
基於歷史數據預測未來人流趨勢
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
class PredictionResult:
    """預測結果"""
    timestamp: datetime
    predicted_count: float
    confidence: float
    prediction_horizon: int  # 預測時間範圍（分鐘）
    features_used: List[str]
    model_type: str


@dataclass
class TrafficPattern:
    """流量模式"""
    hour: int
    day_of_week: int
    average_count: float
    std_dev: float
    pattern_type: str  # peak, normal, low


class TrafficPredictor:
    """人流預測器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history_size = config.get("history_size", 10000)
        self.prediction_horizon = config.get("prediction_horizon", 60)  # 預測60分鐘
        self.min_history_for_prediction = config.get("min_history", 100)
        
        # 數據存儲
        self.traffic_history: deque = deque(maxlen=self.history_size)
        self.patterns: Dict[str, TrafficPattern] = {}
        
        # 預測模型參數
        self.model_params = {
            "trend_weight": 0.3,
            "seasonal_weight": 0.4,
            "recent_weight": 0.3,
            "smoothing_factor": 0.1
        }
        
        # 線程安全
        self.lock = threading.Lock()
        
        logger.info("人流預測器已初始化")
    
    def add_traffic_data(self, timestamp: datetime, count: int, features: Dict[str, Any] = None):
        """添加流量數據"""
        data_point = {
            "timestamp": timestamp,
            "count": count,
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "features": features or {}
        }
        
        with self.lock:
            self.traffic_history.append(data_point)
            
            # 更新模式
            self._update_patterns()
    
    def _update_patterns(self):
        """更新流量模式"""
        if len(self.traffic_history) < 24:  # 至少需要24小時的數據
            return
        
        # 按小時和星期幾分組
        hourly_data = {}
        for data in self.traffic_history:
            key = (data["hour"], data["day_of_week"])
            if key not in hourly_data:
                hourly_data[key] = []
            hourly_data[key].append(data["count"])
        
        # 計算每個時段的統計數據
        for (hour, day_of_week), counts in hourly_data.items():
            if len(counts) < 3:  # 至少需要3個數據點
                continue
            
            pattern_key = f"{hour}_{day_of_week}"
            avg_count = np.mean(counts)
            std_dev = np.std(counts)
            
            # 確定模式類型
            if avg_count > np.mean([np.mean(c) for c in hourly_data.values()]) * 1.5:
                pattern_type = "peak"
            elif avg_count < np.mean([np.mean(c) for c in hourly_data.values()]) * 0.5:
                pattern_type = "low"
            else:
                pattern_type = "normal"
            
            self.patterns[pattern_key] = TrafficPattern(
                hour=hour,
                day_of_week=day_of_week,
                average_count=avg_count,
                std_dev=std_dev,
                pattern_type=pattern_type
            )
    
    def predict_traffic(self, prediction_time: datetime) -> Optional[PredictionResult]:
        """預測指定時間的流量"""
        with self.lock:
            if len(self.traffic_history) < self.min_history_for_prediction:
                logger.warning("歷史數據不足，無法進行預測")
                return None
            
            # 獲取相關模式
            pattern_key = f"{prediction_time.hour}_{prediction_time.weekday()}"
            pattern = self.patterns.get(pattern_key)
            
            if not pattern:
                # 如果沒有該時段的模式，使用最近時段的模式
                pattern = self._find_closest_pattern(prediction_time)
            
            if not pattern:
                logger.warning("無法找到合適的流量模式")
                return None
            
            # 計算趨勢
            trend = self._calculate_trend()
            
            # 計算季節性調整
            seasonal_adjustment = self._calculate_seasonal_adjustment(prediction_time)
            
            # 計算最近數據的影響
            recent_influence = self._calculate_recent_influence()
            
            # 綜合預測
            base_prediction = pattern.average_count
            trend_adjustment = trend * self.model_params["trend_weight"]
            seasonal_adjustment = seasonal_adjustment * self.model_params["seasonal_weight"]
            recent_adjustment = recent_influence * self.model_params["recent_weight"]
            
            predicted_count = base_prediction + trend_adjustment + seasonal_adjustment + recent_adjustment
            predicted_count = max(0, predicted_count)  # 確保非負
            
            # 計算置信度
            confidence = self._calculate_confidence(pattern, len(self.traffic_history))
            
            return PredictionResult(
                timestamp=prediction_time,
                predicted_count=predicted_count,
                confidence=confidence,
                prediction_horizon=self.prediction_horizon,
                features_used=["hour", "day_of_week", "trend", "seasonal", "recent"],
                model_type="hybrid_pattern_trend"
            )
    
    def _find_closest_pattern(self, prediction_time: datetime) -> Optional[TrafficPattern]:
        """尋找最接近的模式"""
        target_hour = prediction_time.hour
        target_day = prediction_time.weekday()
        
        min_distance = float('inf')
        closest_pattern = None
        
        for pattern in self.patterns.values():
            # 計算時間距離（小時差）
            hour_diff = min(abs(pattern.hour - target_hour), 24 - abs(pattern.hour - target_hour))
            day_diff = min(abs(pattern.day_of_week - target_day), 7 - abs(pattern.day_of_week - target_day))
            
            distance = hour_diff + day_diff * 24
            
            if distance < min_distance:
                min_distance = distance
                closest_pattern = pattern
        
        return closest_pattern
    
    def _calculate_trend(self) -> float:
        """計算趨勢"""
        if len(self.traffic_history) < 10:
            return 0.0
        
        # 使用最近100個數據點計算趨勢
        recent_data = list(self.traffic_history)[-100:]
        counts = [d["count"] for d in recent_data]
        
        if len(counts) < 2:
            return 0.0
        
        # 簡單線性回歸
        x = np.arange(len(counts))
        y = np.array(counts)
        
        # 計算斜率
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def _calculate_seasonal_adjustment(self, prediction_time: datetime) -> float:
        """計算季節性調整"""
        # 簡化實現：基於小時的調整
        hour = prediction_time.hour
        
        # 定義一天中的流量模式
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # 高峰時段
            return 0.2
        elif 10 <= hour <= 16:  # 正常時段
            return 0.0
        elif 20 <= hour <= 23 or 0 <= hour <= 6:  # 低峰時段
            return -0.3
        else:
            return 0.0
    
    def _calculate_recent_influence(self) -> float:
        """計算最近數據的影響"""
        if len(self.traffic_history) < 5:
            return 0.0
        
        # 使用最近5個數據點
        recent_data = list(self.traffic_history)[-5:]
        recent_counts = [d["count"] for d in recent_data]
        
        # 計算與歷史平均的偏差
        historical_avg = np.mean([d["count"] for d in self.traffic_history])
        recent_avg = np.mean(recent_counts)
        
        if historical_avg == 0:
            return 0.0
        
        return (recent_avg - historical_avg) / historical_avg
    
    def _calculate_confidence(self, pattern: TrafficPattern, data_size: int) -> float:
        """計算預測置信度"""
        # 基於數據量和模式穩定性
        data_confidence = min(data_size / self.min_history_for_prediction, 1.0)
        
        # 基於模式穩定性（標準差越小越穩定）
        stability_confidence = max(0, 1.0 - pattern.std_dev / (pattern.average_count + 1))
        
        # 綜合置信度
        confidence = (data_confidence * 0.6 + stability_confidence * 0.4)
        return min(confidence, 1.0)
    
    def get_traffic_patterns(self) -> Dict[str, Any]:
        """獲取流量模式"""
        with self.lock:
            patterns_data = {}
            for key, pattern in self.patterns.items():
                patterns_data[key] = {
                    "hour": pattern.hour,
                    "day_of_week": pattern.day_of_week,
                    "average_count": pattern.average_count,
                    "std_dev": pattern.std_dev,
                    "pattern_type": pattern.pattern_type
                }
            
            return {
                "patterns": patterns_data,
                "total_patterns": len(self.patterns),
                "data_points": len(self.traffic_history)
            }
    
    def get_prediction_accuracy(self, days: int = 7) -> Dict[str, Any]:
        """獲取預測準確性"""
        with self.lock:
            if len(self.traffic_history) < 10:
                return {"error": "數據不足"}
            
            # 使用最近幾天的數據進行回測
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_data = [d for d in self.traffic_history if d["timestamp"] > cutoff_time]
            
            if len(recent_data) < 10:
                return {"error": "最近數據不足"}
            
            predictions = []
            actuals = []
            
            # 對每個數據點進行預測（使用之前的數據）
            for i in range(10, len(recent_data)):
                # 使用前i-1個數據點預測第i個
                prediction_time = recent_data[i]["timestamp"]
                prediction = self.predict_traffic(prediction_time)
                
                if prediction:
                    predictions.append(prediction.predicted_count)
                    actuals.append(recent_data[i]["count"])
            
            if len(predictions) == 0:
                return {"error": "無法生成預測"}
            
            # 計算準確性指標
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            mae = np.mean(np.abs(predictions - actuals))  # 平均絕對誤差
            mse = np.mean((predictions - actuals) ** 2)   # 均方誤差
            rmse = np.sqrt(mse)                           # 均方根誤差
            
            # 計算相對誤差
            relative_errors = np.abs(predictions - actuals) / (actuals + 1)
            mape = np.mean(relative_errors) * 100         # 平均絕對百分比誤差
            
            return {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "mape": float(mape),
                "predictions_count": len(predictions),
                "accuracy_score": max(0, 1.0 - mape / 100)
            }


# 全局人流預測器實例
_traffic_predictor = None


def get_traffic_predictor(config: Optional[Dict[str, Any]] = None) -> TrafficPredictor:
    """獲取人流預測器實例"""
    global _traffic_predictor
    if _traffic_predictor is None and config:
        _traffic_predictor = TrafficPredictor(config)
    return _traffic_predictor


def init_traffic_predictor(config: Dict[str, Any]):
    """初始化人流預測器"""
    global _traffic_predictor
    _traffic_predictor = TrafficPredictor(config)
    logger.info("人流預測器已初始化")
