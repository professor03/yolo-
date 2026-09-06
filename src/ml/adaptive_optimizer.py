#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自適應優化器
根據系統性能自動調整參數和配置
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
class OptimizationResult:
    """優化結果"""
    parameter_name: str
    old_value: Any
    new_value: Any
    improvement: float
    confidence: float
    reason: str
    timestamp: datetime


@dataclass
class PerformanceProfile:
    """性能檔案"""
    profile_id: str
    name: str
    conditions: Dict[str, Any]  # 觸發條件
    parameters: Dict[str, Any]  # 優化參數
    performance_score: float
    usage_count: int
    last_used: datetime


class AdaptiveOptimizer:
    """自適應優化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history_size = config.get("history_size", 1000)
        self.optimization_threshold = config.get("optimization_threshold", 0.1)  # 10%改善才優化
        self.learning_rate = config.get("learning_rate", 0.1)
        
        # 數據存儲
        self.performance_history: deque = deque(maxlen=self.history_size)
        self.optimization_history: List[OptimizationResult] = []
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        
        # 當前配置
        self.current_config = config.get("initial_config", {})
        
        # 優化參數範圍
        self.parameter_ranges = {
            "fps_target": (5, 60),
            "batch_size": (1, 16),
            "queue_size": (10, 100),
            "detection_confidence": (0.1, 0.9),
            "nms_threshold": (0.1, 0.9),
            "max_detections": (10, 1000)
        }
        
        # 線程安全
        self.lock = threading.Lock()
        
        # 初始化默認性能檔案
        self._init_default_profiles()
        
        logger.info("自適應優化器已初始化")
    
    def _init_default_profiles(self):
        """初始化默認性能檔案"""
        profiles = [
            PerformanceProfile(
                profile_id="high_performance",
                name="高性能模式",
                conditions={"cpu_percent": (0, 50), "memory_percent": (0, 70)},
                parameters={
                    "fps_target": 30,
                    "batch_size": 8,
                    "queue_size": 50,
                    "detection_confidence": 0.5,
                    "nms_threshold": 0.4
                },
                performance_score=0.0,
                usage_count=0,
                last_used=datetime.now()
            ),
            PerformanceProfile(
                profile_id="balanced",
                name="平衡模式",
                conditions={"cpu_percent": (50, 80), "memory_percent": (70, 85)},
                parameters={
                    "fps_target": 20,
                    "batch_size": 4,
                    "queue_size": 30,
                    "detection_confidence": 0.6,
                    "nms_threshold": 0.5
                },
                performance_score=0.0,
                usage_count=0,
                last_used=datetime.now()
            ),
            PerformanceProfile(
                profile_id="conservative",
                name="保守模式",
                conditions={"cpu_percent": (80, 100), "memory_percent": (85, 100)},
                parameters={
                    "fps_target": 10,
                    "batch_size": 2,
                    "queue_size": 20,
                    "detection_confidence": 0.7,
                    "nms_threshold": 0.6
                },
                performance_score=0.0,
                usage_count=0,
                last_used=datetime.now()
            )
        ]
        
        for profile in profiles:
            self.performance_profiles[profile.profile_id] = profile
    
    def update_performance(self, metrics: Dict[str, Any]):
        """更新性能數據"""
        performance_data = {
            "timestamp": datetime.now(),
            "metrics": metrics.copy(),
            "config": self.current_config.copy()
        }
        
        with self.lock:
            self.performance_history.append(performance_data)
    
    def optimize_parameters(self) -> List[OptimizationResult]:
        """優化參數"""
        if len(self.performance_history) < 10:
            return []  # 需要足夠的歷史數據
        
        optimizations = []
        
        with self.lock:
            # 分析當前性能
            current_performance = self._analyze_current_performance()
            
            # 選擇最佳性能檔案
            best_profile = self._select_best_profile(current_performance)
            
            if best_profile:
                # 應用性能檔案
                profile_optimizations = self._apply_performance_profile(best_profile)
                optimizations.extend(profile_optimizations)
            
            # 細粒度參數優化
            parameter_optimizations = self._optimize_individual_parameters(current_performance)
            optimizations.extend(parameter_optimizations)
        
        return optimizations
    
    def _analyze_current_performance(self) -> Dict[str, Any]:
        """分析當前性能"""
        if not self.performance_history:
            return {}
        
        recent_data = list(self.performance_history)[-10:]  # 最近10個數據點
        
        # 計算平均性能指標
        metrics = {
            "fps_processing": [],
            "fps_inference": [],
            "latency_ms": [],
            "cpu_percent": [],
            "memory_percent": [],
            "gpu_utilization": [],
            "queue_size": [],
            "dropped_frames": []
        }
        
        for data in recent_data:
            for metric in metrics.keys():
                if metric in data["metrics"]:
                    metrics[metric].append(data["metrics"][metric])
        
        # 計算統計數據
        performance = {}
        for metric, values in metrics.items():
            if values:
                performance[f"{metric}_mean"] = np.mean(values)
                performance[f"{metric}_std"] = np.std(values)
                performance[f"{metric}_min"] = np.min(values)
                performance[f"{metric}_max"] = np.max(values)
        
        # 計算綜合性能分數
        performance["overall_score"] = self._calculate_overall_score(performance)
        
        return performance
    
    def _calculate_overall_score(self, performance: Dict[str, Any]) -> float:
        """計算綜合性能分數"""
        score = 0.0
        weight_sum = 0.0
        
        # FPS 分數 (權重 0.3)
        if "fps_processing_mean" in performance:
            fps_score = min(performance["fps_processing_mean"] / 30.0, 1.0)
            score += fps_score * 0.3
            weight_sum += 0.3
        
        # 延遲分數 (權重 0.2)
        if "latency_ms_mean" in performance:
            latency_score = max(0, 1.0 - performance["latency_ms_mean"] / 1000.0)
            score += latency_score * 0.2
            weight_sum += 0.2
        
        # CPU 使用率分數 (權重 0.2)
        if "cpu_percent_mean" in performance:
            cpu_score = max(0, 1.0 - performance["cpu_percent_mean"] / 100.0)
            score += cpu_score * 0.2
            weight_sum += 0.2
        
        # 記憶體使用率分數 (權重 0.15)
        if "memory_percent_mean" in performance:
            memory_score = max(0, 1.0 - performance["memory_percent_mean"] / 100.0)
            score += memory_score * 0.15
            weight_sum += 0.15
        
        # 穩定性分數 (權重 0.15)
        stability_score = 1.0
        for metric in ["fps_processing_std", "latency_ms_std", "cpu_percent_std"]:
            if metric in performance and performance[metric] > 0:
                # 標準差越小越穩定
                metric_stability = max(0, 1.0 - performance[metric] / (performance[metric.replace("_std", "_mean")] + 1))
                stability_score = min(stability_score, metric_stability)
        
        score += stability_score * 0.15
        weight_sum += 0.15
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _select_best_profile(self, performance: Dict[str, Any]) -> Optional[PerformanceProfile]:
        """選擇最佳性能檔案"""
        cpu_percent = performance.get("cpu_percent_mean", 50)
        memory_percent = performance.get("memory_percent_mean", 50)
        
        # 找到符合當前條件的檔案
        matching_profiles = []
        for profile in self.performance_profiles.values():
            cpu_range = profile.conditions.get("cpu_percent", (0, 100))
            memory_range = profile.conditions.get("memory_percent", (0, 100))
            
            if (cpu_range[0] <= cpu_percent <= cpu_range[1] and 
                memory_range[0] <= memory_percent <= memory_range[1]):
                matching_profiles.append(profile)
        
        if not matching_profiles:
            return None
        
        # 選擇使用次數最少或性能分數最高的檔案
        best_profile = min(matching_profiles, key=lambda p: (p.usage_count, -p.performance_score))
        
        # 更新檔案使用統計
        best_profile.usage_count += 1
        best_profile.last_used = datetime.now()
        
        return best_profile
    
    def _apply_performance_profile(self, profile: PerformanceProfile) -> List[OptimizationResult]:
        """應用性能檔案"""
        optimizations = []
        
        for param_name, param_value in profile.parameters.items():
            if param_name in self.current_config:
                old_value = self.current_config[param_name]
                if old_value != param_value:
                    # 檢查參數是否在有效範圍內
                    if param_name in self.parameter_ranges:
                        min_val, max_val = self.parameter_ranges[param_name]
                        param_value = max(min_val, min(max_val, param_value))
                    
                    self.current_config[param_name] = param_value
                    
                    optimization = OptimizationResult(
                        parameter_name=param_name,
                        old_value=old_value,
                        new_value=param_value,
                        improvement=0.0,  # 檔案切換的改善難以量化
                        confidence=0.8,   # 檔案切換的置信度較高
                        reason=f"應用性能檔案: {profile.name}",
                        timestamp=datetime.now()
                    )
                    
                    optimizations.append(optimization)
                    self.optimization_history.append(optimization)
        
        return optimizations
    
    def _optimize_individual_parameters(self, performance: Dict[str, Any]) -> List[OptimizationResult]:
        """優化個別參數"""
        optimizations = []
        
        # FPS 目標優化
        fps_optimization = self._optimize_fps_target(performance)
        if fps_optimization:
            optimizations.append(fps_optimization)
        
        # 批次大小優化
        batch_optimization = self._optimize_batch_size(performance)
        if batch_optimization:
            optimizations.append(batch_optimization)
        
        # 佇列大小優化
        queue_optimization = self._optimize_queue_size(performance)
        if queue_optimization:
            optimizations.append(queue_optimization)
        
        # 檢測置信度優化
        confidence_optimization = self._optimize_detection_confidence(performance)
        if confidence_optimization:
            optimizations.append(confidence_optimization)
        
        return optimizations
    
    def _optimize_fps_target(self, performance: Dict[str, Any]) -> Optional[OptimizationResult]:
        """優化 FPS 目標"""
        current_fps = self.current_config.get("fps_target", 20)
        actual_fps = performance.get("fps_processing_mean", 0)
        cpu_percent = performance.get("cpu_percent_mean", 50)
        
        # 如果實際 FPS 遠低於目標，降低目標
        if actual_fps < current_fps * 0.8 and cpu_percent > 70:
            new_fps = max(5, int(current_fps * 0.8))
            improvement = (current_fps - new_fps) / current_fps
            
            if improvement > self.optimization_threshold:
                self.current_config["fps_target"] = new_fps
                
                return OptimizationResult(
                    parameter_name="fps_target",
                    old_value=current_fps,
                    new_value=new_fps,
                    improvement=improvement,
                    confidence=0.7,
                    reason=f"實際FPS({actual_fps:.1f})低於目標，降低目標以減少CPU負載",
                    timestamp=datetime.now()
                )
        
        # 如果實際 FPS 接近目標且 CPU 使用率低，提高目標
        elif actual_fps >= current_fps * 0.95 and cpu_percent < 50:
            new_fps = min(60, int(current_fps * 1.2))
            improvement = (new_fps - current_fps) / current_fps
            
            if improvement > self.optimization_threshold:
                self.current_config["fps_target"] = new_fps
                
                return OptimizationResult(
                    parameter_name="fps_target",
                    old_value=current_fps,
                    new_value=new_fps,
                    improvement=improvement,
                    confidence=0.6,
                    reason=f"實際FPS({actual_fps:.1f})接近目標且CPU使用率低，提高目標",
                    timestamp=datetime.now()
                )
        
        return None
    
    def _optimize_batch_size(self, performance: Dict[str, Any]) -> Optional[OptimizationResult]:
        """優化批次大小"""
        current_batch = self.current_config.get("batch_size", 4)
        latency = performance.get("latency_ms_mean", 0)
        cpu_percent = performance.get("cpu_percent_mean", 50)
        
        # 如果延遲高且 CPU 使用率低，增加批次大小
        if latency > 500 and cpu_percent < 60:
            new_batch = min(16, current_batch * 2)
            improvement = (new_batch - current_batch) / current_batch
            
            if improvement > self.optimization_threshold:
                self.current_config["batch_size"] = new_batch
                
                return OptimizationResult(
                    parameter_name="batch_size",
                    old_value=current_batch,
                    new_value=new_batch,
                    improvement=improvement,
                    confidence=0.7,
                    reason=f"延遲高({latency:.1f}ms)且CPU使用率低，增加批次大小",
                    timestamp=datetime.now()
                )
        
        # 如果延遲低且 CPU 使用率高，減少批次大小
        elif latency < 200 and cpu_percent > 80:
            new_batch = max(1, current_batch // 2)
            improvement = (current_batch - new_batch) / current_batch
            
            if improvement > self.optimization_threshold:
                self.current_config["batch_size"] = new_batch
                
                return OptimizationResult(
                    parameter_name="batch_size",
                    old_value=current_batch,
                    new_value=new_batch,
                    improvement=improvement,
                    confidence=0.7,
                    reason=f"延遲低({latency:.1f}ms)且CPU使用率高，減少批次大小",
                    timestamp=datetime.now()
                )
        
        return None
    
    def _optimize_queue_size(self, performance: Dict[str, Any]) -> Optional[OptimizationResult]:
        """優化佇列大小"""
        current_queue = self.current_config.get("queue_size", 30)
        dropped_frames = performance.get("dropped_frames_mean", 0)
        memory_percent = performance.get("memory_percent_mean", 50)
        
        # 如果丟幀多且記憶體使用率低，增加佇列大小
        if dropped_frames > 5 and memory_percent < 70:
            new_queue = min(100, int(current_queue * 1.5))
            improvement = (new_queue - current_queue) / current_queue
            
            if improvement > self.optimization_threshold:
                self.current_config["queue_size"] = new_queue
                
                return OptimizationResult(
                    parameter_name="queue_size",
                    old_value=current_queue,
                    new_value=new_queue,
                    improvement=improvement,
                    confidence=0.6,
                    reason=f"丟幀多({dropped_frames:.1f})且記憶體使用率低，增加佇列大小",
                    timestamp=datetime.now()
                )
        
        # 如果丟幀少且記憶體使用率高，減少佇列大小
        elif dropped_frames < 1 and memory_percent > 85:
            new_queue = max(10, int(current_queue * 0.7))
            improvement = (current_queue - new_queue) / current_queue
            
            if improvement > self.optimization_threshold:
                self.current_config["queue_size"] = new_queue
                
                return OptimizationResult(
                    parameter_name="queue_size",
                    old_value=current_queue,
                    new_value=new_queue,
                    improvement=improvement,
                    confidence=0.6,
                    reason=f"丟幀少({dropped_frames:.1f})且記憶體使用率高，減少佇列大小",
                    timestamp=datetime.now()
                )
        
        return None
    
    def _optimize_detection_confidence(self, performance: Dict[str, Any]) -> Optional[OptimizationResult]:
        """優化檢測置信度"""
        current_confidence = self.current_config.get("detection_confidence", 0.5)
        fps = performance.get("fps_processing_mean", 0)
        cpu_percent = performance.get("cpu_percent_mean", 50)
        
        # 如果 FPS 低且 CPU 使用率高，提高置信度以減少檢測
        if fps < 15 and cpu_percent > 75:
            new_confidence = min(0.9, current_confidence + 0.1)
            improvement = (new_confidence - current_confidence) / current_confidence
            
            if improvement > self.optimization_threshold:
                self.current_config["detection_confidence"] = new_confidence
                
                return OptimizationResult(
                    parameter_name="detection_confidence",
                    old_value=current_confidence,
                    new_value=new_confidence,
                    improvement=improvement,
                    confidence=0.6,
                    reason=f"FPS低({fps:.1f})且CPU使用率高，提高檢測置信度",
                    timestamp=datetime.now()
                )
        
        # 如果 FPS 高且 CPU 使用率低，降低置信度以增加檢測
        elif fps > 25 and cpu_percent < 60:
            new_confidence = max(0.1, current_confidence - 0.1)
            improvement = (current_confidence - new_confidence) / current_confidence
            
            if improvement > self.optimization_threshold:
                self.current_config["detection_confidence"] = new_confidence
                
                return OptimizationResult(
                    parameter_name="detection_confidence",
                    old_value=current_confidence,
                    new_value=new_confidence,
                    improvement=improvement,
                    confidence=0.6,
                    reason=f"FPS高({fps:.1f})且CPU使用率低，降低檢測置信度",
                    timestamp=datetime.now()
                )
        
        return None
    
    def get_current_config(self) -> Dict[str, Any]:
        """獲取當前配置"""
        with self.lock:
            return self.current_config.copy()
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """獲取優化統計"""
        with self.lock:
            total_optimizations = len(self.optimization_history)
            
            # 按參數統計
            parameter_stats = {}
            for optimization in self.optimization_history:
                param = optimization.parameter_name
                if param not in parameter_stats:
                    parameter_stats[param] = 0
                parameter_stats[param] += 1
            
            # 最近優化
            recent_optimizations = [
                opt for opt in self.optimization_history
                if (datetime.now() - opt.timestamp).total_seconds() < 3600
            ]
            
            # 平均改善
            if self.optimization_history:
                avg_improvement = np.mean([opt.improvement for opt in self.optimization_history])
                avg_confidence = np.mean([opt.confidence for opt in self.optimization_history])
            else:
                avg_improvement = 0.0
                avg_confidence = 0.0
            
            return {
                "total_optimizations": total_optimizations,
                "recent_optimizations": len(recent_optimizations),
                "parameter_statistics": parameter_stats,
                "average_improvement": avg_improvement,
                "average_confidence": avg_confidence,
                "performance_profiles": len(self.performance_profiles),
                "history_size": len(self.performance_history)
            }
    
    def get_recent_optimizations(self, limit: int = 20) -> List[OptimizationResult]:
        """獲取最近的優化記錄"""
        with self.lock:
            return list(self.optimization_history)[-limit:]


# 全局自適應優化器實例
_adaptive_optimizer = None


def get_adaptive_optimizer(config: Optional[Dict[str, Any]] = None) -> AdaptiveOptimizer:
    """獲取自適應優化器實例"""
    global _adaptive_optimizer
    if _adaptive_optimizer is None and config:
        _adaptive_optimizer = AdaptiveOptimizer(config)
    return _adaptive_optimizer


def init_adaptive_optimizer(config: Dict[str, Any]):
    """初始化自適應優化器"""
    global _adaptive_optimizer
    _adaptive_optimizer = AdaptiveOptimizer(config)
    logger.info("自適應優化器已初始化")
