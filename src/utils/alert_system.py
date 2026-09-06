#!/usr/bin/env python3
"""
智能警報系統
提供規則引擎、自適應閾值、通知系統
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import smtplib
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    # 提供備用實現
    class MimeText:
        def __init__(self, *args, **kwargs):
            pass
    class MimeMultipart:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """警報級別"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """警報類型"""
    PERFORMANCE = "performance"
    DETECTION = "detection"
    SYSTEM = "system"
    SECURITY = "security"
    CUSTOM = "custom"


@dataclass
class AlertRule:
    """警報規則"""
    id: str
    name: str
    description: str
    alert_type: AlertType
    level: AlertLevel
    condition: str  # JSON 字符串，包含條件邏輯
    threshold: float
    duration: int  # 持續時間（秒）
    cooldown: int  # 冷卻時間（秒）
    enabled: bool = True
    created_at: datetime = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Alert:
    """警報實例"""
    id: str
    rule_id: str
    level: AlertLevel
    alert_type: AlertType
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            result['resolved_at'] = self.resolved_at.isoformat()
        if self.acknowledged_at:
            result['acknowledged_at'] = self.acknowledged_at.isoformat()
        return result


class NotificationChannel:
    """通知渠道基類"""
    
    def send(self, alert: Alert) -> bool:
        """發送警報通知"""
        raise NotImplementedError


class EmailNotification(NotificationChannel):
    """郵件通知"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, 
                 from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    def send(self, alert: Alert) -> bool:
        """發送郵件警報"""
        if not EMAIL_AVAILABLE:
            logger.warning("郵件功能不可用，跳過郵件發送")
            return False
            
        try:
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            body = f"""
警報詳情:
- 級別: {alert.level.value.upper()}
- 類型: {alert.alert_type.value}
- 時間: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- 描述: {alert.message}

數據:
{json.dumps(alert.data, indent=2, ensure_ascii=False)}

請及時處理此警報。
            """
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"郵件警報發送成功: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"郵件警報發送失敗: {e}")
            return False


class WebhookNotification(NotificationChannel):
    """Webhook 通知"""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    def send(self, alert: Alert) -> bool:
        """發送 Webhook 警報"""
        try:
            import requests
            
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook 警報發送成功: {alert.id}")
                return True
            else:
                logger.error(f"Webhook 警報發送失敗: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Webhook 警報發送失敗: {e}")
            return False


class AlertSystem:
    """智能警報系統"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.notification_channels: List[NotificationChannel] = []
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        
        # 自適應閾值
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = {}
        self.threshold_learning_rate = 0.1
        
        # 線程安全
        self.lock = threading.Lock()
        
        # 初始化默認規則
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默認警報規則"""
        default_rules = [
            AlertRule(
                id="high_cpu_usage",
                name="高 CPU 使用率",
                description="CPU 使用率超過 80%",
                alert_type=AlertType.PERFORMANCE,
                level=AlertLevel.WARNING,
                condition='{"metric": "cpu_percent", "operator": ">", "value": 80}',
                threshold=80.0,
                duration=30,
                cooldown=300
            ),
            AlertRule(
                id="high_memory_usage",
                name="高記憶體使用率",
                description="記憶體使用率超過 85%",
                alert_type=AlertType.PERFORMANCE,
                level=AlertLevel.WARNING,
                condition='{"metric": "memory_percent", "operator": ">", "value": 85}',
                threshold=85.0,
                duration=30,
                cooldown=300
            ),
            AlertRule(
                id="low_fps",
                name="低 FPS 警報",
                description="處理 FPS 低於 10",
                alert_type=AlertType.PERFORMANCE,
                level=AlertLevel.CRITICAL,
                condition='{"metric": "fps_processing", "operator": "<", "value": 10}',
                threshold=10.0,
                duration=60,
                cooldown=600
            ),
            AlertRule(
                id="high_latency",
                name="高延遲警報",
                description="系統延遲超過 1000ms",
                alert_type=AlertType.PERFORMANCE,
                level=AlertLevel.WARNING,
                condition='{"metric": "latency_ms", "operator": ">", "value": 1000}',
                threshold=1000.0,
                duration=30,
                cooldown=300
            ),
            AlertRule(
                id="detection_failure",
                name="檢測失敗",
                description="連續 5 分鐘無檢測結果",
                alert_type=AlertType.DETECTION,
                level=AlertLevel.CRITICAL,
                condition='{"metric": "detection_count", "operator": "==", "value": 0}',
                threshold=0.0,
                duration=300,
                cooldown=1800
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: AlertRule) -> bool:
        """添加警報規則"""
        try:
            with self.lock:
                self.rules[rule.id] = rule
                logger.info(f"警報規則已添加: {rule.name}")
                return True
        except Exception as e:
            logger.error(f"添加警報規則失敗: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除警報規則"""
        try:
            with self.lock:
                if rule_id in self.rules:
                    del self.rules[rule_id]
                    logger.info(f"警報規則已移除: {rule_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"移除警報規則失敗: {e}")
            return False
    
    def add_notification_channel(self, channel: NotificationChannel):
        """添加通知渠道"""
        self.notification_channels.append(channel)
        logger.info("通知渠道已添加")
    
    def evaluate_metrics(self, metrics: Dict[str, Any]) -> List[Alert]:
        """評估指標並生成警報"""
        triggered_alerts = []
        
        with self.lock:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                # 檢查冷卻時間
                if rule.last_triggered:
                    time_since_trigger = (datetime.now() - rule.last_triggered).total_seconds()
                    if time_since_trigger < rule.cooldown:
                        continue
                
                # 評估條件
                if self._evaluate_condition(rule, metrics):
                    # 檢查持續時間
                    if self._check_duration(rule, metrics):
                        alert = self._create_alert(rule, metrics)
                        triggered_alerts.append(alert)
                        
                        # 更新規則狀態
                        rule.last_triggered = datetime.now()
                        rule.trigger_count += 1
                        
                        # 發送通知
                        self._send_notifications(alert)
        
        return triggered_alerts
    
    def _evaluate_condition(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """評估警報條件"""
        try:
            condition = json.loads(rule.condition)
            metric_name = condition.get("metric")
            operator = condition.get("operator")
            value = condition.get("value")
            
            if metric_name not in metrics:
                return False
            
            metric_value = metrics[metric_name]
            
            # 應用自適應閾值
            adaptive_value = self._get_adaptive_threshold(rule.id, metric_name, value)
            
            if operator == ">":
                return metric_value > adaptive_value
            elif operator == ">=":
                return metric_value >= adaptive_value
            elif operator == "<":
                return metric_value < adaptive_value
            elif operator == "<=":
                return metric_value <= adaptive_value
            elif operator == "==":
                return metric_value == adaptive_value
            elif operator == "!=":
                return metric_value != adaptive_value
            else:
                return False
                
        except Exception as e:
            logger.error(f"評估警報條件失敗: {e}")
            return False
    
    def _check_duration(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """檢查持續時間"""
        # 簡化實現，實際應該檢查歷史數據
        return True
    
    def _create_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> Alert:
        """創建警報實例"""
        alert_id = f"{rule.id}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            level=rule.level,
            alert_type=rule.alert_type,
            title=f"{rule.name} - {rule.level.value.upper()}",
            message=rule.description,
            data=metrics.copy(),
            timestamp=datetime.now()
        )
        
        with self.lock:
            self.alerts.append(alert)
            self.alert_history.append(alert)
            
            # 保持歷史記錄數量限制
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
        
        logger.info(f"警報已創建: {alert.title}")
        return alert
    
    def _send_notifications(self, alert: Alert):
        """發送通知"""
        for channel in self.notification_channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(f"發送通知失敗: {e}")
    
    def _get_adaptive_threshold(self, rule_id: str, metric_name: str, default_value: float) -> float:
        """獲取自適應閾值"""
        if rule_id not in self.adaptive_thresholds:
            self.adaptive_thresholds[rule_id] = {}
        
        if metric_name not in self.adaptive_thresholds[rule_id]:
            self.adaptive_thresholds[rule_id][metric_name] = default_value
        
        return self.adaptive_thresholds[rule_id][metric_name]
    
    def update_adaptive_threshold(self, rule_id: str, metric_name: str, actual_value: float):
        """更新自適應閾值"""
        if rule_id not in self.adaptive_thresholds:
            self.adaptive_thresholds[rule_id] = {}
        
        current_threshold = self.adaptive_thresholds[rule_id].get(metric_name, 0)
        new_threshold = current_threshold + self.threshold_learning_rate * (actual_value - current_threshold)
        self.adaptive_thresholds[rule_id][metric_name] = new_threshold
    
    def get_active_alerts(self) -> List[Alert]:
        """獲取活躍警報"""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """獲取警報歷史"""
        with self.lock:
            return self.alert_history[-limit:]
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """解決警報"""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"警報已解決: {alert_id}")
                    return True
        return False
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """確認警報"""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = datetime.now()
                    logger.info(f"警報已確認: {alert_id} by {acknowledged_by}")
                    return True
        return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """獲取警報統計"""
        with self.lock:
            total_alerts = len(self.alert_history)
            active_alerts = len([a for a in self.alerts if not a.resolved])
            resolved_alerts = len([a for a in self.alert_history if a.resolved])
            
            # 按級別統計
            level_stats = {}
            for level in AlertLevel:
                level_stats[level.value] = len([a for a in self.alert_history if a.level == level])
            
            # 按類型統計
            type_stats = {}
            for alert_type in AlertType:
                type_stats[alert_type.value] = len([a for a in self.alert_history if a.alert_type == alert_type])
            
            return {
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "resolved_alerts": resolved_alerts,
                "level_statistics": level_stats,
                "type_statistics": type_stats,
                "rules_count": len(self.rules),
                "enabled_rules": len([r for r in self.rules.values() if r.enabled])
            }


# 全局警報系統實例
_alert_system = None


def get_alert_system() -> AlertSystem:
    """獲取警報系統實例"""
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertSystem()
    return _alert_system


def init_alert_system():
    """初始化警報系統"""
    global _alert_system
    _alert_system = AlertSystem()
    logger.info("警報系統已初始化")
