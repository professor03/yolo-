#!/usr/bin/env python3
"""
錯誤處理和恢復模組
提供統一的錯誤處理、重試機制和恢復策略
"""

import logging
import time
import functools
from typing import Callable, Any, Optional, Dict, List, Union
from enum import Enum
import traceback
import asyncio

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """錯誤嚴重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """錯誤類別"""
    NETWORK = "network"
    DATABASE = "database"
    MODEL = "model"
    CAMERA = "camera"
    MEMORY = "memory"
    CONFIG = "config"
    UNKNOWN = "unknown"

class ErrorHandler:
    """統一錯誤處理器"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_thresholds = {
            ErrorSeverity.LOW: 10,
            ErrorSeverity.MEDIUM: 5,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 1
        }
        self.recovery_strategies = {}
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """分類錯誤"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        if any(keyword in error_msg for keyword in ['connection', 'network', 'timeout', 'socket']):
            return ErrorCategory.NETWORK
        elif any(keyword in error_msg for keyword in ['database', 'sql', 'query', 'transaction']):
            return ErrorCategory.DATABASE
        elif any(keyword in error_msg for keyword in ['model', 'tensor', 'cuda', 'gpu']):
            return ErrorCategory.MODEL
        elif any(keyword in error_msg for keyword in ['camera', 'video', 'capture', 'frame']):
            return ErrorCategory.CAMERA
        elif any(keyword in error_msg for keyword in ['memory', 'out of memory', 'allocation']):
            return ErrorCategory.MEMORY
        elif any(keyword in error_msg for keyword in ['config', 'yaml', 'json', 'setting']):
            return ErrorCategory.CONFIG
        else:
            return ErrorCategory.UNKNOWN
    
    def get_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """判斷錯誤嚴重程度"""
        error_type = type(error).__name__
        
        # 根據錯誤類型和內容判斷嚴重程度
        if category == ErrorCategory.MEMORY:
            return ErrorSeverity.CRITICAL
        elif category == ErrorCategory.DATABASE:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.MODEL:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.NETWORK:
            return ErrorSeverity.MEDIUM
        elif category == ErrorCategory.CAMERA:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def should_retry(self, error: Exception, retry_count: int) -> bool:
        """判斷是否應該重試"""
        category = self.categorize_error(error)
        severity = self.get_severity(error, category)
        
        # 根據嚴重程度決定重試次數
        max_retries = {
            ErrorSeverity.LOW: 5,
            ErrorSeverity.MEDIUM: 3,
            ErrorSeverity.HIGH: 1,
            ErrorSeverity.CRITICAL: 0
        }
        
        return retry_count < max_retries[severity]
    
    def get_retry_delay(self, error: Exception, retry_count: int) -> float:
        """計算重試延遲時間"""
        category = self.categorize_error(error)
        
        # 根據錯誤類別使用不同的延遲策略
        base_delays = {
            ErrorCategory.NETWORK: 1.0,
            ErrorCategory.DATABASE: 0.5,
            ErrorCategory.MODEL: 2.0,
            ErrorCategory.CAMERA: 1.5,
            ErrorCategory.MEMORY: 5.0,
            ErrorCategory.CONFIG: 0.1,
            ErrorCategory.UNKNOWN: 1.0
        }
        
        base_delay = base_delays[category]
        # 指數退避
        return base_delay * (2 ** retry_count)
    
    def handle_error(self, error: Exception, context: str = "", 
                    retry_count: int = 0) -> bool:
        """處理錯誤並決定是否重試"""
        category = self.categorize_error(error)
        severity = self.get_severity(error, category)
        
        # 記錄錯誤
        error_key = f"{category.value}_{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # 記錄日誌
        log_level = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, 
                  f"錯誤處理 [{category.value}] [{severity.value}]: {error} "
                  f"(上下文: {context}, 重試: {retry_count})")
        
        # 檢查錯誤閾值
        if self.error_counts[error_key] > self.error_thresholds[severity]:
            logger.critical(f"錯誤頻率過高: {error_key} ({self.error_counts[error_key]} 次)")
            return False
        
        # 決定是否重試
        return self.should_retry(error, retry_count)

def retry_on_error(max_retries: int = 3, 
                  delay: float = 1.0,
                  backoff_factor: float = 2.0,
                  exceptions: tuple = (Exception,)):
    """重試裝飾器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if not error_handler.handle_error(e, f"{func.__name__}", attempt):
                        break
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(f"重試 {func.__name__} 在 {wait_time:.2f} 秒後 (嘗試 {attempt + 1}/{max_retries + 1})")
                        time.sleep(wait_time)
            
            # 所有重試都失敗了
            logger.error(f"函數 {func.__name__} 在 {max_retries + 1} 次嘗試後仍然失敗")
            raise last_exception
        
        return wrapper
    return decorator

def async_retry_on_error(max_retries: int = 3,
                        delay: float = 1.0,
                        backoff_factor: float = 2.0,
                        exceptions: tuple = (Exception,)):
    """非同步重試裝飾器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if not error_handler.handle_error(e, f"{func.__name__}", attempt):
                        break
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(f"重試 {func.__name__} 在 {wait_time:.2f} 秒後 (嘗試 {attempt + 1}/{max_retries + 1})")
                        await asyncio.sleep(wait_time)
            
            # 所有重試都失敗了
            logger.error(f"函數 {func.__name__} 在 {max_retries + 1} 次嘗試後仍然失敗")
            raise last_exception
        
        return wrapper
    return decorator

class CircuitBreaker:
    """斷路器模式"""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """執行函數並處理斷路器邏輯"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("斷路器開啟，服務暫時不可用")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """成功時重置計數器"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """失敗時更新計數器"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"斷路器開啟，失敗次數: {self.failure_count}")

def safe_execute(func: Callable, *args, 
                default_return: Any = None,
                log_errors: bool = True,
                **kwargs) -> Any:
    """安全執行函數，捕獲所有異常"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"安全執行失敗: {func.__name__} - {e}")
        return default_return

def create_error_context(func_name: str, **context) -> Dict[str, Any]:
    """創建錯誤上下文"""
    return {
        "function": func_name,
        "timestamp": time.time(),
        "context": context
    }
