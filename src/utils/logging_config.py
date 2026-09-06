"""改進的日誌配置 - P3企業級"""
import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON格式日誌格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # 添加異常信息
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # 添加額外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class StructuredLogger:
    """結構化日誌記錄器"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def _log_with_extra(self, level: int, message: str, **kwargs):
        """記錄帶額外字段的日誌"""
        extra_fields = {k: v for k, v in kwargs.items() if not k.startswith('_')}
        self.logger.log(level, message, extra={'extra_fields': extra_fields})
    
    def debug(self, message: str, **kwargs):
        self._log_with_extra(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log_with_extra(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_extra(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_extra(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log_with_extra(logging.CRITICAL, message, **kwargs)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "text",
    log_file: str = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True
) -> logging.Logger:
    """
    設置企業級日誌配置
    
    Args:
        log_level: 日誌級別 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 日誌格式 (text, json)
        log_file: 日誌檔案路徑
        max_file_size: 最大檔案大小（字節）
        backup_count: 備份檔案數量
        enable_console: 是否啟用控制台輸出
    
    Returns:
        配置好的根日誌記錄器
    """
    # 創建根日誌記錄器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除現有處理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 設置格式化器
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # 控制台處理器
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 檔案處理器
    if log_file:
        # 確保日誌目錄存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 旋轉檔案處理器
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 設置第三方庫日誌級別
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    return root_logger


def get_structured_logger(name: str) -> StructuredLogger:
    """獲取結構化日誌記錄器"""
    return StructuredLogger(name)


# 預設配置
def setup_default_logging():
    """設置預設日誌配置"""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "text")
    log_file = os.getenv("LOG_FILE", "logs/app.log")
    
    return setup_logging(
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        enable_console=True
    )


# 性能日誌裝飾器
def log_performance(logger: logging.Logger):
    """性能日誌裝飾器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Function {func.__name__} completed successfully",
                    function=func.__name__,
                    execution_time=execution_time,
                    status="success"
                )
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Function {func.__name__} failed",
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    status="error"
                )
                raise
        return wrapper
    return decorator


# 檢測日誌記錄器
detection_logger = get_structured_logger("detection")
api_logger = get_structured_logger("api")
performance_logger = get_structured_logger("performance")
