#!/usr/bin/env python3
"""
資料庫依賴注入
"""

from sqlalchemy.orm import Session
from .models import get_database_manager

# 創建資料庫管理器
db_manager = get_database_manager()

def get_db_session():
    """獲取資料庫會話（用於依賴注入）"""
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()
