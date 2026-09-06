#!/usr/bin/env python3
"""
認證依賴項
提供 FastAPI 依賴注入的認證功能
"""

from typing import Optional
import os
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.api_key import APIKeyHeader

from .security import get_security_manager, SecurityManager

# HTTP Bearer 認證
security_scheme = HTTPBearer()

# API Key 認證
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthDependencies:
    """認證依賴項類"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> dict:
        """獲取當前用戶（JWT 認證）"""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="未提供認證憑證",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        payload = self.security_manager.verify_token(credentials.credentials, "access")
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="無效的認證憑證",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return {
            "username": payload.get("username"),
            "role": payload.get("role"),
            "token_type": "jwt",
            "permissions": payload.get("permissions", []),
            "is_active": payload.get("is_active", True),
        }
    
    async def get_current_user_optional(self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)) -> Optional[dict]:
        """獲取當前用戶（可選，用於公開端點）"""
        if not credentials:
            return None
        
        payload = self.security_manager.verify_token(credentials.credentials, "access")
        if payload is None:
            return None
        
        return {
            "username": payload.get("username"),
            "role": payload.get("role"),
            "token_type": "jwt",
            "permissions": payload.get("permissions", []),
            "is_active": payload.get("is_active", True),
        }
    
    async def get_api_user(self, api_key: Optional[str] = Depends(api_key_header)) -> dict:
        """獲取 API 用戶（API Key 認證）"""
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="未提供 API 密鑰",
            )
        
        # No bundled credential; disabled until explicitly configured.
        expected_key = os.getenv("YOLO_API_KEY", "")
        if expected_key and secrets.compare_digest(api_key.encode("utf-8"), expected_key.encode("utf-8")):
            return {
                "username": "api_user",
                "role": "api",
                "token_type": "api_key"
            }
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無效的 API 密鑰",
        )
    
    async def require_admin(self, current_user: dict = Depends(get_current_user)) -> dict:
        """要求管理員權限"""
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="需要認證",
            )
        
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="需要管理員權限",
            )
        
        return current_user
    
    async def require_operator_or_admin(self, current_user: dict = Depends(get_current_user)) -> dict:
        """要求操作員或管理員權限"""
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="需要認證",
            )
        
        role = current_user.get("role")
        if role not in ["admin", "operator"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="需要操作員或管理員權限",
            )
        
        return current_user
    
    async def require_viewer_or_above(self, current_user: dict = Depends(get_current_user)) -> dict:
        """要求查看者或以上權限"""
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="需要認證",
            )
        
        role = current_user.get("role")
        if role not in ["admin", "operator", "viewer"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="需要查看者或以上權限",
            )
        
        return current_user
    
    def require_permission(self, permission: str):
        """要求特定權限 - 返回依賴項函數"""
        async def permission_checker(current_user: dict = Depends(get_current_user)) -> dict:
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="需要認證",
                )
            
            role = current_user.get("role")
            if not self.security_manager.check_permission(role, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"需要權限: {permission}",
                )
            
            return current_user
        
        return permission_checker
    
    def require_any_permission(self, permissions: list):
        """要求任一權限 - 返回依賴項函數"""
        async def permission_checker(current_user: dict = Depends(get_current_user)) -> dict:
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="需要認證",
                )
            
            role = current_user.get("role")
            if not self.security_manager.has_any_permission(role, permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"需要以下任一權限: {', '.join(permissions)}",
                )
            
            return current_user
        
        return permission_checker


# 創建依賴項實例
auth_deps = AuthDependencies(get_security_manager())

# 導出常用的依賴項
get_current_user = auth_deps.get_current_user
get_current_user_optional = auth_deps.get_current_user_optional
get_api_user = auth_deps.get_api_user
require_admin = auth_deps.require_admin
require_operator_or_admin = auth_deps.require_operator_or_admin
require_viewer_or_above = auth_deps.require_viewer_or_above
require_permission = auth_deps.require_permission
require_any_permission = auth_deps.require_any_permission

# 向下相容，提供 require_viewer 別名
require_viewer = auth_deps.require_viewer_or_above

