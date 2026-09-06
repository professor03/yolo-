#!/usr/bin/env python3
"""
認證路由
提供登入、登出、令牌刷新等認證相關的 API 端點
"""

from datetime import timedelta
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from .security import get_security_manager, SecurityManager
from .dependencies import get_current_user


# 創建認證路由器
router = APIRouter(prefix="/auth", tags=["認證"])


class LoginRequest(BaseModel):
    """登入請求模型"""
    username: str
    password: str


class AuthenticatedUser(BaseModel):
    """登入後用戶資訊"""
    username: str
    role: str
    is_active: bool
    permissions: List[str] = Field(default_factory=list)


class LoginResponse(BaseModel):
    """登入響應模型"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: AuthenticatedUser


class RefreshRequest(BaseModel):
    """刷新令牌請求模型"""
    refresh_token: str


class RefreshResponse(BaseModel):
    """刷新令牌響應模型"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: AuthenticatedUser


class UserInfo(BaseModel):
    """用戶信息模型"""
    username: str
    role: str
    is_active: bool
    permissions: List[str] = Field(default_factory=list)


class TokenInfo(BaseModel):
    """令牌信息模型"""
    token_type: str
    expires_in: int
    user: UserInfo


@router.post("/login", response_model=LoginResponse, summary="用戶登入")
async def login(request: LoginRequest):
    """
    用戶登入
    
    - **username**: 用戶名
    - **password**: 密碼
    
    返回訪問令牌和刷新令牌
    """
    security_manager = get_security_manager()
    
    # 認證用戶
    user = security_manager.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用戶名或密碼錯誤",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 創建令牌
    access_token = security_manager.create_access_token(user)
    refresh_token = security_manager.create_refresh_token(user)

    user_model = AuthenticatedUser(**user)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=security_manager.access_token_expire_minutes * 60,
        user=user_model,
    )


@router.post("/refresh", response_model=RefreshResponse, summary="刷新訪問令牌")
async def refresh_token(request: RefreshRequest):
    """
    刷新訪問令牌
    
    - **refresh_token**: 刷新令牌
    
    返回新的訪問令牌和刷新令牌
    """
    security_manager = get_security_manager()
    
    # 刷新令牌
    result = security_manager.refresh_access_token(request.refresh_token)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無效的刷新令牌",
        )
    
    return RefreshResponse(
        access_token=result["access_token"],
        refresh_token=result["refresh_token"],
        token_type="bearer",
        expires_in=result["expires_in"],
        user=AuthenticatedUser(**result["user"]),
    )


@router.get("/me", response_model=UserInfo, summary="獲取當前用戶信息")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    獲取當前用戶信息
    
    需要有效的訪問令牌
    """
    return UserInfo(
        username=current_user["username"],
        role=current_user["role"],
        is_active=current_user.get("is_active", True),
        permissions=current_user.get("permissions", []),
    )


@router.get("/token-info", response_model=TokenInfo, summary="獲取令牌信息")
async def get_token_info(current_user: dict = Depends(get_current_user)):
    """
    獲取當前令牌信息
    
    需要有效的訪問令牌
    """
    security_manager = get_security_manager()
    
    return TokenInfo(
        token_type="bearer",
        expires_in=security_manager.access_token_expire_minutes * 60,
        user=UserInfo(
            username=current_user["username"],
            role=current_user["role"],
            is_active=current_user.get("is_active", True),
            permissions=current_user.get("permissions", []),
        )
    )


@router.post("/logout", summary="用戶登出")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    用戶登出
    
    注意：JWT 是無狀態的，實際登出需要在客戶端刪除令牌
    這裡主要用於記錄登出事件
    """
    return {
        "message": "登出成功",
        "username": current_user["username"]
    }


@router.get("/verify", response_model=UserInfo, summary="驗證令牌")
async def verify_token(current_user: dict = Depends(get_current_user)):
    """
    驗證訪問令牌
    
    用於前端檢查令牌是否有效
    """
    return UserInfo(
        username=current_user["username"],
        role=current_user["role"],
        is_active=current_user.get("is_active", True),
        permissions=current_user.get("permissions", []),
    )


@router.get("/health", summary="認證服務健康檢查")
async def auth_health():
    """
    認證服務健康檢查
    """
    return {
        "status": "healthy",
        "service": "authentication",
        "version": "1.0.0"
    }
