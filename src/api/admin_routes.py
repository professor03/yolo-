#!/usr/bin/env python3
"""
管理員專用 API 路由
提供用戶管理、攝影機管理等管理功能
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime

from src.auth.security import get_security_manager
from src.database.user_repository import UserRepository
from src.database.camera_repository import CameraRepository
from src.database.models import get_database_manager

# 創建路由器
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# 依賴注入
from fastapi import Header

# Phase C: 串流管理器重載輔助函數
def reload_camera_streams():
    """重載攝影機串流管理器"""
    try:
        # 動態導入以避免循環依賴
        import sys
        if 'src.server.app' in sys.modules:
            app_module = sys.modules['src.server.app']
            if hasattr(app_module, 'app') and hasattr(app_module.app.state, 'camera_manager'):
                db_manager = get_database_manager()
                camera_repository = CameraRepository(db_manager)
                # 強制重新從資料庫獲取最新的攝影機配置
                sources = camera_repository.get_enabled_camera_sources(force_refresh=True)
                print(f"Reloading camera streams with sources: {sources}")
                app_module.app.state.camera_manager.reload_sources(sources)
                return True
    except Exception as e:
        print(f"Warning: Failed to reload camera streams: {e}")
    return False

def get_current_admin_user(authorization: str = Header(..., alias="Authorization")):
    """獲取當前管理員用戶"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供認證令牌"
        )
    
    token = authorization.split(" ")[1]
    security_manager = get_security_manager()
    payload = security_manager.verify_token(token, "access")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無效的認證令牌"
        )
    
    # 檢查管理員權限
    if not security_manager.check_permission(payload.get("role"), "user_management"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理員權限"
        )
    
    return payload

# Pydantic 模型
class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=80)
    password: str = Field(..., min_length=6, max_length=128)
    role: str = Field(default="viewer", pattern="^(admin|operator|viewer)$")
    permissions: Optional[List[str]] = None
    is_active: bool = True

class UserUpdateRequest(BaseModel):
    new_username: Optional[str] = Field(None, min_length=3, max_length=80)
    password: Optional[str] = Field(None, min_length=6, max_length=128)
    role: Optional[str] = Field(None, pattern="^(admin|operator|viewer)$")
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None

class CameraCreateRequest(BaseModel):
    slug: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    source_url: str = Field(..., min_length=1, max_length=1024)
    protocol: str = Field(default="rtsp", pattern="^(rtsp|mjpeg|file)$")
    description: Optional[str] = None
    enabled: bool = True
    owner_username: Optional[str] = None

class CameraUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    source_url: Optional[str] = Field(None, min_length=1, max_length=1024)
    protocol: Optional[str] = Field(None, pattern="^(rtsp|mjpeg|file)$")
    description: Optional[str] = None
    enabled: Optional[bool] = None
    owner_username: Optional[str] = None

# ===== 用戶管理端點 =====

@router.get("/users")
async def list_users(current_user: Dict = Depends(get_current_admin_user)):
    """列出所有用戶"""
    db_manager = get_database_manager()
    user_repo = UserRepository(db_manager)
    users = user_repo.list_users()
    return {"users": users, "total": len(users)}

@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreateRequest,
    current_user: Dict = Depends(get_current_admin_user)
):
    """創建新用戶"""
    db_manager = get_database_manager()
    user_repo = UserRepository(db_manager)
    
    try:
        new_user = user_repo.create_user(
            username=user_data.username,
            password=user_data.password,
            role=user_data.role,
            permissions=user_data.permissions,
            is_active=user_data.is_active
        )
        return {
            "message": "用戶創建成功",
            "user": new_user
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/users/{username}")
async def get_user(
    username: str,
    current_user: Dict = Depends(get_current_admin_user)
):
    """獲取特定用戶資訊"""
    db_manager = get_database_manager()
    user_repo = UserRepository(db_manager)
    user = user_repo.get_user_by_username(username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用戶不存在"
        )
    
    return {"user": user}

@router.put("/users/{username}")
async def update_user(
    username: str,
    user_data: UserUpdateRequest,
    current_user: Dict = Depends(get_current_admin_user)
):
    """更新用戶資訊"""
    db_manager = get_database_manager()
    user_repo = UserRepository(db_manager)
    
    try:
        updated_user = user_repo.update_user(
            username=username,
            new_username=user_data.new_username,
            password=user_data.password,
            role=user_data.role,
            permissions=user_data.permissions,
            is_active=user_data.is_active
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用戶不存在"
            )
        
        return {
            "message": "用戶更新成功",
            "user": updated_user
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.delete("/users/{username}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    username: str,
    current_user: Dict = Depends(get_current_admin_user)
):
    """刪除用戶"""
    # 防止刪除自己
    if username == current_user.get("username"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能刪除自己的帳戶"
        )
    
    db_manager = get_database_manager()
    user_repo = UserRepository(db_manager)
    success = user_repo.delete_user(username)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用戶不存在"
        )

# ===== 攝影機管理端點 =====

@router.get("/cameras")
async def list_cameras(current_user: Dict = Depends(get_current_admin_user)):
    """列出所有攝影機（包含敏感資訊）"""
    db_manager = get_database_manager()
    camera_repo = CameraRepository(db_manager)
    cameras = camera_repo.list_cameras(include_source_url=True)
    return {"cameras": cameras, "total": len(cameras)}

@router.post("/cameras", status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera_data: CameraCreateRequest,
    current_user: Dict = Depends(get_current_admin_user)
):
    """創建新攝影機"""
    db_manager = get_database_manager()
    camera_repo = CameraRepository(db_manager)
    user_repo = UserRepository(db_manager)
    
    # 驗證擁有者
    owner_id = None
    if camera_data.owner_username:
        owner = user_repo.get_user_by_username(camera_data.owner_username)
        if not owner:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"用戶 {camera_data.owner_username} 不存在"
            )
        owner_id = owner["id"]
    
    try:
        new_camera = camera_repo.create_camera(
            slug=camera_data.slug,
            name=camera_data.name,
            source_url=camera_data.source_url,
            protocol=camera_data.protocol,
            description=camera_data.description,
            enabled=camera_data.enabled,
            owner_id=owner_id
        )
        # Phase C: 重載串流管理器
        stream_reloaded = reload_camera_streams()
        
        return {
            "message": "攝影機創建成功",
            "camera": new_camera,
            "stream_reloaded": stream_reloaded
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/cameras/{slug}")
async def update_camera(
    slug: str,
    camera_data: CameraUpdateRequest,
    current_user: Dict = Depends(get_current_admin_user)
):
    """更新攝影機資訊"""
    db_manager = get_database_manager()
    camera_repo = CameraRepository(db_manager)
    user_repo = UserRepository(db_manager)
    
    # 驗證擁有者
    owner_id = None
    if camera_data.owner_username:
        owner = user_repo.get_user_by_username(camera_data.owner_username)
        if not owner:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"用戶 {camera_data.owner_username} 不存在"
            )
        owner_id = owner["id"]
    
    try:
        updated_camera = camera_repo.update_camera(
            slug=slug,
            name=camera_data.name,
            source_url=camera_data.source_url,
            protocol=camera_data.protocol,
            description=camera_data.description,
            enabled=camera_data.enabled,
            owner_id=owner_id
        )
        
        if not updated_camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="攝影機不存在"
            )
        
        # Phase C: 重載串流管理器
        stream_reloaded = reload_camera_streams()
        
        return {
            "message": "攝影機更新成功",
            "camera": updated_camera,
            "stream_reloaded": stream_reloaded
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.delete("/cameras/{slug}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    slug: str,
    current_user: Dict = Depends(get_current_admin_user)
):
    """刪除攝影機"""
    db_manager = get_database_manager()
    camera_repo = CameraRepository(db_manager)
    success = camera_repo.delete_camera(slug)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="攝影機不存在"
        )
    
    # Phase C: 重載串流管理器
    reload_camera_streams()

@router.post("/cameras/{slug}/test")
async def test_camera(
    slug: str,
    current_user: Dict = Depends(get_current_admin_user)
):
    """測試攝影機連接"""
    db_manager = get_database_manager()
    camera_repo = CameraRepository(db_manager)
    
    camera = camera_repo.get_camera(slug)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="攝影機不存在"
        )
    
    # Phase C: 執行增強的連接測試
    test_result = camera_repo.test_camera_connection(slug)
    
    # 如果測試成功，更新連接狀態
    if test_result.get("success"):
        camera_repo.update_connection_status(slug, True, reset_reconnect_attempts=True)
    else:
        camera_repo.update_connection_status(slug, False)
    
    return {
        "camera_slug": slug,
        "camera_name": camera.get("name"),
        "test_result": test_result,
        "connection_updated": True
    }

@router.put("/cameras/{slug}/owner")
async def assign_camera_owner(
    slug: str,
    owner_data: dict,
    current_user: Dict = Depends(get_current_admin_user)
):
    """分配攝影機擁有者 - Phase C 多用戶功能"""
    db_manager = get_database_manager()
    camera_repo = CameraRepository(db_manager)
    user_repo = UserRepository(db_manager)
    
    # 檢查攝影機是否存在
    camera = camera_repo.get_camera(slug)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="攝影機不存在"
        )
    
    # 驗證新擁有者
    new_owner_username = owner_data.get("owner_username")
    owner_id = None
    
    if new_owner_username:
        owner = user_repo.get_user_by_username(new_owner_username)
        if not owner:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"用戶 {new_owner_username} 不存在"
            )
        owner_id = owner["id"]
    
    # 更新攝影機擁有者
    try:
        updated_camera = camera_repo.update_camera(
            slug=slug,
            owner_id=owner_id
        )
        
        return {
            "message": "攝影機擁有者更新成功",
            "camera": updated_camera,
            "new_owner": new_owner_username or "無擁有者"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新失敗: {str(e)}"
        )

@router.get("/cameras/by-user/{username}")
async def get_user_cameras(
    username: str,
    current_user: Dict = Depends(get_current_admin_user)
):
    """獲取特定用戶的攝影機列表 - Phase C 多用戶功能"""
    db_manager = get_database_manager()
    user_repo = UserRepository(db_manager)
    db_manager = get_database_manager()
    camera_repo = CameraRepository(db_manager)
    
    # 驗證用戶存在
    user = user_repo.get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用戶 {username} 不存在"
        )
    
    # 獲取用戶的攝影機
    cameras = camera_repo.list_cameras(owner_id=user["id"], include_source_url=True)
    
    return {
        "user": {
            "username": username,
            "id": user["id"]
        },
        "cameras": cameras,
        "total": len(cameras)
    }

# ===== 系統統計端點 =====

@router.get("/stats")
async def get_system_stats(current_user: Dict = Depends(get_current_admin_user)):
    """獲取系統統計資訊"""
    db_manager = get_database_manager()
    user_repo = UserRepository(db_manager)
    db_manager = get_database_manager()
    camera_repo = CameraRepository(db_manager)
    
    users = user_repo.list_users()
    cameras = camera_repo.list_cameras()
    
    active_users = len([u for u in users if u.get("is_active")])
    enabled_cameras = len([c for c in cameras if c.get("enabled")])
    online_cameras = len([c for c in cameras if c.get("is_online")])
    
    return {
        "users": {
            "total": len(users),
            "active": active_users,
            "inactive": len(users) - active_users
        },
        "cameras": {
            "total": len(cameras),
            "enabled": enabled_cameras,
            "disabled": len(cameras) - enabled_cameras,
            "online": online_cameras,
            "offline": enabled_cameras - online_cameras
        },
        "timestamp": datetime.utcnow().isoformat()
    }
