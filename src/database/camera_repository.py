#!/usr/bin/env python3
"""攝影機資料存取層"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional
from datetime import datetime

import cv2
from sqlalchemy.orm import Session

from .models import Camera, get_database_manager, DatabaseManager

DEFAULT_CAMERAS = []  # Configure camera sources explicitly through the admin UI.


@contextmanager
def session_scope(db_manager: DatabaseManager):
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class CameraRepository:
    """封裝攝影機資料庫操作"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db_manager = db_manager or get_database_manager()

    # ------------------------------------------------------------------
    # 查詢操作
    # ------------------------------------------------------------------
    def list_cameras(self, include_disabled: bool = True, include_source_url: bool = False, owner_id: Optional[int] = None) -> List[Dict]:
        with session_scope(self.db_manager) as session:
            query = session.query(Camera).order_by(Camera.created_at.asc())
            if not include_disabled:
                query = query.filter(Camera.enabled.is_(True))
            if owner_id is not None:
                query = query.filter(Camera.owner_id == owner_id)
            return [self._serialize_camera(camera, include_source_url=include_source_url) for camera in query.all()]

    def get_camera(self, slug: str) -> Optional[Dict]:
        with session_scope(self.db_manager) as session:
            camera = (
                session.query(Camera)
                .filter(Camera.slug == slug)
                .one_or_none()
            )
            if not camera:
                return None
            return self._serialize_camera(camera)

    def get_enabled_camera_sources(self, force_refresh: bool = False) -> Dict[str, str]:
        enabled = self.list_cameras(include_disabled=False, include_source_url=True)
        sources = {cam["slug"]: cam["source_url"] for cam in enabled}
        if force_refresh:
            print(f"Force refresh enabled camera sources: {sources}")
        return sources

    # ------------------------------------------------------------------
    # 變更操作
    # ------------------------------------------------------------------
    def create_camera(
        self,
        slug: str,
        name: str,
        source_url: str,
        protocol: str = "rtsp",
        *,
        description: Optional[str] = None,
        enabled: bool = True,
        owner_id: Optional[int] = None,
    ) -> Dict:
        slug = slug.strip()
        if not slug:
            raise ValueError("slug cannot be empty")

        with session_scope(self.db_manager) as session:
            existing = (
                session.query(Camera)
                .filter(Camera.slug == slug)
                .one_or_none()
            )
            if existing:
                raise ValueError("camera slug already exists")

            camera = Camera(
                slug=slug,
                name=name.strip() or slug,
                source_url=source_url.strip(),
                protocol=protocol.strip().lower() or "rtsp",
                enabled=enabled,
                description=description,
                owner_id=owner_id,
            )
            session.add(camera)
            session.flush()
            return self._serialize_camera(camera)

    def update_camera(
        self,
        slug: str,
        *,
        name: Optional[str] = None,
        source_url: Optional[str] = None,
        protocol: Optional[str] = None,
        enabled: Optional[bool] = None,
        description: Optional[str] = None,
        owner_id: Optional[int] = None,
    ) -> Optional[Dict]:
        with session_scope(self.db_manager) as session:
            camera = (
                session.query(Camera)
                .filter(Camera.slug == slug)
                .one_or_none()
            )
            if not camera:
                return None

            if name is not None:
                camera.name = name.strip() or camera.name
            if source_url is not None:
                camera.source_url = source_url.strip()
            if protocol is not None:
                camera.protocol = protocol.strip().lower() or camera.protocol
            if enabled is not None:
                camera.enabled = enabled
            if description is not None:
                camera.description = description
            if owner_id is not None:
                camera.owner_id = owner_id

            session.add(camera)
            session.flush()
            return self._serialize_camera(camera)

    def delete_camera(self, slug: str) -> bool:
        with session_scope(self.db_manager) as session:
            camera = (
                session.query(Camera)
                .filter(Camera.slug == slug)
                .one_or_none()
            )
            if not camera:
                return False
            session.delete(camera)
            return True

    def ensure_default_cameras(self) -> None:
        with session_scope(self.db_manager) as session:
            for item in DEFAULT_CAMERAS:
                existing = (
                    session.query(Camera)
                    .filter(Camera.slug == item["slug"])
                    .one_or_none()
                )
                if existing:
                    updated = False
                    if item.get("name") and existing.name != item["name"]:
                        existing.name = item["name"]
                        updated = True
                    if existing.source_url != item["source_url"]:
                        existing.source_url = item["source_url"]
                        updated = True
                    desired_protocol = item.get("protocol", existing.protocol)
                    if existing.protocol != desired_protocol:
                        existing.protocol = desired_protocol
                        updated = True
                    desired_enabled = item.get("enabled", True)
                    if existing.enabled != desired_enabled:
                        existing.enabled = desired_enabled
                        updated = True
                    if item.get("description") and existing.description != item["description"]:
                        existing.description = item["description"]
                        updated = True
                    if updated:
                        session.add(existing)
                    continue

                camera = Camera(
                    slug=item["slug"],
                    name=item.get("name", item["slug"]),
                    source_url=item["source_url"],
                    protocol=item.get("protocol", "rtsp"),
                    description=item.get("description"),
                    enabled=item.get("enabled", True),
                )
                session.add(camera)

    def record_test_result(
        self, slug: str, success: bool, message: Optional[str] = None
    ) -> None:
        with session_scope(self.db_manager) as session:
            camera = (
                session.query(Camera)
                .filter(Camera.slug == slug)
                .one_or_none()
            )
            if not camera:
                return
            camera.last_test_status = "online" if success else "offline"
            camera.last_test_message = message
            camera.last_tested_at = datetime.utcnow()
            session.add(camera)

    def update_connection_status(
        self, slug: str, is_online: bool, reset_reconnect_attempts: bool = False
    ) -> None:
        """更新攝影機連接狀態"""
        with session_scope(self.db_manager) as session:
            camera = (
                session.query(Camera)
                .filter(Camera.slug == slug)
                .one_or_none()
            )
            if not camera:
                return
            
            camera.is_online = is_online
            camera.last_seen_at = datetime.utcnow()
            
            if reset_reconnect_attempts:
                camera.reconnect_attempts = 0
            elif not is_online:
                camera.reconnect_attempts = (camera.reconnect_attempts or 0) + 1
            
            session.add(camera)

    def get_cameras_for_reconnection(self) -> List[Dict]:
        """獲取需要重連的攝影機"""
        with session_scope(self.db_manager) as session:
            cameras = (
                session.query(Camera)
                .filter(Camera.enabled.is_(True))
                .filter(Camera.is_online.is_(False))
                .filter(Camera.reconnect_attempts < Camera.max_reconnect_attempts)
                .all()
            )
            return [self._serialize_camera(camera) for camera in cameras]

    def test_camera_connection(self, slug: str) -> Dict:
        """測試攝影機連接並記錄結果"""
        # 確保獲取包含 source_url 的攝影機資訊
        with session_scope(self.db_manager) as session:
            camera_obj = (
                session.query(Camera)
                .filter(Camera.slug == slug)
                .one_or_none()
            )
            if not camera_obj:
                return {"success": False, "message": "攝影機不存在"}
            
            # 直接從資料庫物件獲取資訊
            source_url = camera_obj.source_url
            protocol = camera_obj.protocol
        
        test_result = self.test_camera_source(source_url, protocol)
        self.record_test_result(slug, test_result["success"], test_result["message"])
        self.update_connection_status(slug, test_result["success"], reset_reconnect_attempts=test_result["success"])
        
        return test_result

    # ------------------------------------------------------------------
    # 測試工具
    # ------------------------------------------------------------------
    def test_camera_source(self, source_url: str, protocol: str, timeout: int = 10) -> Dict:
        """測試攝影機源連接 - Phase C 增強版本"""
        try:
            import threading
            import time
            from urllib.parse import urlparse
            
            # 驗證 URL 格式
            try:
                parsed = urlparse(source_url)
                if not parsed.scheme:
                    return {"success": False, "message": "無效的 URL 格式"}
            except Exception:
                return {"success": False, "message": "URL 解析失敗"}
            
            # 測試結果容器
            test_result = {"success": False, "message": "測試超時"}
            
            def test_connection():
                params = []
                lowered = (protocol or "").lower()
                
                try:
                    # 根據協議選擇參數
                    if lowered in ("rtsp", "mjpeg", "http"):
                        params = [cv2.CAP_FFMPEG]
                    
                    cap = cv2.VideoCapture(source_url, *params)
                    
                    # 設置緩衝區大小以減少延遲
                    if lowered == "rtsp":
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    try:
                        if not cap.isOpened():
                            test_result.update({
                                "success": False, 
                                "message": "無法開啟串流"
                            })
                            return
                        
                        # 嘗試讀取多幀以確保穩定性
                        successful_reads = 0
                        total_attempts = 3
                        last_frame = None
                        
                        for i in range(total_attempts):
                            ok, frame = cap.read()
                            if ok and frame is not None:
                                successful_reads += 1
                                last_frame = frame
                                time.sleep(0.1)  # 短暫延遲
                        
                        if successful_reads > 0 and last_frame is not None:
                            height, width = last_frame.shape[:2]
                            channels = last_frame.shape[2] if len(last_frame.shape) > 2 else 1
                            
                            # 獲取攝影機屬性
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            # 計算成功率
                            success_rate = (successful_reads / total_attempts) * 100
                            
                            test_result.update({
                                "success": True,
                                "message": f"串流連線正常 ({successful_reads}/{total_attempts} 幀成功)",
                                "details": {
                                    "resolution": f"{width}x{height}",
                                    "channels": channels,
                                    "fps": fps if fps > 0 else "未知",
                                    "success_rate": f"{success_rate:.1f}%",
                                    "protocol": protocol.upper(),
                                    "url_scheme": parsed.scheme
                                }
                            })
                        else:
                            test_result.update({
                                "success": False, 
                                "message": f"無法讀取影格 (0/{total_attempts} 成功)"
                            })
                    finally:
                        cap.release()
                        
                except Exception as e:
                    test_result.update({
                        "success": False, 
                        "message": f"連接錯誤: {str(e)}"
                    })
            
            # 使用線程進行超時控制
            test_thread = threading.Thread(target=test_connection)
            test_thread.daemon = True
            test_thread.start()
            test_thread.join(timeout)
            
            if test_thread.is_alive():
                test_result.update({
                    "success": False, 
                    "message": f"測試超時 ({timeout}秒)"
                })
            
            return test_result
                
        except ImportError:
            return {"success": False, "message": "OpenCV 未安裝，無法進行攝影機測試"}
        except Exception as e:
            return {"success": False, "message": f"測試系統錯誤: {str(e)}"}

    # ------------------------------------------------------------------
    # 私有工具
    # ------------------------------------------------------------------
    def _serialize_camera(self, camera: Camera, include_source_url: bool = False) -> Dict:
        data = {
            "id": camera.id,
            "slug": camera.slug,
            "name": camera.name,
            "protocol": camera.protocol,
            "enabled": camera.enabled,
            "description": camera.description,
            "created_at": camera.created_at,
            "updated_at": camera.updated_at,
            "last_tested_at": camera.last_tested_at,
            "last_test_status": camera.last_test_status,
            "last_test_message": camera.last_test_message,
            "owner_id": camera.owner_id,
            "is_online": camera.is_online,
            "last_seen_at": camera.last_seen_at,
            "reconnect_attempts": camera.reconnect_attempts,
            "max_reconnect_attempts": camera.max_reconnect_attempts,
        }
        
        # 只有在明確要求時才包含敏感的 source_url
        if include_source_url:
            data["source_url"] = camera.source_url
            
        return data
