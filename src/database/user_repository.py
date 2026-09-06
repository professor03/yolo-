#!/usr/bin/env python3
"""使用者資料存取層"""

from __future__ import annotations

from typing import Dict, List, Optional
from contextlib import contextmanager

from sqlalchemy.orm import Session
import bcrypt
import os

from .models import User, get_database_manager, DatabaseManager



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


class UserRepository:
    """封裝使用者資料庫操作"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db_manager = db_manager or get_database_manager()

    def get_user_by_username(
        self, username: str, *, include_sensitive: bool = False
    ) -> Optional[Dict]:
        with session_scope(self.db_manager) as session:
            user: Optional[User] = (
                session.query(User).filter(User.username == username).first()
            )
            if not user:
                return None
            return self._serialize_user(user, include_sensitive=include_sensitive)

    def list_users(self) -> List[Dict]:
        with session_scope(self.db_manager) as session:
            users = session.query(User).order_by(User.id.asc()).all()
            return [self._serialize_user(user) for user in users]

    def list_active_users(self) -> List[Dict]:
        with session_scope(self.db_manager) as session:
            users = (
                session.query(User)
                .filter(User.is_active.is_(True))
                .order_by(User.id.asc())
                .all()
            )
            return [self._serialize_user(user) for user in users]

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "viewer",
        permissions: Optional[List[str]] = None,
        is_active: bool = True,
    ) -> Dict:
        with session_scope(self.db_manager) as session:
            if session.query(User).filter(User.username == username).first():
                raise ValueError("username already exists")

            user = User(
                username=username,
                password_hash=self._hash_password(password),
                role=role,
                is_active=is_active,
            )
            user.set_permissions(permissions)
            session.add(user)
            session.flush()
            return self._serialize_user(user)

    def update_user(
        self,
        username: str,
        *,
        new_username: Optional[str] = None,
        password: Optional[str] = None,
        role: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Dict]:
        with session_scope(self.db_manager) as session:
            user: Optional[User] = (
                session.query(User)
                .filter(User.username == username)
                .one_or_none()
            )
            if not user:
                return None

            if new_username and new_username != username:
                if (
                    session.query(User)
                    .filter(User.username == new_username)
                    .first()
                ):
                    raise ValueError("username already exists")
                user.username = new_username

            if password is not None:
                user.password_hash = self._hash_password(password)
            if role is not None:
                user.role = role
            if permissions is not None:
                user.set_permissions(permissions)
            if is_active is not None:
                user.is_active = is_active

            session.add(user)
            session.flush()
            return self._serialize_user(user)

    def delete_user(self, username: str) -> bool:
        with session_scope(self.db_manager) as session:
            user: Optional[User] = (
                session.query(User)
                .filter(User.username == username)
                .one_or_none()
            )
            if not user:
                return False
            session.delete(user)
            return True

    def ensure_default_users(self) -> None:
        """Kept for compatibility; account creation is explicit."""
        return

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _hash_password(self, password: str) -> str:
        password_bytes = password.encode("utf-8")
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
        return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")

    def _serialize_user(self, user: User, *, include_sensitive: bool = False) -> Dict:
        data = {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "is_active": user.is_active,
            "permissions": user.get_permissions(),
            "created_at": user.created_at,
        }
        if include_sensitive:
            data["password_hash"] = user.password_hash
        return data
