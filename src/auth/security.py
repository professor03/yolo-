#!/usr/bin/env python3
"""
Security helpers for password hashing, JWT management, and API key utilities.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import bcrypt
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

if __package__ is None and __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.database.user_repository import UserRepository

JWT_SECRET_KEY = os.getenv("JWT_SECRET") or secrets.token_urlsafe(48)
JWT_ALGORITHM = os.getenv("JWT_ALG", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "14"))

DEFAULT_ROLE_PERMISSIONS = {
    "admin": [
        "user_management",
        "system_config",
        "camera_management",
        "alert_config",
        "log_view",
        "data_export",
        "detection_control",
        "full_access",
        "monitoring",
        "data_analysis",
        "alert_management",
        "zone_occupancy",
        "data_read",
        "dashboard_view",
        "reports_view",
    ],
    "operator": [
        "monitoring",
        "data_analysis",
        "alert_management",
        "zone_occupancy",
        "data_read",
        "dashboard_view",
        "reports_view",
    ],
    "viewer": [
        "dashboard_view",
        "data_read",
        "reports_view",
    ],
}


class SecurityManager:
    """Provides password hashing, JWT helpers, and permission utilities."""

    def __init__(
        self,
        secret_key: str = JWT_SECRET_KEY,
        *,
        algorithm: str = JWT_ALGORITHM,
        access_token_expire_minutes: int = JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days: int = JWT_REFRESH_TOKEN_EXPIRE_DAYS,
        user_repository: Optional[UserRepository] = None,
    ) -> None:
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.user_repository = user_repository or UserRepository()
        self.role_permissions = DEFAULT_ROLE_PERMISSIONS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_permissions(self, permissions: Optional[Dict]) -> list:
        if not permissions:
            return []
        if isinstance(permissions, list):
            return [str(item) for item in permissions]
        return [str(permissions)]

    def _build_authenticated_user(self, user_record: Dict) -> Dict:
        return {
            "username": user_record.get("username"),
            "role": user_record.get("role"),
            "is_active": bool(user_record.get("is_active", False)),
            "permissions": self._normalize_permissions(
                user_record.get("permissions")
            ),
        }

    def _build_token_payload(
        self, user: Dict, token_type: str, expires_delta: timedelta
    ) -> Dict:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user.get("username"),
            "username": user.get("username"),
            "role": user.get("role"),
            "permissions": self._normalize_permissions(user.get("permissions")),
            "type": token_type,
            "iat": int(now.timestamp()),
            "exp": int((now + expires_delta).timestamp()),
        }
        if token_type == "access":
            payload["is_active"] = bool(user.get("is_active", True))
        return payload

    # ------------------------------------------------------------------
    # Password helpers
    # ------------------------------------------------------------------

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Return True when the plain password matches the stored bcrypt hash."""
        if not hashed_password:
            return False
        try:
            return bcrypt.checkpw(
                plain_password.encode("utf-8"), hashed_password.encode("utf-8")
            )
        except (ValueError, TypeError):
            return False

    def get_password_hash(self, password: str) -> str:
        """Hash a password using bcrypt with safe truncation at 72 bytes."""
        password_bytes = password.encode("utf-8")
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
        return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")

    # ------------------------------------------------------------------
    # Authentication flows
    # ------------------------------------------------------------------

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Look up a user and verify credentials. Returns a minimal user payload."""
        user_record = self.user_repository.get_user_by_username(
            username, include_sensitive=True
        )
        if not user_record:
            return None

        if not user_record.get("is_active", False):
            return None

        if not self.verify_password(password, user_record.get("password_hash", "")):
            return None

        return self._build_authenticated_user(user_record)

    def create_access_token(
        self, data: Dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        """Generate a signed access token carrying role and permissions."""
        expires_delta = expires_delta or timedelta(
            minutes=self.access_token_expire_minutes
        )
        payload = self._build_token_payload(data, "access", expires_delta)
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, data: Dict) -> str:
        """Generate a refresh token for long lived sessions."""
        expires_delta = timedelta(days=self.refresh_token_expire_days)
        payload = self._build_token_payload(data, "refresh", expires_delta)
        payload.pop("is_active", None)
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict]:
        """Decode and validate a JWT, confirming the expected token type."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"require": ["exp", "type", "sub"]},
            )
        except ExpiredSignatureError:
            return None
        except InvalidTokenError:
            return None

        if payload.get("type") != token_type:
            return None

        if "username" not in payload:
            payload["username"] = payload.get("sub")

        payload["permissions"] = self._normalize_permissions(
            payload.get("permissions")
        )

        return payload

    def refresh_access_token(self, refresh_token: str) -> Optional[Dict]:
        """Issue a new access/refresh token pair based on a valid refresh token."""
        payload = self.verify_token(refresh_token, "refresh")
        if not payload:
            return None

        username = payload.get("username") or payload.get("sub")
        if not username:
            return None

        user_record = self.user_repository.get_user_by_username(username)
        if not user_record or not user_record.get("is_active", False):
            return None

        user = self._build_authenticated_user(user_record)
        new_access_token = self.create_access_token(user)
        new_refresh_token = self.create_refresh_token(user)

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "user": user,
            "expires_in": self.access_token_expire_minutes * 60,
        }

    # ------------------------------------------------------------------
    # Permission helpers
    # ------------------------------------------------------------------

    def check_permission(self, user_role: str, required_permission: str) -> bool:
        """Return True if the user's role grants the required permission."""
        if not user_role or not required_permission:
            return False
        user_permissions = self.get_user_permissions(user_role)
        return required_permission in user_permissions

    def get_user_permissions(self, user_role: str) -> list:
        """Retrieve the permission list bound to a specific role."""
        return self.role_permissions.get(user_role, [])

    def has_any_permission(self, user_role: str, permissions: list) -> bool:
        """Return True when any permission from the list is granted."""
        if not user_role or not permissions:
            return False
        user_permissions = self.get_user_permissions(user_role)
        return any(permission in user_permissions for permission in permissions)

    def has_all_permissions(self, user_role: str, permissions: list) -> bool:
        """Return True only when every permission from the list is granted."""
        if not user_role or not permissions:
            return False
        user_permissions = self.get_user_permissions(user_role)
        return all(permission in user_permissions for permission in permissions)

    def has_permission(self, user_role: str, permission: str) -> bool:
        """Convenience wrapper around check_permission."""
        return self.check_permission(user_role, permission)


# ----------------------------------------------------------------------
# Global helpers
# ----------------------------------------------------------------------

security_manager = SecurityManager()


def get_security_manager() -> SecurityManager:
    """Return the shared SecurityManager instance."""
    return security_manager


def generate_api_key() -> str:
    """Generate a random API key token."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Create a deterministic hash of an API key for storage."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Check whether an API key matches its stored hash."""
    return hash_api_key(api_key) == hashed_key
