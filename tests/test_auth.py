"""認證模組與 API 測試"""
import unittest
import secrets

# Legacy tests use a generated test-only password, never a bundled account secret.
TEST_PASSWORD = secrets.token_urlsafe(24)
from datetime import timedelta

from fastapi.testclient import TestClient

from src.auth.security import SecurityManager
from src.server.app import create_app
from src.database.camera_repository import CameraRepository


class TestAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.security_manager = SecurityManager()
        cls.camera_repo = CameraRepository()

    def test_authenticate_user_success(self):
        user = self.security_manager.authenticate_user("admin", TEST_PASSWORD)
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "admin")
        self.assertEqual(user["role"], "admin")
        self.assertTrue(user["permissions"], "預設帳號應該具備權限列表")

    def test_authenticate_user_invalid_password(self):
        user = self.security_manager.authenticate_user("admin", "wrongpassword")
        self.assertIsNone(user)

    def test_access_token_embeds_role_and_permissions(self):
        user_payload = {
            "username": "alice",
            "role": "viewer",
            "is_active": True,
            "permissions": ["dashboard_view"],
        }
        token = self.security_manager.create_access_token(
            user_payload,
            expires_delta=timedelta(minutes=5),
        )
        decoded = self.security_manager.verify_token(token, "access")
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["username"], "alice")
        self.assertEqual(decoded["role"], "viewer")
        self.assertEqual(decoded["permissions"], ["dashboard_view"])

    def test_refresh_token_flow(self):
        user = self.security_manager.authenticate_user("admin", TEST_PASSWORD)
        self.assertIsNotNone(user)
        refresh_token = self.security_manager.create_refresh_token(user)

        refreshed = self.security_manager.refresh_access_token(refresh_token)
        self.assertIsNotNone(refreshed)
        self.assertEqual(refreshed["user"]["username"], "admin")
        self.assertTrue(refreshed["user"]["permissions"])

        new_payload = self.security_manager.verify_token(
            refreshed["access_token"], "access"
        )
        self.assertIsNotNone(new_payload)
        self.assertEqual(new_payload["role"], "admin")

    def test_login_api_returns_user_payload(self):
        response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": TEST_PASSWORD},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)
        self.assertIn("refresh_token", data)
        self.assertEqual(data["user"]["role"], "admin")
        self.assertTrue(data["user"]["permissions"])

    def test_refresh_api_requires_valid_refresh_token(self):
        login_response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": TEST_PASSWORD},
        )
        self.assertEqual(login_response.status_code, 200)
        refresh_token = login_response.json()["refresh_token"]

        refresh_response = self.client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        self.assertEqual(refresh_response.status_code, 200)
        data = refresh_response.json()
        self.assertIn("access_token", data)
        self.assertIn("user", data)
        self.assertEqual(data["user"]["username"], "admin")

    def test_protected_endpoint_requires_auth(self):
        response = self.client.get("/auth/verify")
        self.assertEqual(response.status_code, 401)

    def test_protected_endpoint_with_token(self):
        login_response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": TEST_PASSWORD},
        )
        token = login_response.json()["access_token"]

        verify_response = self.client.get(
            "/auth/verify",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(verify_response.status_code, 200)
        data = verify_response.json()
        self.assertEqual(data["username"], "admin")
        self.assertEqual(data["role"], "admin")
        self.assertTrue(data["permissions"])

    def test_admin_user_crud_flow(self):
        # 取得管理員 token
        login_response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": TEST_PASSWORD},
        )
        self.assertEqual(login_response.status_code, 200)
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 建立新用戶
        username = "test_user_api"
        # 若先前測試異常留下紀錄，先清理
        self.client.delete(
            f"/api/v1/admin/users/{username}",
            headers=headers,
        )
        create_response = self.client.post(
            "/api/v1/admin/users",
            json={
                "username": username,
                "password": secrets.token_urlsafe(24),
                "role": "viewer",
                "permissions": ["dashboard_view"],
            },
            headers=headers,
        )
        self.assertEqual(create_response.status_code, 201)
        created_user = create_response.json()["user"]
        self.assertEqual(created_user["username"], username)
        self.assertEqual(created_user["role"], "viewer")

        # 更新角色與啟用狀態
        update_response = self.client.put(
            f"/api/v1/admin/users/{username}",
            json={"role": "operator", "is_active": False},
            headers=headers,
        )
        self.assertEqual(update_response.status_code, 200)
        updated_user = update_response.json()["user"]
        self.assertEqual(updated_user["role"], "operator")
        self.assertFalse(updated_user["is_active"])

        # 列表應包含該用戶
        list_response = self.client.get(
            "/api/v1/admin/users",
            headers=headers,
        )
        self.assertEqual(list_response.status_code, 200)
        usernames = [user["username"] for user in list_response.json()["users"]]
        self.assertIn(username, usernames)

        # 刪除用戶
        delete_response = self.client.delete(
            f"/api/v1/admin/users/{username}",
            headers=headers,
        )
        self.assertEqual(delete_response.status_code, 204)

        # 列表確認已移除
        list_after_delete = self.client.get(
            "/api/v1/admin/users",
            headers=headers,
        )
        self.assertEqual(list_after_delete.status_code, 200)
        usernames_after = [user["username"] for user in list_after_delete.json()["users"]]
        self.assertNotIn(username, usernames_after)

    def test_camera_admin_crud_flow(self):
        login_response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": TEST_PASSWORD},
        )
        self.assertEqual(login_response.status_code, 200)
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        slug = "test_cam_api"
        # cleanup any leftover camera
        self.camera_repo.delete_camera(slug)

        create_response = self.client.post(
            "/api/v1/admin/cameras",
            json={
                "slug": slug,
                "name": "測試攝影機",
                "source_url": "http://example.com/stream",
                "protocol": "http",
                "enabled": True,
            },
            headers=headers,
        )
        self.assertEqual(create_response.status_code, 201)
        camera = create_response.json()["camera"]
        self.assertEqual(camera["slug"], slug)

        admin_list = self.client.get("/api/v1/admin/cameras", headers=headers)
        self.assertEqual(admin_list.status_code, 200)
        admin_payload = admin_list.json()
        slugs = [item["slug"] for item in admin_payload["cameras"]]
        self.assertIn(slug, slugs)

        update_response = self.client.put(
            f"/api/v1/admin/cameras/{slug}",
            json={"enabled": False, "description": "停用測試"},
            headers=headers,
        )
        self.assertEqual(update_response.status_code, 200)
        updated = update_response.json()["camera"]
        self.assertFalse(updated["enabled"])
        self.assertEqual(updated["description"], "停用測試")

        cameras_response = self.client.get("/api/v1/cameras", headers=headers)
        self.assertEqual(cameras_response.status_code, 200)
        cameras_payload = cameras_response.json()
        self.assertTrue(any(cam["slug"] == slug for cam in cameras_payload["cameras"]))

        delete_response = self.client.delete(
            f"/api/v1/admin/cameras/{slug}", headers=headers
        )
        self.assertEqual(delete_response.status_code, 204)

        # ensure removed
        final_list = self.client.get("/api/v1/admin/cameras", headers=headers)
        self.assertEqual(final_list.status_code, 200)
        final_slugs = [item["slug"] for item in final_list.json()["cameras"]]
        self.assertNotIn(slug, final_slugs)

    def test_metrics_export_endpoints(self):
        login_response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": TEST_PASSWORD},
        )
        self.assertEqual(login_response.status_code, 200)
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        csv_response = self.client.get("/export/metrics.csv", headers=headers)
        self.assertEqual(csv_response.status_code, 200)
        self.assertIn("text/csv", csv_response.headers.get("content-type", ""))
        self.assertTrue(csv_response.text.startswith("timestamp"))

        excel_response = self.client.get("/export/metrics.xlsx", headers=headers)
        self.assertIn(excel_response.status_code, (200, 503))


if __name__ == "__main__":
    unittest.main()
