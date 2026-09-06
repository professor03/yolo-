"""Authenticated API smoke checks: temporary state, no camera access."""
import os
import asyncio
import pytest
import secrets
import tempfile
from pathlib import Path
_temporary = tempfile.TemporaryDirectory(prefix='yolo-release-')
_root = Path(_temporary.name)
os.environ['DATABASE_URL'] = 'sqlite:///' + (_root / 'test.db').as_posix()
os.environ['DATASTORE_PATH'] = str(_root / 'datastore.json')
os.environ['FRAME_OUTPUT_DIR'] = str(_root / 'frames')
os.environ['HLS_OUTPUT_DIR'] = str(_root / 'hls')
os.environ['DISABLE_CAMERA_STREAMS'] = 'true'
from fastapi.testclient import TestClient
from src.server.app import app
from src.database.user_repository import UserRepository
from src.database.camera_repository import CameraRepository
from src.auth.security import DEFAULT_ROLE_PERMISSIONS
from src.api.data_store import data_store
from src.database.models import get_database_manager

def test_no_bundled_credentials(monkeypatch):
    from src.utils.config_validator import ConfigValidator
    monkeypatch.delenv('JWT_SECRET', raising=False)
    first = ConfigValidator().default_config['security']
    second = ConfigValidator().default_config['security']
    assert first['users'] == {}
    assert first['api_keys'] == []
    assert first['jwt_secret_key'] != second['jwt_secret_key']
    login_page = Path('static/login.html').read_text(encoding='utf-8')
    assert '本系統不提供共用示範帳號' in login_page

def test_api_key_requires_explicit_configuration(monkeypatch):
    from fastapi import HTTPException
    from src.auth.dependencies import auth_deps
    key = secrets.token_urlsafe(32)
    monkeypatch.delenv('YOLO_API_KEY', raising=False)
    with pytest.raises(HTTPException) as denied:
        asyncio.run(auth_deps.get_api_user(key))
    assert denied.value.status_code == 401
    monkeypatch.setenv('YOLO_API_KEY', key)
    assert asyncio.run(auth_deps.get_api_user(key))['username'] == 'api_user'
    with pytest.raises(HTTPException):
        asyncio.run(auth_deps.get_api_user(secrets.token_urlsafe(32)))

def teardown_module():
    get_database_manager().close()
    _temporary.cleanup()

def test_safe_startup_and_authenticated_learnsight():
    users = UserRepository()
    assert users.get_user_by_username('admin') is None
    assert CameraRepository().get_enabled_camera_sources() == {}
    password = secrets.token_urlsafe(24)
    users.create_user(username='release-test', password=password,
                      role='admin', permissions=DEFAULT_ROLE_PERMISSIONS['admin'])
    with TestClient(app) as client:
        assert app.state.camera_manager.get_sources() == {}
        # Disabling camera workers must also disable the on-demand fallback.
        assert client.get('/video_feed?camera=rtsp_camera').status_code == 503
        assert client.post('/api/v1/learnsight/sessions', json={
            'study_goal': 'review', 'planned_minutes': 25}).status_code == 401
        login = client.post('/auth/login', json={
            'username': 'release-test', 'password': password})
        assert login.status_code == 200
        headers = {'Authorization': 'Bearer ' + login.json()['access_token']}
        created = client.post('/api/v1/learnsight/sessions', headers=headers,
                              json={'study_goal': 'review', 'planned_minutes': 25})
        assert created.status_code == 201
        sid = created.json()['session_id']
        assert client.post(f'/api/v1/learnsight/sessions/{sid}/sync', headers=headers).status_code == 503
        data_store.update_metrics(people_count=2, fps_processing=5)
        synced = client.post(f'/api/v1/learnsight/sessions/{sid}/sync', headers=headers)
        assert synced.status_code == 200
        assert synced.json()['present_now'] is True
        assert client.post(f'/api/v1/learnsight/sessions/{sid}/end', headers=headers).json()['status'] == 'ended'
        assert client.get('/learnsight.html').status_code == 200
