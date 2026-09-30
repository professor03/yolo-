import io
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from PIL import Image
from src.server.browser_demo import create_browser_app


def jpeg(size=(320, 240)):
    output = io.BytesIO()
    Image.new('RGB', size).save(output, format='JPEG')
    return output.getvalue()


def create_guest_session(client):
    token = client.post('/auth/guest').json()['access_token']
    auth = {'Authorization': 'Bearer ' + token}
    session = client.post('/api/v1/learnsight/sessions', headers=auth,
                          json={'study_goal': '複習', 'planned_minutes': 5}).json()
    return auth, '/api/v1/learnsight/sessions/' + session['session_id']


def test_guest_isolation_and_actual_inference_contract():
    counts = iter([1, 0, 1])
    now = [0]
    def predictor(image):
        return [dict(xyxy=[1, 2, 30, 40], confidence=.9)] * next(counts)
    base = datetime.now(timezone.utc)
    with TestClient(create_browser_app(predictor, lambda: now[0], lambda: base + timedelta(seconds=now[0]))) as client:
        auth, path = create_guest_session(client)
        other, _ = create_guest_session(client)
        assert client.post(path + '/sync', headers=other).status_code == 404
        assert client.post(path + '/sync').status_code == 401
        for count in [1, 0, 1]:
            now[0] += 3
            response = client.post(path + '/frame', content=jpeg(),
                                  headers={**auth, 'Content-Type': 'image/jpeg'})
            assert response.status_code == 200
            session = response.json()['session']
            assert session['person_count'] == count
            assert session['signal_origin'] == 'detector'
        assert session['observation_count'] == 3
        assert client.post(path + '/stop-camera', headers=auth).json()['signal_status'] == 'unavailable'
        assert client.post(path + '/end', headers=auth).json()['status'] == 'ended'
        assert client.post(path + '/frame', content=jpeg(), headers=auth).status_code == 409
        assert client.get('/health').json()['frame_storage'] is False


def test_frame_limits_and_expiry():
    now = [0]
    with TestClient(create_browser_app(lambda image: [], lambda: now[0])) as client:
        auth, path = create_guest_session(client)
        headers = {**auth, 'Content-Type': 'image/jpeg'}
        assert client.post(path + '/frame', content=b'a' * 300001, headers=headers).status_code == 413
        now[0] += 3
        assert client.post(path + '/frame', content=jpeg((1281, 20)), headers=headers).status_code == 413
        now[0] += 3
        assert client.post(path + '/frame', content=b'invalid', headers=headers).status_code == 400
        now[0] += 3
        assert client.post(path + '/frame', content=jpeg(), headers=headers).status_code == 200
        assert client.post(path + '/frame', content=jpeg(), headers=headers).status_code == 429
        now[0] = 7201
        assert client.post(path + '/sync', headers=auth).status_code == 401
