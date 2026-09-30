"""Bounded public browser demo. Never exposes the local administration API.

One opaque capability owns one LearnSightService; frames live only during
inference. This is a temporary, CPU-only demonstration, not a production API.
"""
from __future__ import annotations

import asyncio
import io
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from PIL import Image, UnidentifiedImageError

from src.learnsight.models import DetectorObservationRequest, StartStudySessionRequest
from src.learnsight.service import LearnSightService

MAX_BYTES = 300_000
TOKEN_TTL = 7200


@dataclass
class Guest:
    service: LearnSightService = field(default_factory=LearnSightService)
    expires: float = 0
    sessions: int = 0
    last_frame: float = -100
    revisions: dict = field(default_factory=dict)


def load_predictor():
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')

    def predict(image):
        result = model.predict(image, classes=[0], conf=0.35, imgsz=640,
                               max_det=50, device='cpu', verbose=False)[0]
        return [dict(xyxy=[round(float(v), 1) for v in box.xyxy[0]],
                     confidence=round(float(box.conf[0]), 4)) for box in result.boxes]
    return predict


def create_browser_app(predictor=None, clock=time.monotonic, utc_clock=None):
    guests: dict[str, Guest] = {}
    guest_requests = deque()
    frame_requests = deque()
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = predictor or await asyncio.to_thread(load_predictor)
        yield
        guests.clear()
    app = FastAPI(title='LearnSight Browser Demo', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.predictor = predictor

    def limited(queue, maximum):
        now = clock()
        while queue and now - queue[0] >= 60:
            queue.popleft()
        if len(queue) >= maximum:
            raise HTTPException(429, '展示服務忙碌，請稍後再試。')
        queue.append(now)

    def authorize(authorization):
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(401, '請重新建立訪客工作階段。')
        guest = guests.get(authorization[7:])
        if not guest or guest.expires <= clock():
            raise HTTPException(401, '訪客憑證已過期；請重新整理並開始。')
        return guest

    def owned(guest, session_id):
        try:
            return guest.service.get(session_id)
        except KeyError:
            raise HTTPException(404, '找不到你的讀書時段。')

    @app.middleware('http')
    async def privacy_headers(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/health')
    def health():
        return dict(status='ok', model_ready=app.state.predictor is not None,
                    model='YOLOv8n', device='cpu', frame_storage=False)

    @app.post('/auth/guest')
    async def guest_login():
        limited(guest_requests, 30)
        for token in [key for key, value in guests.items() if value.expires <= clock()]:
            del guests[token]
        if len(guests) >= 64:
            raise HTTPException(503, '展示訪客額度已滿，請稍後再試。')
        token = secrets.token_urlsafe(32)
        guests[token] = Guest(expires=clock() + TOKEN_TTL)
        return dict(access_token=token, expires_in=TOKEN_TTL)

    @app.post('/api/v1/learnsight/sessions')
    async def start(data: StartStudySessionRequest, authorization: str | None = Header(None)):
        guest = authorize(authorization)
        if not data.study_goal.strip():
            raise HTTPException(422, '學習目標不可空白。')
        if guest.sessions >= 20:
            raise HTTPException(429, '每個訪客最多建立 20 個時段。')
        data.detector_source_id = 'browser-camera'
        guest.sessions += 1
        session = guest.service.start(data)
        guest.revisions[session.session_id] = 0
        return session

    @app.post('/api/v1/learnsight/sessions/{session_id}/sync')
    async def sync(session_id: str, authorization: str | None = Header(None)):
        return owned(authorize(authorization), session_id)

    @app.post('/api/v1/learnsight/sessions/{session_id}/end')
    async def end(session_id: str, authorization: str | None = Header(None)):
        guest = authorize(authorization)
        owned(guest, session_id)
        guest.revisions[session_id] += 1
        return guest.service.end(session_id)

    @app.post('/api/v1/learnsight/sessions/{session_id}/stop-camera')
    async def stop(session_id: str, authorization: str | None = Header(None)):
        guest = authorize(authorization)
        current = owned(guest, session_id)
        guest.revisions[session_id] += 1
        if current.status == 'active':
            guest.service.mark_unavailable(session_id, '鏡頭輸入已停止，無法判斷有人或無人。')
        return guest.service.get(session_id)

    @app.post('/api/v1/learnsight/sessions/{session_id}/frame')
    async def frame(session_id: str, request: Request, authorization: str | None = Header(None)):
        guest = authorize(authorization)
        if owned(guest, session_id).status != 'active':
            raise HTTPException(409, '時段已結束，不能再接收影格。')
        if not request.headers.get('content-type', '').startswith('image/jpeg'):
            raise HTTPException(415, '僅接受 JPEG 影格。')
        if clock() - guest.last_frame < 2:
            raise HTTPException(429, '每位訪客至少間隔 2 秒送一張影格。')
        limited(frame_requests, 120)
        if inference_lock.locked():
            raise HTTPException(429, '模型正在服務另一位訪客，請稍後再試。')
        guest.last_frame = clock()
        revision = guest.revisions[session_id]
        async with inference_lock:
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise HTTPException(413, '影格超過 300 KB。')
            try:
                with Image.open(io.BytesIO(data)) as source:
                    width, height = source.size
                    if source.format != 'JPEG' or width > 1280 or height > 960:
                        raise HTTPException(413, '影格尺寸上限 1280×960，需為 JPEG。')
                    image = source.convert('RGB')
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                raise HTTPException(400, '無法讀取影格。')
            if app.state.predictor is None:
                raise HTTPException(503, '模型尚未準備完成。')
            started = time.perf_counter()
            try:
                boxes = await asyncio.to_thread(app.state.predictor, image)
            except Exception:
                raise HTTPException(503, '模型推論失敗，請稍後再試。')
            finally:
                image.close()
                data.clear()
            # A concurrent end/stop request must not re-open or overwrite a session.
            if guest.service.get(session_id).status != 'active' or guest.revisions[session_id] != revision:
                raise HTTPException(409, '時段已結束或鏡頭已停止。')
            observation = DetectorObservationRequest(
                person_count=len(boxes), observed_at=utc_clock() if utc_clock else datetime.now(timezone.utc))
            session = guest.service.observe(session_id, observation,
                                            detector_source_id='browser-camera')
            return dict(session=session, boxes=boxes, width=width, height=height,
                        inference_ms=round((time.perf_counter() - started) * 1000, 1),
                        model='YOLOv8n', device='cpu')
    return app


app = create_browser_app()
