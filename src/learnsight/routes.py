"""Authenticated API routes for the LearnSight MVP."""

from fastapi import APIRouter, Depends, HTTPException, Header, status

from src.api.data_store import data_store
from src.auth.security import get_security_manager

from .models import DetectorObservationRequest, StartStudySessionRequest, StudySessionResponse
from .service import learnsight_service


router = APIRouter(prefix="/api/v1/learnsight", tags=["LearnSight"])


def get_current_user(authorization: str = Header(None, alias="Authorization")) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="請先登入後再使用 LearnSight",
        )
    payload = get_security_manager().verify_token(authorization.split(" ", 1)[1], "access")
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登入憑證已失效，請重新登入",
        )
    return payload


def _not_found_or_invalid(operation):
    try:
        return operation()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/sessions", response_model=StudySessionResponse, status_code=status.HTTP_201_CREATED)
def start_session(
    request: StartStudySessionRequest,
    _: dict = Depends(get_current_user),
) -> StudySessionResponse:
    return learnsight_service.start(request)


@router.get("/sessions/{session_id}", response_model=StudySessionResponse)
def get_session(session_id: str, _: dict = Depends(get_current_user)) -> StudySessionResponse:
    return _not_found_or_invalid(lambda: learnsight_service.get(session_id))


@router.post("/sessions/{session_id}/observations", response_model=StudySessionResponse)
def record_observation(
    session_id: str,
    observation: DetectorObservationRequest,
    _: dict = Depends(get_current_user),
) -> StudySessionResponse:
    return _not_found_or_invalid(lambda: learnsight_service.observe(session_id, observation))


@router.post("/sessions/{session_id}/sync", response_model=StudySessionResponse)
def sync_detector_metrics(session_id: str, _: dict = Depends(get_current_user)) -> StudySessionResponse:
    """Record only the latest aggregate person count from the shared detector store."""
    metrics = data_store.get_current_metrics()
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="目前沒有可用的偵測資料；請先以本機影片啟動偵測流程。",
        )
    try:
        person_count = max(0, int(metrics.get("people_count", 0)))
    except (TypeError, ValueError):
        person_count = 0
    return _not_found_or_invalid(
        lambda: learnsight_service.observe(
            session_id,
            DetectorObservationRequest(person_count=person_count),
        )
    )


@router.post("/sessions/{session_id}/end", response_model=StudySessionResponse)
def end_session(session_id: str, _: dict = Depends(get_current_user)) -> StudySessionResponse:
    return _not_found_or_invalid(lambda: learnsight_service.end(session_id))
