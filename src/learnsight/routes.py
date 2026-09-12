"""Authenticated API routes for the LearnSight MVP."""

from fastapi import APIRouter, Depends, HTTPException, Header, status

from src.api.data_store import data_store
from src.auth.security import get_security_manager

from .models import DetectorObservationRequest, StartStudySessionRequest, StudySessionResponse
from .service import learnsight_service
from .telemetry import select_observation


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
    session = _not_found_or_invalid(lambda: learnsight_service.get(session_id))
    if session.status == 'ended':
        raise HTTPException(status_code=409, detail='已結束的讀書時段不能再同步')
    try:
        person_count, observed_at, source_id = select_observation(
            data_store.get_current_metrics(), data_store.get_sources(), session.detector_source_id)
    except ValueError as exc:
        _not_found_or_invalid(lambda: learnsight_service.mark_unavailable(session_id, str(exc)))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _not_found_or_invalid(
        lambda: learnsight_service.observe(
            session_id,
            DetectorObservationRequest(person_count=person_count, observed_at=observed_at),
            detector_source_id=source_id,
        )
    )


@router.post("/sessions/{session_id}/end", response_model=StudySessionResponse)
def end_session(session_id: str, _: dict = Depends(get_current_user)) -> StudySessionResponse:
    return _not_found_or_invalid(lambda: learnsight_service.end(session_id))

