"""In-memory, privacy-first study-session state for LearnSight.

This is intentionally a small MVP.  It stores no frame, face, name, or
biometric result.  It retains only a study goal and aggregate detector events
for the current process.  A later production iteration can persist an
explicitly consented session summary through the project's database layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, Optional
from uuid import uuid4

from .models import DetectorObservationRequest, StartStudySessionRequest, StudySessionResponse
from .telemetry import MAX_SIGNAL_AGE_SECONDS


AWAY_REMINDER_AFTER = timedelta(seconds=90)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StudySession:
    session_id: str
    study_goal: str
    planned_minutes: int
    source_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    present_now: bool = False
    last_present_at: Optional[datetime] = None
    absence_started_at: Optional[datetime] = None
    observation_count: int = 0
    person_count: Optional[int] = None
    last_observed_at: Optional[datetime] = None
    detector_source_id: Optional[str] = None
    signal_origin: str = "none"
    signal_error: Optional[str] = None

    def as_response(self) -> StudySessionResponse:
        stale = bool(self.signal_origin == 'detector' and self.last_observed_at
                     and (_utc_now() - self.last_observed_at).total_seconds() > MAX_SIGNAL_AGE_SECONDS)
        available = not self.signal_error and not stale and self.last_observed_at is not None
        away_eligible = bool(
            self.status == "active"
            and available
            and self.absence_started_at
            and _utc_now() - self.absence_started_at >= AWAY_REMINDER_AFTER
        )
        return StudySessionResponse(
            session_id=self.session_id,
            study_goal=self.study_goal,
            planned_minutes=self.planned_minutes,
            source_id=self.source_id,
            started_at=self.started_at,
            ended_at=self.ended_at,
            status=self.status,
            present_now=self.present_now if available else False,
            last_present_at=self.last_present_at,
            absence_started_at=self.absence_started_at,
            away_reminder_eligible=away_eligible,
            observation_count=self.observation_count,
            person_count=self.person_count if available else None,
            last_observed_at=self.last_observed_at,
            detector_source_id=self.detector_source_id,
            signal_origin=self.signal_origin,
            signal_status='unavailable' if self.signal_error else ('stale' if stale else ('fresh' if available else 'waiting')),
            signal_max_age_seconds=MAX_SIGNAL_AGE_SECONDS,
            note=(
                self.signal_error or ("人物訊號已過期；不能判斷有人或無人。" if stale else
                "偵測到有人在鏡頭範圍內。這不代表專注度或動作分類。"
                if self.present_now
                else "尚未偵測到人員；僅作為可選的離席提醒訊號，不評價專注度。")
            ),
        )

    @property
    def status(self) -> str:
        return "ended" if self.ended_at else "active"


class LearnSightService:
    def __init__(self) -> None:
        self._sessions: Dict[str, StudySession] = {}
        self._lock = Lock()

    def start(self, request: StartStudySessionRequest) -> StudySessionResponse:
        session = StudySession(
            session_id=uuid4().hex,
            study_goal=request.study_goal.strip(),
            planned_minutes=request.planned_minutes,
            source_id=request.source_id.strip(),
            started_at=_utc_now(),
            detector_source_id=request.detector_source_id,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session.as_response()

    def get(self, session_id: str) -> StudySessionResponse:
        with self._lock:
            session = self._require(session_id)
            return session.as_response()

    def observe(self, session_id: str, observation: DetectorObservationRequest, *, detector_source_id=None) -> StudySessionResponse:
        observed_at = observation.observed_at or _utc_now()
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)

        with self._lock:
            session = self._require(session_id)
            if session.ended_at:
                raise ValueError("已結束的讀書時段不能再接收偵測事件")

            if detector_source_id:
                if session.detector_source_id and session.detector_source_id != detector_source_id:
                    raise ValueError('此時段已綁定其他偵測來源')
                if session.signal_origin == 'detector' and session.last_observed_at and observed_at <= session.last_observed_at:
                    return session.as_response()
                if session.signal_error or (session.last_observed_at and
                        (observed_at - session.last_observed_at).total_seconds() > MAX_SIGNAL_AGE_SECONDS):
                    session.absence_started_at = None
                session.detector_source_id = detector_source_id
            session.signal_origin = 'detector' if detector_source_id else 'manual'
            session.signal_error = None
            session.last_observed_at = observed_at
            session.person_count = observation.person_count

            session.observation_count += 1
            if observation.person_count > 0:
                session.present_now = True
                session.last_present_at = observed_at
                session.absence_started_at = None
            else:
                session.present_now = False
                if session.absence_started_at is None:
                    session.absence_started_at = observed_at
            return session.as_response()

    def mark_unavailable(self, session_id: str, message: str) -> None:
        with self._lock:
            session = self._require(session_id)
            if session.ended_at:
                raise ValueError('已結束的讀書時段不能再接收偵測事件')
            session.signal_error = message
            session.absence_started_at = None
            session.present_now = False

    def end(self, session_id: str) -> StudySessionResponse:
        with self._lock:
            session = self._require(session_id)
            if session.ended_at is None:
                session.ended_at = _utc_now()
            return session.as_response()

    def _require(self, session_id: str) -> StudySession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError("找不到 LearnSight 讀書時段") from exc


learnsight_service = LearnSightService()

