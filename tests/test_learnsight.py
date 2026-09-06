"""Unit tests for LearnSight state transitions; no camera or model is required."""

from datetime import datetime, timedelta, timezone

from src.learnsight.models import DetectorObservationRequest, StartStudySessionRequest
from src.learnsight.service import LearnSightService


def test_presence_then_away_reminder() -> None:
    service = LearnSightService()
    session = service.start(
        StartStudySessionRequest(study_goal="複習演算法", planned_minutes=30, source_id="demo")
    )
    present = service.observe(session.session_id, DetectorObservationRequest(person_count=1))
    assert present.present_now is True
    assert present.away_reminder_eligible is False

    old_timestamp = datetime.now(timezone.utc) - timedelta(seconds=91)
    away = service.observe(
        session.session_id,
        DetectorObservationRequest(person_count=0, observed_at=old_timestamp),
    )
    assert away.present_now is False
    assert away.away_reminder_eligible is True
    assert "專注度" in away.note


def test_session_can_end_once() -> None:
    service = LearnSightService()
    session = service.start(
        StartStudySessionRequest(study_goal="整理筆記", planned_minutes=25)
    )
    ended = service.end(session.session_id)
    assert ended.status == "ended"
    assert service.end(session.session_id).ended_at == ended.ended_at
