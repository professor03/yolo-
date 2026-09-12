"""Signal validity tests; fixture telemetry is explicitly not real inference."""
from datetime import datetime, timedelta, timezone
import pytest
from src.learnsight.telemetry import select_observation
from src.learnsight.models import StartStudySessionRequest, DetectorObservationRequest
from src.learnsight.service import LearnSightService


def metric(count=1, seconds=0):
    return {'timestamp': (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(), 'people_count': count}


@pytest.mark.parametrize('count', [None, -1, 1.2, True, '3', 51])
def test_invalid_count_is_not_absence(count):
    with pytest.raises(ValueError):
        select_observation(metric(count), [])


@pytest.mark.parametrize('seconds', [31, -10])
def test_old_or_future_frame_rejected(seconds):
    with pytest.raises(ValueError):
        select_observation(metric(seconds=seconds), [])


def test_online_source_and_zero_are_valid():
    count, _, source = select_observation({'streams': {'local-video': metric(0)}},
        [{'source_id': 'local-video', 'status': 'online'}], 'local-video')
    assert count == 0 and source == 'local-video'


def test_offline_is_not_zero_and_no_cross_source_sum():
    with pytest.raises(ValueError, match='離線'):
        select_observation({'streams': {'local-video': metric(1)}},
            [{'source_id': 'local-video', 'status': 'offline'}], 'local-video')
    with pytest.raises(ValueError, match='多個'):
        select_observation({'streams': {'a': metric(), 'b': metric()}}, [])
    with pytest.raises(ValueError, match='找不到'):
        select_observation({'streams': {'a': metric()}}, [], 'b')


def test_duplicate_sync_does_not_increment_and_unavailability_resets_absence():
    service = LearnSightService()
    sid = service.start(StartStudySessionRequest(study_goal='study', planned_minutes=25)).session_id
    now = datetime.now(timezone.utc)
    observation = DetectorObservationRequest(person_count=0, observed_at=now)
    first = service.observe(sid, observation, detector_source_id='local-video')
    repeat = service.observe(sid, observation, detector_source_id='local-video')
    assert repeat.observation_count == first.observation_count == 1
    service.mark_unavailable(sid, 'offline')
    missing = service.get(sid)
    assert missing.person_count is None and missing.absence_started_at is None
    assert not missing.away_reminder_eligible and missing.signal_status == 'unavailable'
    resumed = service.observe(sid, DetectorObservationRequest(person_count=0, observed_at=now+timedelta(seconds=1)), detector_source_id='local-video')
    assert resumed.absence_started_at == now+timedelta(seconds=1)
    assert resumed.signal_origin == 'detector'


def test_manual_is_labeled_and_stale_detector_cannot_remind(monkeypatch):
    import src.learnsight.service as module
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(module, '_utc_now', lambda: now)
    service = LearnSightService()
    sid = service.start(StartStudySessionRequest(study_goal='study', planned_minutes=25)).session_id
    assert service.observe(sid, DetectorObservationRequest(person_count=1)).signal_origin == 'manual'
    service.observe(sid, DetectorObservationRequest(person_count=0, observed_at=now), detector_source_id='a')
    monkeypatch.setattr(module, '_utc_now', lambda: now + timedelta(seconds=100))
    state = service.get(sid)
    assert state.signal_status == 'stale' and state.person_count is None
    assert not state.away_reminder_eligible


def test_unscoped_legacy_metrics_can_be_read_again():
    assert select_observation(metric(), [], 'default')[2] == 'default'

