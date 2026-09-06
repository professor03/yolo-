"""Request and response models for the LearnSight MVP.

The module deliberately accepts detector *telemetry*, not a video frame.  It
therefore cannot and must not make a claim about a student's attention, pose,
or identity.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class StartStudySessionRequest(BaseModel):
    study_goal: str = Field(..., min_length=1, max_length=240)
    planned_minutes: int = Field(..., ge=5, le=480)
    source_id: str = Field(default="default", min_length=1, max_length=100)


class DetectorObservationRequest(BaseModel):
    person_count: int = Field(..., ge=0, le=50)
    track_ids: List[int] = Field(default_factory=list, max_length=50)
    average_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    observed_at: Optional[datetime] = None


class StudySessionResponse(BaseModel):
    session_id: str
    study_goal: str
    planned_minutes: int
    source_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: str
    present_now: bool
    last_present_at: Optional[datetime] = None
    absence_started_at: Optional[datetime] = None
    away_reminder_eligible: bool
    observation_count: int
    note: str
