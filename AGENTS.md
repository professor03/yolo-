# Contributor guidance

The supported API entry point is `src.server.app:app`. The detector entry point is
`people_detect.py`; LearnSight is under `src/learnsight`. Read README.md and
docs/architecture.md before changing the public workflow.

Use local recorded video for verification. Never open a configured camera during
tests. Keep credentials, frames, databases, datasets, weights and logs untracked.
Do not claim attention, action-recognition accuracy or learning improvement from
person-count telemetry. Preserve user data and use isolated test databases.

Run `python -m pytest tests/test_public_release.py tests/test_learnsight.py
tests/test_geometry.py tests/test_calibration.py -q` for the public smoke suite.
Historical modules are retained for development; see docs/limitations.md.
