"""Run local webcam detection with an optional built-in-camera fallback."""
import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Preferred OpenCV camera index (default: 0).")
    parser.add_argument("--fallback-camera-index", type=int,
                        help="Try this index only if the preferred camera cannot open.")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--show", action="store_true", help="Show a local preview window.")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than zero.")

    selected_camera = None
    candidates = [args.camera_index]
    if args.fallback_camera_index is not None and args.fallback_camera_index != args.camera_index:
        candidates.append(args.fallback_camera_index)
    for camera_index in candidates:
        capture = cv2.VideoCapture(camera_index)
        opened, frame = capture.isOpened(), None
        if opened:
            opened, frame = capture.read()
        capture.release()
        if opened and frame is not None:
            selected_camera = camera_index
            break
    if selected_camera is None:
        parser.error("Could not open the requested camera or its fallback. Try --camera-index 0, 1, or 2.")
    if selected_camera != args.camera_index:
        print(f"Preferred camera {args.camera_index} unavailable; using fallback {selected_camera}.")
    os.chdir(ROOT)
    os.environ.setdefault("DATABASE_URL", "sqlite:///data/public-demo.db")
    os.environ.setdefault("DATASTORE_PATH", "data/public-demo.json")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics-demo"))
    from people_detect import run_detection
    from src.utils.common import load_yaml_config
    config = load_yaml_config("configs/local_video.yaml")
    config['source_id'] = 'local-camera'
    print(f'Camera input index: {selected_camera}; LearnSight source: local-camera')
    result = run_detection(
        selected_camera, config,
        show=args.show, save=False, duration=args.duration,
        outdir="runs/public-demo", session_id="camera-" + uuid4().hex[:12],
    )
    print(result)


if __name__ == "__main__":
    main()

