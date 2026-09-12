"""Detect people in an explicit local file using serve_demo's isolated state."""
import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--weights', type=Path, help='Existing local model weights (optional).')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('--duration must be greater than zero.')
    video = args.video.resolve()
    weights = args.weights.resolve() if args.weights else None
    if weights and not weights.is_file():
        parser.error('--weights must point to an existing file.')
    if not video.is_file():
        parser.error('Supply an existing local video file. Camera URLs are not accepted.')
    os.chdir(ROOT)
    os.environ.setdefault('DATABASE_URL', 'sqlite:///data/public-demo.db')
    os.environ.setdefault('DATASTORE_PATH', 'data/public-demo.json')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics-demo'))
    from people_detect import run_detection
    from src.utils.common import load_yaml_config
    config = load_yaml_config('configs/local_video.yaml')
    if weights:
        config['model'] = str(weights)
    result = run_detection(str(video), config,
                           show=args.show, save=False, duration=args.duration,
                           outdir='runs/public-demo', session_id='demo-' + uuid4().hex[:12])
    print(result)
if __name__ == '__main__':
    main()

