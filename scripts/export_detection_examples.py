"""Export real YOLO detections for a beginner-friendly visual guide.

Usage: python scripts/export_detection_examples.py --video YOUR_VIDEO --weights yolov8n.pt --outdir runs/visual-guide
Only recorded video is accepted; this command does not open a camera.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def panel(frame, title, detail):
    header = np.full((76, frame.shape[1], 3), (35, 28, 20), dtype=np.uint8)
    cv2.putText(header, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .67, (255,255,255), 2)
    cv2.putText(header, detail, (16, 59), cv2.FONT_HERSHEY_SIMPLEX, .47, (215,225,230), 1)
    return np.vstack([header, frame])


def annotate(result, frame):
    """Keep labels short so nearby people's scores do not overlap."""
    annotated = frame.copy()
    labels = []
    ids = result.boxes.id.cpu().tolist() if result.boxes.id is not None else [None] * len(result.boxes)
    for box, track_id in zip(result.boxes.xyxy.cpu().tolist(), ids):
        x1, y1, x2, y2 = map(int, box)
        label = f'ID {int(track_id)}' if track_id is not None else 'person'
        color = (0, 210, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .42, 1)[0][0]
        label_y = max(18, y1)
        for left, top, right, bottom in labels:
            if x1 < right and x1+width+5 > left and label_y-18 < bottom and label_y > top:
                label_y = min(frame.shape[0]-1, y2+18)
        labels.append((x1, label_y-18, x1+width+5, label_y))
        cv2.rectangle(annotated, (x1, label_y-18), (x1+width+5, label_y), (35,28,20), -1)
        cv2.putText(annotated, label, (x1+2,label_y-5), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
    return annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('yolov8n.pt'))
    parser.add_argument('--outdir', type=Path, default=Path('runs/visual-guide'))
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error('Provide an existing local video file.')
    args.outdir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    records = []
    selected = {100, 110, 120}
    try:
        for idx in range(121):
            ok, frame = cap.read()
            if not ok:
                break
            result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                 classes=[0], conf=.25, iou=.7, imgsz=640,
                                 device='cpu', verbose=False)[0]
            if idx not in selected:
                continue
            count = len(result.boxes)
            annotated = annotate(result, frame)
            label = f'Frame {idx} | {idx/fps:.1f}s | {count} model detections'
            cv2.imwrite(str(args.outdir / f'detection-{idx}.jpg'),
                        panel(annotated, 'YOLOv8n + ByteTrack | real inference', label),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            if idx == 100:
                pair = np.hstack([panel(frame, 'INPUT | recorded sample video', f'Frame {idx} | {idx/fps:.1f}s'),
                                  panel(annotated, 'OUTPUT | person boxes + tracking IDs', label)])
                cv2.imwrite(str(args.outdir / 'before-after.jpg'), pair,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
            records.append({'frame_index':idx, 'video_seconds':idx/fps,
                            'count':count, 'boxes_xyxy':result.boxes.xyxy.cpu().tolist(),
                            'confidence':result.boxes.conf.cpu().tolist(),
                            'track_ids':result.boxes.id.cpu().tolist() if result.boxes.id is not None else []})
    finally:
        cap.release()
    if len(records) != 3:
        raise RuntimeError('The video must contain at least 121 frames.')
    manifest = {'video_file':args.video.name, 'video_sha256':hashlib.sha256(args.video.read_bytes()).hexdigest(),
                'weights_file':args.weights.name, 'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                'model':'YOLOv8n', 'device':'cpu', 'tracker':'bytetrack.yaml',
                'conf':.25,'iou':.7,'imgsz':640,'classes':[0],
                'sequence':'Frames 0 through 120 processed in order; exports 100,110,120.',
                'records':records, 'accuracy_evaluation':False}
    (args.outdir/'inference-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'exports':str(args.outdir), 'frames':[(r['frame_index'],r['count'],r['track_ids']) for r in records]}))


if __name__ == '__main__':
    main()

