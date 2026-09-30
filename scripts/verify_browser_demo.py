"""Real model/API smoke test on an authorized still image, not webcam footage."""
import argparse
import io
import json
import sys
from pathlib import Path
from time import perf_counter, sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from PIL import Image
from ultralytics import YOLO
from src.server.browser_demo import create_browser_app

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolov8n.pt')
parser.add_argument('--image', required=True)
parser.add_argument('--output', default='runs/browser-demo/verification.json')
parser.add_argument('--left-half', action='store_true', help='Use only the original left half of a before/after comparison.')
args = parser.parse_args()
model = YOLO(args.model)


def predictor(image):
    result = model.predict(image, classes=[0], conf=.35, imgsz=640, max_det=50,
                           device='cpu', verbose=False)[0]
    return [dict(xyxy=[float(value) for value in box.xyxy[0]],
                 confidence=float(box.conf[0])) for box in result.boxes]


with Image.open(args.image) as source:
    if args.left_half: source = source.crop((0, 0, source.width // 2, source.height))
    source = source.convert('RGB'); source.thumbnail((640, 480))
    image = io.BytesIO(); source.save(image, 'JPEG')
blank = io.BytesIO(); Image.new('RGB', (640, 480), 'white').save(blank, 'JPEG')
with TestClient(create_browser_app(predictor)) as client:
    token = client.post('/auth/guest').json()['access_token']
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'image/jpeg'}
    session = client.post('/api/v1/learnsight/sessions', headers={'Authorization': 'Bearer ' + token},
                          json={'study_goal': 'API smoke test', 'planned_minutes': 5}).json()
    path = '/api/v1/learnsight/sessions/' + session['session_id']
    results = []
    for data in [image.getvalue(), blank.getvalue(), image.getvalue()]:
        sleep(2.1)
        started = perf_counter()
        response = client.post(path + '/frame', content=data, headers=headers)
        assert response.status_code == 200, response.text
        result = response.json()
        results.append(dict(count=result['session']['person_count'],
                            inference_ms=result['inference_ms'],
                            request_ms=round((perf_counter() - started) * 1000, 1)))
    assert results[0]['count'] > 0 and results[1]['count'] == 0 and results[2]['count'] > 0
    ended = client.post(path + '/end', headers=headers).json()
    assert ended['status'] == 'ended' and ended['observation_count'] == 3
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(input='authorized still image / blank / same image; not webcam', cropped_original=args.left_half,
                                     device='cpu', model='YOLOv8n', results=results,
                                     session_ended=True), indent=2), encoding='utf-8')
    print(output.read_text(encoding='utf-8'))
