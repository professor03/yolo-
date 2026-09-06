# YOLO People Detection · LearnSight

[![Public smoke tests](https://github.com/professor03/yolo-/actions/workflows/ci.yml/badge.svg)](https://github.com/professor03/yolo-/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB)
![Research prototype](https://img.shields.io/badge/status-research_prototype-0f766e)

**把影片中的人物偵測結果轉成可查詢的人數、追蹤與場域事件，再連接到 LearnSight 讀書時段。**

這是一個以 YOLOv8、FastAPI、SQLite 與網頁儀表板構成的資訊工程專題。主要流程涵蓋影片讀取、模型推論、事件處理、登入、資料儲存及前端展示。

English: A computer-vision prototype for person detection, tracking, line/zone events and a FastAPI dashboard. LearnSight connects aggregate person counts to study sessions. No personal footage or model weights are bundled.

## 導覽

| 想了解 | 閱讀入口 |
|---|---|
| 評審：做了什麼、如何驗證 | [專題與驗證紀錄](docs/project-overview.md) |
| 使用者：安裝與操作 | 下方快速開始及 [操作指南](docs/usage.md) |
| 開發者：模組與資料流 | [系統架構](docs/architecture.md) |
| LearnSight：API 與限制 | [LearnSight 說明](docs/learnsight.md) |
| 問題回報 | [Issues](https://github.com/professor03/yolo-/issues) · [貢獻指南](CONTRIBUTING.md) |

## 已有功能

- **人物偵測與追蹤**：YOLOv8 人物框、追蹤 ID、CPU 推論、可設定的門檻與追蹤器。
- **場域事件**：線段進出計數、區域占用、平面校正與事件匯出模組。
- **後端及儀表板**：FastAPI、JWT、角色權限、SQLite、JSON 聚合狀態、管理網頁與監控指標。
- **LearnSight MVP**：讀書目標與時段、同步人物數量、無人持續 90 秒的可選提醒資格、結束時段。
- **評估工具**：線段事件比對與 Precision／Recall／F1 等評估程式。

目前視覺模型辨識「人物」，尚未具備學生動作分類。人物出現不能證明專注、理解或學習成效。LearnSight 時段存在服務記憶體，重啟會清除。

## 快速開始：本機影片 demo

建議 **Python 3.12、64 位元**。首次安裝需要下載 PyTorch 等套件，CPU 即可運行。Windows PowerShell：

```powershell
git clone https://github.com/professor03/yolo-.git
cd yolo-
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/serve_demo.py
```

第一次啟動會讓你自行建立帳號與密碼，之後用同一個帳號登入。

1. 登入：<http://127.0.0.1:8000/login.html>
2. 儀表板：<http://127.0.0.1:8000/index.html>
3. LearnSight：<http://127.0.0.1:8000/learnsight.html>
4. API 文件：<http://127.0.0.1:8000/docs>

Linux/macOS 使用 `python3.12 -m venv .venv` 與 `source .venv/bin/activate`；後續命令相同，目前本機影片實測平台是 Windows。

第二個終端機啟用同一環境，提供你有權使用的本機影片：

```powershell
python scripts/detect_demo.py "C:\path\to\your-video.mp4" --duration 30
```

預設使用 `yolov8n.pt`，缺少權重時由 Ultralytics 下載。到 LearnSight 按「開始時段」→「同步影片偵測人數」→「結束時段」。手動「有人／無人」按鈕是模擬流程訊號，展示時請說明來源。

demo 僅監聽 `127.0.0.1`，攝影機工作者停用。資料位於 `data/public-demo.db` 與 `data/public-demo.json`，影片不會被上傳。demo 不輸出帶人像的影片；事件與效能輸出位於 `runs/public-demo/`。加上 `--show` 可開啟本機偵測視窗。

## 一般使用

```powershell
Copy-Item .env.example .env
# 編輯 .env，為 JWT_SECRET 設定自行產生的隨機值
python scripts/create_admin.py
python -m uvicorn src.server.app:app --env-file .env --host 127.0.0.1 --port 8000
```

`DISABLE_CAMERA_STREAMS=true` 預設停用背景攝影機。使用自己的攝影機時再明確設定來源及開啟串流。詳見 [操作指南](docs/usage.md)。保留既有帳號與資料庫；不要以刪除資料庫作為一般升級方式。

## 專案結構

```text
people_detect.py       主要影片推論流程
realtime_detection.py  歷史即時偵測入口
scripts/               帳號初始化、本機 demo
configs/               本機影片與進階設定範例
src/
  api/                 聚合狀態、管理與匯出 API
  auth/                JWT、密碼雜湊、權限
  database/            SQLAlchemy 模型與資料存取
  learnsight/          讀書時段與人物訊號
  logic/               幾何、線段／區域、校正
  server/              主 FastAPI 應用及歷史實驗入口
  video/               影片／串流管理
  eval/                事件評估
  ml/, streaming/      研究延伸模組
static/                登入、儀表板、管理、LearnSight
tests/                 公開流程測試與歷史測試
docs/                  專題、架構、操作、限制
```

私人影片、資料集、權重、快取、資料庫、日誌及本機備份由 `.gitignore` 排除。完整公開專案包含原始碼、設定範例與啟動步驟；實驗資料需自行準備。

## 測試與成果界線

```powershell
python -m pytest tests/test_public_release.py tests/test_learnsight.py tests/test_geometry.py tests/test_calibration.py -q
```

CI 使用 `requirements-smoke.txt` 執行相同測試，不下載模型、不連攝影機。涵蓋真實登入、受保護 API、無資料回應、人物同步、時段結束、幾何與校正。歷史測試未全部遷移，綠色 CI 不代表整份原型完成生產驗證。

最新發布檢查：乾淨環境公開測試 **23 項通過**；本機 yolov8n / CPU
試跑處理 **21 個影格、約 3.48 FPS**。完整環境、指令及限制見
[發布驗證](docs/release-validation.md)。目前適用本機展示，LearnSight
時段持久化、多使用者隔離與進階串流授權仍待完成。

2026-09-06 的既有本機 `yolov8m` 管線測試在約 10.68 秒處理 13 個影格，約 1.22 fps。這是單次 CPU 系統測試，硬體資訊未完整記錄，不是準確率或正式 benchmark。公開 demo 預設改用 `yolov8n`；詳見 [驗證紀錄](docs/project-overview.md) 與 [已知限制](docs/limitations.md)。

## 使用範圍與第三方

模型與套件的權利由各自權利人持有，本倉庫不散布權重。此版本未新增專案授權許可；若要再散布或用於產品，請先確認作者及第三方授權。詳見 [第三方說明](THIRD_PARTY_NOTICES.md)。
