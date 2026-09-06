# 操作指南

先依 [README](../README.md) 安裝 Python 3.12 與 requirements.txt。
所有命令從專案根目錄執行。建議獨立虛擬環境，避免沿用其他專案的套件。

## 本機展示

1. 執行 `python scripts/serve_demo.py`。
2. 首次啟動依提示建立帳號。密碼至少 12 字元、最多 72 UTF-8 bytes。
3. 開啟 http://127.0.0.1:8000/login.html 登入，進入 LearnSight。
4. 第二個終端機啟用同一虛擬環境，執行：

```powershell
python scripts/detect_demo.py "C:\path\to\your-video.mp4" --duration 30
```

5. 在推論執行期間建立讀書時段並按「同步影片偵測人數」。
6. 結束時段、查看回傳狀態；保存需要的展示紀錄後再關閉 API。

影片需自行提供且有權使用。首次推論可能下載 yolov8n.pt；
離線環境須預先準備相容權重放在專案根目錄。
加上 `--show` 可觀看本機推論視窗。預設不輸出標註影片。
儀表板的攝影機影像區不會播放此本機影片；本機 demo 共享的是人數等資料。

模型設定在 `configs/local_video.yaml`：預設 CPU、imgsz 640、
只辨識 person（class 0），使用 ByteTrack 追蹤。
`--duration` 是程式執行時間上限；不是精準裁切的影片秒數，
初始化與推論時間會影響實際處理影格數。

## 本機資料

| 路徑 | 內容 |
|---|---|
| data/public-demo.db | demo 專用帳號等 SQLite 資料 |
| data/public-demo.json | 偵測聚合狀態，可供另一個進程讀取 |
| runs/public-demo/ | 推論事件與效能輸出 |
| 服務記憶體 | LearnSight 時段；重啟後不保留 |

資料與私人影片不提交至 Git。既有資料庫不會被 demo 腳本刪除。
JSON 可能保留上一次推論資料，停止推論後不應把同步結果解釋為即時狀態。

## 自訂環境

若已有 `.env`，請先備份並手動核對，不要直接覆蓋。
初次使用可複製 `.env.example`，設定自己的 JWT_SECRET，再執行：

```powershell
python scripts/create_admin.py
python -m uvicorn src.server.app:app --env-file .env --host 127.0.0.1 --port 8000
```

沒有固定 JWT_SECRET 時，服務使用啟動時產生的隨機秘密，
重啟會使既有 token 失效；多進程部署需要另外設計一致的認證設定。

目前正式支援本機影片 demo。接入攝影機屬進階實驗流程：
需自行設定來源、權限及 `DISABLE_CAMERA_STREAMS=false`。
舊 YAML 中的空白帳密與保留網域只是格式範例，不是可連線來源，
也不保證每個歷史模組都會載入環境變數。
正式網路部署前請閱讀 [限制](limitations.md)。

## 常見問題

| 現象 | 檢查方式 |
|---|---|
| 找不到 Python 套件 | 確認已啟用正確虛擬環境並安裝 requirements.txt |
| 模型無法下載 | 核對網路，或自行準備 yolov8n.pt |
| 同步回傳 503 | 先執行影片推論；尚無聚合資料時不會捏造人數 |
| video_feed 回傳 503 | demo 關閉攝影機串流，屬預期行為 |
| 登入失效 | 重新登入；API 重啟可能已換 JWT 秘密 |
| 影片無法開啟 | 確认路徑、檔案權限與 OpenCV 支援的編碼 |
| FPS 偏低 | 用小模型與較低解析度比較，記錄硬體和設定 |

## 檢查指令

```powershell
python -m pytest tests/test_public_release.py tests/test_learnsight.py tests/test_geometry.py tests/test_calibration.py -q
```

這是公開流程的驗證範圍，不涵蓋全部歷史測試、真實攝影機或模型準確率。
