# 發布驗證 · 2026-09-06

## 公開 API 流程

以全新 .venv-release 安裝 requirements-smoke.txt，
執行下列指定測試：25 passed，4 個 UTC 時間 API 棄用警告。
pip check 回報沒有損壞的套件依賴。

```powershell
python -m pytest tests/test_public_release.py tests/test_learnsight.py tests/test_geometry.py tests/test_calibration.py -q -p no:cacheprovider
```

其中包含攝影機停用時 /video_feed 回傳 503，
避免背景工作者已停用但按需串流仍嘗試連線。
另驗證設定範本沒有預設帳號/金鑰，未設定 YOLO_API_KEY 時拒絕 API-key 登入，
以及明確設定後才允許相符金鑰。
測試以暫存資料庫和資料目錄執行，不使用既有帳號、影像或攝影機。

## 本機模型推論

環境：Windows、Python 3.12.14、Intel Core i5-12500H（CPU 推論）。
PyTorch 2.4.1、torchvision 0.19.1、Ultralytics 8.3.203、
NumPy 2.2.6、OpenCV 4.12.0.88。
本輪推論使用既有虛擬環境，並非全新安裝全部 requirements.txt。

命令：`python scripts/detect_demo.py <local-video.mp4> --duration 5`。
使用 configs/local_video.yaml：yolov8n、CPU、imgsz=640、
conf=0.25、iou=0.7、person-only、ByteTrack、show=false、save=false。

| 程式回傳指標 | 結果 |
|---|---|
| 處理影格 | 21 |
| total_time | 6.029 秒 |
| fps_avg | 3.483 |
| 最後一幀人物數 | 3 |
| 線段事件 | 0（本次未配置計數線） |

影像讀取、模型推論、追蹤與資料寫入流程正常結束，exit code 0。
測試影片為本機私人素材，未上傳；最後人物數沒有人工標註核對，
所以這份結果不代表人物偵測準確率或可重現的公開 benchmark。
事件 CSV 僅在有事件時生成；日誌顯示輸出路徑不代表一定生成非空 CSV。

## 整理內容

- 原始素材、權重、快取、資料庫與舊備份由 .gitignore 排除。
- 移除設定檔中的硬編碼密碼；帳號改為自行建立。
- 補回推論必要的異常事件模組至 src/logic，修正舊匯入路徑。
- 補齊操作、架構、LearnSight、限制與第三方說明。
- 對待發布文字檔檢查秘密、私人 IP、含帳密 URL 與文件連結。

完整歷史測試與生產部署不在上述通過結果內；
待辦與功能界線見 [limitations.md](limitations.md)。

## 歷史測試抽查

另安裝 pandas 2.2.3 後執行
`pytest tests/test_counter.py tests/test_evaluate.py -q --maxfail=5`：
20 passed、5 failed，達到失敗上限後停止。失敗均在旧版
test_evaluate.py，包括 EvaluationResult 欄位、match_events 回傳值、
calculate_idf1 呼叫方式及字串 track_id 格式與目前模組不一致。
這些歷史測試仍保留供遷移，沒有藉由刪除測試宣稱全套通過。
