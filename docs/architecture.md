# 系統架構

本次公開版支援的入口是 `scripts/serve_demo.py`（網頁/API）與
`scripts/detect_demo.py`（本機影片推論）。兩者從專案根目錄執行，
共用 demo 專用的 SQLite 與 JSON 檔案。

## 資料流

```text
本機影片 → OpenCV → YOLOv8 + ByteTrack → 人物/線段/區域事件
                                             ↓
                                   JSON 聚合狀態 + SQLite
                                             ↓
                                FastAPI → 儀表板 / LearnSight
                                             ↓
                                手動同步人數 → 讀書時段狀態
```

| 模組 | 職責 | 邊界 |
|---|---|---|
| people_detect.py | 影片讀取、推論、追蹤、輸出 | 執行預訓練模型，不是自行訓練的動作分類器 |
| src/logic | 線段、區域、座標轉換與事件 | 依賴場景設定與追蹤品質 |
| src/api/data_store.py | 聚合狀態、JSON 檔案交換 | 不等同跨機器訊息佇列 |
| src/database | SQLAlchemy、帳號/攝影機等資料存取 | SQLite；本機輸出不納入 Git |
| src/server/app.py | 主要 FastAPI、登入、儀表板及路由 | 公開 demo 限本機使用 |
| src/learnsight | 讀書時段、出現/離席訊號 | 時段存記憶體、同步聚合人數 |
| static | 登入、儀表板、管理與 LearnSight 頁面 | 模擬訊號需與真實推論區分 |

## 支援範圍

本機 demo 預設關閉背景攝影機，使用者自行建立管理帳號。
尚未完成多使用者時段隔離、跨服務 session 儲存及完整串流權限驗證，
因此目前不提供公開網路部署保證。

`realtime_detection.py`、其他 server 入口、`src/ml` 與
`src/streaming` 保留為歷史研究模組。這些檔案存在不代表其全部功能
已經接入本次 demo。請以 [操作指南](usage.md) 與
[驗證紀錄](project-overview.md) 列出的路徑為準。
