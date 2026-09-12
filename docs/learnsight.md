# LearnSight MVP

LearnSight 把讀書目標與人物出現訊號連接起來。2026-09-12 已完成與 AI-Learning-System 的真實本機影片聯測，包含跨頁同步、停止偵測與結束後的瀏覽器摘要保存。這仍是單人本機原型，不是正式多使用者部署。

[完整操作、實跑圖片與驗證紀錄](https://github.com/professor03/AI-Learning-System/blob/main/docs/yolo-integration.md)

## 來源與訊號有效性（2026-09-12）

建立時段時可指定 `detector_source_id`：本機影片 `local-video`、明確啟動的筆電鏡頭 `local-camera`。`source_id` 是學習端標籤，不是鏡頭選擇。未指定時只接受單一來源，首次同步後綁定它；多來源時不能擅自混合人數。自動測試不開鏡頭，硬體路徑尚未現場實測。

| 新回應欄位 | 意義 |
| --- | --- |
| `signal_origin` | `none`／`manual`／`detector`，區分手動輸入與真實摘要 |
| `signal_status` | `waiting`／`fresh`／`stale`／`unavailable` |
| `person_count` | 有效人物框數；未知為 null，不是零 |
| `last_observed_at` | 偵測器寫入的 UTC 時間，不是按下同步的時間 |
| `detector_source_id` | 此時段綁定的視覺來源 |
| `signal_max_age_seconds` | 目前為 30 秒；舊資料不能判斷現在有人或無人 |

相同或較舊的真實訊號時間戳不增加觀察次數。來源離線、資料超過 30 秒、來源不存在、錯誤時間或人物數格式不合法，同步回傳 503 且不增加次數。未知訊號不轉成零人；中斷後離席累計重設。`/observations` 標記為手動輸入，不能說成模型推論。

AI 端每 5 秒嘗試同步，切去筆記頁仍繼續；瀏覽器休眠或背景節流仍可能造成延遲。CORS 預設已允許學習系統 5173／8787 的 localhost 和 127.0.0.1，其他前端位址需設定 `CORS_ORIGINS`。兩個 YOLO 進程必須使用一致的 `DATABASE_URL`／`DATASTORE_PATH`。

## API

先在 /auth/login 取得 access_token，以下端點皆需
`Authorization: Bearer <access_token>`。
互動式 API 文件在 http://127.0.0.1:8000/docs。

| 方法 | 路徑（前綴 /api/v1/learnsight） | 用途 |
|---|---|---|
| POST | /sessions | 以 study_goal、planned_minutes 建立時段 |
| GET | /sessions/{session_id} | 讀取時段 |
| POST | /sessions/{session_id}/observations | 傳入 person_count 作為觀察值 |
| POST | /sessions/{session_id}/sync | 從聚合資料讀取最新人數 |
| POST | /sessions/{session_id}/end | 結束時段 |

沒有 token 回傳 401、未知時段回傳 404、
已結束時段再接收觀察值回傳 409、尚無偵測資料時同步回傳 503。

## 解讀結果

`present_now` 只代表觀察到的人數大於零。
在有效且持續更新的無人訊號下，自首次無人觀察開始持續 90 秒後，讀取時段可得到
`away_reminder_eligible=true`。它是提醒資格，不代表已送出推播；
也不能直接推論個人的專注度、理解或學習成效。

前端手動「有人／無人」是模擬輸入。
實際推論的展示需啟動本機影片流程並按同步。
AI 學習系統已有跨頁自動同步；此倉庫的靜態示範頁仍以手動操作為主。沒有人物身分識別與坐姿／站姿分類。

## 後續工作

- 將時段持久化並綁定登入使用者，驗證跨帳號存取隔離。
- 現場攝影機、遮擋與背光驗證；目前僅完成本機影片聯測及訊號有效性測試。
- 記錄離席區間與可匯出的時段摘要。
- 進一步驗證多使用者與服務重啟；本機跨專案聯測腳本位於學習系統 `scripts/verify-yolo-integration.mjs`。

2026-09-12 公開測試加上 `tests/test_learnsight_telemetry.py` 共 38 項通過；不是模型準確率。LearnSight 不保存原始影像，但 YOLO 偵測端仍產生本機最新標記 JPEG，不能宣稱整個專案從不保存影像。

目前使用單一進程的記憶體字典儲存時段，不適用多 worker 或共享多使用者部署。

