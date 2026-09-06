# LearnSight MVP

LearnSight 把讀書目標與人物出現訊號連接起來。這個 repository
提供視覺端的讀書時段原型，尚未等同完整 AI-Learning-Assistant 系統整合。

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
自首次無人觀察開始持續 90 秒後，讀取時段可得到
`away_reminder_eligible=true`。它是提醒資格，不代表已送出推播；
也不能直接推論個人的專注度、理解或學習成效。

前端手動「有人／無人」是模擬輸入。
實際推論的展示需啟動本機影片流程並按同步。
目前沒有自动背景同步、人物身分識別與坐姿/站姿分類。

## 後續工作

- 將時段持久化並綁定登入使用者，驗證跨帳號存取隔離。
- 同步時檢查訊號時間與來源，區分「無人」與「攝影機斷線/過期資料」。
- 記錄離席區間與可匯出的時段摘要。
- 與學習系統建立明確 API 契約，加入端到端整合測試。

目前使用單一進程的記憶體字典儲存時段，不適用多 worker 或共享多使用者部署。
