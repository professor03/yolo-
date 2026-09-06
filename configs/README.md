# 設定檔導覽

公開 demo 使用 `local_video.yaml`，由 `scripts/detect_demo.py` 載入。
預設 CPU、yolov8n、640 輸入尺寸、人物類別 0 與 ByteTrack。

| 檔案 | 用途 |
|---|---|
| local_video.yaml | 已測試的本機影片入口 |
| sample.yaml、mjpg_camera.yaml | 歷史單攝影機參數範例 |
| multi_cam.yaml | 歷史多攝影機參數範例，不保證所有旗標已實作 |
| camera_streams.yaml | 串流清單格式範例 |
| roles.yaml | 權限定義範例，無預設帳號 |

舊設定的保留網域與空白密碼不是可使用的連線資訊。
請勿把真實帳密寫進 Git。主 API 的環境變數範例位於根目錄
`.env.example`，它與歷史 YAML 格式不是自動對等映射。

conf 是偵測門檻；降低門檻可能增加召回也可能增加誤報。
FPS 與準確率需依模型、硬體和場景實測。

操作流程見 [usage.md](../docs/usage.md)，功能界線見
[limitations.md](../docs/limitations.md)。
