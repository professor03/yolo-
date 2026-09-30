# 瀏覽器／雲端 Demo 驗證（2026-09-30）

新增安全入口 `src.server.browser_demo:app`，只開放訪客 LearnSight 與 JPEG
影格推論，不開放原本管理介面。AI Learning System 的 8787 閘道轉送
指定 API 到 loopback 8000，並提供明確影格傳送同意。

## 本機程式驗證

公開測試與新增瀏覽器服務測試合計 **40 項通過**。
新測試以注入的假推論器驗證 API 契約、訪客隔離、大小／尺寸限制、
頻率限制、憑證過期、停止及已結束時段拒收影格。
**這些單元測試不是辨識準確率或真實鏡頭驗證。**

另外執行真正的 YOLOv8n／CPU，將已公開比較圖的**原始左半部**縮小後
送入新 API，再送空白 JPEG，最後送相同原圖，結果為 **6 → 0 → 6**。
三次 API 更新均進入同一時段，最後成功結束。沒有預製 JSON 代替模型。
圖片路徑 `docs/assets/detection/before-after.jpg`，未使用右半部帶框成果圖。
這是既有範例圖片的 API 整合 smoke test，不是作者本人的新鏡頭錄影。

```bash
python scripts/verify_browser_demo.py --image docs/assets/detection/before-after.jpg --left-half
```

該次 CPU 模型耗時約 159.5／61.8／57.8 ms。這是單次本機數值，
不能作為 Codespaces 效能、完整端到端延遲或跨硬體 benchmark。
真正鏡頭證據仍以 2026-09-14 作者本人 `1 → 0 → 1` 影片為準。

## 雲端驗證尚待部署後補記

Codespaces 的 Linux 全新安裝、外部 HTTPS 網址、未登入存取、訪客自己的
相機權限與雙訪客隔離，必須以實際部署測試確認，不能拿以上本機測試取代。
具體操作見 AI Learning System 的 `docs/codespaces-demo.md`。
