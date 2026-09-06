# -*- coding: utf-8 -*-
"""
多攝影機 Web 界面
支援同時查看多台攝影機的即時串流
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import json
import base64
import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional
from src.streaming.multi_camera_manager import MultiCameraManager, CameraConfig, CameraFrame

logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Camera YOLO Stream Interface")

# 多攝影機管理器
camera_manager = MultiCameraManager(max_cameras=8)

# WebSocket 連接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.camera_subscriptions: Dict[str, List[str]] = {}  # camera_name -> [client_ids]
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket 連接已建立: {client_id}")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
        # 清理訂閱
        for camera_name, subscribers in self.camera_subscriptions.items():
            if client_id in subscribers:
                subscribers.remove(client_id)
                
        logger.info(f"WebSocket 連接已斷開: {client_id}")
        
    async def subscribe_camera(self, client_id: str, camera_name: str):
        """訂閱攝影機串流"""
        if camera_name not in self.camera_subscriptions:
            self.camera_subscriptions[camera_name] = []
        
        if client_id not in self.camera_subscriptions[camera_name]:
            self.camera_subscriptions[camera_name].append(client_id)
            logger.info(f"客戶端 {client_id} 訂閱攝影機 {camera_name}")
            
    async def unsubscribe_camera(self, client_id: str, camera_name: str):
        """取消訂閱攝影機串流"""
        if camera_name in self.camera_subscriptions:
            if client_id in self.camera_subscriptions[camera_name]:
                self.camera_subscriptions[camera_name].remove(client_id)
                logger.info(f"客戶端 {client_id} 取消訂閱攝影機 {camera_name}")
                
    async def send_to_subscribers(self, camera_name: str, message: str):
        """發送消息給訂閱者"""
        if camera_name in self.camera_subscriptions:
            for client_id in self.camera_subscriptions[camera_name]:
                if client_id in self.active_connections:
                    try:
                        await self.active_connections[client_id].send_text(message)
                    except:
                        # 連接已斷開，清理
                        self.disconnect(client_id)

manager = ConnectionManager()

# 幀回調函數
def frame_callback(camera_frame: CameraFrame):
    """處理攝影機幀"""
    try:
        # 轉換為 JPEG
        _, jpeg_data = cv2.imencode('.jpg', camera_frame.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(jpeg_data).decode('utf-8')
        
        # 創建消息
        message = json.dumps({
            "type": "camera_frame",
            "camera_name": camera_frame.camera_name,
            "frame": frame_base64,
            "timestamp": camera_frame.timestamp,
            "frame_number": camera_frame.frame_number
        })
        
        # 發送給訂閱者
        asyncio.create_task(manager.send_to_subscribers(camera_frame.camera_name, message))
        
    except Exception as e:
        logger.error(f"幀回調錯誤: {e}")

# 添加幀回調
camera_manager.add_frame_callback(frame_callback)

@app.get("/", response_class=HTMLResponse)
async def get_multi_camera_interface():
    """多攝影機 Web 界面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>多攝影機 YOLO 串流界面</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #1a1a1a;
                color: #ffffff;
                overflow: hidden;
            }
            
            .header {
                background: #2d2d2d;
                padding: 15px 20px;
                border-bottom: 1px solid #444;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .logo {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
            }
            
            .header-controls {
                display: flex;
                gap: 15px;
                align-items: center;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #4CAF50;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .main-content {
                display: flex;
                height: calc(100vh - 70px);
            }
            
            .sidebar {
                width: 280px;
                background: #2d2d2d;
                border-right: 1px solid #444;
                padding: 20px;
                overflow-y: auto;
            }
            
            .camera-grid {
                flex: 1;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                padding: 20px;
                overflow-y: auto;
            }
            
            .camera-item {
                background: #3d3d3d;
                border-radius: 12px;
                overflow: hidden;
                border: 2px solid transparent;
                transition: all 0.3s;
                cursor: pointer;
            }
            
            .camera-item:hover {
                border-color: #4CAF50;
                transform: translateY(-2px);
            }
            
            .camera-item.active {
                border-color: #4CAF50;
                box-shadow: 0 0 20px rgba(76, 175, 80, 0.3);
            }
            
            .camera-header {
                padding: 15px;
                background: #4d4d4d;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .camera-name {
                font-weight: bold;
                font-size: 16px;
            }
            
            .camera-status {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 12px;
            }
            
            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
            }
            
            .status-connected { background: #4CAF50; }
            .status-connecting { background: #FF9800; }
            .status-error { background: #F44336; }
            .status-disconnected { background: #666; }
            
            .camera-stream {
                position: relative;
                background: #000;
                min-height: 300px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .stream-video {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            
            .stream-placeholder {
                text-align: center;
                color: #888;
                font-size: 14px;
            }
            
            .stream-info {
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.7);
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            
            .camera-controls {
                padding: 15px;
                background: #4d4d4d;
                display: flex;
                gap: 10px;
            }
            
            .control-btn {
                padding: 8px 16px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: background 0.3s;
            }
            
            .control-btn:hover {
                background: #45a049;
            }
            
            .control-btn.secondary {
                background: #666;
            }
            
            .control-btn.secondary:hover {
                background: #777;
            }
            
            .control-btn.danger {
                background: #F44336;
            }
            
            .control-btn.danger:hover {
                background: #d32f2f;
            }
            
            .loading {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 15px;
                color: #888;
            }
            
            .spinner {
                width: 30px;
                height: 30px;
                border: 3px solid #333;
                border-top: 3px solid #4CAF50;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .add-camera-btn {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                margin-bottom: 20px;
                width: 100%;
                transition: background 0.3s;
            }
            
            .add-camera-btn:hover {
                background: #45a049;
            }
            
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
            }
            
            .modal-content {
                background: #2d2d2d;
                margin: 5% auto;
                padding: 20px;
                border-radius: 8px;
                width: 80%;
                max-width: 500px;
            }
            
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            
            .close {
                color: #aaa;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            
            .close:hover {
                color: white;
            }
            
            .form-group {
                margin-bottom: 15px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            
            .form-group input {
                width: 100%;
                padding: 10px;
                border: 1px solid #555;
                border-radius: 4px;
                background: #3d3d3d;
                color: white;
            }
            
            .form-group input:focus {
                outline: none;
                border-color: #4CAF50;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">🎥 多攝影機 YOLO 監控</div>
            <div class="header-controls">
                <div class="status-indicator"></div>
                <span>即時監控</span>
                <button class="control-btn" onclick="startAllCameras()">全部啟動</button>
                <button class="control-btn secondary" onclick="stopAllCameras()">全部停止</button>
            </div>
        </div>
        
        <div class="main-content">
            <div class="sidebar">
                <button class="add-camera-btn" onclick="showAddCameraModal()">+ 添加攝影機</button>
                <div id="cameraList">
                    <!-- 攝影機列表將在這裡動態生成 -->
                </div>
            </div>
            
            <div class="camera-grid" id="cameraGrid">
                <div class="stream-placeholder">
                    <div class="loading">
                        <div class="spinner"></div>
                        <div>請添加攝影機開始監控</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 添加攝影機模態框 -->
        <div id="addCameraModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>添加攝影機</h3>
                    <span class="close" onclick="hideAddCameraModal()">&times;</span>
                </div>
                <form id="addCameraForm">
                    <div class="form-group">
                        <label>攝影機名稱:</label>
                        <input type="text" id="cameraName" required>
                    </div>
                    <div class="form-group">
                        <label>RTSP URL:</label>
                        <input type="text" id="rtspUrl" placeholder="rtsp://camera.example.invalid/stream" required>
                    </div>
                    <div class="form-group">
                        <label>解析度寬度:</label>
                        <input type="number" id="cameraWidth" value="1280">
                    </div>
                    <div class="form-group">
                        <label>解析度高度:</label>
                        <input type="number" id="cameraHeight" value="720">
                    </div>
                    <div class="form-group">
                        <label>FPS:</label>
                        <input type="number" id="cameraFps" value="15">
                    </div>
                    <button type="submit" class="control-btn">添加攝影機</button>
                </form>
            </div>
        </div>
        
        <script>
            let websocket = null;
            let cameras = {};
            let selectedCamera = null;
            
            // 初始化
            document.addEventListener('DOMContentLoaded', function() {
                connectWebSocket();
                loadCameras();
            });
            
            // 連接 WebSocket
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/multi-camera`;
                
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = function() {
                    console.log('WebSocket 連接已建立');
                };
                
                websocket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                };
                
                websocket.onclose = function() {
                    console.log('WebSocket 連接已關閉');
                    setTimeout(connectWebSocket, 3000);
                };
                
                websocket.onerror = function(error) {
                    console.error('WebSocket 錯誤:', error);
                };
            }
            
            // 處理 WebSocket 消息
            function handleWebSocketMessage(data) {
                if (data.type === 'camera_frame') {
                    displayCameraFrame(data);
                } else if (data.type === 'camera_status') {
                    updateCameraStatus(data);
                } else if (data.type === 'camera_list') {
                    updateCameraList(data.cameras);
                }
            }
            
            // 顯示攝影機幀
            function displayCameraFrame(data) {
                const cameraName = data.camera_name;
                const frameElement = document.getElementById(`frame_${cameraName}`);
                
                if (frameElement) {
                    frameElement.innerHTML = `
                        <img class="stream-video" src="data:image/jpeg;base64,${data.frame}" alt="串流畫面">
                        <div class="stream-info">
                            <div>攝影機: ${cameraName}</div>
                            <div>幀號: ${data.frame_number}</div>
                            <div>時間: ${new Date(data.timestamp * 1000).toLocaleTimeString()}</div>
                        </div>
                    `;
                }
            }
            
            // 更新攝影機狀態
            function updateCameraStatus(data) {
                const statusElement = document.getElementById(`status_${data.camera_name}`);
                if (statusElement) {
                    statusElement.className = `status-dot status-${data.status}`;
                }
            }
            
            // 更新攝影機列表
            function updateCameraList(cameraList) {
                cameras = {};
                cameraList.forEach(camera => {
                    cameras[camera.name] = camera;
                });
                renderCameraList();
                renderCameraGrid();
            }
            
            // 渲染攝影機列表
            function renderCameraList() {
                const cameraList = document.getElementById('cameraList');
                cameraList.innerHTML = '';
                
                Object.values(cameras).forEach(camera => {
                    const item = document.createElement('div');
                    item.className = 'camera-item';
                    item.onclick = () => selectCamera(camera.name);
                    item.innerHTML = `
                        <div class="camera-name">${camera.name}</div>
                        <div class="camera-status">
                            <div class="status-dot status-${camera.status}" id="status_${camera.name}"></div>
                            <span>${camera.status}</span>
                        </div>
                    `;
                    cameraList.appendChild(item);
                });
            }
            
            // 渲染攝影機網格
            function renderCameraGrid() {
                const cameraGrid = document.getElementById('cameraGrid');
                cameraGrid.innerHTML = '';
                
                Object.values(cameras).forEach(camera => {
                    const item = document.createElement('div');
                    item.className = 'camera-item';
                    item.id = `camera_${camera.name}`;
                    item.innerHTML = `
                        <div class="camera-header">
                            <div class="camera-name">${camera.name}</div>
                            <div class="camera-status">
                                <div class="status-dot status-${camera.status}" id="status_${camera.name}"></div>
                                <span>${camera.status}</span>
                            </div>
                        </div>
                        <div class="camera-stream" id="frame_${camera.name}">
                            <div class="stream-placeholder">
                                <div class="loading">
                                    <div class="spinner"></div>
                                    <div>等待串流...</div>
                                </div>
                            </div>
                        </div>
                        <div class="camera-controls">
                            <button class="control-btn" onclick="startCamera('${camera.name}')">啟動</button>
                            <button class="control-btn secondary" onclick="stopCamera('${camera.name}')">停止</button>
                            <button class="control-btn danger" onclick="removeCamera('${camera.name}')">移除</button>
                        </div>
                    `;
                    cameraGrid.appendChild(item);
                });
            }
            
            // 選擇攝影機
            function selectCamera(cameraName) {
                selectedCamera = cameraName;
                // 更新 UI 選中狀態
                document.querySelectorAll('.camera-item').forEach(item => {
                    item.classList.remove('active');
                });
                document.getElementById(`camera_${cameraName}`).classList.add('active');
            }
            
            // 啟動攝影機
            function startCamera(cameraName) {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'start_camera',
                        camera_name: cameraName
                    }));
                }
            }
            
            // 停止攝影機
            function stopCamera(cameraName) {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'stop_camera',
                        camera_name: cameraName
                    }));
                }
            }
            
            // 移除攝影機
            function removeCamera(cameraName) {
                if (confirm(`確定要移除攝影機 ${cameraName} 嗎？`)) {
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify({
                            type: 'remove_camera',
                            camera_name: cameraName
                        }));
                    }
                }
            }
            
            // 啟動所有攝影機
            function startAllCameras() {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'start_all_cameras'
                    }));
                }
            }
            
            // 停止所有攝影機
            function stopAllCameras() {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'stop_all_cameras'
                    }));
                }
            }
            
            // 顯示添加攝影機模態框
            function showAddCameraModal() {
                document.getElementById('addCameraModal').style.display = 'block';
            }
            
            // 隱藏添加攝影機模態框
            function hideAddCameraModal() {
                document.getElementById('addCameraModal').style.display = 'none';
            }
            
            // 添加攝影機表單提交
            document.getElementById('addCameraForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const cameraData = {
                    name: document.getElementById('cameraName').value,
                    rtsp_url: document.getElementById('rtspUrl').value,
                    width: parseInt(document.getElementById('cameraWidth').value),
                    height: parseInt(document.getElementById('cameraHeight').value),
                    fps: parseInt(document.getElementById('cameraFps').value)
                };
                
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'add_camera',
                        camera_data: cameraData
                    }));
                }
                
                hideAddCameraModal();
                document.getElementById('addCameraForm').reset();
            });
            
            // 點擊模態框外部關閉
            window.onclick = function(event) {
                const modal = document.getElementById('addCameraModal');
                if (event.target === modal) {
                    hideAddCameraModal();
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws/multi-camera")
async def websocket_multi_camera(websocket: WebSocket):
    """多攝影機 WebSocket 端點"""
    client_id = f"client_{id(websocket)}"
    await manager.connect(websocket, client_id)
    
    try:
        # 發送初始攝影機列表
        cameras = camera_manager.get_all_camera_info()
        await websocket.send_text(json.dumps({
            "type": "camera_list",
            "cameras": cameras
        }))
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "add_camera":
                camera_data = message.get("camera_data")
                config = CameraConfig(
                    name=camera_data["name"],
                    rtsp_url=camera_data["rtsp_url"],
                    width=camera_data["width"],
                    height=camera_data["height"],
                    fps=camera_data["fps"]
                )
                success = camera_manager.add_camera(config)
                
                # 發送更新後的攝影機列表
                cameras = camera_manager.get_all_camera_info()
                await websocket.send_text(json.dumps({
                    "type": "camera_list",
                    "cameras": cameras
                }))
                
            elif message.get("type") == "start_camera":
                camera_name = message.get("camera_name")
                success = camera_manager.start_camera(camera_name)
                
                # 訂閱攝影機串流
                await manager.subscribe_camera(client_id, camera_name)
                
            elif message.get("type") == "stop_camera":
                camera_name = message.get("camera_name")
                success = camera_manager.stop_camera(camera_name)
                
                # 取消訂閱
                await manager.unsubscribe_camera(client_id, camera_name)
                
            elif message.get("type") == "remove_camera":
                camera_name = message.get("camera_name")
                success = camera_manager.remove_camera(camera_name)
                
                # 發送更新後的攝影機列表
                cameras = camera_manager.get_all_camera_info()
                await websocket.send_text(json.dumps({
                    "type": "camera_list",
                    "cameras": cameras
                }))
                
            elif message.get("type") == "start_all_cameras":
                results = camera_manager.start_all_cameras()
                for camera_name in results.keys():
                    if results[camera_name]:
                        await manager.subscribe_camera(client_id, camera_name)
                        
            elif message.get("type") == "stop_all_cameras":
                results = camera_manager.stop_all_cameras()
                for camera_name in results.keys():
                    await manager.unsubscribe_camera(client_id, camera_name)
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket 錯誤: {e}")
        manager.disconnect(client_id)

@app.get("/api/cameras")
async def get_cameras():
    """獲取攝影機列表 API"""
    return {
        "cameras": camera_manager.get_all_camera_info(),
        "max_cameras": camera_manager.max_cameras
    }

@app.post("/api/cameras")
async def add_camera(camera_data: dict):
    """添加攝影機 API"""
    config = CameraConfig(
        name=camera_data["name"],
        rtsp_url=camera_data["rtsp_url"],
        width=camera_data.get("width", 1280),
        height=camera_data.get("height", 720),
        fps=camera_data.get("fps", 15)
    )
    
    success = camera_manager.add_camera(config)
    return {"success": success, "camera": camera_data["name"]}

@app.post("/api/cameras/{camera_name}/start")
async def start_camera(camera_name: str):
    """啟動攝影機 API"""
    success = camera_manager.start_camera(camera_name)
    return {"success": success, "camera": camera_name}

@app.post("/api/cameras/{camera_name}/stop")
async def stop_camera(camera_name: str):
    """停止攝影機 API"""
    success = camera_manager.stop_camera(camera_name)
    return {"success": success, "camera": camera_name}

@app.delete("/api/cameras/{camera_name}")
async def remove_camera(camera_name: str):
    """移除攝影機 API"""
    success = camera_manager.remove_camera(camera_name)
    return {"success": success, "camera": camera_name}

@app.get("/api/cameras/{camera_name}/mjpeg")
async def mjpeg_stream(camera_name: str):
    """MJPEG 串流端點"""
    def generate_mjpeg():
        while camera_name in camera_manager.camera_captures:
            frame = camera_manager.get_latest_frame(camera_name)
            if frame is not None:
                _, jpeg_data = cv2.imencode('.jpg', frame.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       jpeg_data.tobytes() + b'\r\n')
            
            time.sleep(1/15)  # 15 FPS
    
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    
    # 添加預設攝影機
    default_cameras = [
        CameraConfig(
            name="camera_1",
            rtsp_url="rtsp://camera.example.invalid/stream",
            width=1280,
            height=720,
            fps=15
        ),
        CameraConfig(
            name="camera_2",
            rtsp_url="rtsp://camera.example.invalid/stream",
            width=1280,
            height=720,
            fps=15
        )
    ]
    
    for camera in default_cameras:
        camera_manager.add_camera(camera)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
