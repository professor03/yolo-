# -*- coding: utf-8 -*-
"""
Frigate 啟發的 Web 界面
支援 WebRTC 和 MJPEG 串流顯示
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import base64
import cv2
import numpy as np
import time
from typing import Dict, List
import logging
from src.streaming.frigate_inspired import FrigateInspiredStreamManager

logger = logging.getLogger(__name__)

app = FastAPI(title="Frigate-Inspired YOLO Stream Interface")

# 串流管理器
stream_manager = FrigateInspiredStreamManager()

# WebSocket 連接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket 連接已建立: {client_id}")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket 連接已斷開: {client_id}")
            
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
            
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_frigate_interface():
    """Frigate 風格的 Web 界面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Frigate-Inspired YOLO 串流界面</title>
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
                padding: 10px 20px;
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
            
            .status {
                display: flex;
                align-items: center;
                gap: 20px;
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
                height: calc(100vh - 60px);
            }
            
            .sidebar {
                width: 250px;
                background: #2d2d2d;
                border-right: 1px solid #444;
                padding: 20px;
                overflow-y: auto;
            }
            
            .camera-list {
                list-style: none;
            }
            
            .camera-item {
                padding: 15px;
                margin-bottom: 10px;
                background: #3d3d3d;
                border-radius: 8px;
                cursor: pointer;
                transition: background 0.3s;
                border: 2px solid transparent;
            }
            
            .camera-item:hover {
                background: #4d4d4d;
            }
            
            .camera-item.active {
                border-color: #4CAF50;
                background: #4d4d4d;
            }
            
            .camera-name {
                font-weight: bold;
                margin-bottom: 5px;
            }
            
            .camera-status {
                font-size: 12px;
                color: #888;
            }
            
            .stream-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: #1a1a1a;
            }
            
            .stream-header {
                padding: 15px 20px;
                background: #2d2d2d;
                border-bottom: 1px solid #444;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .stream-title {
                font-size: 18px;
                font-weight: bold;
            }
            
            .stream-controls {
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
                font-size: 14px;
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
            
            .stream-display {
                flex: 1;
                position: relative;
                background: #000;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .video-container {
                position: relative;
                max-width: 100%;
                max-height: 100%;
            }
            
            .video-stream {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            
            .stream-info {
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.7);
                padding: 10px;
                border-radius: 4px;
                font-size: 12px;
            }
            
            .no-stream {
                text-align: center;
                color: #888;
                font-size: 18px;
            }
            
            .loading {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 20px;
            }
            
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid #333;
                border-top: 4px solid #4CAF50;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">🎥 Frigate-Inspired YOLO</div>
            <div class="status">
                <div class="status-indicator"></div>
                <span>即時串流</span>
            </div>
        </div>
        
        <div class="main-content">
            <div class="sidebar">
                <h3>攝影機列表</h3>
                <ul class="camera-list" id="cameraList">
                    <li class="camera-item" data-camera="ip_camera">
                        <div class="camera-name">IP 攝影機</div>
                        <div class="camera-status">RTSP 串流</div>
                    </li>
                </ul>
            </div>
            
            <div class="stream-container">
                <div class="stream-header">
                    <div class="stream-title" id="streamTitle">選擇攝影機</div>
                    <div class="stream-controls">
                        <button class="control-btn" onclick="startStream()">開始串流</button>
                        <button class="control-btn secondary" onclick="stopStream()">停止串流</button>
                        <button class="control-btn secondary" onclick="toggleFullscreen()">全螢幕</button>
                    </div>
                </div>
                
                <div class="stream-display" id="streamDisplay">
                    <div class="no-stream">
                        <div class="loading">
                            <div class="spinner"></div>
                            <div>請選擇攝影機開始串流</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let currentCamera = null;
            let websocket = null;
            let streamInterval = null;
            
            // 初始化
            document.addEventListener('DOMContentLoaded', function() {
                setupCameraList();
                connectWebSocket();
            });
            
            // 設置攝影機列表
            function setupCameraList() {
                const cameraItems = document.querySelectorAll('.camera-item');
                cameraItems.forEach(item => {
                    item.addEventListener('click', function() {
                        // 移除其他項目的 active 類
                        cameraItems.forEach(i => i.classList.remove('active'));
                        // 添加當前項目的 active 類
                        this.classList.add('active');
                        
                        const cameraName = this.dataset.camera;
                        selectCamera(cameraName);
                    });
                });
            }
            
            // 選擇攝影機
            function selectCamera(cameraName) {
                currentCamera = cameraName;
                document.getElementById('streamTitle').textContent = `攝影機: ${cameraName}`;
                
                // 如果 WebSocket 已連接，請求串流
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    requestStream(cameraName);
                }
            }
            
            // 連接 WebSocket
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
                
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = function() {
                    console.log('WebSocket 連接已建立');
                    if (currentCamera) {
                        requestStream(currentCamera);
                    }
                };
                
                websocket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleStreamData(data);
                };
                
                websocket.onclose = function() {
                    console.log('WebSocket 連接已關閉');
                    // 嘗試重連
                    setTimeout(connectWebSocket, 3000);
                };
                
                websocket.onerror = function(error) {
                    console.error('WebSocket 錯誤:', error);
                };
            }
            
            // 請求串流
            function requestStream(cameraName) {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'request_stream',
                        camera: cameraName
                    }));
                }
            }
            
            // 處理串流數據
            function handleStreamData(data) {
                if (data.type === 'stream_frame' && data.camera === currentCamera) {
                    displayFrame(data.frame);
                }
            }
            
            // 顯示幀
            function displayFrame(frameData) {
                const streamDisplay = document.getElementById('streamDisplay');
                streamDisplay.innerHTML = `
                    <div class="video-container">
                        <img class="video-stream" src="data:image/jpeg;base64,${frameData}" alt="串流畫面">
                        <div class="stream-info">
                            <div>攝影機: ${currentCamera}</div>
                            <div>解析度: 1280x720</div>
                            <div>FPS: 15</div>
                        </div>
                    </div>
                `;
            }
            
            // 開始串流
            function startStream() {
                if (currentCamera) {
                    requestStream(currentCamera);
                }
            }
            
            // 停止串流
            function stopStream() {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'stop_stream',
                        camera: currentCamera
                    }));
                }
                
                document.getElementById('streamDisplay').innerHTML = `
                    <div class="no-stream">
                        <div class="loading">
                            <div class="spinner"></div>
                            <div>串流已停止</div>
                        </div>
                    </div>
                `;
            }
            
            // 全螢幕
            function toggleFullscreen() {
                const streamDisplay = document.getElementById('streamDisplay');
                if (!document.fullscreenElement) {
                    streamDisplay.requestFullscreen();
                } else {
                    document.exitFullscreen();
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端點處理串流"""
    client_id = f"client_{id(websocket)}"
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "request_stream":
                camera_name = message.get("camera")
                if camera_name:
                    # 啟動串流
                    stream_manager.start_camera(camera_name)
                    
                    # 開始發送幀
                    await send_stream_frames(websocket, camera_name)
                    
            elif message.get("type") == "stop_stream":
                camera_name = message.get("camera")
                if camera_name:
                    stream_manager.stop_camera(camera_name)
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket 錯誤: {e}")
        manager.disconnect(client_id)

async def send_stream_frames(websocket: WebSocket, camera_name: str):
    """發送串流幀"""
    try:
        while camera_name in stream_manager.converter.active_streams:
            frame = stream_manager.get_frame(camera_name)
            if frame is not None:
                # 轉換為 JPEG
                _, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(jpeg_data).decode('utf-8')
                
                # 發送幀數據
                await websocket.send_text(json.dumps({
                    "type": "stream_frame",
                    "camera": camera_name,
                    "frame": frame_base64,
                    "timestamp": time.time()
                }))
            
            await asyncio.sleep(1/15)  # 15 FPS
            
    except Exception as e:
        logger.error(f"發送串流幀錯誤: {e}")

@app.get("/stream/{camera_name}/mjpeg")
async def mjpeg_stream(camera_name: str):
    """MJPEG 串流端點"""
    def generate_mjpeg():
        while camera_name in stream_manager.converter.active_streams:
            frame = stream_manager.get_frame(camera_name)
            if frame is not None:
                _, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       jpeg_data.tobytes() + b'\r\n')
            
            time.sleep(1/15)  # 15 FPS
    
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/cameras")
async def get_cameras():
    """獲取攝影機列表"""
    return {
        "cameras": list(stream_manager.converter.streams.keys()),
        "active": list(stream_manager.converter.active_streams.keys())
    }

@app.post("/api/cameras/{camera_name}/start")
async def start_camera(camera_name: str):
    """啟動攝影機"""
    success = stream_manager.start_camera(camera_name)
    return {"success": success, "camera": camera_name}

@app.post("/api/cameras/{camera_name}/stop")
async def stop_camera(camera_name: str):
    """停止攝影機"""
    success = stream_manager.stop_camera(camera_name)
    return {"success": success, "camera": camera_name}

if __name__ == "__main__":
    import uvicorn
    
    # 添加預設攝影機
    stream_manager.add_camera(
        name="ip_camera",
        rtsp_url="rtsp://camera.example.invalid/stream",
        width=1280,
        height=720,
        fps=15
    )
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
