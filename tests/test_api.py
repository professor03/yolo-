"""API服務器單元測試 - P2企業級"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pytest
import asyncio
from fastapi.testclient import TestClient
from src.server.app import create_app
from src.api.data_store import data_store


class TestAPI(unittest.TestCase):
    def setUp(self):
        """設置測試環境"""
        self.app = create_app()
        self.client = TestClient(self.app)
        # 清空數據存儲
        data_store.sources.clear()
        data_store.metrics_history.clear()
        data_store.line_counts.clear()
        data_store.alerts.clear()
    
    def test_health_check(self):
        """測試健康檢查端點"""
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
    
    def test_sources_endpoint(self):
        """測試來源狀態端點"""
        # 添加測試來源
        data_store.update_source("test_camera", "rtsp://test", "online")
        
        response = self.client.get("/api/v1/sources")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("sources", data)
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["source_id"], "test_camera")
    
    def test_metrics_endpoint(self):
        """測試指標端點"""
        # 添加測試指標
        data_store.update_metrics({
            "timestamp": "2024-01-01T00:00:00",
            "fps_processing": 30.0,
            "fps_source": 25.0,
            "people_count": 5,
            "latency_ms": 33.3,
            "cpu_percent": 50.0,
            "memory_mb": 1024.0
        })
        
        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("current", data)
        self.assertIn("history", data)
        self.assertEqual(data["current"]["people_count"], 5)
    
    def test_counts_endpoint(self):
        """測試計數端點"""
        # 添加測試計數
        data_store.update_line_counts({"line1": {"in": 10, "out": 8}})
        
        response = self.client.get("/api/v1/counts")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("line_counts", data)
        self.assertIn("zone_occupancy", data)
    
    def test_alerts_endpoint(self):
        """測試警報端點"""
        # 添加測試警報
        data_store.add_alert({
            "level": "warning",
            "title": "Test Alert",
            "message": "This is a test alert",
            "timestamp": "2024-01-01T00:00:00"
        })
        
        response = self.client.get("/api/v1/alerts")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("alerts", data)
        self.assertEqual(len(data["alerts"]), 1)
        self.assertEqual(data["alerts"][0]["level"], "warning")
    
    def test_dashboard_endpoint(self):
        """測試儀表板端點"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("YOLO People Detection", response.text)
    
    def test_websocket_connection(self):
        """測試WebSocket連接"""
        with self.client.websocket_connect("/ws/stream") as websocket:
            # 發送測試數據
            data_store.update_metrics({
                "timestamp": "2024-01-01T00:00:00",
                "fps_processing": 30.0,
                "fps_source": 25.0,
                "people_count": 3,
                "latency_ms": 33.3,
                "cpu_percent": 50.0,
                "memory_mb": 1024.0
            })
            
            # 接收WebSocket消息
            data = websocket.receive_json()
            self.assertIn("type", data)
            self.assertIn("data", data)
            self.assertEqual(data["type"], "dashboard_update")
    
    def test_api_error_handling(self):
        """測試API錯誤處理"""
        # 測試不存在的端點
        response = self.client.get("/api/v1/nonexistent")
        self.assertEqual(response.status_code, 404)
    
    def test_cors_headers(self):
        """測試CORS標頭"""
        response = self.client.options("/api/v1/sources")
        self.assertIn("access-control-allow-origin", response.headers)
    
    def test_metrics_data_validation(self):
        """測試指標數據驗證"""
        # 測試無效數據
        data_store.update_metrics(None)
        
        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        # 應該返回默認值或空數據
    
    def test_line_counts_data_validation(self):
        """測試計數數據驗證"""
        # 測試無效計數數據
        data_store.update_line_counts(None)
        
        response = self.client.get("/api/v1/counts")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data["line_counts"], list)
    
    def test_alerts_data_validation(self):
        """測試警報數據驗證"""
        # 測試無效警報數據
        data_store.add_alert(None)
        
        response = self.client.get("/api/v1/alerts")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data["alerts"], list)
    
    def test_concurrent_requests(self):
        """測試並發請求"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = self.client.get("/api/v1/sources")
            results.append(response.status_code)
        
        # 創建多個並發請求
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有線程完成
        for thread in threads:
            thread.join()
        
        # 所有請求都應該成功
        self.assertEqual(len(results), 10)
        self.assertTrue(all(status == 200 for status in results))
    
    def test_data_persistence(self):
        """測試數據持久性"""
        # 添加數據
        data_store.update_source("camera1", "rtsp://test1", "online")
        data_store.update_metrics({"people_count": 5})
        data_store.update_line_counts({"line1": {"in": 10, "out": 8}})
        
        # 驗證數據存在
        self.assertEqual(len(data_store.sources), 1)
        self.assertEqual(len(data_store.metrics_history), 1)
        self.assertEqual(len(data_store.line_counts), 1)
    
    def test_websocket_multiple_clients(self):
        """測試多個WebSocket客戶端"""
        with self.client.websocket_connect("/ws/stream") as ws1:
            with self.client.websocket_connect("/ws/stream") as ws2:
                # 更新數據
                data_store.update_metrics({"people_count": 10})
                
                # 兩個客戶端都應該收到消息
                data1 = ws1.receive_json()
                data2 = ws2.receive_json()
                
                self.assertEqual(data1["type"], "dashboard_update")
                self.assertEqual(data2["type"], "dashboard_update")


if __name__ == "__main__":
    unittest.main()
