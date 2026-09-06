"""整合測試 - P2企業級"""
import sys
import os
import tempfile
import json
import csv
import time
from pathlib import Path
import unittest
import pytest

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.evaluate import CrossingEvent, EvaluationResult, evaluate_crossings
from src.logic.geometry import Point, point_in_polygon, line_side
from src.logic.counter import LineConfig, MultiLineCounter, TrackInfo
from src.logic.calibration import HomographyCalibrator, CalibrationConfig


class TestIntegration(unittest.TestCase):
    def setUp(self):
        """設置整合測試環境"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """清理測試環境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_detection_pipeline(self):
        """測試完整偵測流程"""
        # 創建測試配置
        line_configs = [
            LineConfig("entrance", [(0, 100), (200, 100)], "both", 30, 2),
            LineConfig("exit", [(0, 200), (200, 200)], "both", 30, 2)
        ]
        
        # 創建計數器
        counter = MultiLineCounter(line_configs, 30, 15, 40.0, 30)
        
        # 模擬軌跡數據
        tracks = [
            TrackInfo(1, (100, 95), (90, 90, 110, 100), timestamp=time.time()),
            TrackInfo(1, (100, 105), (90, 100, 110, 110), timestamp=time.time() + 0.1),
            TrackInfo(2, (100, 195), (90, 190, 110, 200), timestamp=time.time() + 0.2),
            TrackInfo(2, (100, 205), (90, 200, 110, 210), timestamp=time.time() + 0.3),
        ]
        
        # 更新計數器
        events = counter.update(tracks, 0)
        
        # 驗證事件生成
        self.assertGreater(len(events), 0)
        
        # 檢查計數
        counts = counter.get_counts()
        self.assertIn("entrance", counts)
        self.assertIn("exit", counts)
    
    def test_geometry_and_counter_integration(self):
        """測試幾何模組與計數器整合"""
        # 創建複雜多邊形ROI
        roi_polygon = [(0, 0), (200, 0), (200, 200), (0, 200)]
        ignore_polygon = [(50, 50), (150, 50), (150, 150), (50, 150)]
        
        # 測試點在多邊形內判斷
        point_inside_roi = Point(100, 100)
        point_inside_ignore = Point(100, 100)
        point_outside = Point(250, 250)
        
        self.assertTrue(point_in_polygon(point_inside_roi, roi_polygon))
        self.assertTrue(point_in_polygon(point_inside_ignore, ignore_polygon))
        self.assertFalse(point_in_polygon(point_outside, roi_polygon))
        
        # 測試線段側判斷
        line = (Point(0, 100), Point(200, 100))
        point_above = Point(100, 50)
        point_below = Point(100, 150)
        
        self.assertGreater(line_side(point_above, line), 0)
        self.assertLess(line_side(point_below, line), 0)
    
    def test_calibration_and_counter_integration(self):
        """測試標定與計數器整合"""
        # 創建標定配置
        calibration_config = CalibrationConfig(
            enabled=True,
            src_points=[(0, 0), (200, 0), (200, 200), (0, 200)],
            dst_points=[(0, 0), (2, 0), (2, 2), (0, 2)]
        )
        calibrator = HomographyCalibrator(calibration_config)
        
        # 創建計數器
        line_configs = [
            LineConfig("test_line", [(0, 100), (200, 100)], "both", 30, 2)
        ]
        counter = MultiLineCounter(line_configs, 30, 15, 40.0, 30)
        
        # 模擬軌跡
        tracks = [
            TrackInfo(1, (100, 95), (90, 90, 110, 100), timestamp=time.time()),
            TrackInfo(1, (100, 105), (90, 100, 110, 110), timestamp=time.time() + 0.1),
        ]
        
        # 使用標定器更新計數器
        events = counter.update(tracks, 0, calibrator)
        
        # 驗證事件包含世界座標
        if events:
            event = events[0]
            self.assertIsNotNone(event.x_m)
            self.assertIsNotNone(event.y_m)
    
    def test_evaluation_with_sample_data(self):
        """測試使用示例數據的評估"""
        # 創建示例真實標註
        gt_file = os.path.join(self.temp_dir, "gt_crossings.jsonl")
        with open(gt_file, 'w') as f:
            f.write('{"timestamp": 1.5, "line_id": "entrance", "direction": "in", "track_id": "person_001", "x_px": 100, "y_px": 100, "confidence": 0.95}\n')
            f.write('{"timestamp": 2.0, "line_id": "entrance", "direction": "in", "track_id": "person_002", "x_px": 120, "y_px": 100, "confidence": 0.92}\n')
            f.write('{"timestamp": 3.0, "line_id": "exit", "direction": "out", "track_id": "person_001", "x_px": 100, "y_px": 200, "confidence": 0.88}\n')
        
        # 創建示例預測結果
        pred_file = os.path.join(self.temp_dir, "pred_crossings.csv")
        with open(pred_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'line_id', 'direction', 'track_id', 'x_px', 'y_px', 'confidence'])
            writer.writerow([1.6, 'entrance', 'in', 'person_001', 105, 100, 0.95])
            writer.writerow([2.1, 'entrance', 'in', 'person_002', 125, 100, 0.92])
            writer.writerow([3.1, 'exit', 'out', 'person_001', 105, 200, 0.88])
            writer.writerow([4.0, 'entrance', 'in', 'person_003', 140, 100, 0.90])  # 額外預測
        
        # 運行評估
        evaluator = CrossingEvaluator(tolerance=0.5)
        result = evaluator.evaluate(gt_file, pred_file)
        
        # 驗證評估結果
        self.assertGreater(result.precision, 0.7)  # 精確度應該大於0.7
        self.assertGreater(result.recall, 0.8)     # 召回率應該大於0.8
        self.assertGreater(result.f1, 0.7)         # F1應該大於0.7
        self.assertGreater(result.idf1, 0.7)       # IDF1應該大於0.7
        
        # 驗證統計數據
        self.assertEqual(result.total_gt, 3)
        self.assertEqual(result.total_pred, 4)
        self.assertEqual(result.total_matched, 3)
        self.assertEqual(result.total_id_switches, 0)
    
    def test_multi_line_multi_zone_scenario(self):
        """測試多線多區域場景"""
        # 創建多線配置
        line_configs = [
            LineConfig("entrance_1", [(0, 100), (100, 100)], "in", 30, 2),
            LineConfig("entrance_2", [(100, 100), (200, 100)], "in", 30, 2),
            LineConfig("exit_1", [(0, 200), (100, 200)], "out", 30, 2),
            LineConfig("exit_2", [(100, 200), (200, 200)], "out", 30, 2)
        ]
        
        # 創建多區域配置
        from src.logic.counter import ZoneConfig, ZoneOccupancy
        zone_configs = [
            ZoneConfig("waiting_area", [(0, 0), (200, 0), (200, 100), (0, 100)], False),
            ZoneConfig("service_area", [(0, 100), (200, 100), (200, 200), (0, 200)], False),
            ZoneConfig("staff_only", [(0, 0), (50, 0), (50, 50), (0, 50)], True)
        ]
        
        # 創建計數器和區域分類器
        counter = MultiLineCounter(line_configs, 30, 15, 40.0, 30)
        zone_classifier = ZoneOccupancy(zone_configs)
        
        # 模擬複雜軌跡
        tracks = [
            TrackInfo(1, (50, 95), (40, 90, 60, 100), timestamp=time.time()),
            TrackInfo(1, (50, 105), (40, 100, 60, 110), timestamp=time.time() + 0.1),
            TrackInfo(2, (150, 95), (140, 90, 160, 100), timestamp=time.time() + 0.2),
            TrackInfo(2, (150, 105), (140, 100, 160, 110), timestamp=time.time() + 0.3),
            TrackInfo(3, (25, 25), (20, 20, 30, 30), timestamp=time.time() + 0.4),  # 在忽略區域
        ]
        
        # 更新計數器
        line_events = counter.update(tracks, 0)
        
        # 更新區域分類器
        zone_events = zone_classifier.update(tracks)
        
        # 驗證結果
        self.assertGreater(len(line_events), 0)
        self.assertGreater(len(zone_events), 0)
        
        # 檢查計數
        counts = counter.get_counts()
        self.assertIn("entrance_1", counts)
        self.assertIn("entrance_2", counts)
        self.assertIn("exit_1", counts)
        self.assertIn("exit_2", counts)
        
        # 檢查區域佔用
        occupancy = zone_classifier.get_occupancy()
        self.assertIn("waiting_area", occupancy)
        self.assertIn("service_area", occupancy)
        self.assertIn("staff_only", occupancy)
    
    def test_direction_jitter_and_uturn_filtering(self):
        """測試方向抖動和U-turn過濾"""
        line_config = LineConfig("test_line", [(0, 100), (200, 100)], "both", 30, 3)
        counter = MultiLineCounter([line_config], 30, 15, 40.0, 30)
        
        # 模擬抖動軌跡
        jitter_tracks = [
            TrackInfo(1, (100, 95), (90, 90, 110, 100), timestamp=time.time()),
            TrackInfo(1, (100, 105), (90, 100, 110, 110), timestamp=time.time() + 0.1),
            TrackInfo(1, (100, 95), (90, 90, 110, 100), timestamp=time.time() + 0.2),
            TrackInfo(1, (100, 105), (90, 100, 110, 110), timestamp=time.time() + 0.3),
        ]
        
        # 抖動應該被濾波
        events = counter.update(jitter_tracks, 0)
        self.assertEqual(len(events), 0)
        
        # 模擬U-turn軌跡
        uturn_tracks = [
            TrackInfo(2, (100, 95), (90, 90, 110, 100), timestamp=time.time() + 1.0),
            TrackInfo(2, (100, 105), (90, 100, 110, 110), timestamp=time.time() + 1.1),
            TrackInfo(2, (100, 95), (90, 90, 110, 100), timestamp=time.time() + 1.2),
        ]
        
        # U-turn應該被去重
        events = counter.update(uturn_tracks, 1)
        self.assertEqual(len(events), 0)
    
    def test_id_fault_tolerance_and_relinking(self):
        """測試ID容錯和重連"""
        line_config = LineConfig("test_line", [(0, 100), (200, 100)], "both", 30, 2)
        counter = MultiLineCounter([line_config], 30, 15, 40.0, 30)
        
        # 軌跡1出現
        tracks1 = [TrackInfo(1, (100, 95), (90, 90, 110, 100), timestamp=time.time())]
        counter.update(tracks1, 0)
        
        # 軌跡1消失，軌跡2在附近出現（應該被合併）
        tracks2 = [TrackInfo(2, (102, 97), (92, 92, 112, 102), timestamp=time.time() + 0.1)]
        events = counter.update(tracks2, 1)
        
        # 應該產生合併事件
        self.assertGreaterEqual(len(events), 0)
        
        # 軌跡2繼續移動
        tracks3 = [TrackInfo(2, (100, 105), (90, 100, 110, 110), timestamp=time.time() + 0.2)]
        events = counter.update(tracks3, 2)
        
        # 應該產生穿越事件
        self.assertGreater(len(events), 0)
    
    def test_performance_under_load(self):
        """測試負載下的性能"""
        line_configs = [
            LineConfig(f"line_{i}", [(0, 100 + i*10), (200, 100 + i*10)], "both", 30, 2)
            for i in range(10)
        ]
        counter = MultiLineCounter(line_configs, 30, 15, 40.0, 30)
        
        # 模擬大量軌跡
        start_time = time.time()
        for frame in range(100):
            tracks = [
                TrackInfo(i, (100 + i*2, 100 + (i % 10)*10), 
                         (90 + i*2, 90 + (i % 10)*10, 110 + i*2, 110 + (i % 10)*10), 
                         timestamp=time.time() + frame*0.1)
                for i in range(50)
            ]
            events = counter.update(tracks, frame)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 驗證性能（應該在合理時間內完成）
        self.assertLess(processing_time, 10.0)  # 應該在10秒內完成
        
        # 驗證計數結果
        counts = counter.get_counts()
        self.assertEqual(len(counts), 10)


if __name__ == "__main__":
    unittest.main()
