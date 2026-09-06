"""整合測試 - P3企業交付級"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pytest
import json
import csv
import time
from pathlib import Path
from src.eval.evaluate import evaluate_crossings, EvaluationResult
from src.logic.counter import MultiLineCounter, LineConfig
from src.logic.calibration import HomographyCalibrator, CalibrationConfig
from src.logic.geometry import point_in_polygon


class TestIntegrationP3(unittest.TestCase):
    """P3企業級整合測試"""
    
    def setUp(self):
        """測試前準備"""
        self.test_data_dir = Path("datasets/sample_eval")
        self.results_dir = Path("reports")
        self.results_dir.mkdir(exist_ok=True)
    
    def test_full_pipeline_evaluation(self):
        """測試完整流程評估 - P3企業級"""
        # 檢查測試數據是否存在
        crossings_file = self.test_data_dir / "sample_video1_crossings.jsonl"
        predictions_file = self.test_data_dir / "sample_video1_predictions.csv"
        
        if not crossings_file.exists() or not predictions_file.exists():
            self.skipTest("測試數據檔案不存在")
        
        # 讀取真實標註
        gt_crossings = []
        with open(crossings_file, 'r') as f:
            for line in f:
                gt_crossings.append(json.loads(line.strip()))
        
        # 讀取預測結果
        predictions = []
        with open(predictions_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions.append(row)
        
        # 創建臨時檔案進行評估
        gt_file = self.test_data_dir / "temp_gt.jsonl"
        pred_file = self.test_data_dir / "temp_pred.csv"
        
        # 寫入GT檔案
        with open(gt_file, 'w') as f:
            for crossing in gt_crossings:
                f.write(json.dumps(crossing) + '\n')
        
        # 寫入預測檔案
        with open(pred_file, 'w', newline='') as f:
            if predictions:
                writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
                writer.writeheader()
                writer.writerows(predictions)
        
        # 執行評估
        results = evaluate_crossings(str(gt_file), str(pred_file))
        
        # 清理臨時檔案
        gt_file.unlink(missing_ok=True)
        pred_file.unlink(missing_ok=True)
        
        # 驗證結果格式
        self.assertIsInstance(results, EvaluationResult)
        self.assertIsInstance(results.precision, float)
        self.assertIsInstance(results.recall, float)
        self.assertIsInstance(results.f1, float)
        
        # 驗證數值範圍
        self.assertGreaterEqual(results.precision, 0.0)
        self.assertLessEqual(results.precision, 1.0)
        self.assertGreaterEqual(results.recall, 0.0)
        self.assertLessEqual(results.recall, 1.0)
        self.assertGreaterEqual(results.f1, 0.0)
        self.assertLessEqual(results.f1, 1.0)
        
        # P3企業級門檻驗證
        self.assertGreaterEqual(results.precision, 0.8, "Precision 應 ≥ 80%")
        self.assertGreaterEqual(results.recall, 0.8, "Recall 應 ≥ 80%")
        self.assertGreaterEqual(results.f1, 0.8, "F1 應 ≥ 80%")
        
        print(f"✅ 評估結果: P={results.precision:.3f}, R={results.recall:.3f}, F1={results.f1:.3f}")
    
    def test_multi_line_counter_integration(self):
        """測試多線計數器整合 - P3企業級"""
        # 創建多個線段配置
        line_configs = [
            LineConfig(id="entrance", points=[(0, 0), (100, 0)], direction="in"),
            LineConfig(id="exit", points=[(0, 50), (100, 50)], direction="out"),
            LineConfig(id="sidewalk", points=[(0, 25), (100, 25)], direction="both")
        ]
        
        counter = MultiLineCounter(line_configs, 1, 15, 40.0, 30)  # 冷卻期設為1幀
        
        # 模擬複雜軌跡場景
        from src.logic.counter import TrackInfo
        
        # 場景1：正常通過 - 需要滿足抖動過濾條件
        normal_track = [
            TrackInfo(1, (50, -5), (45, -10, 55, 0)),   # 在入口下方
            TrackInfo(1, (50, -5), (45, -10, 55, 0)),   # 重複位置
            TrackInfo(1, (50, 5), (45, 0, 55, 10)),     # 通過入口 (in) - 位置改變
            TrackInfo(1, (50, 5), (45, 0, 55, 10)),     # 重複位置（抖動過濾）
            TrackInfo(1, (50, 5), (45, 0, 55, 10)),     # 再次重複位置（抖動過濾）
            TrackInfo(1, (50, 25), (45, 20, 55, 30)),   # 通過人行道
            TrackInfo(1, (50, 45), (45, 40, 55, 50)),   # 在出口下方
            TrackInfo(1, (50, 45), (45, 40, 55, 50)),   # 重複位置
            TrackInfo(1, (50, 55), (45, 50, 55, 60)),   # 通過出口 (out) - 位置改變
            TrackInfo(1, (50, 55), (45, 50, 55, 60)),   # 重複位置（抖動過濾）
            TrackInfo(1, (50, 55), (45, 50, 55, 60)),   # 再次重複位置（抖動過濾）
        ]
        
        # 場景2：U-turn
        uturn_track = [
            TrackInfo(2, (50, -5), (45, -10, 55, 0)),   # 在入口下方
            TrackInfo(2, (50, -5), (45, -10, 55, 0)),   # 重複位置
            TrackInfo(2, (50, 5), (45, 0, 55, 10)),     # 通過入口 (in) - 位置改變
            TrackInfo(2, (50, 5), (45, 0, 55, 10)),     # 重複位置（抖動過濾）
            TrackInfo(2, (50, 5), (45, 0, 55, 10)),     # 再次重複位置（抖動過濾）
            TrackInfo(2, (50, 25), (45, 20, 55, 30)),   # 通過人行道
            TrackInfo(2, (50, 5), (45, 0, 55, 10)),     # U-turn回到入口
        ]
        
        # 場景3：側向移動
        side_track = [
            TrackInfo(3, (-5, 25), (-10, 20, 0, 30)),   # 在左側
            TrackInfo(3, (-5, 25), (-10, 20, 0, 30)),   # 重複位置
            TrackInfo(3, (5, 25), (0, 20, 10, 30)),     # 通過左側人行道 - 位置改變
            TrackInfo(3, (5, 25), (0, 20, 10, 30)),     # 重複位置（抖動過濾）
            TrackInfo(3, (5, 25), (0, 20, 10, 30)),     # 再次重複位置（抖動過濾）
            TrackInfo(3, (50, 25), (45, 20, 55, 30)),   # 移動到中間
            TrackInfo(3, (95, 25), (90, 20, 100, 30)),  # 在右側
            TrackInfo(3, (95, 25), (90, 20, 100, 30)),  # 重複位置
            TrackInfo(3, (105, 25), (100, 20, 110, 30)), # 通過右側人行道 - 位置改變
            TrackInfo(3, (105, 25), (100, 20, 110, 30)), # 重複位置（抖動過濾）
            TrackInfo(3, (105, 25), (100, 20, 110, 30)), # 再次重複位置（抖動過濾）
        ]
        
        all_events = []
        
        # 處理所有軌跡
        frame_number = 0
        for track_list in [normal_track, uturn_track, side_track]:
            for track in track_list:
                events = counter.update([track], frame_number)
                all_events.extend(events)
                frame_number += 1
        
        # 驗證事件產生
        self.assertGreater(len(all_events), 0, "應該產生計數事件")
        
        # 分析事件類型
        event_types = [e.event_type for e in all_events]
        line_ids = [e.line_id for e in all_events]
        
        # 至少應該有進入事件
        self.assertIn("in", event_types, "應該有進入事件")
        print(f"✅ 多線計數器整合測試: 總事件數={len(all_events)}, 事件類型={set(event_types)}, 線段ID={set(line_ids)}")
        
        # 驗證線段ID
        for line_id in line_ids:
            self.assertIn(line_id, ["entrance", "exit", "sidewalk"], f"未知線段ID: {line_id}")
    
    def test_calibration_integration(self):
        """測試標定整合 - P3企業級"""
        # 創建標定配置
        calibration_config = CalibrationConfig(
            enabled=True,
            src_points=[(0, 0), (640, 0), (640, 480), (0, 480)],  # 像素座標
            dst_points=[(0, 0), (10, 0), (10, 8), (0, 8)]          # 實際座標(公尺)
        )
        
        calibrator = HomographyCalibrator(calibration_config)
        
        # 測試像素到實際座標轉換
        test_pixels = [(320, 240), (160, 120), (480, 360)]
        expected_meters = [(5, 4), (2.5, 2), (7.5, 6)]
        
        for pixel, expected_meter in zip(test_pixels, expected_meters):
            actual_meter = calibrator.pixel_to_world(pixel)
            
            # 驗證轉換精度（允許1%誤差）
            self.assertAlmostEqual(actual_meter[0], expected_meter[0], delta=0.1)
            self.assertAlmostEqual(actual_meter[1], expected_meter[1], delta=0.1)
        
        # 測試實際座標到像素轉換
        for meter, expected_pixel in zip(expected_meters, test_pixels):
            actual_pixel = calibrator.world_to_pixel(meter)
            
            # 驗證轉換精度（允許1像素誤差）
            self.assertAlmostEqual(actual_pixel[0], expected_pixel[0], delta=1)
            self.assertAlmostEqual(actual_pixel[1], expected_pixel[1], delta=1)
    
    def test_roi_ignore_zone_integration(self):
        """測試ROI和忽略區域整合 - P3企業級"""
        # 定義ROI區域
        roi_polygon = [(0, 0), (640, 0), (640, 480), (0, 480)]
        
        # 定義忽略區域
        ignore_zones = [
            [(100, 100), (200, 100), (200, 200), (100, 200)],  # 忽略區域1
            [(400, 300), (500, 300), (500, 400), (400, 400)]   # 忽略區域2
        ]
        
        # 測試點
        test_points = [
            ((50, 50), True, False),    # 在ROI內，不在忽略區
            ((150, 150), True, True),   # 在ROI內，在忽略區1
            ((450, 350), True, True),   # 在ROI內，在忽略區2
            ((700, 500), False, False), # 不在ROI內
            ((300, 300), True, False),  # 在ROI內，不在任何忽略區
        ]
        
        for point, should_be_in_roi, should_be_ignored in test_points:
            # 檢查ROI
            in_roi = point_in_polygon(point, roi_polygon)
            self.assertEqual(in_roi, should_be_in_roi, f"ROI檢查失敗: {point}")
            
            # 檢查忽略區域
            in_ignore = any(point_in_polygon(point, zone) for zone in ignore_zones)
            self.assertEqual(in_ignore, should_be_ignored, f"忽略區域檢查失敗: {point}")
    
    def test_performance_under_load(self):
        """測試負載下的性能 - P3企業級"""
        # 創建大量線段和區域
        line_configs = [
            LineConfig(id=f"line_{i}", points=[(i*10, 0), (i*10+50, 0)], direction="both")
            for i in range(20)
        ]
        counter = MultiLineCounter(line_configs)
        
        # 創建大量軌跡，確保穿過線段
        from src.logic.counter import TrackInfo
        track_infos = []
        for i in range(200):
            # 確保軌跡穿過至少一個線段
            x = (i * 5) % 200  # 循環使用x座標
            track_infos.append(TrackInfo(i, (x, 0), (x-2, -2, x+2, 2)))
        
        # 性能測試
        start_time = time.time()
        events = counter.update(track_infos, 0)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # 驗證性能要求
        self.assertLess(processing_time, 0.5, f"處理時間過長: {processing_time:.3f}s")
        # 注意：由於軌跡可能不穿過線段，所以不強制要求產生事件
        # 主要測試性能，即處理時間
        
        print(f"✅ 性能測試: 處理200個軌跡用時 {processing_time:.3f}s")
    
    def test_error_handling_robustness(self):
        """測試錯誤處理健壯性 - P3企業級"""
        # 測試空輸入
        counter = MultiLineCounter([])
        events = counter.update([], 0)
        self.assertEqual(len(events), 0)
        
        # 測試無效配置
        invalid_config = LineConfig(
            id="invalid",
            points=[],  # 空點列表
            direction="both"
        )
        
        # 應該能夠處理無效配置而不崩潰
        try:
            counter = MultiLineCounter([invalid_config])
            events = counter.update([], 0)
            self.assertIsInstance(events, list)
        except Exception as e:
            # 如果拋出異常，應該是預期的錯誤類型
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_data_consistency(self):
        """測試數據一致性 - P3企業級"""
        # 創建計數器
        line_config = LineConfig(
            id="test_line",
            points=[(0, 0), (100, 0)],
            direction="both"
        )
        counter = MultiLineCounter([line_config])
        
        # 模擬軌跡
        from src.logic.counter import TrackInfo
        track = TrackInfo(1, (50, 0), (45, -5, 55, 5))
        
        # 多次更新，檢查一致性
        events1 = counter.update([track], 0)
        events2 = counter.update([track], 1)
        
        # 相同輸入應該產生相同結果
        self.assertEqual(len(events1), len(events2))
        
        # 檢查事件屬性
        for event in events1 + events2:
            self.assertIsNotNone(event.timestamp)
            self.assertIsNotNone(event.track_id)
            self.assertIsNotNone(event.event_type)
            self.assertIn(event.event_type, ["in", "out"])


if __name__ == "__main__":
    unittest.main()
