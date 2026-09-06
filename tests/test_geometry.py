"""幾何模組單元測試 - P3企業交付級"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pytest
import numpy as np
from src.logic.geometry import Point, Point2D, centroid, point_in_polygon, line_side, vector_angle_deg, uturn_detected, distance_between_points, point_to_line_distance


class TestGeometry(unittest.TestCase):
    def test_centroid(self):
        """測試質心計算"""
        points = [(0, 0), (2, 0), (2, 2), (0, 2)]
        cx, cy = centroid(points)
        self.assertAlmostEqual(cx, 1.0)
        self.assertAlmostEqual(cy, 1.0)
    
    def test_point_in_polygon(self):
        """測試點在多邊形內判斷 - P3增強版"""
        # 正方形
        square = [(0, 0), (2, 0), (2, 2), (0, 2)]
        
        # 內部點
        self.assertTrue(point_in_polygon((1, 1), square))
        
        # 外部點
        self.assertFalse(point_in_polygon((3, 3), square))
        self.assertFalse(point_in_polygon((-1, -1), square))
        
        # 邊界點測試
        self.assertTrue(point_in_polygon((0, 0), square))  # 頂點
        self.assertTrue(point_in_polygon((1, 0), square))  # 邊上點
        
        # 複雜多邊形測試
        complex_poly = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        self.assertTrue(point_in_polygon((1, 1), complex_poly))
        self.assertFalse(point_in_polygon((3, 3), complex_poly))
        
        # 空多邊形測試
        empty_poly = []
        self.assertFalse(point_in_polygon((0, 0), empty_poly))
        
        # 單點多邊形測試
        single_point = [(0, 0)]
        self.assertFalse(point_in_polygon((0, 0), single_point))
    
    def test_line_side(self):
        """測試點相對於線段的位置"""
        line = ((0, 0), (2, 0))  # 水平線
        
        # 線上方
        self.assertGreater(line_side((1, 1), line), 0)
        
        # 線下方
        self.assertLess(line_side((1, -1), line), 0)
        
        # 線上
        self.assertEqual(line_side((1, 0), line), 0)
    
    def test_vector_angle_deg(self):
        """測試向量夾角計算"""
        # 相同方向
        angle = vector_angle_deg((1, 0), (2, 0))
        self.assertAlmostEqual(angle, 0.0, places=1)
        
        # 垂直
        angle = vector_angle_deg((1, 0), (0, 1))
        self.assertAlmostEqual(angle, 90.0, places=1)
        
        # 相反方向
        angle = vector_angle_deg((1, 0), (-1, 0))
        self.assertAlmostEqual(angle, 180.0, places=1)
    
    def test_uturn_detected(self):
        """測試U-turn檢測"""
        # 直線移動
        history = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self.assertFalse(uturn_detected(history, 120.0))
        
        # U-turn移動（更明顯的角度變化）
        history = [(0, 0), (1, 0), (1, 1), (0, 0)]
        self.assertTrue(uturn_detected(history, 120.0))
    
    def test_distance_between_points(self):
        """測試點間距離計算"""
        p1 = (0, 0)
        p2 = (3, 4)
        distance = distance_between_points(p1, p2)
        self.assertAlmostEqual(distance, 5.0, places=2)
    
    def test_point_to_line_distance(self):
        """測試點到線段距離計算"""
        line_start = (0, 0)
        line_end = (4, 0)
        point = (2, 3)
        distance = point_to_line_distance(point, line_start, line_end)
        self.assertAlmostEqual(distance, 3.0, places=2)
    
    def test_complex_polygon_roi(self):
        """測試複雜多邊形ROI"""
        # L形多邊形
        l_shape = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        
        # 內部點
        self.assertTrue(point_in_polygon((1, 1), l_shape))
        self.assertTrue(point_in_polygon((3, 1), l_shape))
        self.assertTrue(point_in_polygon((1, 3), l_shape))
        
        # 外部點
        self.assertFalse(point_in_polygon((3, 3), l_shape))  # L形缺口
        self.assertFalse(point_in_polygon((5, 5), l_shape))  # 完全外部
    
    def test_ignore_zone_detection(self):
        """測試忽略區域檢測"""
        # 矩形忽略區域
        ignore_zone = [(1, 1), (3, 1), (3, 3), (1, 3)]
        
        # 在忽略區域內
        self.assertTrue(point_in_polygon((2, 2), ignore_zone))
        
        # 在忽略區域外
        self.assertFalse(point_in_polygon((0, 0), ignore_zone))
        self.assertFalse(point_in_polygon((4, 4), ignore_zone))
    
    def test_edge_cases(self):
        """測試邊界情況"""
        # 空多邊形
        empty_polygon = []
        self.assertFalse(point_in_polygon((0, 0), empty_polygon))
        
        # 單點多邊形
        single_point = [(0, 0)]
        self.assertFalse(point_in_polygon((0, 0), single_point))
        
        # 線段多邊形
        line_polygon = [(0, 0), (1, 0)]
        self.assertFalse(point_in_polygon((0.5, 0), line_polygon))


    def test_performance_large_polygon(self):
        """測試大多邊形性能 - P3企業級"""
        import time
        
        # 創建一個大多邊形（1000個頂點）
        large_polygon = [(i, i % 10) for i in range(1000)]
        
        # 測試內部點
        start_time = time.time()
        result = point_in_polygon((500, 5), large_polygon)
        end_time = time.time()
        
        # 性能測試：應該在合理時間內完成
        self.assertLess(end_time - start_time, 1.0)  # 1秒內完成
        self.assertIsInstance(result, bool)
    
    def test_roi_ignore_zone_integration(self):
        """測試ROI和忽略區域整合 - P3企業級"""
        # 主ROI區域
        main_roi = [(0, 0), (10, 0), (10, 10), (0, 10)]
        
        # 忽略區域（在ROI內部）
        ignore_zone = [(2, 2), (8, 2), (8, 8), (2, 8)]
        
        # 測試點在不同區域
        test_points = [
            ((1, 1), True, False),   # 在ROI內，不在忽略區
            ((5, 5), True, True),    # 在ROI內，在忽略區
            ((11, 11), False, False) # 不在ROI內
        ]
        
        for point, in_roi, in_ignore in test_points:
            roi_result = point_in_polygon(point, main_roi)
            ignore_result = point_in_polygon(point, ignore_zone)
            
            self.assertEqual(roi_result, in_roi, f"Point {point} ROI check failed")
            self.assertEqual(ignore_result, in_ignore, f"Point {point} ignore check failed")
    
    def test_precision_handling(self):
        """測試精度處理 - P3企業級"""
        # 測試浮點數精度
        points = [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)]
        self.assertTrue(point_in_polygon((0.15, 0.15), points))
        
        # 測試負數座標
        neg_poly = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        self.assertTrue(point_in_polygon((0, 0), neg_poly))
        self.assertFalse(point_in_polygon((2, 2), neg_poly))
        
        # 測試重複頂點
        duplicate_poly = [(0, 0), (1, 0), (1, 0), (1, 1), (0, 1)]
        self.assertTrue(point_in_polygon((0.5, 0.5), duplicate_poly))
    
    def test_uturn_advanced_scenarios(self):
        """測試U-turn高級場景 - P3企業級"""
        # 測試不同角度閾值
        track_points = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (2, 1), (1, 1), (0, 1)]
        
        # 這個軌跡可能不會被檢測為U-turn，取決於實現
        # 我們測試函數能正常運行即可
        result_45 = uturn_detected(track_points, threshold_deg=45)
        result_90 = uturn_detected(track_points, threshold_deg=90)
        result_180 = uturn_detected(track_points, threshold_deg=180)
        
        # 至少應該有一個結果為True（較低閾值）
        self.assertTrue(result_45 or result_90, "至少一個閾值應該檢測到U-turn")
        self.assertFalse(result_180, "180度閾值不應該檢測到U-turn")
        
        # 測試邊界情況
        self.assertFalse(uturn_detected([], threshold_deg=90))
        self.assertFalse(uturn_detected([(0, 0)], threshold_deg=90))
        self.assertFalse(uturn_detected([(0, 0), (1, 0)], threshold_deg=90))
        
        # 測試緩慢轉彎（不應該被檢測為U-turn）
        slow_turn = [(0, 0), (1, 0.1), (2, 0.2), (3, 0.3), (3, 1.3)]
        self.assertFalse(uturn_detected(slow_turn, threshold_deg=90))


if __name__ == "__main__":
    unittest.main()
