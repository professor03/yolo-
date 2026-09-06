#!/usr/bin/env python3
"""評估系統單元測試 - P2企業級擴展"""
import sys
import os
import tempfile
import json
import pandas as pd
from pathlib import Path
import unittest
import pytest

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.evaluate import CrossingEvent, EvaluationResult, evaluate_crossings, load_ground_truth, load_predictions, match_events, calculate_idf1


def test_crossing_event():
    """測試CrossingEvent數據類"""
    event = CrossingEvent(
        timestamp=1.5,
        line_id="line_1",
        track_id=1,
        event_type="in",
        x_px=150.0,
        y_px=200.0
    )
    
    assert event.timestamp == 1.5
    assert event.line_id == "line_1"
    assert event.event_type == "in"
    assert event.track_id == 1
    assert event.x_px == 150.0
    assert event.y_px == 200.0


def test_evaluation_result():
    """測試EvaluationResult數據類"""
    result = EvaluationResult(
        precision=0.8,
        recall=0.9,
        f1=0.85,
        idf1=0.82,
        total_gt=10,
        total_pred=12,
        total_matched=8,
        total_id_switches=1
    )
    
    assert result.precision == 0.8
    assert result.recall == 0.9
    assert result.f1 == 0.85
    assert result.idf1 == 0.82
    assert result.total_gt == 10
    assert result.total_pred == 12
    assert result.total_matched == 8
    assert result.total_id_switches == 1


def test_load_ground_truth():
    """測試載入真實標註數據"""
    # 創建臨時文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"timestamp": 1.5, "line_id": "line_1", "event_type": "in", "track_id": 1, "x_px": 150, "y_px": 200}\n')
        f.write('{"timestamp": 2.0, "line_id": "line_1", "event_type": "in", "track_id": 2, "x_px": 180, "y_px": 200}\n')
        temp_file = f.name
    
    try:
        events = load_ground_truth(temp_file)
        assert len(events) == 2
        assert events[0].timestamp == 1.5
        assert events[0].line_id == "line_1"
        assert events[1].track_id == 2
    finally:
        os.unlink(temp_file)


def test_load_predictions():
    """測試載入預測結果"""
    
    # 創建臨時CSV文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('timestamp,line_id,direction,track_id,x_px,y_px,confidence\n')
        f.write('1.6,line_1,in,person_001,155,200,0.95\n')
        f.write('2.1,line_1,in,person_002,185,200,0.92\n')
        temp_file = f.name
    
    try:
        events = load_predictions(temp_file)
        assert len(events) == 2
        assert events[0].timestamp == 1.6
        assert events[0].line_id == "line_1"
        assert events[1].track_id == 2
    finally:
        os.unlink(temp_file)


def test_match_events():
    """測試事件匹配"""
    # 創建測試事件
    gt_events = [
        CrossingEvent(1.5, "line_1", 1, "in", 150, 200),
        CrossingEvent(2.0, "line_1", 2, "in", 180, 200),
        CrossingEvent(3.0, "line_2", 1, "out", 200, 300)
    ]
    
    pred_events = [
        CrossingEvent(1.6, "line_1", 1, "in", 155, 200),
        CrossingEvent(2.1, "line_1", 2, "in", 185, 200),
        CrossingEvent(3.1, "line_2", 1, "out", 205, 300),
        CrossingEvent(4.0, "line_1", 3, "in", 220, 200)  # 額外的預測
    ]
    
    matched, unmatched_gt, unmatched_pred = match_events(gt_events, pred_events, tolerance=0.5)
    
    assert len(matched) == 3  # 所有真實事件都匹配
    assert len(unmatched_gt) == 0  # 沒有未匹配的真實事件
    assert len(unmatched_pred) == 1  # 有一個未匹配的預測事件


def test_calculate_idf1():
    """測試IDF1計算"""
    # 創建匹配事件（包含ID切換）
    matched = [
        (CrossingEvent(1.0, "line_1", 1, "in", 150, 200), 
         CrossingEvent(1.1, "line_1", 1, "in", 155, 200)),
        (CrossingEvent(2.0, "line_1", 1, "in", 180, 200), 
         CrossingEvent(2.1, "line_1", 2, "in", 185, 200))  # ID切換
    ]
    
    unmatched_gt = []
    unmatched_pred = []
    
    idf1 = calculate_idf1(matched, unmatched_gt, unmatched_pred)
    
    assert 0.0 <= idf1 <= 1.0
    assert idf1 < 1.0  # 因為有ID切換，應該小於1.0


def test_evaluate_perfect_match():
    """測試完美匹配的評估"""
    
    # 創建臨時文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as gt_file:
        gt_file.write('{"timestamp": 1.5, "line_id": "line_1", "direction": "in", "track_id": "person_001", "x_px": 150, "y_px": 200, "confidence": 0.95}\n')
        gt_file.write('{"timestamp": 2.0, "line_id": "line_1", "direction": "in", "track_id": "person_002", "x_px": 180, "y_px": 200, "confidence": 0.92}\n')
        gt_path = gt_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as pred_file:
        pred_file.write('timestamp,line_id,direction,track_id,x_px,y_px,confidence\n')
        pred_file.write('1.6,line_1,in,person_001,155,200,0.95\n')
        pred_file.write('2.1,line_1,in,person_002,185,200,0.92\n')
        pred_path = pred_file.name
    
    try:
        result = evaluate_crossings(gt_path, pred_path, tolerance=0.5)
        
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.idf1 == 1.0
        assert result.total_gt == 2
        assert result.total_pred == 2
        assert result.total_matched == 2
        assert result.total_id_switches == 0
    finally:
        os.unlink(gt_path)
        os.unlink(pred_path)


def test_evaluate_with_errors():
    """測試包含錯誤的評估"""
    
    # 創建臨時文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as gt_file:
        gt_file.write('{"timestamp": 1.5, "line_id": "line_1", "direction": "in", "track_id": "person_001", "x_px": 150, "y_px": 200, "confidence": 0.95}\n')
        gt_file.write('{"timestamp": 2.0, "line_id": "line_1", "direction": "in", "track_id": "person_002", "x_px": 180, "y_px": 200, "confidence": 0.92}\n')
        gt_path = gt_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as pred_file:
        pred_file.write('timestamp,line_id,direction,track_id,x_px,y_px,confidence\n')
        pred_file.write('1.6,line_1,in,person_001,155,200,0.95\n')
        pred_file.write('2.1,line_1,in,person_002,185,200,0.92\n')
        pred_file.write('3.0,line_1,in,person_003,220,200,0.90\n')  # 額外的預測
        pred_path = pred_file.name
    
    try:
        result = evaluate_crossings(gt_path, pred_path, tolerance=0.5)
        
        assert result.precision < 1.0  # 精確度小於1.0
        assert result.recall == 1.0    # 召回率為1.0
        assert result.f1 < 1.0         # F1小於1.0
        assert result.total_gt == 2
        assert result.total_pred == 3
        assert result.total_matched == 2
    finally:
        os.unlink(gt_path)
        os.unlink(pred_path)


if __name__ == "__main__":
    print("運行評估系統單元測試...")
    
    test_crossing_event()
    print("✓ CrossingEvent 測試通過")
    
    test_evaluation_result()
    print("✓ EvaluationResult 測試通過")
    
    test_load_ground_truth()
    print("✓ 載入真實標註測試通過")
    
    test_load_predictions()
    print("✓ 載入預測結果測試通過")
    
    test_match_events()
    print("✓ 事件匹配測試通過")
    
    test_calculate_idf1()
    print("✓ IDF1計算測試通過")
    
    test_evaluate_perfect_match()
    print("✓ 完美匹配評估測試通過")
    
    test_evaluate_with_errors()
    print("✓ 錯誤評估測試通過")
    
    print("\n所有測試通過！🎉")
