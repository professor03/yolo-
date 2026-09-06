from __future__ import annotations

import argparse
import json
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import time
from datetime import datetime


@dataclass
class CrossingEvent:
    """穿越事件"""
    timestamp: float
    line_id: str
    track_id: int
    event_type: str  # "in" or "out"
    x_px: float
    y_px: float
    x_m: Optional[float] = None
    y_m: Optional[float] = None


@dataclass
class EvaluationResult:
    """評估結果"""
    precision: float
    recall: float
    f1: float
    idf1: float
    total_gt: int
    total_pred: int
    matched_gt: int
    matched_pred: int
    per_line_results: Dict[str, Dict[str, float]]


def load_ground_truth(file_path: str) -> List[CrossingEvent]:
    """載入真實標註數據"""
    events = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                event = CrossingEvent(
                    timestamp=float(data['timestamp']),
                    line_id=str(data['line_id']),
                    track_id=int(data['track_id']),
                    event_type=str(data['event_type']),
                    x_px=float(data['x_px']),
                    y_px=float(data['y_px']),
                    x_m=data.get('x_m'),
                    y_m=data.get('y_m')
                )
                events.append(event)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"警告: 跳過無效的標註行: {line[:50]}... 錯誤: {e}")
                continue
    
    return events


def load_predictions(file_path: str) -> List[CrossingEvent]:
    """載入預測結果"""
    events = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                event = CrossingEvent(
                    timestamp=float(row['timestamp']),
                    line_id=str(row['line_id']),
                    track_id=int(row['track_id']),
                    event_type=str(row['event_type']),
                    x_px=float(row['x_px']),
                    y_px=float(row['y_px']),
                    x_m=float(row['x_m']) if row.get('x_m') and row['x_m'] != '' else None,
                    y_m=float(row['y_m']) if row.get('y_m') and row['y_m'] != '' else None
                )
                events.append(event)
            except (KeyError, ValueError) as e:
                print(f"警告: 跳過無效的預測行: {row} 錯誤: {e}")
                continue
    
    return events


def match_events(gt_events: List[CrossingEvent], pred_events: List[CrossingEvent], 
                tolerance: float = 0.5) -> Tuple[Set[int], Set[int], Dict[int, int]]:
    """匹配真實標註和預測事件
    
    返回:
        matched_gt_indices: 匹配的真實事件索引
        matched_pred_indices: 匹配的預測事件索引
        gt_to_pred_mapping: 真實事件到預測事件的映射
    """
    matched_gt = set()
    matched_pred = set()
    gt_to_pred = {}
    
    # 按時間排序
    gt_events.sort(key=lambda x: x.timestamp)
    pred_events.sort(key=lambda x: x.timestamp)
    
    for i, gt_event in enumerate(gt_events):
        best_match = None
        best_score = float('inf')
        
        for j, pred_event in enumerate(pred_events):
            if j in matched_pred:
                continue
            
            # 檢查是否匹配
            if (gt_event.line_id == pred_event.line_id and 
                gt_event.event_type == pred_event.event_type):
                
                # 計算時間差
                time_diff = abs(gt_event.timestamp - pred_event.timestamp)
                
                if time_diff <= tolerance:
                    # 計算空間距離（如果都有座標）
                    spatial_score = 0
                    if (gt_event.x_px is not None and gt_event.y_px is not None and
                        pred_event.x_px is not None and pred_event.y_px is not None):
                        spatial_score = ((gt_event.x_px - pred_event.x_px) ** 2 + 
                                       (gt_event.y_px - pred_event.y_px) ** 2) ** 0.5
                    
                    # 綜合評分（時間差 + 空間距離）
                    total_score = time_diff + spatial_score * 0.001
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_match = j
        
        if best_match is not None:
            matched_gt.add(i)
            matched_pred.add(best_match)
            gt_to_pred[i] = best_match
    
    return matched_gt, matched_pred, gt_to_pred


def calculate_idf1(gt_events: List[CrossingEvent], pred_events: List[CrossingEvent],
                  matched_gt: Set[int], matched_pred: Set[int], 
                  gt_to_pred: Dict[int, int]) -> float:
    """計算IDF1分數（簡化版）"""
    if not gt_events or not pred_events:
        return 0.0
    
    # 按軌跡ID分組
    gt_tracks = {}
    pred_tracks = {}
    
    for i, event in enumerate(gt_events):
        if i in matched_gt:
            track_id = event.track_id
            if track_id not in gt_tracks:
                gt_tracks[track_id] = []
            gt_tracks[track_id].append(i)
    
    for i, event in enumerate(pred_events):
        if i in matched_pred:
            track_id = event.track_id
            if track_id not in pred_tracks:
                pred_tracks[track_id] = []
            pred_tracks[track_id].append(i)
    
    # 計算IDF1
    total_gt_tracks = len(gt_tracks)
    total_pred_tracks = len(pred_tracks)
    
    if total_gt_tracks == 0 and total_pred_tracks == 0:
        return 1.0
    
    if total_gt_tracks == 0 or total_pred_tracks == 0:
        return 0.0
    
    # 簡化的IDF1計算
    matched_tracks = 0
    for gt_track_id in gt_tracks:
        if gt_track_id in pred_tracks:
            matched_tracks += 1
    
    precision = matched_tracks / total_pred_tracks if total_pred_tracks > 0 else 0
    recall = matched_tracks / total_gt_tracks if total_gt_tracks > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def evaluate_crossings(gt_file: str, pred_file: str, tolerance: float = 0.5) -> EvaluationResult:
    """評估穿越事件"""
    print(f"載入真實標註: {gt_file}")
    gt_events = load_ground_truth(gt_file)
    print(f"載入 {len(gt_events)} 個真實事件")
    
    print(f"載入預測結果: {pred_file}")
    pred_events = load_predictions(pred_file)
    print(f"載入 {len(pred_events)} 個預測事件")
    
    print(f"匹配事件 (容忍度: {tolerance}秒)")
    matched_gt, matched_pred, gt_to_pred = match_events(gt_events, pred_events, tolerance)
    
    print(f"匹配結果: {len(matched_gt)}/{len(gt_events)} 真實事件, {len(matched_pred)}/{len(pred_events)} 預測事件")
    
    # 計算整體指標
    precision = len(matched_pred) / len(pred_events) if pred_events else 0.0
    recall = len(matched_gt) / len(gt_events) if gt_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 計算IDF1
    idf1 = calculate_idf1(gt_events, pred_events, matched_gt, matched_pred, gt_to_pred)
    
    # 按線段計算結果
    per_line_results = {}
    line_ids = set(event.line_id for event in gt_events + pred_events)
    
    for line_id in line_ids:
        line_gt = [i for i, event in enumerate(gt_events) if event.line_id == line_id]
        line_pred = [i for i, event in enumerate(pred_events) if event.line_id == line_id]
        
        line_matched_gt = [i for i in line_gt if i in matched_gt]
        line_matched_pred = [i for i in line_pred if i in matched_pred]
        
        line_precision = len(line_matched_pred) / len(line_pred) if line_pred else 0.0
        line_recall = len(line_matched_gt) / len(line_gt) if line_gt else 0.0
        line_f1 = 2 * line_precision * line_recall / (line_precision + line_recall) if (line_precision + line_recall) > 0 else 0.0
        
        per_line_results[line_id] = {
            'precision': line_precision,
            'recall': line_recall,
            'f1': line_f1,
            'gt_count': len(line_gt),
            'pred_count': len(line_pred),
            'matched_gt': len(line_matched_gt),
            'matched_pred': len(line_matched_pred)
        }
    
    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        idf1=idf1,
        total_gt=len(gt_events),
        total_pred=len(pred_events),
        matched_gt=len(matched_gt),
        matched_pred=len(matched_pred),
        per_line_results=per_line_results
    )


def save_evaluation_report(result: EvaluationResult, output_file: str):
    """保存評估報告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_metrics': {
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1,
            'idf1': result.idf1,
            'total_gt_events': result.total_gt,
            'total_pred_events': result.total_pred,
            'matched_gt_events': result.matched_gt,
            'matched_pred_events': result.matched_pred
        },
        'per_line_metrics': result.per_line_results
    }
    
    # 確保輸出目錄存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"評估報告已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='評估人流偵測結果')
    parser.add_argument('--pred', required=True, help='預測結果CSV檔案')
    parser.add_argument('--gt', required=True, help='真實標註JSONL檔案')
    parser.add_argument('--tolerance', type=float, default=0.5, help='時間容忍度（秒）')
    parser.add_argument('--output', default='reports/last_eval.json', help='輸出報告檔案')
    parser.add_argument('--verbose', action='store_true', help='詳細輸出')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("=== YOLO 人流偵測評估系統 ===")
        print(f"預測檔案: {args.pred}")
        print(f"真實標註: {args.gt}")
        print(f"時間容忍度: {args.tolerance}秒")
        print()
    
    # 執行評估
    result = evaluate_crossings(args.gt, args.pred, args.tolerance)
    
    # 保存報告
    save_evaluation_report(result, args.output)
    
    # 輸出結果
    print("\n=== 評估結果 ===")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1 Score:  {result.f1:.4f}")
    print(f"IDF1:      {result.idf1:.4f}")
    print()
    print(f"真實事件: {result.total_gt}")
    print(f"預測事件: {result.total_pred}")
    print(f"匹配事件: {result.matched_gt} (GT) / {result.matched_pred} (Pred)")
    print()
    
    if args.verbose and result.per_line_results:
        print("=== 各線段結果 ===")
        for line_id, metrics in result.per_line_results.items():
            print(f"{line_id}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  GT/Pred:   {metrics['gt_count']}/{metrics['pred_count']}")
            print(f"  Matched:   {metrics['matched_gt']}/{metrics['matched_pred']}")
            print()


if __name__ == '__main__':
    main()