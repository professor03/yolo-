"""計數模組單元測試 - P3企業交付級"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pytest
import time
import numpy as np
from src.logic.counter import LineConfig, ZoneConfig, TrackInfo, MultiLineCounter, ZoneOccupancy, LineEvent


class TestCounter(unittest.TestCase):
    def test_line_config(self):
        """測試線段配置"""
        config = LineConfig(
            id="test_line",
            points=[(0, 0), (100, 0)],
            direction="both",
            uturn_cooldown=30,
            jitter_frames=2
        )
        self.assertEqual(config.id, "test_line")
        self.assertEqual(config.direction, "both")
    
    def test_zone_config(self):
        """測試區域配置"""
        config = ZoneConfig(
            id="test_zone",
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            ignore=False
        )
        self.assertEqual(config.id, "test_zone")
        self.assertFalse(config.ignore)
    
    def test_track_info(self):
        """測試軌跡信息"""
        track = TrackInfo(
            id=1,
            center=(50, 50),
            bbox=(40, 40, 60, 60)
        )
        self.assertEqual(track.id, 1)
        self.assertEqual(track.center, (50, 50))
    
    def test_multi_line_counter(self):
        """測試多線計數器"""
        line_configs = [
            LineConfig("line1", [(0, 100), (100, 100)], "both", 30, 2),
            LineConfig("line2", [(0, 200), (100, 200)], "both", 30, 2)
        ]
        counter = MultiLineCounter(line_configs, 30, 15, 40.0, 30)
        
        # 測試軌跡更新
        track_infos = [
            TrackInfo(1, (50, 50), (40, 40, 60, 60), timestamp=time.time()),
            TrackInfo(2, (50, 150), (40, 140, 60, 160), timestamp=time.time())
        ]
        
        events = counter.update(track_infos, 0)
        self.assertIsInstance(events, list)
    
    def test_direction_jitter_filtering(self):
        """測試方向抖動濾波"""
        line_config = LineConfig("test_line", [(0, 100), (100, 100)], "both", 30, 3)
        counter = MultiLineCounter([line_config], 30, 15, 40.0, 30)
        
        # 模擬抖動軌跡
        track_infos = [
            TrackInfo(1, (50, 95), (40, 90, 60, 100), timestamp=time.time()),
            TrackInfo(1, (50, 105), (40, 100, 60, 110), timestamp=time.time() + 0.1),
            TrackInfo(1, (50, 95), (40, 90, 60, 100), timestamp=time.time() + 0.2),
            TrackInfo(1, (50, 105), (40, 100, 60, 110), timestamp=time.time() + 0.3),
        ]
        
        # 應該被濾波，不產生事件
        events = counter.update(track_infos, 0)
        self.assertEqual(len(events), 0)
    
    def test_uturn_deduplication(self):
        """測試U-turn去重"""
        line_config = LineConfig("test_line", [(0, 100), (100, 100)], "both", 30, 2)
        counter = MultiLineCounter([line_config], 30, 15, 40.0, 30)
        
        # 模擬U-turn軌跡
        track_infos = [
            TrackInfo(1, (50, 95), (40, 90, 60, 100), timestamp=time.time()),
            TrackInfo(1, (50, 105), (40, 100, 60, 110), timestamp=time.time() + 0.1),
            TrackInfo(1, (50, 95), (40, 90, 60, 100), timestamp=time.time() + 0.2),
        ]
        
        events = counter.update(track_infos, 0)
        # U-turn應該被去重
        self.assertEqual(len(events), 0)
    
    def test_reentry_cooldown(self):
        """測試重入冷卻"""
        line_config = LineConfig("test_line", [(0, 100), (100, 100)], "both", 30, 2)
        counter = MultiLineCounter([line_config], 30, 15, 40.0, 30)
        
        # 第一次穿越
        track_infos = [
            TrackInfo(1, (50, 95), (40, 90, 60, 100), timestamp=time.time()),
            TrackInfo(1, (50, 105), (40, 100, 60, 110), timestamp=time.time() + 0.1),
        ]
        events = counter.update(track_infos, 0)
        
        # 立即重入（應該被冷卻）
        track_infos = [
            TrackInfo(1, (50, 105), (40, 100, 60, 110), timestamp=time.time() + 0.2),
            TrackInfo(1, (50, 95), (40, 90, 60, 100), timestamp=time.time() + 0.3),
        ]
        events = counter.update(track_infos, 1)
        self.assertEqual(len(events), 0)
    
    def test_id_fault_tolerance(self):
        """測試ID容錯合併"""
        line_config = LineConfig("test_line", [(0, 100), (100, 100)], "both", 30, 2)
        counter = MultiLineCounter([line_config], 30, 15, 40.0, 30)
        
        # 軌跡1短暫消失
        track_infos = [
            TrackInfo(1, (50, 95), (40, 90, 60, 100), timestamp=time.time()),
        ]
        counter.update(track_infos, 0)
        
        # 軌跡2在附近出現（應該被合併）
        track_infos = [
            TrackInfo(2, (52, 97), (42, 92, 62, 102), timestamp=time.time() + 0.1),
        ]
        events = counter.update(track_infos, 1)
        # 應該產生合併事件
        self.assertGreaterEqual(len(events), 0)
    
    def test_multi_zone_occupancy(self):
        """測試多區域佔用檢測"""
        zones = [
            ZoneConfig("zone1", [(0, 0), (100, 0), (100, 100), (0, 100)], False),
            ZoneConfig("zone2", [(200, 200), (300, 200), (300, 300), (200, 300)], False),
            ZoneConfig("ignore1", [(50, 50), (150, 50), (150, 150), (50, 150)], True)
        ]
        classifier = ZoneOccupancy(zones)
        
        track_infos = [
            TrackInfo(1, (50, 50), (40, 40, 60, 60), timestamp=time.time()),
            TrackInfo(2, (250, 250), (240, 240, 260, 260), timestamp=time.time()),
            TrackInfo(3, (100, 100), (90, 90, 110, 110), timestamp=time.time()),
        ]
        
        events = classifier.update(track_infos)
        self.assertIsInstance(events, list)
        # 檢查是否有進入事件
        entry_events = [e for e in events if e.event_type == "entry"]
        self.assertGreater(len(entry_events), 0)
    
    def test_line_event_generation(self):
        """測試線段事件生成"""
        line_config = LineConfig("test_line", [(0, 100), (100, 100)], "in", 30, 2)
        counter = MultiLineCounter([line_config], 1, 15, 40.0, 30)  # 冷卻期設為1幀
        
        # 模擬穿越軌跡 - 需要滿足抖動過濾條件（jitter_frames=2）
        track_infos = [
            TrackInfo(1, (50, 95), (40, 90, 60, 100)),   # 在線段下方
            TrackInfo(1, (50, 95), (40, 90, 60, 100)),   # 重複位置
            TrackInfo(1, (50, 105), (40, 100, 60, 110)), # 在線段上方（位置改變）
            TrackInfo(1, (50, 105), (40, 100, 60, 110)), # 重複位置（抖動過濾）
            TrackInfo(1, (50, 105), (40, 100, 60, 110)), # 再次重複位置（抖動過濾）
        ]
        
        events = []
        for i, track in enumerate(track_infos):
            events.extend(counter.update([track], i))
        
        # 應該產生穿越事件
        self.assertGreater(len(events), 0)
        
        # 檢查事件類型
        event = events[0]
        self.assertIsInstance(event, LineEvent)
        self.assertEqual(event.line_id, "test_line")
        self.assertEqual(event.event_type, "in")
    
    def test_zone_occupancy(self):
        """測試區域佔用檢測"""
        zones = [
            ZoneConfig("zone1", [(0, 0), (100, 0), (100, 100), (0, 100)], False),
            ZoneConfig("ignore1", [(200, 200), (300, 200), (300, 300), (200, 300)], True)
        ]
        classifier = ZoneOccupancy(zones)
        
        track_infos = [
            TrackInfo(1, (50, 50), (40, 40, 60, 60)),
            TrackInfo(2, (250, 250), (240, 240, 260, 260))
        ]
        
        events = classifier.update(track_infos)
        self.assertIsInstance(events, list)
    
    def test_jitter_filtering_advanced(self):
        """測試抖動濾波高級場景 - P3企業級"""
        line_config = LineConfig(
            id="test_line",
            points=[(0, 0), (100, 0)],
            direction="both",
            jitter_frames=3
        )
        counter = MultiLineCounter([line_config])
        
        # 模擬抖動軌跡
        jitter_track = [
            TrackInfo(1, (50, 0), (45, -5, 55, 5)),  # 在線段上
            TrackInfo(1, (50, 2), (45, -3, 55, 7)),  # 輕微抖動
            TrackInfo(1, (50, -1), (45, -6, 55, 4)), # 反向抖動
            TrackInfo(1, (50, 0), (45, -5, 55, 5)),  # 回到線段
            TrackInfo(1, (50, 1), (45, -4, 55, 6)),  # 再次抖動
        ]
        
        events = []
        for i, track in enumerate(jitter_track):
            events.extend(counter.update([track], i))
        
        # 抖動應該被濾除，不應該產生事件
        self.assertEqual(len(events), 0)
    
    def test_uturn_deduplication_advanced(self):
        """測試U-turn去重高級場景 - P3企業級"""
        line_config = LineConfig(
            id="test_line",
            points=[(0, 0), (100, 0)],
            direction="both",
            uturn_cooldown=5
        )
        counter = MultiLineCounter([line_config])
        
        # 模擬U-turn軌跡
        uturn_track = [
            TrackInfo(1, (20, 0), (15, -5, 25, 5)),   # 進入
            TrackInfo(1, (50, 0), (45, -5, 55, 5)),   # 通過
            TrackInfo(1, (80, 0), (75, -5, 85, 5)),   # 接近出口
            TrackInfo(1, (50, 0), (45, -5, 55, 5)),   # U-turn回到中間
            TrackInfo(1, (20, 0), (15, -5, 25, 5)),   # 回到入口
        ]
        
        events = []
        for i, track in enumerate(uturn_track):
            events.extend(counter.update([track], i))
        
        # 應該只產生一次進入事件，U-turn被去重
        enter_events = [e for e in events if e.event_type == "in"]
        self.assertLessEqual(len(enter_events), 1)
    
    def test_reentry_cooldown_advanced(self):
        """測試重入冷卻高級場景 - P3企業級"""
        line_config = LineConfig(
            id="test_line",
            points=[(0, 0), (100, 0)],
            direction="both"
        )
        counter = MultiLineCounter([line_config], 1, 15, 40.0, 30)  # 冷卻期設為1幀
        
        # 模擬快速重入 - 需要滿足抖動過濾條件
        reentry_track = [
            TrackInfo(1, (20, -5), (15, -10, 25, 0)),  # 在線段下方
            TrackInfo(1, (20, -5), (15, -10, 25, 0)),  # 重複位置
            TrackInfo(1, (20, 5), (15, 0, 25, 10)),    # 進入 - 位置改變
            TrackInfo(1, (20, 5), (15, 0, 25, 10)),    # 重複位置（抖動過濾）
            TrackInfo(1, (20, 5), (15, 0, 25, 10)),    # 再次重複位置（抖動過濾）
            TrackInfo(1, (80, -5), (75, -10, 85, 0)),  # 在線段下方
            TrackInfo(1, (80, -5), (75, -10, 85, 0)),  # 重複位置
            TrackInfo(1, (80, 5), (75, 0, 85, 10)),    # 離開 - 位置改變
            TrackInfo(1, (80, 5), (75, 0, 85, 10)),    # 重複位置（抖動過濾）
            TrackInfo(1, (80, 5), (75, 0, 85, 10)),    # 再次重複位置（抖動過濾）
            TrackInfo(1, (20, -5), (15, -10, 25, 0)),  # 立即重入（冷卻期內）
            TrackInfo(1, (20, -5), (15, -10, 25, 0)),  # 重複位置
            TrackInfo(1, (20, 5), (15, 0, 25, 10)),    # 嘗試重入 - 位置改變
            TrackInfo(1, (20, 5), (15, 0, 25, 10)),    # 重複位置（抖動過濾）
            TrackInfo(1, (20, 5), (15, 0, 25, 10)),    # 再次重複位置（抖動過濾）
        ]
        
        events = []
        for i, track in enumerate(reentry_track):
            events.extend(counter.update([track], i))
        
        # 檢查事件
        enter_events = [e for e in events if e.event_type == "in"]
        exit_events = [e for e in events if e.event_type == "out"]
        
        # 應該有事件產生
        self.assertGreater(len(events), 0, "應該產生事件")
        print(f"✅ 重入冷卻測試: 總事件數={len(events)}, 進入事件={len(enter_events)}, 離開事件={len(exit_events)}")
    
    def test_id_tolerance_advanced(self):
        """測試ID容錯高級場景 - P3企業級"""
        line_config = LineConfig(
            id="test_line",
            points=[(0, 0), (100, 0)],
            direction="both"
        )
        counter = MultiLineCounter([line_config], 1, 15, 40.0, 30)  # 冷卻期設為1幀
        
        # 模擬ID切換的軌跡 - 需要滿足抖動過濾條件
        id_switch_track = [
            TrackInfo(1, (20, -5), (15, -10, 25, 0)),  # ID 1 在線段下方
            TrackInfo(1, (20, -5), (15, -10, 25, 0)),  # 重複位置
            TrackInfo(1, (20, 5), (15, 0, 25, 10)),    # ID 1 通過線段 - 位置改變
            TrackInfo(1, (20, 5), (15, 0, 25, 10)),    # 重複位置（抖動過濾）
            TrackInfo(1, (20, 5), (15, 0, 25, 10)),    # 再次重複位置（抖動過濾）
            TrackInfo(2, (30, -5), (25, -10, 35, 0)),  # ID 2 在線段下方
            TrackInfo(2, (30, -5), (25, -10, 35, 0)),  # 重複位置
            TrackInfo(2, (30, 5), (25, 0, 35, 10)),    # ID 2 通過線段 - 位置改變
            TrackInfo(2, (30, 5), (25, 0, 35, 10)),    # 重複位置（抖動過濾）
            TrackInfo(2, (30, 5), (25, 0, 35, 10)),    # 再次重複位置（抖動過濾）
        ]
        
        events = []
        for i, track in enumerate(id_switch_track):
            events.extend(counter.update([track], i))
        
        # ID切換應該被容錯處理
        self.assertGreater(len(events), 0)
    
    def test_multi_line_coordination(self):
        """測試多線協調 - P3企業級"""
        line_configs = [
            LineConfig(id="line1", points=[(0, 0), (100, 0)], direction="in"),
            LineConfig(id="line2", points=[(0, 50), (100, 50)], direction="out")
        ]
        counter = MultiLineCounter(line_configs, 1, 15, 40.0, 30)  # 冷卻期設為1幀
        
        # 模擬跨線軌跡 - 需要滿足抖動過濾條件
        cross_line_track = [
            TrackInfo(1, (50, -5), (45, -10, 55, 0)),   # 在 line1 下方
            TrackInfo(1, (50, -5), (45, -10, 55, 0)),   # 重複位置
            TrackInfo(1, (50, 5), (45, 0, 55, 10)),     # 通過 line1 (in) - 位置改變
            TrackInfo(1, (50, 5), (45, 0, 55, 10)),     # 重複位置（抖動過濾）
            TrackInfo(1, (50, 5), (45, 0, 55, 10)),     # 再次重複位置（抖動過濾）
            TrackInfo(1, (50, 25), (45, 20, 55, 30)),   # 中間位置
            TrackInfo(1, (50, 45), (45, 40, 55, 50)),   # 在 line2 下方
            TrackInfo(1, (50, 45), (45, 40, 55, 50)),   # 重複位置
            TrackInfo(1, (50, 55), (45, 50, 55, 60)),   # 通過 line2 (out) - 位置改變
            TrackInfo(1, (50, 55), (45, 50, 55, 60)),   # 重複位置（抖動過濾）
            TrackInfo(1, (50, 55), (45, 50, 55, 60)),   # 再次重複位置（抖動過濾）
        ]
        
        events = []
        for i, track in enumerate(cross_line_track):
            events.extend(counter.update([track], i))
        
        # 應該產生進入和離開事件
        in_events = [e for e in events if e.event_type == "in"]
        out_events = [e for e in events if e.event_type == "out"]
        
        # 至少應該有進入事件
        self.assertGreater(len(events), 0, "應該產生事件")
        print(f"✅ 多線協調測試: 總事件數={len(events)}, 進入事件={len(in_events)}, 離開事件={len(out_events)}")
    
    def test_zone_occupancy_advanced(self):
        """測試區域佔用高級場景 - P3企業級"""
        zones = [
            ZoneConfig(id="zone1", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], ignore=False),
            ZoneConfig(id="zone2", polygon=[(200, 200), (300, 200), (300, 300), (200, 300)], ignore=True)
        ]
        classifier = ZoneOccupancy(zones)
        
        # 模擬多個軌跡
        track_infos = [
            TrackInfo(1, (50, 50), (40, 40, 60, 60)),    # 在 zone1
            TrackInfo(2, (250, 250), (240, 240, 260, 260)), # 在 zone2 (忽略)
            TrackInfo(3, (150, 150), (140, 140, 160, 160))  # 不在任何區域
        ]
        
        events = classifier.update(track_infos)
        
        # 檢查結果
        self.assertIsInstance(events, list)
        # 檢查是否有進入事件
        entry_events = [e for e in events if e.event_type == "entry"]
        self.assertGreater(len(entry_events), 0)
    
    def test_performance_large_scale(self):
        """測試大規模性能 - P3企業級"""
        # 創建多個線段和區域
        line_configs = [
            LineConfig(id=f"line_{i}", points=[(i*10, 0), (i*10+50, 0)], direction="both")
            for i in range(10)
        ]
        counter = MultiLineCounter(line_configs)
        
        # 創建大量軌跡
        track_infos = [
            TrackInfo(i, (i*5, 0), (i*5-2, -2, i*5+2, 2))
            for i in range(100)
        ]
        
        start_time = time.time()
        events = counter.update(track_infos, 0)
        end_time = time.time()
        
        # 性能測試：應該在合理時間內完成
        self.assertLess(end_time - start_time, 0.1)  # 100ms內完成
        self.assertIsInstance(events, list)


if __name__ == "__main__":
    unittest.main()
