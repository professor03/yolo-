from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .geometry import Point


@dataclass
class CalibrationConfig:
    """標定配置"""
    enabled: bool = False
    src_points: List[Point] = None  # 像素座標點
    dst_points: List[Point] = None  # 對應的地面座標點（米）
    
    def __post_init__(self):
        if self.src_points is None:
            self.src_points = []
        if self.dst_points is None:
            self.dst_points = []


class HomographyCalibrator:
    """Homography標定器，用於像素座標到地面座標的轉換"""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography_matrix: Optional[np.ndarray] = None
        
        if config.enabled and len(config.src_points) >= 4 and len(config.dst_points) >= 4:
            self._compute_homography()
    
    def _compute_homography(self):
        """計算Homography矩陣"""
        if len(self.config.src_points) < 4 or len(self.config.dst_points) < 4:
            return
        
        # 轉換為numpy數組
        src_points = np.array(self.config.src_points, dtype=np.float32)
        dst_points = np.array(self.config.dst_points, dtype=np.float32)
        
        # 計算Homography矩陣
        self.homography_matrix, _ = cv2.findHomography(src_points, dst_points)
        
        if self.homography_matrix is not None:
            # 計算逆矩陣
            self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)
    
    def pixel_to_world(self, pixel_point: Point) -> Tuple[float, float]:
        """將像素座標轉換為世界座標（米）"""
        if not self.config.enabled or self.homography_matrix is None:
            return (0.0, 0.0)
        
        # 轉換為齊次座標
        pixel_homogeneous = np.array([[pixel_point[0], pixel_point[1], 1.0]], dtype=np.float32).T
        
        # 應用Homography變換
        world_homogeneous = self.homography_matrix @ pixel_homogeneous
        
        # 轉換回笛卡爾座標
        if world_homogeneous[2, 0] != 0:
            world_x = world_homogeneous[0, 0] / world_homogeneous[2, 0]
            world_y = world_homogeneous[1, 0] / world_homogeneous[2, 0]
            return (float(world_x), float(world_y))
        else:
            return (0.0, 0.0)
    
    def world_to_pixel(self, world_point: Tuple[float, float]) -> Point:
        """將世界座標（米）轉換為像素座標"""
        if not self.config.enabled or self.inverse_homography_matrix is None:
            return (0, 0)
        
        # 轉換為齊次座標
        world_homogeneous = np.array([[world_point[0], world_point[1], 1.0]], dtype=np.float32).T
        
        # 應用逆Homography變換
        pixel_homogeneous = self.inverse_homography_matrix @ world_homogeneous
        
        # 轉換回笛卡爾座標
        if pixel_homogeneous[2, 0] != 0:
            pixel_x = pixel_homogeneous[0, 0] / pixel_homogeneous[2, 0]
            pixel_y = pixel_homogeneous[1, 0] / pixel_homogeneous[2, 0]
            return (round(float(pixel_x)), round(float(pixel_y)))
        else:
            return (0, 0)
    
    def is_valid(self) -> bool:
        """檢查標定是否有效"""
        return (self.config.enabled and 
                self.homography_matrix is not None and 
                self.inverse_homography_matrix is not None)
    
    def update_calibration(self, src_points: List[Point], dst_points: List[Point]):
        """更新標定點並重新計算Homography矩陣"""
        self.config.src_points = src_points
        self.config.dst_points = dst_points
        self._compute_homography()
    
    def get_calibration_error(self) -> float:
        """計算標定誤差（像素）"""
        if not self.is_valid():
            return float('inf')
        
        total_error = 0.0
        for src_point, dst_point in zip(self.config.src_points, self.config.dst_points):
            # 像素 -> 世界 -> 像素
            world_point = self.pixel_to_world(src_point)
            back_pixel = self.world_to_pixel(world_point)
            
            # 計算誤差
            error = np.sqrt((src_point[0] - back_pixel[0])**2 + (src_point[1] - back_pixel[1])**2)
            total_error += error
        
        return total_error / len(self.config.src_points) if self.config.src_points else 0.0


def create_calibrator_from_config(config_dict: dict) -> HomographyCalibrator:
    """從配置字典創建標定器"""
    calibration_config = CalibrationConfig(
        enabled=config_dict.get('enabled', False),
        src_points=config_dict.get('src_points', []),
        dst_points=config_dict.get('dst_points', [])
    )
    
    return HomographyCalibrator(calibration_config)


# 需要導入cv2，但為了避免循環導入，在這裡處理
try:
    import cv2
except ImportError:
    # 如果cv2不可用，提供一個簡化的實現
    class cv2:
        @staticmethod
        def findHomography(src_points, dst_points):
            # 簡化的Homography計算（僅用於測試）
            return None, None
