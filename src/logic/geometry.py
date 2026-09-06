from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass
class Point2D:
    """二維點類別，提供更多幾何運算功能"""
    x: float
    y: float
    
    def __add__(self, other: Point2D) -> Point2D:
        return Point2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> Point2D:
        return Point2D(self.x * scalar, self.y * scalar)
    
    def distance_to(self, other: Point2D) -> float:
        """計算到另一點的距離"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_tuple(self) -> Point:
        return (self.x, self.y)
    
    @classmethod
    def from_tuple(cls, point: Point) -> Point2D:
        return cls(point[0], point[1])


def centroid(points: Sequence[Point]) -> Point:
    """計算點集的質心"""
    if not points:
        raise ValueError("points cannot be empty")
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    n = len(points)
    return (sx / n, sy / n)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """判斷點是否在多邊形內部（射線法）"""
    if hasattr(point, 'x') and hasattr(point, 'y'):
        x, y = point.x, point.y
    else:
        x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def line_side(point: Point, line: Tuple[Point, Point]) -> float:
    """計算點相對於線段的位置
    返回值: >0 在左側, <0 在右側, =0 在線上
    """
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """計算兩線段交點
    返回交點或None（如果線段不相交）
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # 平行線
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    
    return None


def line_segment_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """判斷兩線段是否相交"""
    return line_intersection(p1, p2, p3, p4) is not None


def point_to_line_distance(point: Point, line_start: Point, line_end: Point) -> float:
    """計算點到線段的最短距離"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:
        return math.sqrt(A * A + B * B)
    
    param = dot / len_sq
    
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    
    dx = px - xx
    dy = py - yy
    return math.sqrt(dx * dx + dy * dy)


def vector_angle_deg(a: Point, b: Point) -> float:
    """計算兩向量間的角度（度）"""
    ax, ay = a
    bx, by = b
    norm_a = math.hypot(ax, ay)
    norm_b = math.hypot(bx, by)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    dot = (ax * bx + ay * by) / (norm_a * norm_b)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def uturn_detected(history: Sequence[Point], threshold_deg: float = 120.0) -> bool:
    """檢測U-turn行為
    history: 軌跡點列表（按時間順序）
    threshold_deg: 角度變化閾值（度）
    """
    if len(history) < 3:
        return False
    
    # 計算方向向量
    vectors = []
    for i in range(1, len(history)):
        dx = history[i][0] - history[i-1][0]
        dy = history[i][1] - history[i-1][1]
        vectors.append((dx, dy))
    
    if len(vectors) < 2:
        return False
    
    # 計算角度變化
    for i in range(1, len(vectors)):
        v1 = vectors[i-1]
        v2 = vectors[i]
        
        angle = vector_angle_deg(v1, v2)
        if angle > threshold_deg:
            return True
    
    return False


def polygon_area(polygon: Sequence[Point]) -> float:
    """計算多邊形面積（Shoelace公式）"""
    if len(polygon) < 3:
        return 0.0
    
    area = 0.0
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    
    return abs(area) / 2.0


def is_convex_polygon(polygon: Sequence[Point]) -> bool:
    """判斷多邊形是否為凸多邊形"""
    if len(polygon) < 3:
        return False
    
    n = len(polygon)
    prev_cross = 0
    
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        p3 = polygon[(i + 2) % n]
        
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        
        if cross != 0:
            if prev_cross == 0:
                prev_cross = cross
            elif cross * prev_cross < 0:
                return False
    
    return True


def bounding_box(points: Sequence[Point]) -> Tuple[Point, Point]:
    """計算點集的邊界框"""
    if not points:
        return (0, 0), (0, 0)
    
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    
    return (min_x, min_y), (max_x, max_y)


def point_in_rectangle(point: Point, rect: Tuple[Point, Point]) -> bool:
    """判斷點是否在矩形內"""
    (min_x, min_y), (max_x, max_y) = rect
    x, y = point
    return min_x <= x <= max_x and min_y <= y <= max_y


def distance_between_points(p1: Point, p2: Point) -> float:
    """計算兩點間距離"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def normalize_vector(v: Point) -> Point:
    """正規化向量"""
    x, y = v
    length = math.sqrt(x*x + y*y)
    if length < 1e-10:
        return (0, 0)
    return (x/length, y/length)


def dot_product(v1: Point, v2: Point) -> float:
    """計算兩向量的點積"""
    return v1[0] * v2[0] + v1[1] * v2[1]


def cross_product(v1: Point, v2: Point) -> float:
    """計算兩向量的叉積（2D）"""
    return v1[0] * v2[1] - v1[1] * v2[0]