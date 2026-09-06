#!/usr/bin/env python3
"""
常數定義模組
統一管理系統中的所有常數和配置值
"""

from enum import Enum
from typing import Dict, List, Tuple

class ModelType(Enum):
    """模型類型"""
    YOLO_NANO = "yolov8n.pt"
    YOLO_MEDIUM = "yolov8m.pt"
    YOLO_LARGE = "yolov8l.pt"

class DeviceType(Enum):
    """設備類型"""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = ""

class DetectionClass(Enum):
    """檢測類別"""
    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORCYCLE = 3
    AIRPLANE = 4
    BUS = 5
    TRAIN = 6
    TRUCK = 7
    BOAT = 8
    TRAFFIC_LIGHT = 9
    FIRE_HYDRANT = 10
    STOP_SIGN = 11
    PARKING_METER = 12
    BENCH = 13
    BIRD = 14
    CAT = 15
    DOG = 16
    HORSE = 17
    SHEEP = 18
    COW = 19
    ELEPHANT = 20
    BEAR = 21
    ZEBRA = 22
    GIRAFFE = 23
    BACKPACK = 24
    UMBRELLA = 25
    HANDBAG = 26
    TIE = 27
    SUITCASE = 28
    FRISBEE = 29
    SKIS = 30
    SNOWBOARD = 31
    SPORTS_BALL = 32
    KITE = 33
    BASEBALL_BAT = 34
    BASEBALL_GLOVE = 35
    SKATEBOARD = 36
    SURFBOARD = 37
    TENNIS_RACKET = 38
    BOTTLE = 39
    WINE_GLASS = 40
    CUP = 41
    FORK = 42
    KNIFE = 43
    SPOON = 44
    BOWL = 45
    BANANA = 46
    APPLE = 47
    SANDWICH = 48
    ORANGE = 49
    BROCCOLI = 50
    CARROT = 51
    HOT_DOG = 52
    PIZZA = 53
    DONUT = 54
    CAKE = 55
    CHAIR = 56
    COUCH = 57
    POTTED_PLANT = 58
    BED = 59
    DINING_TABLE = 60
    TOILET = 61
    TV = 62
    LAPTOP = 63
    MOUSE = 64
    REMOTE = 65
    KEYBOARD = 66
    CELL_PHONE = 67
    MICROWAVE = 68
    OVEN = 69
    TOASTER = 70
    SINK = 71
    REFRIGERATOR = 72
    BOOK = 73
    CLOCK = 74
    VASE = 75
    SCISSORS = 76
    TEDDY_BEAR = 77
    HAIR_DRIER = 78
    TOOTHBRUSH = 79

class EventType(Enum):
    """事件類型"""
    LINE_CROSSING_IN = "in"
    LINE_CROSSING_OUT = "out"
    ZONE_ENTRY = "entry"
    ZONE_EXIT = "exit"
    ZONE_OCCUPANCY = "occupancy"
    ANOMALY_DETECTED = "anomaly"
    ALERT_TRIGGERED = "alert"

class AlertLevel(Enum):
    """警報級別"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class Direction(Enum):
    """方向"""
    IN = "in"
    OUT = "out"
    BOTH = "both"

class LogLevel(Enum):
    """日誌級別"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# 系統常數
class SystemConstants:
    """系統常數"""
    
    # 預設配置
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_IOU_THRESHOLD = 0.45
    DEFAULT_MAX_DETECTIONS = 1000
    
    # 性能設定
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_NUM_WORKERS = 4
    DEFAULT_QUEUE_SIZE = 10
    DEFAULT_FPS_TARGET = 30
    
    # 記憶體管理
    DEFAULT_MEMORY_THRESHOLD = 0.8
    DEFAULT_GC_THRESHOLD = 1000
    
    # 網路設定
    DEFAULT_API_PORT = 8000
    DEFAULT_WEBSOCKET_PORT = 8000
    DEFAULT_RTSP_PORT = 554
    DEFAULT_HTTP_PORT = 80
    
    # 檔案路徑
    DEFAULT_CONFIG_FILE = "config.yaml"
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_OUTPUT_DIR = "runs/people_detect"
    DEFAULT_MODEL_DIR = "models"
    DEFAULT_DATA_DIR = "data"
    
    # 資料庫設定
    DEFAULT_DB_FILE = "data/yolo_detection.db"
    DEFAULT_DB_POOL_SIZE = 10
    DEFAULT_DB_TIMEOUT = 30
    
    # 重試設定
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    DEFAULT_BACKOFF_FACTOR = 2.0
    
    # 超時設定
    DEFAULT_CONNECTION_TIMEOUT = 10.0
    DEFAULT_READ_TIMEOUT = 30.0
    DEFAULT_WRITE_TIMEOUT = 30.0

# 顏色常數 (BGR 格式)
class Colors:
    """顏色常數"""
    
    # 基本顏色
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    
    # 檢測框顏色
    PERSON_BOX = GREEN
    VEHICLE_BOX = BLUE
    ANIMAL_BOX = YELLOW
    OBJECT_BOX = CYAN
    
    # 線段顏色
    LINE_IN = GREEN
    LINE_OUT = RED
    LINE_BOTH = YELLOW
    
    # 區域顏色
    ROI_COLOR = (0, 255, 0)
    IGNORE_COLOR = (0, 0, 255)
    
    # 文字顏色
    TEXT_COLOR = WHITE
    TEXT_BACKGROUND = BLACK

# 字體常數
class Fonts:
    """字體常數"""
    
    DEFAULT_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    DEFAULT_SCALE = 0.5
    DEFAULT_THICKNESS = 1
    DEFAULT_LINE_TYPE = 8  # cv2.LINE_AA

# 鍵盤常數
class Keys:
    """鍵盤常數"""
    
    ESC = 27
    Q = ord('q')
    SPACE = 32
    ENTER = 13
    TAB = 9

# 錯誤代碼
class ErrorCodes:
    """錯誤代碼"""
    
    SUCCESS = 0
    GENERAL_ERROR = 1
    CONFIG_ERROR = 2
    MODEL_ERROR = 3
    CAMERA_ERROR = 4
    DATABASE_ERROR = 5
    NETWORK_ERROR = 6
    MEMORY_ERROR = 7
    PERMISSION_ERROR = 8
    FILE_ERROR = 9
    VALIDATION_ERROR = 10

# 狀態常數
class Status:
    """狀態常數"""
    
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    INITIALIZING = "initializing"
    SHUTTING_DOWN = "shutting_down"

# 配置預設值
DEFAULT_CONFIG = {
    "model": {
        "weights": ModelType.YOLO_NANO.value,
        "device": DeviceType.AUTO.value,
        "conf_threshold": SystemConstants.DEFAULT_CONFIDENCE_THRESHOLD,
        "iou_threshold": SystemConstants.DEFAULT_IOU_THRESHOLD,
        "max_det": SystemConstants.DEFAULT_MAX_DETECTIONS
    },
    "performance": {
        "batch_size": SystemConstants.DEFAULT_BATCH_SIZE,
        "num_workers": SystemConstants.DEFAULT_NUM_WORKERS,
        "queue_size": SystemConstants.DEFAULT_QUEUE_SIZE,
        "use_fp16": False,
        "pin_memory": True
    },
    "api": {
        "host": "0.0.0.0",
        "port": SystemConstants.DEFAULT_API_PORT,
        "enabled": True
    },
    "logging": {
        "level": LogLevel.INFO.value,
        "console": True,
        "file": True,
        "json_format": False
    }
}

# 支援的檔案格式
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
SUPPORTED_CONFIG_FORMATS = ['.yaml', '.yml', '.json']

# 支援的串流協議
SUPPORTED_STREAM_PROTOCOLS = ['rtsp', 'http', 'https', 'tcp', 'udp']

# 支援的資料庫類型
SUPPORTED_DATABASES = ['sqlite', 'postgresql', 'mysql', 'mariadb']

# 支援的模型格式
SUPPORTED_MODEL_FORMATS = ['.pt', '.onnx', '.engine', '.trt', '.xml', '.bin']

# 預設的 ROI 和線段配置
DEFAULT_LINES = [
    {
        "id": "line_1",
        "points": [(100, 200), (500, 200)],
        "direction": Direction.BOTH.value,
        "uturn_cooldown": 30,
        "jitter_frames": 2
    }
]

DEFAULT_ROIS = [
    {
        "id": "roi_1",
        "points": [(50, 50), (600, 50), (600, 400), (50, 400)],
        "ignore": False
    }
]

DEFAULT_ZONES = [
    {
        "id": "zone_1",
        "points": [(100, 100), (300, 100), (300, 300), (100, 300)],
        "ignore": False
    }
]
