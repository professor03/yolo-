#!/usr/bin/env python3
"""
配置驗證模組
提供配置文件的驗證和預設值設定
"""

import yaml
import os
import secrets
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """配置驗證器"""
    
    def __init__(self):
        self.required_sections = [
            'model', 'detection', 'performance', 'streaming'
        ]
        
        self.default_config = {
            'model': {
                'weights': 'yolov8n.pt',
                'device': '',
                'conf_threshold': 0.5,
                'iou_threshold': 0.45,
                'max_det': 1000
            },
            'detection': {
                'conf_threshold': 0.5,
                'iou_threshold': 0.45,
                'max_det': 1000
            },
            'performance': {
                'use_fp16': False,
                'batch_size': 1,
                'num_workers': 4,
                'pin_memory': True,
                'queue_size': 10,
                'drop_frame_policy': 'drop_oldest',
                'max_queue_wait': 0.1,
                'enable_queue_monitoring': True,
                'enable_gpu_monitoring': True,
                'enable_cpu_monitoring': True,
                'enable_memory_monitoring': True,
                'metrics_interval': 1.0,
                'inference_optimization': {
                    'enable_tensorrt': False,
                    'enable_onnx': False,
                    'enable_openvino': False,
                    'precision': 'fp32'
                },
                'memory_management': {
                    'max_memory_usage': 0.8,
                    'gc_threshold': 1000,
                    'enable_memory_pool': True
                }
            },
            'streaming': {
                'rtsp': {
                    'enabled': False,
                    'url': '',
                    'username': '',
                    'password': '',
                    'reconnect_interval': 5,
                    'buffer_size': 1
                },
                'usb_camera': {
                    'enabled': False,
                    'device_id': 0,
                    'resolution': [1920, 1080],
                    'fps': 30
                },
                'ip_camera': {
                    'enabled': False,
                    'url': '',
                    'username': '',
                    'password': ''
                },
                'realtime': {
                    'enable_tracking': True,
                    'enable_counting': True,
                    'enable_roi': True,
                    'enable_calibration': True,
                    'save_video': False,
                    'save_events': True,
                    'show_display': True
                }
            },
            'security': {
                'enabled': True,
                'jwt_secret_key': os.getenv('JWT_SECRET') or secrets.token_urlsafe(48),
                'jwt_algorithm': 'HS256',
                'access_token_expire_minutes': 30,
                'refresh_token_expire_days': 7,
                'users': {},
                'api_keys': []
            },
            'cors': {
                'enabled': True,
                'allow_origins': [
                    'http://localhost:3000',
                    'http://localhost:8080',
                    'http://127.0.0.1:3000',
                    'http://127.0.0.1:8080'
                ],
                'allow_credentials': True,
                'allow_methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
                'allow_headers': [
                    'Accept', 'Accept-Language', 'Content-Language',
                    'Content-Type', 'Authorization', 'X-API-Key'
                ]
            },
            'machine_learning': {
                'anomaly_detection': {
                    'enabled': True,
                    'history_size': 1000,
                    'anomaly_threshold': 0.7,
                    'crowd_threshold': 5,
                    'stay_threshold': 10.0,
                    'speed_threshold': 50.0
                },
                'traffic_prediction': {
                    'enabled': True,
                    'prediction_horizon': 60,
                    'history_window': 300,
                    'update_interval': 30
                },
                'behavior_analysis': {
                    'enabled': True,
                    'analysis_interval': 10,
                    'pattern_window': 60,
                    'min_samples': 10
                },
                'adaptive_optimization': {
                    'enabled': True,
                    'optimization_interval': 30,
                    'performance_window': 120,
                    'learning_rate': 0.01
                }
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """驗證並修復配置"""
        validated_config = self._deep_merge(self.default_config, config)
        
        # 驗證必要欄位
        self._validate_required_fields(validated_config)
        
        # 驗證數值範圍
        self._validate_value_ranges(validated_config)
        
        # 驗證檔案路徑
        self._validate_file_paths(validated_config)
        
        # 驗證網路設定
        self._validate_network_settings(validated_config)
        
        logger.info("配置驗證完成")
        return validated_config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合併字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_required_fields(self, config: Dict[str, Any]):
        """驗證必要欄位"""
        for section in self.required_sections:
            if section not in config:
                logger.warning(f"缺少必要配置區段: {section}")
                config[section] = self.default_config[section]
    
    def _validate_value_ranges(self, config: Dict[str, Any]):
        """驗證數值範圍"""
        # 驗證檢測閾值
        if 'detection' in config:
            detection = config['detection']
            detection['conf_threshold'] = max(0.0, min(1.0, detection.get('conf_threshold', 0.5)))
            detection['iou_threshold'] = max(0.0, min(1.0, detection.get('iou_threshold', 0.45)))
            detection['max_det'] = max(1, detection.get('max_det', 1000))
        
        # 驗證性能設定
        if 'performance' in config:
            perf = config['performance']
            perf['batch_size'] = max(1, perf.get('batch_size', 1))
            perf['num_workers'] = max(1, perf.get('num_workers', 4))
            perf['queue_size'] = max(1, perf.get('queue_size', 10))
            perf['metrics_interval'] = max(0.1, perf.get('metrics_interval', 1.0))
    
    def _validate_file_paths(self, config: Dict[str, Any]):
        """驗證檔案路徑"""
        if 'model' in config and 'weights' in config['model']:
            weights_path = config['model']['weights']
            if not os.path.exists(weights_path):
                logger.warning(f"模型檔案不存在: {weights_path}")
                # 嘗試下載或使用預設模型
                config['model']['weights'] = 'yolov8n.pt'
    
    def _validate_network_settings(self, config: Dict[str, Any]):
        """驗證網路設定"""
        if 'streaming' in config:
            streaming = config['streaming']
            
            # 驗證 RTSP 設定
            if 'rtsp' in streaming and streaming['rtsp'].get('enabled', False):
                rtsp = streaming['rtsp']
                if not rtsp.get('url'):
                    logger.warning("RTSP 已啟用但未設定 URL")
                    streaming['rtsp']['enabled'] = False
            
            # 驗證 USB 攝影機設定
            if 'usb_camera' in streaming and streaming['usb_camera'].get('enabled', False):
                usb = streaming['usb_camera']
                device_id = usb.get('device_id', 0)
                if not isinstance(device_id, int) or device_id < 0:
                    logger.warning(f"無效的 USB 設備 ID: {device_id}")
                    streaming['usb_camera']['device_id'] = 0
    
    def create_default_config(self, file_path: str = "config.yaml"):
        """創建預設配置文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.default_config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"預設配置文件已創建: {file_path}")
    
    def load_and_validate_config(self, file_path: str = "config.yaml") -> Dict[str, Any]:
        """載入並驗證配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            return self.validate_config(config)
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {file_path}，使用預設配置")
            self.create_default_config(file_path)
            return self.default_config
        except yaml.YAMLError as e:
            logger.error(f"配置文件格式錯誤: {e}")
            return self.default_config
        except Exception as e:
            logger.error(f"載入配置文件失敗: {e}")
            return self.default_config
