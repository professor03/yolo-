# -*- coding: utf-8 -*-
"""
多攝影機配置管理器
支援從配置文件載入多組 IP 攝影機設定
"""

import yaml
import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """攝影機配置數據類"""
    name: str
    enabled: bool
    rtsp_url: str
    username: str
    password: str
    width: int = 1280
    height: int = 720
    fps: int = 15
    detection_enabled: bool = True
    recording_enabled: bool = False
    audio_enabled: bool = False

class MultiCameraConfigManager:
    """多攝影機配置管理器"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.cameras: Dict[str, CameraConfig] = {}
        self.config_data: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> bool:
        """載入配置文件"""
        try:
            if not os.path.exists(self.config_file):
                logger.error(f"配置文件不存在: {self.config_file}")
                return False
                
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            
            # 載入攝影機配置
            self._load_camera_configs()
            logger.info(f"成功載入配置文件: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"載入配置文件失敗: {e}")
            return False
    
    def _load_camera_configs(self):
        """載入攝影機配置"""
        try:
            cameras_config = self.config_data.get('streaming', {}).get('cameras', {})
            
            for camera_id, camera_data in cameras_config.items():
                config = CameraConfig(
                    name=camera_data.get('name', camera_id),
                    enabled=camera_data.get('enabled', True),
                    rtsp_url=camera_data.get('rtsp_url', ''),
                    username=camera_data.get('username', ''),
                    password=camera_data.get('password', ''),
                    width=camera_data.get('width', 1280),
                    height=camera_data.get('height', 720),
                    fps=camera_data.get('fps', 15),
                    detection_enabled=camera_data.get('detection_enabled', True),
                    recording_enabled=camera_data.get('recording_enabled', False),
                    audio_enabled=camera_data.get('audio_enabled', False)
                )
                
                self.cameras[camera_id] = config
                logger.info(f"載入攝影機配置: {config.name} -> {config.rtsp_url}")
                
        except Exception as e:
            logger.error(f"載入攝影機配置失敗: {e}")
    
    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """獲取攝影機配置"""
        return self.cameras.get(camera_id)
    
    def get_all_cameras(self) -> Dict[str, CameraConfig]:
        """獲取所有攝影機配置"""
        return self.cameras.copy()
    
    def get_enabled_cameras(self) -> Dict[str, CameraConfig]:
        """獲取啟用的攝影機配置"""
        return {k: v for k, v in self.cameras.items() if v.enabled}
    
    def add_camera(self, camera_id: str, config: CameraConfig) -> bool:
        """添加攝影機配置"""
        try:
            self.cameras[camera_id] = config
            
            # 更新配置文件
            if 'streaming' not in self.config_data:
                self.config_data['streaming'] = {}
            if 'cameras' not in self.config_data['streaming']:
                self.config_data['streaming']['cameras'] = {}
                
            self.config_data['streaming']['cameras'][camera_id] = asdict(config)
            
            logger.info(f"添加攝影機配置: {camera_id} -> {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"添加攝影機配置失敗: {e}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """移除攝影機配置"""
        try:
            if camera_id in self.cameras:
                del self.cameras[camera_id]
                
                # 更新配置文件
                if 'streaming' in self.config_data and 'cameras' in self.config_data['streaming']:
                    if camera_id in self.config_data['streaming']['cameras']:
                        del self.config_data['streaming']['cameras'][camera_id]
                
                logger.info(f"移除攝影機配置: {camera_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"移除攝影機配置失敗: {e}")
            return False
    
    def update_camera(self, camera_id: str, config: CameraConfig) -> bool:
        """更新攝影機配置"""
        try:
            if camera_id in self.cameras:
                self.cameras[camera_id] = config
                
                # 更新配置文件
                if 'streaming' in self.config_data and 'cameras' in self.config_data['streaming']:
                    self.config_data['streaming']['cameras'][camera_id] = asdict(config)
                
                logger.info(f"更新攝影機配置: {camera_id} -> {config.name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"更新攝影機配置失敗: {e}")
            return False
    
    def save_config(self) -> bool:
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置文件已保存: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件失敗: {e}")
            return False
    
    def get_camera_list(self) -> List[Dict[str, Any]]:
        """獲取攝影機列表（用於 API）"""
        camera_list = []
        for camera_id, config in self.cameras.items():
            camera_list.append({
                "id": camera_id,
                "name": config.name,
                "enabled": config.enabled,
                "rtsp_url": config.rtsp_url,
                "username": config.username,
                "width": config.width,
                "height": config.height,
                "fps": config.fps,
                "detection_enabled": config.detection_enabled,
                "recording_enabled": config.recording_enabled,
                "audio_enabled": config.audio_enabled
            })
        return camera_list
    
    def validate_camera_config(self, config: CameraConfig) -> List[str]:
        """驗證攝影機配置"""
        errors = []
        
        if not config.name:
            errors.append("攝影機名稱不能為空")
        
        if not config.rtsp_url:
            errors.append("RTSP URL 不能為空")
        elif not config.rtsp_url.startswith(('rtsp://', 'http://', 'https://')):
            errors.append("RTSP URL 格式不正確")
        
        if config.width <= 0 or config.height <= 0:
            errors.append("解析度必須大於 0")
        
        if config.fps <= 0:
            errors.append("FPS 必須大於 0")
        
        return errors
    
    def get_system_config(self) -> Dict[str, Any]:
        """獲取系統配置"""
        return self.config_data.get('streaming', {}).get('multi_camera', {})
    
    def reload_config(self) -> bool:
        """重新載入配置文件"""
        self.cameras.clear()
        return self.load_config()

# 全局配置管理器實例
config_manager = MultiCameraConfigManager()

def get_config_manager() -> MultiCameraConfigManager:
    """獲取配置管理器實例"""
    return config_manager

# 使用示例
if __name__ == "__main__":
    # 創建配置管理器
    manager = MultiCameraConfigManager()
    
    # 獲取所有攝影機
    cameras = manager.get_all_cameras()
    print(f"載入的攝影機數量: {len(cameras)}")
    
    for camera_id, config in cameras.items():
        print(f"攝影機 {camera_id}: {config.name} -> {config.rtsp_url}")
    
    # 獲取啟用的攝影機
    enabled_cameras = manager.get_enabled_cameras()
    print(f"啟用的攝影機數量: {len(enabled_cameras)}")
    
    # 獲取攝影機列表
    camera_list = manager.get_camera_list()
    print(f"攝影機列表: {camera_list}")
