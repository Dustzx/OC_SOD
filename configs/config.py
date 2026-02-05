"""
OC_SOD 配置文件
"""

import os
from typing import Dict, Any


class Config:
    """基础配置类"""
    
    # 模型配置
    SAM2_CHECKPOINT = "facebook/sam2-hiera-large"
    VLM_MODEL_NAME = "qwen3-vl-8b-instruct"
    DEVICE = "cuda"  # cuda or cpu
    
    # 处理配置
    TARGET_SIZE = 1000
    CONFIDENCE_THRESHOLD = 0.5
    MAX_ITERATIONS = 3
    QUALITY_THRESHOLD = 0.85
    
    # API配置
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 输出配置
    OUTPUT_DIR = "./outputs"
    SAVE_VISUALIZATION = True
    SAVE_MASKS = True
    OUTPUT_FORMAT = "png"  # png or jpg
    
    # 数据配置
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # 颜色定义 (BGR格式)
    COLORS = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'purple': (255, 0, 255),
        'cyan': (0, 255, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
    }
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and key.isupper()
        }
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print("OC_SOD Configuration")
        print("=" * 60)
        for key, value in cls.to_dict().items():
            print(f"{key:.<40} {value}")
        print("=" * 60)


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    DEVICE = "cuda"


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    DEVICE = "cuda"
    MAX_ITERATIONS = 5


class CPUConfig(Config):
    """CPU环境配置"""
    DEVICE = "cpu"
    TARGET_SIZE = 512  # 减小尺寸以节省内存
    MAX_ITERATIONS = 2


# 环境变量确定当前配置
_env = os.getenv("OC_SOD_ENV", "development").lower()

if _env == "production":
    config = ProductionConfig()
elif _env == "cpu":
    config = CPUConfig()
else:
    config = DevelopmentConfig()
