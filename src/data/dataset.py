"""
Data module - 数据处理和加载
"""

import json
import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class ImageDataset:
    """
    简单的图像数据集加载器
    """
    
    def __init__(self,
                 image_dir: str,
                 annotation_file: Optional[str] = None):
        """
        初始化数据集
        
        Args:
            image_dir: 图像目录
            annotation_file: 标注文件路径（JSON格式）
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.annotations = {}
        self.image_files = []
        
        self._load_images()
        if annotation_file and os.path.exists(annotation_file):
            self._load_annotations()
    
    def _load_images(self):
        """加载图像文件列表"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for file in os.listdir(self.image_dir):
            if Path(file).suffix.lower() in image_extensions:
                self.image_files.append(file)
        
        print(f"加载了{len(self.image_files)}个图像文件")
    
    def _load_annotations(self):
        """加载标注文件"""
        if not os.path.exists(self.annotation_file):
            return
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        print(f"加载了{len(self.annotations)}个标注")
    
    def __len__(self) -> int:
        """获取数据集大小"""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取数据集项
        
        Args:
            idx: 索引
            
        Returns:
            包含图像和标注的字典
        """
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"无法加载图像: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取标注
        item_id = Path(image_file).stem
        annotation = self.annotations.get(item_id, {})
        
        return {
            "image": image,
            "image_path": image_path,
            "image_id": item_id,
            "annotation": annotation
        }
    
    def get_by_id(self, image_id: str) -> Dict:
        """
        通过ID获取数据
        
        Args:
            image_id: 图像ID
            
        Returns:
            数据字典
        """
        # 查找匹配的文件
        for file in self.image_files:
            if Path(file).stem == image_id:
                image_path = os.path.join(self.image_dir, file)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    annotation = self.annotations.get(image_id, {})
                    return {
                        "image": image,
                        "image_path": image_path,
                        "image_id": image_id,
                        "annotation": annotation
                    }
        
        return None


class AnnotationLoader:
    """
    标注文件加载器，支持多种格式
    """
    
    @staticmethod
    def load_json(file_path: str) -> Dict:
        """加载JSON标注文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Dict, file_path: str):
        """保存为JSON格式"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_coco_format(annotation_file: str) -> Dict:
        """加载COCO格式标注"""
        with open(annotation_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 转换为简单的ID->标注映射
        annotations = {}
        for img_info in coco_data.get('images', []):
            img_id = img_info['id']
            annotations[str(img_id)] = {
                'image_id': img_id,
                'file_name': img_info.get('file_name'),
                'height': img_info.get('height'),
                'width': img_info.get('width'),
                'annotations': []
            }
        
        # 添加对象标注
        for ann in coco_data.get('annotations', []):
            img_id = str(ann['image_id'])
            if img_id in annotations:
                annotations[img_id]['annotations'].append(ann)
        
        return annotations


__all__ = [
    'ImageDataset',
    'AnnotationLoader',
]
