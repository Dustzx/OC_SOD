"""
SAM2 Model Wrapper - Segment Anything Model 2
处理图像分割的核心模型
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


class SAM2Model:
    """
    SAM2模型的封装类，提供统一的分割接口
    """
    
    def __init__(self, model_name: str = "facebook/sam2-hiera-large", device: str = "cuda"):
        """
        初始化SAM2模型
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """加载SAM2预训练模型"""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
            print(f"✓ SAM2模型加载成功: {self.model_name}")
        except ImportError:
            raise ImportError("请安装SAM2: pip install git+https://github.com/facebookresearch/sam2.git")
        except Exception as e:
            raise RuntimeError(f"SAM2模型加载失败: {e}")
    
    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/sam2-hiera-large", **kwargs):
        """
        从预训练模型加载
        
        Args:
            model_name: 模型名称或路径
            **kwargs: 其他参数
            
        Returns:
            SAM2Model实例
        """
        return cls(model_name=model_name, **kwargs)
    
    def set_image(self, image: np.ndarray):
        """
        设置要分割的图像（只需设置一次）
        
        Args:
            image: 输入图像，shape为 (H, W, 3)
        """
        if self.predictor is None:
            raise RuntimeError("模型未加载")
        self.predictor.set_image(image)
    
    def predict(self, 
                bbox: Optional[np.ndarray] = None,
                points: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                multimask_output: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对给定输入进行分割预测
        
        Args:
            bbox: 边界框 [x1, y1, x2, y2]
            points: 点坐标，shape为 (N, 2)
            point_labels: 点标签，shape为 (N,)，1表示前景，0表示背景
            multimask_output: 是否返回多个mask
            
        Returns:
            masks: 分割mask，shape为 (N, H, W)
            iou_preds: 预测的IoU，shape为 (N,)
            low_res_logits: 低分辨率logits
        """
        if self.predictor is None:
            raise RuntimeError("模型未加载")
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, iou_preds, low_res_logits = self.predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                box=bbox[None, :] if bbox is not None else None,
                multimask_output=multimask_output,
            )
        
        return masks, iou_preds, low_res_logits
    
    def segment_bboxes(self, image: np.ndarray, bboxes: List[List[int]]) -> List[np.ndarray]:
        """
        对多个边界框进行分割
        
        Args:
            image: 输入图像
            bboxes: 边界框列表，每个元素为 [x1, y1, x2, y2]
            
        Returns:
            masks列表，每个mask对应一个bbox
        """
        self.set_image(image)
        masks = []
        
        for bbox in bboxes:
            try:
                mask, _, _ = self.predict(bbox=np.array(bbox))
                masks.append((mask[0] > 0.5).astype(bool))
            except Exception as e:
                print(f"警告: 分割bbox {bbox} 失败: {e}")
                # 返回默认简单mask
                simple_mask = self._create_simple_mask(image.shape[:2], bbox)
                masks.append(simple_mask)
        
        return masks
    
    def _create_simple_mask(self, image_shape: Tuple[int, int], bbox: List[int]) -> np.ndarray:
        """
        创建简单的矩形mask（当分割失败时使用）
        
        Args:
            image_shape: 图像形状 (H, W)
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            布尔mask
        """
        mask = np.zeros(image_shape, dtype=bool)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_shape[1], x2), min(image_shape[0], y2)
        mask[y1:y2, x1:x2] = True
        return mask
    
    def reset(self):
        """重置模型状态"""
        if self.predictor is not None:
            self.predictor.reset_image()
    
    def __del__(self):
        """清理资源"""
        self.reset()
