"""
Mask utilities - Mask处理工具
"""

import cv2
import numpy as np
import pycocotools.mask as maskUtils
from typing import List, Tuple, Dict


def rle_to_mask(rle_dict: Dict) -> np.ndarray:
    """
    将RLE格式转换为二值mask
    
    Args:
        rle_dict: RLE格式字典，包含'size'和'counts'
        
    Returns:
        二值mask
    """
    if isinstance(rle_dict['counts'], str):
        rle_dict = rle_dict.copy()
        rle_dict['counts'] = rle_dict['counts'].encode('utf-8')
    
    mask = maskUtils.decode(rle_dict)
    return mask.astype(bool)


def mask_to_rle(mask: np.ndarray) -> Dict:
    """
    将二值mask转换为RLE格式
    
    Args:
        mask: 二值mask
        
    Returns:
        RLE格式字典
    """
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(mask_uint8))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def mask_to_bbox(mask: np.ndarray) -> List[int]:
    """
    从mask提取边界框
    
    Args:
        mask: 二值mask
        
    Returns:
        边界框 [x1, y1, x2, y2]
    """
    ys, xs = np.where(mask > 0.5)
    
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 1, 1]  # 返回默认bbox
    
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def bbox_to_mask(image_shape: Tuple[int, int], 
                bbox: List[int]) -> np.ndarray:
    """
    从边界框生成二值mask
    
    Args:
        image_shape: 图像形状 (H, W)
        bbox: 边界框 [x1, y1, x2, y2]
        
    Returns:
        二值mask
    """
    mask = np.zeros(image_shape, dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_shape[1], x2)
    y2 = min(image_shape[0], y2)
    
    mask[y1:y2, x1:x2] = True
    return mask


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    合并多个mask为一个（并集）
    
    Args:
        masks: mask列表
        
    Returns:
        合并后的mask
    """
    if len(masks) == 0:
        return None
    
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined = np.logical_or(combined, mask)
    
    return combined


def intersect_masks(mask1: np.ndarray, 
                   mask2: np.ndarray) -> np.ndarray:
    """
    计算两个mask的交集
    
    Args:
        mask1: 第一个mask
        mask2: 第二个mask
        
    Returns:
        交集mask
    """
    return np.logical_and(mask1, mask2)


def union_masks(mask1: np.ndarray,
               mask2: np.ndarray) -> np.ndarray:
    """
    计算两个mask的并集
    
    Args:
        mask1: 第一个mask
        mask2: 第二个mask
        
    Returns:
        并集mask
    """
    return np.logical_or(mask1, mask2)


def compute_mask_area(mask: np.ndarray) -> int:
    """
    计算mask的面积（像素数）
    
    Args:
        mask: 二值mask
        
    Returns:
        面积
    """
    return np.count_nonzero(mask)


def compute_mask_iou(mask1: np.ndarray,
                    mask2: np.ndarray) -> float:
    """
    计算两个mask的IoU
    
    Args:
        mask1: 第一个mask
        mask2: 第二个mask
        
    Returns:
        IoU值 [0, 1]
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def dilate_mask(mask: np.ndarray, 
               kernel_size: int = 5,
               iterations: int = 1) -> np.ndarray:
    """
    膨胀mask
    
    Args:
        mask: 输入mask
        kernel_size: 内核大小
        iterations: 迭代次数
        
    Returns:
        膨胀后的mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    dilated = cv2.dilate(mask_uint8, kernel, iterations=iterations)
    return dilated.astype(bool)


def erode_mask(mask: np.ndarray,
              kernel_size: int = 5,
              iterations: int = 1) -> np.ndarray:
    """
    腐蚀mask
    
    Args:
        mask: 输入mask
        kernel_size: 内核大小
        iterations: 迭代次数
        
    Returns:
        腐蚀后的mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    eroded = cv2.erode(mask_uint8, kernel, iterations=iterations)
    return eroded.astype(bool)


def get_mask_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    从mask提取轮廓
    
    Args:
        mask: 二值mask
        
    Returns:
        轮廓列表
    """
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
