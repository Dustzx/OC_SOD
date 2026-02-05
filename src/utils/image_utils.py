"""
Image utilities - 图像处理工具
"""

import cv2
import numpy as np
from typing import Tuple, List


def resize_image_with_padding(image: np.ndarray, 
                             target_size: int = 1000) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
    """
    将图像缩放到目标大小并添加填充以保持宽高比
    
    Args:
        image: 输入图像
        target_size: 目标大小
        
    Returns:
        缩放后的图像, 缩放因子, 填充参数(top, bottom, left, right)
    """
    h, w = image.shape[:2]
    
    # 计算缩放因子
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h))
    
    # 计算填充
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    
    # 添加填充
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    
    return padded, scale, (top, bottom, left, right)


def map_coords_to_original(bbox_resized: List[int], 
                          scale: float, 
                          padding: Tuple[int, int, int, int],
                          original_size: Tuple[int, int]) -> List[int]:
    """
    将缩放后图像中的坐标映射回原始图像坐标
    
    Args:
        bbox_resized: 缩放图像中的边界框 [x1, y1, x2, y2]
        scale: 缩放因子
        padding: 填充参数 (top, bottom, left, right)
        original_size: 原始图像大小 (width, height)
        
    Returns:
        原始图像坐标系中的边界框
    """
    top, bottom, left, right = padding
    x1, y1, x2, y2 = bbox_resized
    
    # 去除填充
    x1 = max(0, x1 - left)
    y1 = max(0, y1 - top)
    x2 = min(original_size[0], (x2 - left) if x2 > left else 0)
    y2 = min(original_size[1], (y2 - top) if y2 > top else 0)
    
    # 反向缩放
    x1 = int(x1 / scale)
    y1 = int(y1 / scale)
    x2 = int(x2 / scale)
    y2 = int(y2 / scale)
    
    # 确保在原始图像范围内
    x1 = max(0, min(x1, original_size[0]))
    y1 = max(0, min(y1, original_size[1]))
    x2 = max(0, min(x2, original_size[0]))
    y2 = max(0, min(y2, original_size[1]))
    
    return [x1, y1, x2, y2]


def draw_bbox(image: np.ndarray, 
              bbox: List[int], 
              color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制边界框
    
    Args:
        image: 输入图像
        bbox: 边界框 [x1, y1, x2, y2]
        color: 颜色 (B, G, R)
        thickness: 线条厚度
        
    Returns:
        绘制后的图像
    """
    vis_image = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
    return vis_image


def draw_bboxes(image: np.ndarray,
                bboxes: List[List[int]],
                colors: List[Tuple[int, int, int]] = None,
                thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制多个边界框
    
    Args:
        image: 输入图像
        bboxes: 边界框列表
        colors: 颜色列表
        thickness: 线条厚度
        
    Returns:
        绘制后的图像
    """
    if colors is None:
        colors = [(0, 255, 0)] * len(bboxes)
    
    vis_image = image.copy()
    for bbox, color in zip(bboxes, colors):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
    
    return vis_image


def draw_mask_contour(image: np.ndarray,
                     mask: np.ndarray,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制mask的轮廓
    
    Args:
        image: 输入图像
        mask: 二值mask
        color: 颜色
        thickness: 线条厚度
        
    Returns:
        绘制后的图像
    """
    vis_image = image.copy()
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, color, thickness)
    return vis_image


def overlay_mask(image: np.ndarray,
                mask: np.ndarray,
                color: Tuple[int, int, int] = (0, 255, 0),
                alpha: float = 0.3) -> np.ndarray:
    """
    在图像上叠加mask
    
    Args:
        image: 输入图像
        mask: 二值mask
        color: 颜色
        alpha: 透明度
        
    Returns:
        叠加后的图像
    """
    vis_image = image.copy()
    overlay = vis_image.copy()
    
    mask_bool = (mask > 0.5).astype(bool)
    overlay_color = np.zeros_like(vis_image)
    overlay_color[:, :] = color
    overlay[mask_bool] = overlay_color[mask_bool]
    
    cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)
    return vis_image


def crop_image(image: np.ndarray, 
              bbox: List[int]) -> np.ndarray:
    """
    按边界框裁剪图像
    
    Args:
        image: 输入图像
        bbox: 边界框 [x1, y1, x2, y2]
        
    Returns:
        裁剪后的图像
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    return image[y1:y2, x1:x2]
