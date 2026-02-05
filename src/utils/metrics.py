"""
Metrics - 评估指标
"""

import numpy as np
from typing import List, Tuple


def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        mask_pred: 预测的mask
        mask_gt: 真值mask
        
    Returns:
        IoU值 [0, 1]
    """
    mask_pred = (mask_pred > 0.5).astype(bool)
    mask_gt = (mask_gt > 0.5).astype(bool)
    
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_bbox_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    计算两个边界框的IoU
    
    Args:
        bbox1: 第一个边界框 [x1, y1, x2, y2]
        bbox2: 第二个边界框 [x1, y1, x2, y2]
        
    Returns:
        IoU值 [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 计算交集
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # 计算并集
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_giou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    计算GIoU (Generalized IoU)
    
    Args:
        bbox1: 第一个边界框 [x1, y1, x2, y2]
        bbox2: 第二个边界框 [x1, y1, x2, y2]
        
    Returns:
        GIoU值 [-1, 1]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 计算IoU
    iou = compute_bbox_iou(bbox1, bbox2)
    
    # 计算最小包含框
    enclose_xmin = min(x1_min, x2_min)
    enclose_ymin = min(y1_min, y2_min)
    enclose_xmax = max(x1_max, x2_max)
    enclose_ymax = max(y1_max, y2_max)
    
    enclose_area = (enclose_xmax - enclose_xmin) * (enclose_ymax - enclose_ymin)
    
    # 计算GIoU
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - (
        max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) *
        max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    )
    
    if enclose_area == 0:
        return iou
    
    giou = iou - (enclose_area - union_area) / enclose_area
    return giou


def compute_ciou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    计算CIoU (Complete IoU)
    
    Args:
        bbox1: 第一个边界框 [x1, y1, x2, y2]
        bbox2: 第二个边界框 [x1, y1, x2, y2]
        
    Returns:
        CIoU值 [-1, 1]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 计算IoU
    iou = compute_bbox_iou(bbox1, bbox2)
    
    if iou == 0:
        return 0.0
    
    # 计算中心距离
    c1_x = (x1_min + x1_max) / 2
    c1_y = (y1_min + y1_max) / 2
    c2_x = (x2_min + x2_max) / 2
    c2_y = (y2_min + y2_max) / 2
    
    center_distance_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2
    
    # 计算对角线距离
    c_x_min = min(x1_min, x2_min)
    c_y_min = min(y1_min, y2_min)
    c_x_max = max(x1_max, x2_max)
    c_y_max = max(y1_max, y2_max)
    
    diagonal_distance_sq = (c_x_max - c_x_min) ** 2 + (c_y_max - c_y_min) ** 2
    
    # 计算宽高比
    w1 = x1_max - x1_min
    h1 = y1_max - y1_min
    w2 = x2_max - x2_min
    h2 = y2_max - y2_min
    
    # 计算atan2距离
    atan1 = np.arctan(w1 / h1) if h1 != 0 else 0
    atan2 = np.arctan(w2 / h2) if h2 != 0 else 0
    atan_distance = (2 / np.pi ** 2) * ((atan1 - atan2) ** 2)
    
    # 计算CIoU
    if diagonal_distance_sq == 0:
        ciou = iou
    else:
        ciou = iou - (center_distance_sq / diagonal_distance_sq + atan_distance)
    
    return ciou


def compute_dice(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    计算Dice系数 (F1 Score)
    
    Args:
        mask_pred: 预测的mask
        mask_gt: 真值mask
        
    Returns:
        Dice值 [0, 1]
    """
    mask_pred = (mask_pred > 0.5).astype(bool)
    mask_gt = (mask_gt > 0.5).astype(bool)
    
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    
    if mask_pred.sum() + mask_gt.sum() == 0:
        return 1.0 if np.array_equal(mask_pred, mask_gt) else 0.0
    
    return 2 * intersection / (mask_pred.sum() + mask_gt.sum())


def compute_mae(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    Args:
        mask_pred: 预测的mask
        mask_gt: 真值mask
        
    Returns:
        MAE值 [0, 1]
    """
    mask_pred = (mask_pred > 0.5).astype(float)
    mask_gt = (mask_gt > 0.5).astype(float)
    
    mae = np.mean(np.abs(mask_pred - mask_gt))
    return mae
