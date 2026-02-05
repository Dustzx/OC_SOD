"""
OC-SOD Agent - Object-Centric Salient Object Detection Agent
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from ..models import SAM2Model, VisionLanguageModel
from ..utils import (
    combine_masks, overlay_mask
)

class OCSODAgent:
    """
    Object-Centric Salient Object Detection Agent.
    
    This agent uses an iterative refinement process that combines:
    1. A Segmentation Model (SAM2) for generating masks.
    2. A Vision Language Model (VLM) for evaluating quality and suggest corrections.
    """
    
    def __init__(self,
                 sam2_model: Optional[SAM2Model] = None,
                 vlm_model: Optional[VisionLanguageModel] = None,
                 target_size: int = 1000,
                 max_iterations: int = 3,
                 quality_threshold: float = 0.85):
        """
        Initialize the OC-SOD Agent.
        
        Args:
            sam2_model: Instance of SAM2Model.
            vlm_model: Instance of VisionLanguageModel.
            target_size: Target size for image processing.
            max_iterations: Maximum number of refinement iterations.
            quality_threshold: Quality threshold for early stopping.
        """
        self.sam2_model = sam2_model or SAM2Model.from_pretrained()
        self.vlm_model = vlm_model or VisionLanguageModel.from_pretrained()
        self.target_size = target_size
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
    
    def segment(self, 
               image: np.ndarray,
               bboxes: List[List[int]]) -> List[np.ndarray]:
        """
        Segment objects in the image based on bounding boxes.
        
        Args:
            image: Input image (H, W, 3).
            bboxes: List of bounding boxes [x1, y1, x2, y2].
            
        Returns:
            List of binary masks.
        """
        return self.sam2_model.segment_bboxes(image, bboxes)
    
    def evaluate_quality(self,
                        image: np.ndarray,
                        masks: List[np.ndarray],
                        bboxes: List[List[int]],
                        referring_expression: str = "object",
                        intent: str = "segment the object") -> Dict[str, Any]:
        """
        Evaluate the quality of segmentation using VLM.
        
        Args:
            image: Input image.
            masks: List of segmentation masks.
            bboxes: List of bounding boxes used for segmentation.
            referring_expression: Referring expression for the object.
            intent: The intent behind the segmentation.
            
        Returns:
            Dictionary containing evaluation results (accurate: bool, reason: str, corrected_bboxes: List).
        """
        if not masks:
            return {
                "accurate": False,
                "reason": "No masks provided",
                "corrected_bboxes": None
            }
        
        # Call VLM to evaluate
        result = self.vlm_model.evaluate_segmentation_quality(
            image=image,
            bboxes=bboxes,
            referring_expression=referring_expression,
            intent=intent
        )
        
        return result
    
    def iterative_refine(self,
                        image: np.ndarray,
                        bboxes: List[List[int]],
                        intent: str = "segment the object",
                        referring_expression: str = "object") -> Tuple[List[np.ndarray], Dict]:
        """
        Iteratively refine segmentation results.
        
        Args:
            image: Input image.
            bboxes: Initial bounding boxes.
            intent: User intent (e.g., "segment the cat").
            referring_expression: Expression referring to the target.
            
        Returns:
            Tuple of (best_masks, info_dictionary).
        """
        current_bboxes = bboxes
        best_masks = None
        refinement_history = []
        
        for iteration in range(self.max_iterations):
            # 1. Segment
            masks = self.segment(image, current_bboxes)
            
            # 2. Evaluate
            quality_result = self.evaluate_quality(
                image=image,
                masks=masks,
                bboxes=current_bboxes,
                referring_expression=referring_expression,
                intent=intent
            )
            
            refinement_history.append({
                "iteration": iteration,
                "bboxes": current_bboxes,
                "quality": quality_result
            })
            
            # 3. Check for convergence / quality
            if quality_result.get("accurate", False):
                best_masks = masks
                break
            
            # 4. Refine Bounding Boxes if suggested
            corrected_bboxes = quality_result.get("corrected_bboxes")
            if corrected_bboxes:
                current_bboxes = corrected_bboxes
            else:
                # If inaccurate but no correction suggested, maybe keep current or break
                best_masks = masks
                break
        
        if best_masks is None:
            # Fallback to the last segmentation
            best_masks = self.segment(image, current_bboxes)
        
        return best_masks, {
            "history": refinement_history,
            "final_accuracy": refinement_history[-1]["quality"].get("accurate", False)
        }
    
    def process_with_visualization(self,
                                   image: np.ndarray,
                                   bboxes: List[List[int]],
                                   intent: str = "segment the object") -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Process an image and return visualization of the result.
        """
        masks, info = self.iterative_refine(
            image=image,
            bboxes=bboxes,
            intent=intent
        )
        
        vis_image = image.copy()
        
        # Draw final bounding boxes (from usage, not necessarily inputs if refined)
        final_bboxes = info["history"][-1]["bboxes"]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (bbox, color) in enumerate(zip(final_bboxes, colors)):
            if i >= len(colors): break
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Overlay masks
        for mask, color in zip(masks, colors):
            if i >= len(colors): break
            vis_image = overlay_mask(vis_image, mask, color, alpha=0.3)
            
        return vis_image, masks
