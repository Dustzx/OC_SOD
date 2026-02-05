"""
基础分割示例
演示如何使用OC_SOD进行简单的图像分割
"""

import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Segmentor
from src.models import SAM2Model, VisionLanguageModel


def example_basic_segmentation():
    """
    基础分割示例
    """
    print("=" * 60)
    print("OC_SOD - 基础分割示例")
    print("=" * 60)
    
    # 1. 初始化模型
    print("\n[1] 初始化模型...")
    try:
        sam2_model = SAM2Model.from_pretrained("facebook/sam2-hiera-large")
        vlm_model = VisionLanguageModel.from_pretrained("qwen3-vl-8b-instruct")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        print("请确保已安装依赖: pip install -r requirements.txt")
        return
    
    # 2. 创建分割器
    print("\n[2] 创建分割器...")
    segmentor = Segmentor(
        sam2_model=sam2_model,
        vlm_model=vlm_model,
        target_size=1000,
        max_iterations=3
    )
    
    # 3. 准备示例图像
    print("\n[3] 准备示例图像...")
    # 创建一个示例图像（实际应用中应该加载真实图像）
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 在图像上绘制几个对象
    cv2.circle(image, (200, 200), 80, (0, 0, 255), -1)  # 红圆
    cv2.rectangle(image, (400, 100), (600, 300), (0, 255, 0), -1)  # 绿矩形
    cv2.ellipse(image, (700, 400), (60, 100), 45, 0, 360, (255, 0, 0), -1)  # 蓝椭圆
    
    # 定义感兴趣的对象的边界框
    bboxes = [
        [150, 150, 250, 250],      # 圆形对象
        [400, 100, 600, 300],      # 矩形对象
        [650, 300, 750, 500],      # 椭圆对象
    ]
    
    print(f"  图像大小: {image.shape}")
    print(f"  对象数量: {len(bboxes)}")
    
    # 4. 执行分割
    print("\n[4] 执行分割...")
    try:
        masks = segmentor.segment(image, bboxes)
        print(f"✓ 分割成功，得到{len(masks)}个mask")
    except Exception as e:
        print(f"✗ 分割失败: {e}")
        return
    
    # 5. 评估质量
    print("\n[5] 评估分割质量...")
    try:
        quality = segmentor.evaluate_quality(
            image=image,
            masks=masks,
            bboxes=bboxes,
            referring_expression="three objects in different shapes",
            intent="segment each object accurately"
        )
        print(f"✓ 质量评估完成")
        print(f"  准确性: {quality.get('accurate', 'unknown')}")
        print(f"  理由: {quality.get('reason', 'N/A')}")
    except Exception as e:
        print(f"  质量评估需要配置VLM API密钥: {e}")
    
    # 6. 生成可视化
    print("\n[6] 生成可视化...")
    vis_image, _ = segmentor.process_with_visualization(
        image=image,
        bboxes=bboxes,
        intent="segment objects"
    )
    
    # 保存可视化结果
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "basic_segmentation_result.jpg"
    
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), vis_image_bgr)
    print(f"✓ 可视化结果已保存: {output_path}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    example_basic_segmentation()
