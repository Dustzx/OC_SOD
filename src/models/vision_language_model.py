"""
Vision Language Model Wrapper - 视觉语言模型
用于分割质量评估和自动优化
"""

import json
import base64
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class VisionLanguageModel:
    """
    视觉语言模型的统一接口，支持多种VLM后端
    """
    
    def __init__(self, 
                 model_name: str = "qwen3-vl-8b-instruct",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        初始化VLM
        
        Args:
            model_name: 模型名称
            api_key: API密钥
            base_url: API基础URL
        """
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url or self._get_base_url()
        self.client = self._init_client()
    
    def _get_api_key(self) -> str:
        """获取API密钥，可从环境变量或配置文件读取"""
        import os
        return os.getenv("DASHSCOPE_API_KEY", "")
    
    def _get_base_url(self) -> str:
        """获取API基础URL"""
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    def _init_client(self):
        """初始化OpenAI兼容客户端"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            print(f"✓ VLM模型初始化成功: {self.model_name}")
            return client
        except ImportError:
            raise ImportError("请安装openai SDK: pip install openai")
        except Exception as e:
            raise RuntimeError(f"VLM初始化失败: {e}")
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """
        从预训练模型加载
        
        Args:
            model_name: 模型名称
            **kwargs: 其他参数
            
        Returns:
            VisionLanguageModel实例
        """
        return cls(model_name=model_name, **kwargs)
    
    def encode_image_to_base64(self, image_array: np.ndarray) -> str:
        """
        将numpy数组编码为base64
        
        Args:
            image_array: numpy图像数组
            
        Returns:
            base64编码字符串
        """
        import cv2
        _, buffer = cv2.imencode('.jpg', image_array)
        return base64.b64encode(buffer).decode('utf-8')
    
    def evaluate_segmentation_quality(self,
                                     image: np.ndarray,
                                     bboxes: List[List[int]],
                                     referring_expression: str,
                                     intent: str) -> Dict[str, Any]:
        """
        评估分割质量
        
        Args:
            image: 输入图像
            bboxes: 边界框列表
            referring_expression: 对象描述表达式
            intent: 分割意图
            
        Returns:
            评估结果字典
        """
        image_base64 = self.encode_image_to_base64(image)
        
        prompt = f"""请评估分割的质量。
        
对象描述: {referring_expression}
分割意图: {intent}
边界框数量: {len(bboxes)}
边界框位置: {bboxes}

请判断分割是否准确。如果不准确，请提供改进的边界框。
响应格式:
- 如果准确: "accurate: true; reason: 分割准确的原因"
- 如果不准确: "accurate: false; corrected_bboxes: [[x1,y1,x2,y2], ...]; reason: 改进理由"
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_quality_response(content)
            
        except Exception as e:
            print(f"分割质量评估失败: {e}")
            return {
                "success": False,
                "accurate": None,
                "reason": str(e)
            }
    
    def _parse_quality_response(self, response: str) -> Dict[str, Any]:
        """
        解析质量评估响应
        
        Args:
            response: VLM响应文本
            
        Returns:
            解析结果
        """
        result = {
            "success": True,
            "accurate": None,
            "reason": "",
            "corrected_bboxes": None,
            "raw_response": response
        }
        
        # 解析准确性
        if "accurate: true" in response.lower():
            result["accurate"] = True
        elif "accurate: false" in response.lower():
            result["accurate"] = False
        
        # 解析理由
        reason_match = re.search(r'reason:\s*([^;]+)', response, re.IGNORECASE)
        if reason_match:
            result["reason"] = reason_match.group(1).strip()
        
        # 解析纠正的边界框
        if not result["accurate"]:
            bbox_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
            matches = re.findall(bbox_pattern, response)
            if matches:
                result["corrected_bboxes"] = [[int(m[0]), int(m[1]), int(m[2]), int(m[3])] for m in matches]
        
        return result
    
    def generate_referring_expression(self, 
                                     image: np.ndarray,
                                     bbox: List[int]) -> str:
        """
        生成对象的自然语言描述表达式
        
        Args:
            image: 输入图像
            bbox: 边界框
            
        Returns:
            描述表达式
        """
        image_base64 = self.encode_image_to_base64(image)
        
        prompt = """请根据红色边界框中的对象生成简洁的自然语言描述，用一句话描述主要特征。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"生成描述表达式失败: {e}")
            return f"Object at bbox {bbox}"
    
    def chat(self, 
             image: Optional[np.ndarray] = None,
             text: str = "") -> str:
        """
        通用的chat接口
        
        Args:
            image: 输入图像（可选）
            text: 文本提示
            
        Returns:
            模型响应
        """
        if image is None:
            # 仅文本
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )
        else:
            # 图文多模态
            image_base64 = self.encode_image_to_base64(image)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": text
                            }
                        ]
                    }
                ]
            )
        
        return response.choices[0].message.content.strip()
