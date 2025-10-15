"""
多角度融合识别模块
实现多张图片结果的智能融合
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from loguru import logger
import numpy as np


class MultiAngleFusion:
    """多角度融合识别器"""
    
    def __init__(self, config: dict):
        """
        初始化多角度融合器
        
        Args:
            config: 多角度融合配置
        """
        self.config = config
        self.fusion_method = config.get("fusion_method", "voting")
        self.min_images = config.get("min_images", 2)
        self.max_images = config.get("max_images", 10)
        self.alignment_method = config.get("alignment_method", "hybrid")
        self.return_alternatives = config.get("return_alternatives", True)
        self.alternative_threshold = config.get("alternative_threshold", 0.85)
        
    def fuse_results(self, recognition_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        融合多个识别结果
        
        Args:
            recognition_results: 多个图片的识别结果列表
            
        Returns:
            融合后的结果
        """
        if not recognition_results:
            return {
                "success": False,
                "error": "无识别结果可融合"
            }
            
        if len(recognition_results) < self.min_images:
            logger.warning(f"图片数量不足({len(recognition_results)}),建议至少{self.min_images}张")
            
        if len(recognition_results) > self.max_images:
            logger.warning(f"图片数量过多({len(recognition_results)}),将只使用前{self.max_images}张")
            recognition_results = recognition_results[:self.max_images]
            
        # 提取所有识别的文字
        all_texts = []
        for result in recognition_results:
            if result.get("success", False):
                for item in result.get("results", []):
                    all_texts.append({
                        "text": item.get("text", ""),
                        "confidence": item.get("confidence", 0),
                        "source_index": recognition_results.index(result)
                    })
                    
        if not all_texts:
            return {
                "success": False,
                "error": "所有图片识别失败"
            }
            
        # 根据融合方法进行融合
        if self.fusion_method == "voting":
            merged_text, confidence, alternatives = self._voting_fusion(all_texts)
        elif self.fusion_method == "weighted":
            merged_text, confidence, alternatives = self._weighted_fusion(all_texts)
        elif self.fusion_method == "smart":
            merged_text, confidence, alternatives = self._smart_fusion(all_texts)
        elif self.fusion_method == "merge":
            merged_text, confidence, alternatives = self._merge_fusion(all_texts)
        else:
            logger.error(f"不支持的融合方法: {self.fusion_method}")
            return {
                "success": False,
                "error": f"不支持的融合方法: {self.fusion_method}"
            }
            
        # 构建返回结果
        result = {
            "success": True,
            "merged_text": merged_text,
            "confidence": confidence,
            "source_count": len(recognition_results),
            "individual_results": recognition_results,
            "fusion_method": self.fusion_method
        }
        
        if self.return_alternatives and alternatives:
            result["alternatives"] = alternatives
            
        return result
        
    def _voting_fusion(self, all_texts: List[Dict[str, Any]]) -> Tuple[str, float, List[Dict]]:
        """
        投票融合:选择出现频率最高的文字
        
        Args:
            all_texts: 所有识别的文字列表
            
        Returns:
            (融合文字, 综合置信度, 备选结果)
        """
        # 统计每个文字的出现次数和平均置信度
        text_stats = defaultdict(lambda: {"count": 0, "confidences": []})
        
        for item in all_texts:
            text = item["text"]
            confidence = item["confidence"]
            text_stats[text]["count"] += 1
            text_stats[text]["confidences"].append(confidence)
            
        # 计算综合得分
        scored_texts = []
        total_count = len(all_texts)
        
        for text, stats in text_stats.items():
            count = stats["count"]
            avg_confidence = np.mean(stats["confidences"])
            # 综合得分 = 出现频率 × 平均置信度 × 长度权重
            frequency = count / total_count
            length_weight = min(len(text) / 3, 1.5)  # 偏好较长的结果
            score = frequency * avg_confidence * length_weight
            
            scored_texts.append({
                "text": text,
                "score": score,
                "count": count,
                "frequency": frequency,
                "avg_confidence": avg_confidence
            })
            
        # 按得分排序
        scored_texts.sort(key=lambda x: x["score"], reverse=True)
        
        if not scored_texts:
            return "", 0.0, []
            
        # 最佳结果
        best = scored_texts[0]
        merged_text = best["text"]
        confidence = best["avg_confidence"]
        
        # 备选结果
        alternatives = []
        for item in scored_texts[1:]:
            if item["score"] >= best["score"] * self.alternative_threshold:
                alternatives.append({
                    "text": item["text"],
                    "score": item["score"],
                    "confidence": item["avg_confidence"]
                })
                
        logger.info(f"投票融合: {merged_text} (得分: {best['score']:.3f}, 频率: {best['frequency']:.2f})")
        return merged_text, confidence, alternatives
        
    def _weighted_fusion(self, all_texts: List[Dict[str, Any]]) -> Tuple[str, float, List[Dict]]:
        """
        加权融合:根据置信度加权
        
        Args:
            all_texts: 所有识别的文字列表
            
        Returns:
            (融合文字, 综合置信度, 备选结果)
        """
        # 按置信度加权
        text_weights = defaultdict(lambda: {"total_weight": 0, "confidences": []})
        
        for item in all_texts:
            text = item["text"]
            confidence = item["confidence"]
            text_weights[text]["total_weight"] += confidence
            text_weights[text]["confidences"].append(confidence)
            
        # 计算加权得分
        scored_texts = []
        
        for text, weights in text_weights.items():
            total_weight = weights["total_weight"]
            avg_confidence = np.mean(weights["confidences"])
            
            scored_texts.append({
                "text": text,
                "score": total_weight,
                "avg_confidence": avg_confidence
            })
            
        # 按得分排序
        scored_texts.sort(key=lambda x: x["score"], reverse=True)
        
        if not scored_texts:
            return "", 0.0, []
            
        # 最佳结果
        best = scored_texts[0]
        merged_text = best["text"]
        confidence = best["avg_confidence"]
        
        # 备选结果
        alternatives = []
        for item in scored_texts[1:]:
            if item["score"] >= best["score"] * self.alternative_threshold:
                alternatives.append({
                    "text": item["text"],
                    "score": item["score"],
                    "confidence": item["avg_confidence"]
                })
                
        logger.info(f"加权融合: {merged_text} (总权重: {best['score']:.3f})")
        return merged_text, confidence, alternatives
        
    def _smart_fusion(self, all_texts: List[Dict[str, Any]]) -> Tuple[str, float, List[Dict]]:
        """
        智能融合:选择最高置信度的结果
        
        Args:
            all_texts: 所有识别的文字列表
            
        Returns:
            (融合文字, 综合置信度, 备选结果)
        """
        # 按置信度排序
        sorted_texts = sorted(all_texts, key=lambda x: x["confidence"], reverse=True)
        
        if not sorted_texts:
            return "", 0.0, []
            
        # 最佳结果
        best = sorted_texts[0]
        merged_text = best["text"]
        confidence = best["confidence"]
        
        # 备选结果
        alternatives = []
        seen_texts = {merged_text}
        
        for item in sorted_texts[1:]:
            text = item["text"]
            if text not in seen_texts and item["confidence"] >= confidence * self.alternative_threshold:
                alternatives.append({
                    "text": text,
                    "confidence": item["confidence"]
                })
                seen_texts.add(text)
                
        logger.info(f"智能融合: {merged_text} (置信度: {confidence:.3f})")
        return merged_text, confidence, alternatives
        
    def _merge_fusion(self, all_texts: List[Dict[str, Any]]) -> Tuple[str, float, List[Dict]]:
        """
        合并融合:合并所有不重复的文字
        
        Args:
            all_texts: 所有识别的文字列表
            
        Returns:
            (融合文字, 综合置信度, 备选结果)
        """
        # 去重并合并
        unique_texts = {}
        
        for item in all_texts:
            text = item["text"]
            confidence = item["confidence"]
            
            if text not in unique_texts or confidence > unique_texts[text]["confidence"]:
                unique_texts[text] = {
                    "text": text,
                    "confidence": confidence
                }
                
        # 按置信度排序
        sorted_texts = sorted(unique_texts.values(), key=lambda x: x["confidence"], reverse=True)
        
        # 合并所有文字
        merged_text = " ".join([item["text"] for item in sorted_texts])
        avg_confidence = np.mean([item["confidence"] for item in sorted_texts])
        
        logger.info(f"合并融合: {merged_text} (平均置信度: {avg_confidence:.3f})")
        return merged_text, avg_confidence, []
