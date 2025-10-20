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
        
        # 按行融合(从每个图片的行分组结果中融合)
        fused_lines = self._fuse_lines(recognition_results)
            
        # 构建返回结果
        result = {
            "success": True,
            "merged_text": merged_text,
            "confidence": confidence,
            "total_merged_texts": len(merged_text.split()),
            "total_lines": len(fused_lines),
            "lines": fused_lines,
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
    
    def _fuse_lines(self, recognition_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        融合多个图片的行分组结果
        
        Args:
            recognition_results: 多个图片的识别结果列表
            
        Returns:
            融合后的行结果列表
        """
        # 收集所有图片的行结果
        all_lines_by_image = []
        
        for idx, result in enumerate(recognition_results):
            if result.get("success", False) and "lines" in result:
                lines = []
                for line in result.get("lines", []):
                    line_text = line.get("text", "")
                    lines.append({
                        "text": line_text,
                        "confidence": line.get("confidence", 0),
                        "item_count": line.get("item_count", 0)
                    })
                    logger.debug(f"图片{idx+1} 识别: {line_text} (置信度: {line.get('confidence', 0):.2f})")
                if lines:
                    all_lines_by_image.append(lines)
        
        if not all_lines_by_image:
            return []
        
        # 找到最大行数
        max_lines = max(len(lines) for lines in all_lines_by_image)
        
        if max_lines == 0:
            return []
        
        # 按位置融合: 假设每张图片的第 i 行对应同一个物理位置
        fused_lines = []
        
        for line_index in range(max_lines):
            # 收集所有图片在该位置的行
            lines_at_position = []
            
            for image_idx, image_lines in enumerate(all_lines_by_image):
                if line_index < len(image_lines):
                    line_data = image_lines[line_index].copy()
                    line_data['source_image'] = image_idx
                    lines_at_position.append(line_data)
            
            if not lines_at_position:
                continue
            
            logger.info(f"\n=== 位置 {line_index + 1} 的融合 ===")
            for line in lines_at_position:
                logger.info(f"  图片{line['source_image']+1}: '{line['text']}' (置信度: {line['confidence']:.2f})")
            
            # 融合该位置的行: 使用相似度匹配 + 投票机制
            text_groups = self._group_similar_texts(lines_at_position)
            
            if not text_groups:
                continue
            
            # 选择最佳文本组
            best_group = max(text_groups, key=lambda g: (g['count'], g['avg_confidence']))
            
            logger.info(f"  → 融合结果: '{best_group['text']}' (出现{best_group['count']}次, 置信度: {best_group['avg_confidence']:.2f})")
            
            fused_lines.append({
                "text": best_group['text'],
                "confidence": best_group['avg_confidence'],
                "occurrence_count": best_group['count'],
                "item_count": 1
            })
        
        logger.info(f"\n按位置融合后共 {len(fused_lines)} 行")
        return fused_lines
    
    def _group_similar_texts(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将相似的文本分组(支持模糊匹配)
        
        Args:
            lines: 行数据列表
            
        Returns:
            分组后的文本列表
        """
        groups = []
        
        for line in lines:
            text = line['text'].strip()
            if not text:
                continue
            
            # 查找相似的组
            matched = False
            for group in groups:
                if self._is_similar(text, group['text']):
                    group['confidences'].append(line['confidence'])
                    group['count'] += 1
                    # 如果新文本置信度更高,更新代表文本
                    if line['confidence'] > group['max_confidence']:
                        group['text'] = text
                        group['max_confidence'] = line['confidence']
                    matched = True
                    break
            
            if not matched:
                # 创建新组
                groups.append({
                    'text': text,
                    'confidences': [line['confidence']],
                    'count': 1,
                    'max_confidence': line['confidence']
                })
        
        # 计算平均置信度
        for group in groups:
            group['avg_confidence'] = np.mean(group['confidences'])
        
        return groups
    
    def _is_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        判断两个文本是否相似
        
        Args:
            text1: 文本1
            text2: 文本2
            threshold: 相似度阈值
            
        Returns:
            是否相似
        """
        # 完全匹配
        if text1 == text2:
            return True
        
        # 去除空格后匹配
        norm1 = text1.replace(' ', '').upper()
        norm2 = text2.replace(' ', '').upper()
        if norm1 == norm2:
            return True
        
        # 长度差异过大
        if abs(len(norm1) - len(norm2)) > max(len(norm1), len(norm2)) * 0.3:
            return False
        
        # 计算编辑距离相似度
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        return similarity >= threshold
