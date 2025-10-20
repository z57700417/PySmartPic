"""
后处理模块
实现结果过滤、置信度评估、字符纠正等功能
"""

import re
from typing import List, Dict, Any, Tuple
from loguru import logger


class PostProcessor:
    """后处理器"""
    
    def __init__(self, config: dict):
        """
        初始化后处理器
        
        Args:
            config: 后处理配置字典
        """
        self.config = config
        self.min_confidence = config.get("min_confidence", 0.6)
        self.min_length = config.get("min_length", 1)
        self.max_length = config.get("max_length", 30)
        self.enable_char_filter = config.get("enable_char_filter", True)
        self.allowed_chars = config.get("allowed_chars", "")
        self.enable_correction = config.get("enable_correction", True)
        self.correction_rules = config.get("correction_rules", [])
        self.enable_deduplication = config.get("enable_deduplication", True)
        self.similarity_threshold = config.get("similarity_threshold", 0.9)
        self.min_results = config.get("min_results", 0)
        
        # 区域过滤配置(过滤非轮毂文字)
        self.enable_region_filter = config.get("enable_region_filter", False)
        self.region_filter_config = {
            "min_area_ratio": config.get("min_area_ratio", 0.0001),  # 最小面积比例(相对图像)
            "max_area_ratio": config.get("max_area_ratio", 0.1),     # 最大面积比例
            "min_aspect_ratio": config.get("min_aspect_ratio", 0.2), # 最小宽高比
            "max_aspect_ratio": config.get("max_aspect_ratio", 10),  # 最大宽高比
            "center_region_only": config.get("center_region_only", False),  # 是否只保留中心区域
            "center_region_ratio": config.get("center_region_ratio", 0.6)  # 中心区域比例
        }
        
    def process(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行完整的后处理流程
        
        Args:
            results: 原始识别结果列表
            
        Returns:
            后处理后的结果列表
        """
        if not results:
            return []
            
        orig_results = list(results)
        
        # 区域过滤(过滤非轮毂文字,如贴纸、标签等)
        if self.enable_region_filter:
            results = self._filter_by_region(results)
        
        # 置信度过滤
        results = self._filter_by_confidence(results)
        
        # 长度过滤
        results = self._filter_by_length(results)
        
        # 字符类型过滤
        if self.enable_char_filter:
            results = self._filter_by_chars(results)
            
        # 相似字符纠正
        if self.enable_correction:
            results = self._correct_characters(results)
            
        # 结果去重
        if self.enable_deduplication:
            results = self._deduplicate(results)
            
        # 若结果数不足，按置信度补足到最少条数
        if self.min_results and len(results) < self.min_results:
            supplemented = []
            existing_texts = [str(r.get("text", "")) for r in results]
            for item in sorted(orig_results, key=lambda x: x.get("confidence", 0), reverse=True):
                txt = str(item.get("text", ""))
                # 避免重复
                if any(self._calculate_similarity(txt, et) >= self.similarity_threshold for et in existing_texts):
                    continue
                # 仅保留允许字符
                allowed_set = set(self.allowed_chars) if self.allowed_chars else None
                if allowed_set is not None:
                    txt_clean = ''.join(c for c in txt if c in allowed_set)
                else:
                    txt_clean = txt
                if len(txt_clean) >= 4:
                    new_item = dict(item)
                    new_item["text"] = txt_clean
                    supplemented.append(new_item)
                    existing_texts.append(txt_clean)
                if len(results) + len(supplemented) >= self.min_results:
                    break
            if supplemented:
                results = results + supplemented
            # 截断到min_results
            results = results[:self.min_results]
        
        # 排序：优先 'AT' 前缀与长度符合的结果，其次按置信度
        try:
            for r in results:
                t = str(r.get("text", ""))
                conf = float(r.get("confidence", 0) or 0)
                bonus = 0.0
                if re.match(r"^AT[0-9]{3,}$", t):
                    bonus += 2.0
                elif t.startswith("AT"):
                    bonus += 1.5
                if 6 <= len(t) <= 8:
                    bonus += 0.5
                if t.isdigit():
                    bonus += 0.5
                r["_sort_score"] = conf + bonus
            results = sorted(results, key=lambda x: x.get("_sort_score", x.get("confidence", 0)), reverse=True)
            for r in results:
                if "_sort_score" in r:
                    del r["_sort_score"]
        except Exception:
            pass
        
        # 统一类型：将文本转换为字符串
        for r in results:
            r["text"] = str(r.get("text", ""))
            
        return results
        
    def _filter_by_confidence(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按置信度过滤
        
        Args:
            results: 识别结果列表
            
        Returns:
            过滤后的结果列表
        """
        filtered = []
        for result in results:
            if result.get("confidence", 0) >= self.min_confidence:
                filtered.append(result)
            else:
                logger.debug(f"过滤低置信度结果: {result.get('text')} (置信度: {result.get('confidence')})")
                
        logger.info(f"置信度过滤: {len(results)} -> {len(filtered)}")
        return filtered
    
    def _filter_by_region(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据区域特征过滤(过滤贴纸、标签等非轮毂文字)
        
        Args:
            results: 识别结果列表
            
        Returns:
            过滤后的结果列表
        """
        if not results:
            return []
        
        # 计算图像尺寸(从第一个bbox推断,假设图像大小)
        image_width = 0
        image_height = 0
        for result in results:
            bbox = result.get("bbox", [])
            if bbox and len(bbox) >= 2:
                try:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    image_width = max(image_width, max(x_coords))
                    image_height = max(image_height, max(y_coords))
                except (IndexError, TypeError):
                    continue
        
        if image_width == 0 or image_height == 0:
            logger.warning("无法获取图像尺寸,跳过区域过滤")
            return results
        
        image_area = image_width * image_height
        center_x = image_width / 2
        center_y = image_height / 2
        
        filtered = []
        for result in results:
            bbox = result.get("bbox", [])
            text = result.get("text", "")
            
            if not bbox or len(bbox) < 2:
                filtered.append(result)
                continue
            
            try:
                # 计算区域特征
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                width = max_x - min_x
                height = max_y - min_y
                area = width * height
                
                # 计算中心点
                bbox_center_x = (min_x + max_x) / 2
                bbox_center_y = (min_y + max_y) / 2
                
                # 计算面积比例
                area_ratio = area / image_area
                
                # 计算宽高比
                aspect_ratio = width / height if height > 0 else 0
                
                # 计算与图像中心的距离
                dist_to_center = ((bbox_center_x - center_x)**2 + (bbox_center_y - center_y)**2)**0.5
                max_dist = (center_x**2 + center_y**2)**0.5
                dist_ratio = dist_to_center / max_dist if max_dist > 0 else 1
                
                # 过滤条件
                filter_reasons = []
                
                # 1. 面积过大或过小(可能是标签或噪点)
                if area_ratio < self.region_filter_config["min_area_ratio"]:
                    filter_reasons.append(f"面积过小({area_ratio:.4f})")
                elif area_ratio > self.region_filter_config["max_area_ratio"]:
                    filter_reasons.append(f"面积过大({area_ratio:.4f})")
                
                # 2. 宽高比异常(可能是标签或贴纸)
                if aspect_ratio < self.region_filter_config["min_aspect_ratio"]:
                    filter_reasons.append(f"宽高比过小({aspect_ratio:.2f})")
                elif aspect_ratio > self.region_filter_config["max_aspect_ratio"]:
                    filter_reasons.append(f"宽高比过大({aspect_ratio:.2f})")
                
                # 3. 位置过偏(只保留中心区域)
                if self.region_filter_config["center_region_only"]:
                    center_threshold = 1 - self.region_filter_config["center_region_ratio"]
                    if dist_ratio > center_threshold:
                        filter_reasons.append(f"位置过偏({dist_ratio:.2f})")
                
                # 4. 特殊过滤:检测标签特征
                is_likely_label = False
                
                # 4.1 如果文本全是数字且长度>6,且宽高比接近矩形(0.8-3之间)
                if text.isdigit() and len(text) >= 7:
                    if 0.8 <= aspect_ratio <= 3.0:
                        is_likely_label = True
                        filter_reasons.append(f"疑似标签:长数字+矩形({aspect_ratio:.2f})")
                
                # 4.2 如果区域面积相对较大且形状规整(可能是贴纸/标签)
                if area_ratio > 0.015 and 0.8 <= aspect_ratio <= 4.0:
                    is_likely_label = True
                    filter_reasons.append(f"疑似标签:大面积({area_ratio:.4f})+规整形状")
                
                # 4.3 如果位于图像边缘(距离边界很近)
                edge_threshold = 0.15  # 距离边缘15%以内
                near_edge = (
                    bbox_center_x < image_width * edge_threshold or 
                    bbox_center_x > image_width * (1 - edge_threshold) or
                    bbox_center_y < image_height * edge_threshold or 
                    bbox_center_y > image_height * (1 - edge_threshold)
                )
                if near_edge and area_ratio > 0.005:
                    is_likely_label = True
                    filter_reasons.append("疑似标签:靠近边缘")
                
                # 4.4 轮毂文字通常特征:较小、嵌入金属表面
                # 如果面积比例在0.0005-0.008之间,更可能是轮毂雕刻文字
                is_likely_wheel_text = (0.0005 <= area_ratio <= 0.008)
                
                # 如果被判定为标签,且不是轮毂文字,则过滤
                if is_likely_label and not is_likely_wheel_text:
                    if not filter_reasons:
                        filter_reasons.append("综合判断为标签")
                
                if filter_reasons:
                    logger.debug(f"区域过滤: {text} - {', '.join(filter_reasons)}")
                else:
                    filtered.append(result)
                    
            except Exception as e:
                logger.debug(f"区域过滤处理错误: {e}, 保留结果")
                filtered.append(result)
        
        logger.info(f"区域过滤: {len(results)} -> {len(filtered)}")
        return filtered
        
    def _filter_by_length(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按文字长度过滤
        
        Args:
            results: 识别结果列表
            
        Returns:
            过滤后的结果列表
        """
        filtered = []
        for result in results:
            text = str(result.get("text", ""))
            length = len(text.strip())
            if self.min_length <= length <= self.max_length:
                filtered.append(result)
            else:
                logger.debug(f"过滤异常长度结果: {text} (长度: {length})")
                
        logger.info(f"长度过滤: {len(results)} -> {len(filtered)}")
        return filtered
        
    def _filter_by_chars(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按字符类型过滤
        
        Args:
            results: 识别结果列表
            
        Returns:
            过滤后的结果列表
        """
        if not self.allowed_chars:
            return results
            
        filtered = []
        allowed_set = set(self.allowed_chars)
        
        for result in results:
            text = str(result.get("text", ""))
            # 检查是否所有字符都在允许列表中
            if all(c in allowed_set for c in text):
                filtered.append(result)
            else:
                # 过滤掉不允许的字符
                filtered_text = ''.join(c for c in text if c in allowed_set)
                if filtered_text:
                    result["text"] = filtered_text
                    filtered.append(result)
                    logger.debug(f"字符过滤: {text} -> {filtered_text}")
                else:
                    logger.debug(f"过滤非法字符结果: {text}")
                    
        logger.info(f"字符过滤: {len(results)} -> {len(filtered)}")
        return filtered
        
    def _correct_characters(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        纠正相似字符
        
        Args:
            results: 识别结果列表
            
        Returns:
            纠正后的结果列表
        """
        for result in results:
            text = str(result.get("text", ""))
            original_text = text
            
            # 应用纠正规则
            # 这里实现简单的基于上下文的纠正
            text = self._apply_correction_rules(text)
            
            if text != original_text:
                result["text"] = text
                result["corrected"] = True
                result["original_text"] = original_text
                logger.debug(f"字符纠正: {original_text} -> {text}")
                
        return results
        
    def _apply_correction_rules(self, text: str) -> str:
        """
        应用字符纠正规则
        
        Args:
            text: 原始文本
            
        Returns:
            纠正后的文本
        """
        # 简单的启发式纠正规则
        corrected = text
        
        # 如果文本主要是字母,将数字0纠正为字母O
        if self._is_mostly_letters(text):
            corrected = corrected.replace('0', 'O')
            corrected = corrected.replace('1', 'I')
            corrected = corrected.replace('5', 'S')
            
        # 如果文本主要是数字,将字母纠正为数字
        elif self._is_mostly_digits(text):
            corrected = corrected.replace('O', '0')
            corrected = corrected.replace('I', '1')
            
        # 处理混合字母数字的常见混淆（上下文修正）
        try:
            # 轮毂编号特殊模式: AT + 数字
            if len(corrected) >= 7 and corrected[0:2] == 'AT':
                chars = list(corrected)
                # 后面应该是数字,纠正常见混淆
                for i in range(2, len(chars)):
                    c = chars[i]
                    if c.isalpha():
                        # 字母转数字的常见混淆
                        if c == 'O' or c == 'Q' or c == 'D':
                            chars[i] = '0'
                        elif c == 'I' or c == 'l':
                            chars[i] = '1'
                        elif c == 'Z':
                            chars[i] = '2'
                        elif c == 'B':
                            chars[i] = '8'
                        elif c == 'S':
                            chars[i] = '5'
                        elif c == 'G':
                            chars[i] = '6'
                        elif c == 'T':
                            chars[i] = '7'
                        elif c == 'A':
                            chars[i] = '4'
                        elif c == 'g' or c == 'q':
                            chars[i] = '9'
                corrected = ''.join(chars)
            elif re.fullmatch(r'[A-Z0-9]{3,}', corrected):
                chars = list(corrected)
                for i, c in enumerate(chars):
                    prev = chars[i-1] if i > 0 else ''
                    nxt = chars[i+1] if i+1 < len(chars) else ''
                    if prev.isdigit() and nxt.isdigit():
                        if c == 'L':
                            chars[i] = '4'
                        elif c == 'I':
                            chars[i] = '1'
                        elif c == 'O':
                            chars[i] = '0'
                        elif c == 'B':
                            chars[i] = '8'
                        elif c == 'S':
                            chars[i] = '5'
                        elif c == 'Z':
                            chars[i] = '2'
                        elif c == 'G':
                            chars[i] = '6'
                        elif c == 'T':
                            chars[i] = '7'
                corrected = ''.join(chars)
        except Exception:
            pass
        
        return corrected
        
    def _is_mostly_letters(self, text: str) -> bool:
        """判断文本是否主要由字母组成"""
        if not text:
            return False
        letter_count = sum(1 for c in text if c.isalpha())
        return letter_count / len(text) > 0.6
        
    def _is_mostly_digits(self, text: str) -> bool:
        """判断文本是否主要由数字组成"""
        if not text:
            return False
        digit_count = sum(1 for c in text if c.isdigit())
        return digit_count / len(text) > 0.6
        
    def _deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去除重复结果
        
        Args:
            results: 识别结果列表
            
        Returns:
            去重后的结果列表
        """
        if not results:
            return []
            
        unique_results = []
        seen_texts = set()
        
        for result in results:
            text = str(result.get("text", ""))
            
            # 检查是否已存在相似文本
            is_duplicate = False
            for seen_text in seen_texts:
                if self._calculate_similarity(text, seen_text) >= self.similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"去除重复结果: {text} (与 {seen_text} 相似)")
                    break
                    
            if not is_duplicate:
                unique_results.append(result)
                seen_texts.add(text)
                
        logger.info(f"去重: {len(results)} -> {len(unique_results)}")
        return unique_results
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度(0-1之间)
        """
        if text1 == text2:
            return 1.0
            
        # 使用编辑距离计算相似度
        distance = self._levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 1.0
            
        return 1.0 - (distance / max_len)
        
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        计算编辑距离
        
        Args:
            s1: 字符串1
            s2: 字符串2
            
        Returns:
            编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
