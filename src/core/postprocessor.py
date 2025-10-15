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
            if re.fullmatch(r'[A-Z0-9]{3,}', corrected):
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
