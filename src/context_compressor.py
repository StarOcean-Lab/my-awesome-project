"""
上下文压缩模块
只保留与问题最相关的段落，避免LLM截断并提高回答质量
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
import jieba
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class CompressedContext:
    """压缩后的上下文"""
    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    relevant_paragraphs: List[Dict]
    keywords_found: List[str]

class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self, 
                 max_length: int = 1500,
                 min_paragraph_length: int = 20,
                 keyword_weight: float = 0.4,
                 similarity_weight: float = 0.6):
        """
        初始化上下文压缩器
        
        Args:
            max_length: 压缩后的最大字符长度
            min_paragraph_length: 段落最小长度（字符）
            keyword_weight: 关键词匹配权重
            similarity_weight: 语义相似度权重
        """
        self.max_length = max_length
        self.min_paragraph_length = min_paragraph_length
        self.keyword_weight = keyword_weight
        self.similarity_weight = similarity_weight
        
        # 停用词列表
        self.stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', 
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', 
            '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '们', '与',
            '或', '但', '然而', '因为', '所以', '如果', '虽然', '而且', '并且'
        }
        
        logger.info(f"上下文压缩器初始化完成，最大长度: {max_length}")
    
    def compress_context(self, question: str, context: str) -> CompressedContext:
        """
        压缩上下文，只保留与问题最相关的段落
        
        Args:
            question: 用户问题
            context: 原始上下文
        
        Returns:
            CompressedContext: 压缩结果
        """
        if not context:
            return CompressedContext(
                compressed_text="",
                original_length=0,
                compressed_length=0,
                compression_ratio=0.0,
                relevant_paragraphs=[],
                keywords_found=[]
            )
        
        original_length = len(context)
        
        # 如果原始文本已经很短，直接返回
        if original_length <= self.max_length:
            return CompressedContext(
                compressed_text=context,
                original_length=original_length,
                compressed_length=original_length,
                compression_ratio=1.0,
                relevant_paragraphs=[],
                keywords_found=[]
            )
        
        logger.info(f"开始压缩上下文，原始长度: {original_length}, 目标长度: {self.max_length}")
        
        # 1. 提取问题关键词
        question_keywords = self._extract_keywords(question)
        logger.debug(f"问题关键词: {question_keywords}")
        
        # 2. 分段
        paragraphs = self._split_paragraphs(context)
        logger.debug(f"分段数量: {len(paragraphs)}")
        
        # 3. 计算段落相关性分数
        paragraph_scores = self._score_paragraphs(question, question_keywords, paragraphs)
        
        # 4. 选择最相关的段落
        selected_paragraphs, keywords_found = self._select_paragraphs(
            paragraph_scores, question_keywords
        )
        
        # 5. 组合压缩后的文本
        compressed_text = self._combine_paragraphs(selected_paragraphs)
        compressed_length = len(compressed_text)
        
        compression_ratio = compressed_length / original_length if original_length > 0 else 0
        
        logger.info(f"压缩完成，压缩后长度: {compressed_length}, 压缩比: {compression_ratio:.2f}")
        
        return CompressedContext(
            compressed_text=compressed_text,
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=compression_ratio,
            relevant_paragraphs=selected_paragraphs,
            keywords_found=keywords_found
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 使用jieba分词
        words = jieba.lcut(text)
        
        # 过滤停用词和短词
        keywords = [
            word for word in words 
            if len(word) > 1 and word not in self.stopwords
        ]
        
        # 去重并保持顺序
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords[:10]  # 最多返回10个关键词
    
    def _split_paragraphs(self, context: str) -> List[str]:
        """分段"""
        # 按多种分割符分段
        separators = ['\n\n', '\n', '。', '！', '？']
        
        paragraphs = [context]
        
        for separator in separators:
            new_paragraphs = []
            for para in paragraphs:
                if separator in para:
                    splits = para.split(separator)
                    for split in splits:
                        split = split.strip()
                        if split:
                            new_paragraphs.append(split)
                else:
                    new_paragraphs.append(para)
            paragraphs = new_paragraphs
        
        # 过滤太短的段落
        filtered_paragraphs = [
            para for para in paragraphs 
            if len(para) >= self.min_paragraph_length
        ]
        
        return filtered_paragraphs if filtered_paragraphs else paragraphs
    
    def _score_paragraphs(self, question: str, keywords: List[str], 
                         paragraphs: List[str]) -> List[Tuple[str, float]]:
        """计算段落相关性分数"""
        scored_paragraphs = []
        
        for paragraph in paragraphs:
            # 1. 关键词匹配分数
            keyword_score = self._calculate_keyword_score(paragraph, keywords)
            
            # 2. 语义相似度分数（简化版）
            similarity_score = self._calculate_similarity_score(question, paragraph)
            
            # 3. 位置权重（前面的段落权重稍高）
            position_weight = 1.0  # 简化处理，可以根据需要调整
            
            # 4. 长度权重（避免过短或过长的段落）
            length_weight = self._calculate_length_weight(paragraph)
            
            # 综合分数
            final_score = (
                self.keyword_weight * keyword_score +
                self.similarity_weight * similarity_score
            ) * position_weight * length_weight
            
            scored_paragraphs.append((paragraph, final_score))
        
        # 按分数排序
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_paragraphs
    
    def _calculate_keyword_score(self, paragraph: str, keywords: List[str]) -> float:
        """计算关键词匹配分数"""
        if not keywords:
            return 0.0
        
        paragraph_lower = paragraph.lower()
        matched_keywords = 0
        
        for keyword in keywords:
            if keyword.lower() in paragraph_lower:
                matched_keywords += 1
        
        return matched_keywords / len(keywords)
    
    def _calculate_similarity_score(self, question: str, paragraph: str) -> float:
        """计算语义相似度分数（简化版）"""
        # 使用TF-IDF + 余弦相似度的简化方法
        try:
            # 使用字符级别的n-gram来处理中文
            vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(1, 3),
                max_features=1000,
                stop_words=None
            )
            
            # 构建文档集合
            documents = [question, paragraph]
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # 计算余弦相似度
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"相似度计算失败: {e}")
            # 降级方案：使用简单的词汇重叠
            question_words = set(jieba.lcut(question))
            paragraph_words = set(jieba.lcut(paragraph))
            
            if not question_words:
                return 0.0
            
            intersection = question_words.intersection(paragraph_words)
            return len(intersection) / len(question_words)
    
    def _calculate_length_weight(self, paragraph: str) -> float:
        """计算长度权重"""
        length = len(paragraph)
        
        # 理想长度范围：50-300字符
        if 50 <= length <= 300:
            return 1.0
        elif length < 50:
            return 0.7 + (length / 50) * 0.3  # 短段落轻微降权
        else:
            return max(0.5, 1.0 - (length - 300) / 500)  # 长段落降权
    
    def _select_paragraphs(self, scored_paragraphs: List[Tuple[str, float]], 
                          keywords: List[str]) -> Tuple[List[Dict], List[str]]:
        """选择最相关的段落"""
        selected_paragraphs = []
        current_length = 0
        keywords_found = set()
        
        for paragraph, score in scored_paragraphs:
            # 检查是否还有空间
            if current_length + len(paragraph) > self.max_length:
                # 如果是第一个段落且超出限制，进行截断
                if not selected_paragraphs:
                    remaining_space = self.max_length - current_length
                    if remaining_space > 100:  # 至少保留100个字符
                        truncated = paragraph[:remaining_space - 3] + "..."
                        selected_paragraphs.append({
                            'text': truncated,
                            'score': score,
                            'truncated': True
                        })
                        current_length += len(truncated)
                break
            
            # 添加段落
            selected_paragraphs.append({
                'text': paragraph,
                'score': score,
                'truncated': False
            })
            current_length += len(paragraph)
            
            # 记录找到的关键词
            paragraph_lower = paragraph.lower()
            for keyword in keywords:
                if keyword.lower() in paragraph_lower:
                    keywords_found.add(keyword)
        
        return selected_paragraphs, list(keywords_found)
    
    def _combine_paragraphs(self, selected_paragraphs: List[Dict]) -> str:
        """组合选中的段落"""
        if not selected_paragraphs:
            return ""
        
        # 按分数排序，确保最相关的在前面
        selected_paragraphs.sort(key=lambda x: x['score'], reverse=True)
        
        # 组合文本
        texts = [para['text'] for para in selected_paragraphs]
        return '\n\n'.join(texts)
    
    def batch_compress(self, questions_contexts: List[Tuple[str, str]]) -> List[CompressedContext]:
        """批量压缩"""
        results = []
        for question, context in questions_contexts:
            result = self.compress_context(question, context)
            results.append(result)
        
        return results
    
    def get_compression_stats(self, results: List[CompressedContext]) -> Dict:
        """获取压缩统计信息"""
        if not results:
            return {}
        
        compression_ratios = [r.compression_ratio for r in results]
        original_lengths = [r.original_length for r in results]
        compressed_lengths = [r.compressed_length for r in results]
        
        return {
            "total_compressions": len(results),
            "avg_compression_ratio": np.mean(compression_ratios),
            "avg_original_length": np.mean(original_lengths),
            "avg_compressed_length": np.mean(compressed_lengths),
            "min_compression_ratio": np.min(compression_ratios),
            "max_compression_ratio": np.max(compression_ratios),
            "total_chars_saved": sum(original_lengths) - sum(compressed_lengths)
        }

class AdaptiveContextCompressor(ContextCompressor):
    """自适应上下文压缩器"""
    
    def __init__(self, **kwargs):
        """初始化自适应压缩器"""
        super().__init__(**kwargs)
        
        # 问题类型特定的压缩策略
        self.type_strategies = {
            "时间查询": {
                "keywords": ['时间', '日期', '年', '月', '日', '截止', '开始', '结束'],
                "keyword_weight": 0.7,
                "similarity_weight": 0.3
            },
            "统计问题": {
                "keywords": ['个', '项', '类', '共', '总', '数量', '统计'],
                "keyword_weight": 0.8,
                "similarity_weight": 0.2
            },
            "联系查询": {
                "keywords": ['电话', '邮箱', '联系', '咨询', '@', 'QQ', '微信'],
                "keyword_weight": 0.9,
                "similarity_weight": 0.1
            },
            "定义问题": {
                "keywords": ['定义', '概念', '含义', '是什么'],
                "keyword_weight": 0.5,
                "similarity_weight": 0.5
            }
        }
    
    def _classify_question_type(self, question: str) -> str:
        """分类问题类型"""
        if re.search(r'什么时候|何时|时间|日期', question):
            return "时间查询"
        elif re.search(r'多少|几个|数量', question):
            return "统计问题"
        elif re.search(r'联系|电话|邮箱', question):
            return "联系查询"
        elif re.search(r'什么是|是什么|定义', question):
            return "定义问题"
        else:
            return "通用查询"
    
    def compress_context(self, question: str, context: str) -> CompressedContext:
        """自适应压缩上下文"""
        # 识别问题类型并调整策略
        question_type = self._classify_question_type(question)
        
        if question_type in self.type_strategies:
            strategy = self.type_strategies[question_type]
            
            # 临时调整权重
            original_keyword_weight = self.keyword_weight
            original_similarity_weight = self.similarity_weight
            
            self.keyword_weight = strategy["keyword_weight"]
            self.similarity_weight = strategy["similarity_weight"]
            
            logger.info(f"使用{question_type}压缩策略，关键词权重: {self.keyword_weight}")
            
            try:
                result = super().compress_context(question, context)
            finally:
                # 恢复原始权重
                self.keyword_weight = original_keyword_weight
                self.similarity_weight = original_similarity_weight
        else:
            result = super().compress_context(question, context)
        
        return result 