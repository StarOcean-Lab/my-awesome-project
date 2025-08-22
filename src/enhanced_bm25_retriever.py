"""
增强的BM25关键词检索系统
支持精确短语匹配、强制关键词召回和优化的中文分词
"""

import pickle
import jieba
import re
from typing import List, Dict, Tuple, Set
from rank_bm25 import BM25Okapi
from collections import defaultdict
from loguru import logger

class EnhancedBM25Retriever:
    """增强的BM25关键词检索器 - 支持精确短语匹配"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化增强BM25检索器
        
        Args:
            k1: 控制词频饱和点的参数
            b: 控制文档长度归一化的参数
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        self.raw_texts = []  # 保存原始文本用于精确匹配
        
        # 关键词词典 - 用于强制召回
        self.force_recall_keywords = {
            "未来校园智能应用专项赛",
            "智能交通信号灯", 
            "基本要求",
            "任务描述",
            "技术要求",
            "评分标准",
            "竞赛通知"
        }
        
        # 初始化jieba分词
        jieba.initialize()
        
        # 添加专业词汇到jieba词典
        self._add_domain_words()
        
    def _add_domain_words(self):
        """添加专业领域词汇到jieba词典"""
        domain_words = [
            "未来校园", "智能应用", "交通信号灯", "专项赛",
            "基本要求", "技术要求", "任务描述", "评分标准",
            "竞赛通知", "泰迪杯", "数据挖掘", "人工智能"
        ]
        
        for word in domain_words:
            jieba.add_word(word, freq=1000, tag='domain')
        
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
            
        # 去除特殊字符，保留中文、英文、数字和重要标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：（）【】\-\.]', ' ', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def _tokenize(self, text: str) -> List[str]:
        """文本分词 - 增强版"""
        if not text:
            return []
            
        # 预处理
        text = self._preprocess_text(text)
        
        # jieba分词
        tokens = list(jieba.cut(text, cut_all=False))
        
        # 过滤和处理token
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
            if len(token) > 1 and not self._is_stopword(token):
                filtered_tokens.append(token)
                
        return filtered_tokens
        
    def _is_stopword(self, word: str) -> bool:
        """判断是否为停用词"""
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', 
            '不', '人', '都', '一', '一个', '上', '也', '很', 
            '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '年', '那', '现在', '可以',
            '但是', '这个', '什么', '然后', '比较', '还是', '应该'
        }
        return word in stopwords
    
    def build_index(self, documents: List[Dict]):
        """
        构建BM25索引
        
        Args:
            documents: 文档列表，每个文档是包含'content'和'metadata'的字典
        """
        logger.info(f"开始构建增强BM25索引，文档数量: {len(documents)}")
        
        self.documents = documents
        self.raw_texts = []
        self.tokenized_docs = []
        
        # 处理每个文档
        for i, doc in enumerate(documents):
            content = doc.get('content', '')  # 没有就返回空字符串
            self.raw_texts.append(content)
            
            # 分词
            tokens = self._tokenize(content)
            self.tokenized_docs.append(tokens)  # self.tokenized_docs为二维列表
            
            if i < 3:  # 显示前3个文档的分词结果用于调试
                logger.debug(f"文档 {i+1} 分词结果: {tokens[:10]}...")
        
        # 构建BM25索引
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
            logger.info("增强BM25索引构建完成")
        else:
            logger.error("没有有效的分词文档，索引构建失败")
    
    def exact_phrase_search(self, phrase: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """
        精确短语搜索
        
        Args:
            phrase: 要搜索的精确短语
            k: 返回结果数量
            
        Returns:
            (文档, 分数)元组列表，按相关性排序
        """
        if not phrase or not self.raw_texts:
            return []
        
        logger.info(f"执行精确短语搜索: '{phrase}'")
        
        # 在原始文本中搜索精确短语
        results = []
        for i, text in enumerate(self.raw_texts):
            if phrase in text:
                # 计算短语在文档中的出现次数
                count = text.count(phrase)
                # 计算相对分数（考虑文档长度）
                score = count / (len(text) + 1) * 1000  # 放大分数便于比较
                
                results.append((i, score, count))
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 转换为返回格式
        phrase_results = []
        for doc_idx, score, count in results[:k]:
            if doc_idx < len(self.documents):
                doc = self.documents[doc_idx].copy()
                doc['metadata']['phrase_matches'] = count
                doc['metadata']['exact_match'] = True
                phrase_results.append((doc, score))
        
        logger.info(f"精确短语搜索完成，找到 {len(phrase_results)} 个匹配文档")
        return phrase_results
    
    def force_recall_search(self, query: str, k: int = 20) -> List[Tuple[Dict, float]]:
        """
        强制召回搜索 - 确保包含重要关键词的文档被召回
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            
        Returns:
            强制召回的文档列表
        """
        if not query or not self.raw_texts:
            return []
        
        logger.info(f"执行强制召回搜索: '{query}'")
        
        # 检查查询中是否包含强制召回关键词
        force_keywords = []
        for keyword in self.force_recall_keywords:
            if keyword in query:
                force_keywords.append(keyword)
        
        if not force_keywords:
            return []
        
        logger.info(f"检测到强制召回关键词: {force_keywords}")
        
        # 搜索包含这些关键词的文档
        forced_results = []
        for keyword in force_keywords:
            keyword_results = self.exact_phrase_search(keyword, k=k)
            for doc, score in keyword_results:
                # 给强制召回的文档额外加分
                boosted_score = score + 1000  # 大幅提升分数确保优先返回
                doc['metadata']['force_recalled'] = True
                doc['metadata']['force_keyword'] = keyword
                forced_results.append((doc, boosted_score))
        
        # 去重并排序
        seen_contents = set()
        unique_results = []
        for doc, score in forced_results:
            content = doc.get('content', '')
            if content not in seen_contents:
                seen_contents.add(content)
                unique_results.append((doc, score))
        
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"强制召回完成，召回 {len(unique_results)} 个文档")
        return unique_results[:k]
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        增强搜索 - 结合常规BM25、精确短语匹配和强制召回
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (文档, 分数)元组列表
        """
        try:
            # 确保query是字符串类型
            if not isinstance(query, str):
                if isinstance(query, dict):
                    query = query.get('text', '') or query.get('content', '') or str(query)
                else:
                    query = str(query)
            
            query = query.strip()
            if not query:
                logger.warning("空查询，返回空结果")
                return []
                
            if not self.bm25:
                logger.error("BM25索引未构建")
                return []
            
            logger.info(f"开始增强BM25搜索: '{query[:50]}...'")
            
            all_results = {}  # content -> (doc, max_score)
            
            # 1. 强制召回搜索（优先级最高）
            force_results = self.force_recall_search(query, k=k*2)
            for doc, score in force_results:
                content = doc.get('content', '')
                if content not in all_results or score > all_results[content][1]:
                    all_results[content] = (doc, score)
            
            # 2. 常规BM25搜索
            query_tokens = self._tokenize(query)
            if query_tokens:
                scores = self.bm25.get_scores(query_tokens)
                doc_scores = [(i, score) for i, score in enumerate(scores)]
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                
                for i, score in doc_scores[:k*2]:
                    if i < len(self.documents):
                        doc = self.documents[i].copy()
                        doc['metadata']['bm25_score'] = score
                        content = doc.get('content', '')
                        
                        # 如果已有强制召回结果，不覆盖
                        if content not in all_results:
                            all_results[content] = (doc, score)
            
            # 3. 精确短语搜索（针对重要短语）
            important_phrases = ["未来校园智能应用专项赛", "智能交通信号灯", "基本要求"]
            for phrase in important_phrases:
                if phrase in query:
                    phrase_results = self.exact_phrase_search(phrase, k=k)
                    for doc, score in phrase_results:
                        content = doc.get('content', '')
                        # 精确匹配给予额外加分
                        boosted_score = score + 500
                        if content not in all_results or boosted_score > all_results[content][1]:
                            doc['metadata']['exact_phrase_match'] = phrase
                            all_results[content] = (doc, boosted_score)
            
            # 整理最终结果
            final_results = list(all_results.values())
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"增强BM25搜索完成，查询: {query[:50]}..., 返回: {len(final_results[:k])}个结果")
            
            # 显示搜索结果摘要
            for i, (doc, score) in enumerate(final_results[:3]):
                source = doc['metadata'].get('source', 'unknown')
                force_recalled = doc['metadata'].get('force_recalled', False)
                exact_match = doc['metadata'].get('exact_match', False)
                logger.debug(f"  结果{i+1}: {source}, 分数={score:.2f}, 强制召回={force_recalled}, 精确匹配={exact_match}")
            
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"增强BM25检索失败: {e}")
            return []
    
    def save_index(self, file_path: str):
        """保存索引到文件"""
        try:
            index_data = {
                'bm25': self.bm25,
                'documents': self.documents,
                'tokenized_docs': self.tokenized_docs,
                'raw_texts': self.raw_texts,
                'k1': self.k1,
                'b': self.b
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            logger.info(f"增强BM25索引已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    def load_index(self, file_path: str) -> bool:
        """从文件加载索引"""
        try:
            with open(file_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.bm25 = index_data['bm25']
            self.documents = index_data['documents']
            self.tokenized_docs = index_data['tokenized_docs']
            self.raw_texts = index_data['raw_texts']
            self.k1 = index_data.get('k1', 1.5)
            self.b = index_data.get('b', 0.75)
            
            logger.info(f"增强BM25索引已从文件加载: {file_path}")
            return True
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False 