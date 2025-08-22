"""
BM25关键词检索系统
用于基于关键词的精确匹配检索
"""

import pickle
import jieba
import re
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from collections import defaultdict
from loguru import logger

class BM25Retriever:
    """BM25关键词检索器"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25检索器
        
        Args:
            k1: 控制词频饱和点的参数
            b: 控制文档长度归一化的参数
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        
        # 初始化jieba分词（预热）
        jieba.initialize()
        
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 去除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def _tokenize(self, text: str) -> List[str]:
        """文本分词"""
        if not text:
            return []
            
        # 预处理
        text = self._preprocess_text(text)
        
        # jieba分词
        tokens = list(jieba.cut(text))
        
        # 过滤停用词和短词
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
            if len(token) > 1 and not self._is_stopword(token):
                filtered_tokens.append(token)
                
        return filtered_tokens
    
    def _is_stopword(self, word: str) -> bool:
        """判断是否为停用词"""
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', 
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他',
            '她', '它', '我们', '你们', '他们', '这个', '那个', '这些', '那些',
            '什么', '怎么', '为什么', '哪里', '哪个', '如何', '多少', '几', 
            '年', '月', '日', '时', '分', '秒'
        }
        return word in stopwords
    
    def add_documents(self, documents: List[Dict]):
        """添加文档到BM25索引"""
        if not documents:
            return
            
        logger.info(f"添加 {len(documents)} 个文档到BM25索引")
        
        for doc in documents:
            self.documents.append(doc)
            
            # 分词
            content = doc.get('content', '')
            tokens = self._tokenize(content)
            self.tokenized_docs.append(tokens)
            
        # 重建BM25索引
        self._build_index()
        
    def _build_index(self):
        """构建BM25索引"""
        if not self.tokenized_docs:
            return
            
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
        logger.info(f"BM25索引已构建，包含 {len(self.tokenized_docs)} 个文档")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        搜索相关文档
        
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
                    # 如果是字典，尝试提取文本内容
                    query = query.get('text', '') or query.get('content', '') or str(query)
                else:
                    query = str(query)
            
            # 清理查询文本
            query = query.strip()
            if not query:
                logger.warning("空查询，返回空结果")
                return []
                
            if not self.bm25:
                logger.error("BM25索引未构建")
                return []
            
            # 分词
            query_tokens = self._tokenize(query)
            if not query_tokens:
                logger.warning("查询分词后为空，返回空结果")
                return []
            
            # 获取分数
            scores = self.bm25.get_scores(query_tokens)
            
            # 排序并返回top-k结果
            doc_scores = [(i, score) for i, score in enumerate(scores)]
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, score in doc_scores[:k]:
                if i < len(self.documents):
                    results.append((self.documents[i], score))
                    
            logger.info(f"BM25搜索完成，查询: {query[:50]}..., 返回: {len(results)}个结果")
            return results
            
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            return []
    
    def search_with_keywords(self, keywords: List[str], k: int = 5) -> List[Tuple[Dict, float]]:
        """基于关键词列表检索"""
        if not keywords:
            return []
            
        # 合并关键词为查询
        query = " ".join(keywords)
        return self.search(query, k)
    
    def get_document_keywords(self, doc_index: int, top_k: int = 10) -> List[str]:
        """获取文档的关键词"""
        if doc_index >= len(self.tokenized_docs):
            return []
            
        tokens = self.tokenized_docs[doc_index]
        
        # 统计词频
        word_freq = defaultdict(int)
        for token in tokens:
            word_freq[token] += 1
            
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:top_k]]
        
        return keywords
    
    def save_index(self, index_path: str = "bm25_index"):
        """保存BM25索引"""
        try:
            data = {
                'bm25': self.bm25,
                'documents': self.documents,
                'tokenized_docs': self.tokenized_docs,
                'k1': self.k1,
                'b': self.b
            }
            
            with open(f"{index_path}.pkl", 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"BM25索引已保存到: {index_path}.pkl")
            
        except Exception as e:
            logger.error(f"BM25索引保存失败: {e}")
    
    def load_index(self, index_path: str = "bm25_index"):
        """加载BM25索引"""
        try:
            with open(f"{index_path}.pkl", 'rb') as f:
                data = pickle.load(f)
                
            self.bm25 = data['bm25']
            self.documents = data['documents']
            self.tokenized_docs = data['tokenized_docs']
            self.k1 = data.get('k1', 1.5)
            self.b = data.get('b', 0.75)
            
            logger.info(f"BM25索引已加载，包含 {len(self.documents)} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"BM25索引加载失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取BM25统计信息"""
        stats = {
            'total_documents': len(self.documents),
            'avg_doc_length': sum(len(doc) for doc in self.tokenized_docs) / len(self.tokenized_docs) if self.tokenized_docs else 0,
            'k1': self.k1,
            'b': self.b
        }
        return stats
    
    def clear(self):
        """清空BM25索引"""
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        logger.info("BM25索引已清空")
    
    def remove_documents_by_source(self, source: str):
        """根据来源删除文档"""
        original_count = len(self.documents)
        
        # 找到要删除的文档索引
        indices_to_remove = []
        for i, doc in enumerate(self.documents):
            if doc.get('source', '') == source or doc.get('metadata', {}).get('file_path', '') == source:
                indices_to_remove.append(i)
        
        # 从后往前删除，避免索引变化
        for idx in reversed(indices_to_remove):
            del self.documents[idx]
            del self.tokenized_docs[idx]
        
        removed_count = original_count - len(self.documents)
        
        if removed_count > 0:
            # 重建索引
            self._build_index()
            logger.info(f"已从BM25索引删除 {removed_count} 个来自 {source} 的文档")
        
        return removed_count
    
    def extract_query_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        tokens = self._tokenize(query)
        
        # 识别重要词汇
        important_keywords = []
        for token in tokens:
            # 优先保留名词、动词、形容词等实词
            if len(token) >= 2:
                important_keywords.append(token)
                
        return important_keywords[:10]  # 限制关键词数量 