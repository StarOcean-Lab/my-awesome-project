"""
多样性增强检索器
解决检索偏向性问题，确保检索结果的多样性和平衡性
"""

from typing import List, Dict, Any
from collections import defaultdict, Counter
import numpy as np
from langchain_core.documents import Document
from loguru import logger

from .langchain_retriever import LangChainHybridRetriever
from .reranker import Reranker, RerankResult

class DiversityEnhancedRetriever:
    """多样性增强检索器"""
    
    def __init__(self, 
                 base_retriever: LangChainHybridRetriever,
                 diversity_weight: float = 0.3,
                 max_docs_per_source: int = 2,
                 enable_diversity: bool = True):
        """
        初始化多样性增强检索器
        
        Args:
            base_retriever: 基础检索器
            diversity_weight: 多样性权重（0-1）
            max_docs_per_source: 每个来源最大文档数量
            enable_diversity: 是否启用多样性检索
        """
        self.base_retriever = base_retriever
        self.diversity_weight = diversity_weight
        self.max_docs_per_source = max_docs_per_source
        self.enable_diversity = enable_diversity
        
        # 增强基础检索器
        self._enhance_base_retriever()
        
        logger.info(f"多样性增强检索器初始化完成")
        logger.info(f"  - 多样性权重: {diversity_weight}")
        logger.info(f"  - 每源最大文档数: {max_docs_per_source}")
        logger.info(f"  - 多样性检索: {'启用' if enable_diversity else '禁用'}")
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档（多样性版本）"""
        if not self.enable_diversity:
            return self.base_retriever.get_relevant_documents(query)
        
        try:
            # 1. 获取更多的候选文档
            raw_docs = self.base_retriever.get_relevant_documents(query)
            
            # 如果基础检索器支持扩展检索，使用它
            if hasattr(self.base_retriever, '_get_relevant_documents_raw'):
                raw_docs = self.base_retriever._get_relevant_documents_raw(
                    query, k=self.base_retriever.k * 3
                )
            else:
                # 否则使用标准方法并重复调用获取更多文档
                raw_docs = self.base_retriever.get_relevant_documents(query)
            
            if not raw_docs:
                logger.warning("未检索到任何文档")
                return []
            
            # 2. 按来源分组
            docs_by_source = defaultdict(list)
            for doc in raw_docs:
                source = doc.metadata.get('source', 'unknown')
                docs_by_source[source].append(doc)
            
            logger.debug(f"检索到 {len(raw_docs)} 个候选文档，来源: {len(docs_by_source)} 个")
            
            # 3. 多样性选择
            diverse_docs = self._select_diverse_documents(docs_by_source, query)
            
            # 4. 多样性重排序
            if self.base_retriever.reranker and diverse_docs:
                diverse_docs = self._diversity_rerank(query, diverse_docs)
            
            final_docs = diverse_docs[:self.base_retriever.k]
            
            # 记录最终结果的多样性
            final_sources = [doc.metadata.get('source', 'unknown') for doc in final_docs]
            source_counter = Counter(final_sources)
            logger.debug(f"最终返回 {len(final_docs)} 个文档，来源分布: {dict(source_counter)}")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"多样性检索失败，降级到基础检索: {e}")
            return self.base_retriever._get_relevant_documents(query)
    
    def _enhance_base_retriever(self):
        """增强基础检索器功能"""
        def _get_relevant_documents_raw(query: str, k: int = None):
            """获取原始检索结果（不重排序）"""
            if k is None:
                k = self.base_retriever.k * 2
            
            try:
                # 1. 向量检索
                vector_docs = []
                vector_scores = {}
                if self.base_retriever.vectorstore.get_vectorstore():
                    try:
                        docs_with_scores = self.base_retriever.vectorstore.similarity_search_with_score(query, k=k)
                        for doc, score in docs_with_scores:
                            vector_docs.append(doc)
                            vector_scores[doc.page_content] = 1.0 / (1.0 + score)
                    except Exception as e:
                        logger.warning(f"向量检索失败: {e}")
                        vector_docs = self.base_retriever.vectorstore.similarity_search(query, k=k)
                        for i, doc in enumerate(vector_docs):
                            vector_scores[doc.page_content] = 1.0 / (1.0 + i)
                
                # 2. BM25检索
                bm25_docs = []
                bm25_scores = {}
                if self.base_retriever.enhanced_bm25:
                    try:
                        results = self.base_retriever.enhanced_bm25.search(query, k=k)
                        for result, score in results:
                            doc = Document(
                                page_content=result['content'],
                                metadata=result['metadata']
                            )
                            bm25_docs.append(doc)
                            bm25_scores[doc.page_content] = score
                    except Exception as e:
                        logger.warning(f"BM25检索失败: {e}")
                
                # 3. 合并结果（不重排序）
                final_docs = self.base_retriever._merge_results_enhanced(
                    vector_docs, bm25_docs, vector_scores, bm25_scores, query
                )
                
                return final_docs
                
            except Exception as e:
                logger.error(f"原始检索失败: {e}")
                return []
        
        # 绑定方法到基础检索器
        self.base_retriever._get_relevant_documents_raw = _get_relevant_documents_raw
    
    def _select_diverse_documents(self, docs_by_source: Dict[str, List[Document]], query: str) -> List[Document]:
        """选择多样性文档"""
        selected_docs = []
        source_counts = defaultdict(int)
        
        # 计算每个文档的相关性分数
        doc_scores = []
        for source, docs in docs_by_source.items():
            for doc in docs:
                score = self._calculate_relevance_score(query, doc.page_content)
                doc_scores.append((doc, score, source))
        
        # 按分数排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 第一轮：多样性选择
        for doc, score, source in doc_scores:
            if source_counts[source] < self.max_docs_per_source:
                selected_docs.append(doc)
                source_counts[source] += 1
                
                if len(selected_docs) >= self.base_retriever.k * 2:
                    break
        
        # 第二轮：如果还没选够，适当放宽限制
        if len(selected_docs) < self.base_retriever.k:
            for doc, score, source in doc_scores:
                if doc not in selected_docs and source_counts[source] < self.max_docs_per_source + 1:
                    selected_docs.append(doc)
                    source_counts[source] += 1
                    
                    if len(selected_docs) >= self.base_retriever.k * 2:
                        break
        
        logger.debug(f"多样性选择完成，选中 {len(selected_docs)} 个文档，来源分布: {dict(source_counts)}")
        return selected_docs
    
    def _calculate_relevance_score(self, query: str, content: str) -> float:
        """计算相关性分数"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # 关键词匹配分数
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        keyword_score = len(intersection) / len(query_words)
        
        # 长度惩罚/奖励
        content_length = len(content)
        if 200 <= content_length <= 800:
            length_factor = 1.0
        elif content_length < 200:
            length_factor = 0.6 + (content_length / 200) * 0.4
        else:
            length_factor = max(0.4, 1.0 - (content_length - 800) / 1500)
        
        # 位置奖励（关键词在前部的奖励）
        position_bonus = 0.0
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1 and pos < len(content_lower) * 0.3:
                position_bonus += 0.1
        
        final_score = keyword_score * length_factor + min(position_bonus, 0.2)
        return min(final_score, 1.0)
    
    def _diversity_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """多样性重排序"""
        try:
            # 转换为重排序器格式
            rerank_docs = []
            for doc in docs:
                rerank_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', 'unknown'),
                    'score': 0.5
                })
            
            # 使用重排序器（如果有平衡重排序器更好）
            if hasattr(self.base_retriever.reranker, '_apply_diversity_constraints'):
                # 使用平衡重排序器，返回Document对象列表
                reranked_docs = self.base_retriever.reranker.rerank(query, rerank_docs, top_k=len(docs))
                
                # 直接返回Document对象，它们已经包含了重排序分数在metadata中
                return reranked_docs
            else:
                # 使用普通重排序器的内部方法获取RerankResult对象
                if self.base_retriever.reranker.rerank_model is not None:
                    rerank_results = self.base_retriever.reranker._crossencoder_rerank(query, rerank_docs)
                else:
                    rerank_results = self.base_retriever.reranker._fallback_rerank(query, rerank_docs)
                
                # 排序
                rerank_results.sort(key=lambda x: x.final_score, reverse=True)
                
                # 应用多样性调整
                rerank_results = self._apply_diversity_bonus(rerank_results[:len(docs)])
                
                # 转换回Document格式，保留重排序分数
                final_docs = []
                for i, result in enumerate(rerank_results):
                    # 保留重排序分数信息
                    enhanced_metadata = result.metadata.copy()
                    enhanced_metadata.update({
                        'rerank_score': result.rerank_score,
                        'original_score': result.original_score,
                        'final_score': result.final_score,
                        'rerank_rank': i + 1
                    })
                    
                    doc = Document(
                        page_content=result.content,
                        metadata=enhanced_metadata
                    )
                    final_docs.append(doc)
                
                return final_docs
            
        except Exception as e:
            logger.warning(f"多样性重排序失败，使用原始顺序: {e}")
            return docs
    
    def _apply_diversity_bonus(self, rerank_results: List[RerankResult]) -> List[RerankResult]:
        """应用多样性奖励"""
        # 统计来源分布
        source_counts = defaultdict(int)
        for result in rerank_results:
            source = result.metadata.get('source', 'unknown')
            source_counts[source] += 1
        
        # 应用多样性奖励
        for result in rerank_results:
            source = result.metadata.get('source', 'unknown')
            source_count = source_counts[source]
            
            # 来源越稀少，奖励越多
            diversity_bonus = self.diversity_weight * (1.0 / (1.0 + source_count - 1))
            result.final_score = result.final_score * (1 - self.diversity_weight) + diversity_bonus
        
        # 重新排序
        rerank_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return rerank_results

class BalancedReranker(Reranker):
    """平衡重排序器"""
    
    def __init__(self, 
                 model_name: str = './cross-encoder/ms-marco-MiniLM-L6-v2',
                 diversity_weight: float = 0.2,
                 max_same_source: int = 2,
                 fallback_enabled: bool = True):
        """
        初始化平衡重排序器
        
        Args:
            model_name: 重排序模型名称
            diversity_weight: 多样性权重
            max_same_source: 同一来源最大数量
            fallback_enabled: 是否启用降级方案
        """
        super().__init__(model_name, fallback_enabled)
        self.diversity_weight = diversity_weight
        self.max_same_source = max_same_source
        logger.info(f"平衡重排序器初始化完成，多样性权重: {diversity_weight}")
    
    def rerank(self, question: str, docs: List, top_k: int = None) -> List[Document]:
        """平衡重排序"""
        # 1. 基础重排序 - 直接调用内部方法获取RerankResult
        if self.rerank_model is not None:
            base_results = self._crossencoder_rerank(question, docs)
        else:
            base_results = self._fallback_rerank(question, docs)
        
        # 排序
        base_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # 2. 应用多样性约束
        balanced_results = self._apply_diversity_constraints(base_results)
        
        # 3. 返回top-k结果
        if top_k:
            balanced_results = balanced_results[:top_k]
        
        # 4. 转换为Document对象，包含重排序分数
        reranked_docs = []
        for i, result in enumerate(balanced_results):
            # 保留重排序分数信息
            enhanced_metadata = result.metadata.copy()
            enhanced_metadata.update({
                'rerank_score': result.rerank_score,
                'original_score': result.original_score,
                'final_score': result.final_score,
                'rerank_rank': i + 1
            })
            
            doc = Document(
                page_content=result.content,
                metadata=enhanced_metadata
            )
            reranked_docs.append(doc)
        
        logger.debug(f"平衡重排序完成，返回 {len(reranked_docs)} 个Document对象")
        return reranked_docs
    
    def _apply_diversity_constraints(self, results: List[RerankResult]) -> List[RerankResult]:
        """应用多样性约束"""
        balanced_results = []
        source_counts = defaultdict(int)
        
        # 第一轮：优先选择高分且多样性好的结果
        for result in results:
            source = result.metadata.get('source', 'unknown')
            
            if source_counts[source] < self.max_same_source:
                balanced_results.append(result)
                source_counts[source] += 1
        
        # 第二轮：如果还有空位，适当放宽限制
        remaining_slots = len(results) - len(balanced_results)
        if remaining_slots > 0:
            for result in results:
                if result not in balanced_results:
                    source = result.metadata.get('source', 'unknown')
                    if source_counts[source] < self.max_same_source + 1:
                        balanced_results.append(result)
                        source_counts[source] += 1
                        
                        remaining_slots -= 1
                        if remaining_slots <= 0:
                            break
        
        # 重新设置排名
        for i, result in enumerate(balanced_results):
            result.rank = i + 1
        
        logger.debug(f"多样性约束应用完成，来源分布: {dict(source_counts)}")
        return balanced_results 