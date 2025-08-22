"""
LangChain增强混合检索器
结合BM25和向量检索，集成重排序功能，提供更精准的搜索结果
"""

from typing import List, Dict, Optional, Tuple, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# 修复导入 - 使用兼容的导入方式
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    try:
        from langchain.retrievers import BM25Retriever
    except ImportError:
        BM25Retriever = None

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever  
    except ImportError:
        EnsembleRetriever = None

import numpy as np
from dataclasses import dataclass
from loguru import logger
import os

try:
    from .langchain_vectorstore import LangChainVectorStore
    from .bm25_retriever import BM25Retriever as EnhancedBM25Retriever
    from .reranker import Reranker
except ImportError:
    from langchain_vectorstore import LangChainVectorStore
    from bm25_retriever import BM25Retriever as EnhancedBM25Retriever
    from reranker import Reranker


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float
    source: str  # 'bm25', 'vector', 'hybrid'
    rank: int


class SimpleBM25Retriever:
    """简单的BM25检索器（降级方案）"""
    
    def __init__(self, documents: List[Document], k: int = 5):
        self.documents = documents
        self.k = k
        
        # 简单的关键词索引
        self.keyword_index = {}
        self._build_index()
    
    def _build_index(self):
        """构建关键词索引"""
        import jieba
        
        for i, doc in enumerate(self.documents):
            words = jieba.lcut(doc.page_content.lower())
            for word in words:
                if len(word) > 1:  # 忽略单字符
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    self.keyword_index[word].append(i)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档"""
        import jieba
        
        query_words = jieba.lcut(query.lower())
        doc_scores = {}
        
        for word in query_words:
            if word in self.keyword_index:
                for doc_idx in self.keyword_index[word]:
                    if doc_idx not in doc_scores:
                        doc_scores[doc_idx] = 0
                    doc_scores[doc_idx] += 1
        
        # 按分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回top-k文档
        results = []
        for doc_idx, score in sorted_docs[:self.k]:
            results.append(self.documents[doc_idx])
        
        return results


class LangChainHybridRetriever(BaseRetriever):
    """基于LangChain的混合检索器"""
    
    # Pydantic字段定义
    vectorstore: LangChainVectorStore
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    k: int = 5
    bm25_retriever: Optional[Any] = None
    enhanced_bm25: Optional[Any] = None
    reranker: Optional[Any] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, 
                 vectorstore: LangChainVectorStore,
                 documents: List[Document] = None,
                 bm25_weight: float = 0.5,
                 vector_weight: float = 0.5,
                 k: int = 5,
                 enable_reranking: bool = True):
        """
        初始化增强混合检索器
        
        Args:
            vectorstore: LangChain向量存储
            documents: 文档列表（用于BM25）
            bm25_weight: BM25权重
            vector_weight: 向量检索权重
            k: 返回结果数量
            enable_reranking: 是否启用重排序
        """
        super().__init__(
            vectorstore=vectorstore,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            k=k
        )
        
        # 初始化增强BM25检索器
        self.enhanced_bm25 = None
        if documents:
            try:
                # 转换LangChain文档格式为我们的格式
                doc_data = []
                for doc in documents:
                    doc_data.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'source': doc.metadata.get('source', 'unknown')
                    })
                
                self.enhanced_bm25 = EnhancedBM25Retriever()
                self.enhanced_bm25.add_documents(doc_data)
                logger.info("增强BM25检索器初始化成功")
            except Exception as e:
                logger.warning(f"增强BM25检索器初始化失败: {e}")
        
        # 初始化标准BM25作为降级方案
        if documents and BM25Retriever is not None:
            try:
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                if hasattr(self.bm25_retriever, 'k'):
                    self.bm25_retriever.k = k
                logger.info("标准BM25检索器作为降级方案")
            except Exception as e:
                logger.warning(f"标准BM25检索器初始化失败: {e}")
                self.bm25_retriever = SimpleBM25Retriever(documents, k)
        elif documents:
            self.bm25_retriever = SimpleBM25Retriever(documents, k)
        else:
            self.bm25_retriever = None
            
        # 初始化重排序器
        self.reranker = None
        if enable_reranking:
            try:
                self.reranker = Reranker()
                logger.info("重排序器初始化成功")
            except Exception as e:
                logger.warning(f"重排序器初始化失败: {e}")
                
        logger.info(f"增强混合检索器初始化完成，BM25权重: {bm25_weight}, 向量权重: {vector_weight}, 重排序: {enable_reranking}")
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        智能混合检索：查询优化 → 策略选择 → 多阶段检索 → 竞赛过滤
        
        Args:
            query: 查询文本
            
        Returns:
            排序后的相关文档列表
        """
        try:
            try:
                from config import Config
            except ImportError:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from config import Config
            
            # === 1. 查询优化和策略选择 ===
            try:
                try:
                    from query_enhancer import get_enhanced_query_optimizer
                except ImportError:
                    from .query_enhancer import get_enhanced_query_optimizer
                optimizer = get_enhanced_query_optimizer()
                optimized_query, strategy = optimizer.optimize_query_for_retrieval(query)
                logger.info(f"🔍 查询优化: '{query}' -> '{optimized_query}'")
                logger.info(f"🎯 检索策略: {strategy}")
            except Exception as e:
                logger.warning(f"查询优化失败，使用原始查询: {e}")
                optimized_query = query
                strategy = {
                    "question_type": "basic",
                    "competition_filter": None,
                    "alpha": 0.5,
                    "vector_k": 20,
                    "bm25_k": 30
                }
            
            # === 2. 动态调整检索参数 ===
            # 根据策略调整权重
            original_vector_weight = self.vector_weight
            original_bm25_weight = self.bm25_weight
            
            alpha = strategy.get("alpha", 0.5)
            self.vector_weight = alpha
            self.bm25_weight = 1.0 - alpha
            
            # 获取多阶段检索配置
            stages = getattr(Config, 'RETRIEVAL_STAGES', {})
            enable_multi_stage = stages.get('enable_multi_stage', True)
            
            # 根据策略调整检索数量
            if strategy.get("question_type") == "competition":
                stages = stages.copy()  # 避免修改原配置
                stages["stage1_vector_k"] = strategy.get("vector_k", 15)
                stages["stage1_bm25_k"] = strategy.get("bm25_k", 35)
                logger.info(f"🏆 竞赛模式检索: 向量{stages['stage1_vector_k']}, BM25{stages['stage1_bm25_k']}")
            
            # === 3. 执行检索 ===
            try:
                if enable_multi_stage:
                    results = self._multi_stage_retrieval_with_filter(optimized_query, stages, strategy)
                else:
                    results = self._traditional_retrieval_with_filter(optimized_query, strategy)
            finally:
                # 恢复原始权重
                self.vector_weight = original_vector_weight
                self.bm25_weight = original_bm25_weight
            
            return results
                
        except Exception as e:
            logger.error(f"检索失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 降级到基础向量检索
            if self.vectorstore:
                logger.info("🔄 降级到基础向量检索")
                return self.vectorstore.similarity_search(query, k=self.k)
            else:
                return []
    
    def _multi_stage_retrieval_with_filter(self, query: str, stages: Dict, strategy: Dict) -> List[Document]:
        """带竞赛过滤的多阶段检索实现"""
        stage1_vector_k = stages.get('stage1_vector_k', 50)
        stage1_bm25_k = stages.get('stage1_bm25_k', 50)
        stage2_candidate_k = stages.get('stage2_candidate_k', 80)
        final_k = stages.get('final_k', 10)
        
        logger.info(f"🔍 === 开始智能多阶段检索 ===")
        logger.info(f"📝 查询: {query}")
        logger.info(f"🎯 检索配置: 向量{stage1_vector_k} + BM25{stage1_bm25_k} → 候选{stage2_candidate_k} → 最终{final_k}")
        
        competition_filter = strategy.get("competition_filter")
        if competition_filter:
            logger.info(f"🏆 竞赛过滤: {competition_filter}")
            
        # 执行原有的多阶段检索逻辑
        results = self._multi_stage_retrieval_core(query, stages)
        
        # 应用竞赛过滤和加权
        if competition_filter:
            results = self._apply_competition_filter(results, competition_filter, strategy)
        
        return results
    
    def _traditional_retrieval_with_filter(self, query: str, strategy: Dict) -> List[Document]:
        """带竞赛过滤的传统检索实现"""
        logger.info(f"🔍 === 使用智能传统检索方法 ===")
        logger.info(f"📝 查询: {query}")
        
        competition_filter = strategy.get("competition_filter")
        if competition_filter:
            logger.info(f"🏆 竞赛过滤: {competition_filter}")
            
        # 执行原有的传统检索逻辑
        results = self._traditional_retrieval_core(query)
        
        # 应用竞赛过滤和加权
        if competition_filter:
            results = self._apply_competition_filter(results, competition_filter, strategy)
        
        return results

    def _multi_stage_retrieval_core(self, query: str, stages: Dict) -> List[Document]:
        """多阶段检索核心实现（原_multi_stage_retrieval）"""
        stage1_vector_k = stages.get('stage1_vector_k', 50)
        stage1_bm25_k = stages.get('stage1_bm25_k', 50)
        stage2_candidate_k = stages.get('stage2_candidate_k', 80)
        final_k = stages.get('final_k', 10)
        
        # === 第一阶段：广泛检索 ===
        logger.info("🚀 第一阶段：广泛检索...")
        
        # 1.1 向量检索 - 获取更多候选
        logger.info(f"🧠 向量检索（目标：{stage1_vector_k}个）...")
        vector_docs = []
        vector_scores = {}
        if self.vectorstore:
            try:
                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=stage1_vector_k)
                for doc, score in docs_with_scores:
                    vector_docs.append(doc)
                    vector_scores[doc.page_content] = 1.0 / (1.0 + score)
                logger.info(f"📄 向量检索返回 {len(vector_docs)} 个文档")
            except Exception as e:
                logger.warning(f"向量检索失败: {e}")
                vector_docs = self.vectorstore.similarity_search(query, k=stage1_vector_k)
                for i, doc in enumerate(vector_docs):
                    vector_scores[doc.page_content] = 1.0 / (1.0 + i)
        else:
            logger.warning("⚠️ 向量存储不可用")

        # 1.2 BM25检索 - 获取更多候选
        logger.info(f"📊 BM25检索（目标：{stage1_bm25_k}个）...")
        bm25_docs = []
        bm25_scores = {}
        if self.enhanced_bm25:
            try:
                results = self.enhanced_bm25.search(query, k=stage1_bm25_k)
                for result, score in results:
                    doc = Document(
                        page_content=result['content'],
                        metadata=result['metadata']
                    )
                    doc.metadata['source_type'] = 'bm25'
                    bm25_docs.append(doc)
                    bm25_scores[doc.page_content] = score
                logger.info(f"📊 增强BM25返回 {len(bm25_docs)} 个文档")
            except Exception as e:
                logger.warning(f"增强BM25检索失败: {e}")
        elif self.bm25_retriever:
            try:
                if hasattr(self.bm25_retriever, 'get_relevant_documents'):
                    bm25_docs = self.bm25_retriever.get_relevant_documents(query)[:stage1_bm25_k]
                else:
                    bm25_docs = self.bm25_retriever.invoke(query)[:stage1_bm25_k]
                for i, doc in enumerate(bm25_docs):
                    doc.metadata['source_type'] = 'bm25'
                    bm25_scores[doc.page_content] = 1.0 / (1.0 + i)
                logger.info(f"📊 标准BM25返回 {len(bm25_docs)} 个文档")
            except Exception as e:
                logger.warning(f"标准BM25检索失败: {e}")
        else:
            logger.warning("⚠️ BM25检索器不可用")

        # === 第二阶段：合并去重 ===
        logger.info("🔄 第二阶段：合并去重...")
        all_candidates = self._merge_candidates(vector_docs, bm25_docs, vector_scores, bm25_scores)
        
        # 限制候选数量
        candidates = all_candidates[:stage2_candidate_k]
        logger.info(f"📋 合并后候选文档: {len(candidates)} 个（目标: {stage2_candidate_k}）")
        
        # 显示候选文档来源分布
        candidate_sources = [os.path.basename(doc.metadata.get('source', 'unknown')) for doc in candidates]
        from collections import Counter
        source_counter = Counter(candidate_sources)
        logger.info("📊 候选文档来源分布:")
        for source, count in source_counter.most_common():
            logger.info(f"  📄 {source}: {count} 个片段")

        # === 第三阶段：重排序选择 ===
        logger.info("🎯 第三阶段：重排序选择...")
        if self.reranker and candidates:
            try:
                logger.info(f"🔧 使用重排序器从 {len(candidates)} 个候选中选择 {final_k} 个")
                reranked_docs = self.reranker.rerank(query, candidates, top_k=final_k)
                logger.info(f"✅ 重排序完成，返回 {len(reranked_docs)} 个文档")
                
                # 显示最终结果统计
                final_sources = [os.path.basename(doc.metadata.get('source', 'unknown')) for doc in reranked_docs]
                final_counter = Counter(final_sources)
                logger.info("📊 最终结果来源分布:")
                for source, count in final_counter.most_common():
                    logger.info(f"  📄 {source}: {count} 个片段")
                
                return reranked_docs
            except Exception as e:
                logger.warning(f"重排序失败，使用候选排序: {e}")
                return candidates[:final_k]
        else:
            logger.info("⏭️ 跳过重排序，直接返回候选文档")
            return candidates[:final_k]

    def _merge_candidates(self, vector_docs: List[Document], bm25_docs: List[Document], 
                         vector_scores: Dict, bm25_scores: Dict) -> List[Document]:
        """合并候选文档并计算综合分数"""
        all_docs = {}
        
        # 处理向量检索结果
        for doc in vector_docs:
            content = doc.page_content
            source = doc.metadata.get('source', 'unknown')
            unique_key = f"{source}_{content[:100]}"
            
            vector_score = vector_scores.get(content, 0.0)
            
            doc.metadata['source_type'] = 'vector'
            doc.metadata['vector_score'] = vector_score
            doc.metadata['bm25_score'] = 0.0
            doc.metadata['hybrid_score'] = self.vector_weight * vector_score
            
            all_docs[unique_key] = {
                'document': doc,
                'score': self.vector_weight * vector_score,
                'sources': {'vector'}
            }
        
        # 处理BM25检索结果
        for doc in bm25_docs:
            content = doc.page_content
            source = doc.metadata.get('source', 'unknown')
            unique_key = f"{source}_{content[:100]}"
            
            bm25_score = bm25_scores.get(content, 0.0)
            
            if unique_key in all_docs:
                # 文档已存在，更新分数
                existing_doc = all_docs[unique_key]['document']
                existing_doc.metadata['source_type'] = 'hybrid'
                existing_doc.metadata['bm25_score'] = bm25_score
                existing_doc.metadata['hybrid_score'] += self.bm25_weight * bm25_score
                all_docs[unique_key]['score'] += self.bm25_weight * bm25_score
                all_docs[unique_key]['sources'].add('bm25')
            else:
                # 新文档
                doc.metadata['source_type'] = 'bm25'
                doc.metadata['vector_score'] = 0.0
                doc.metadata['bm25_score'] = bm25_score
                doc.metadata['hybrid_score'] = self.bm25_weight * bm25_score
                
                all_docs[unique_key] = {
                    'document': doc,
                    'score': self.bm25_weight * bm25_score,
                    'sources': {'bm25'}
                }
        
        # 按综合分数排序
        sorted_results = sorted(all_docs.values(), key=lambda x: x['score'], reverse=True)
        return [result['document'] for result in sorted_results]
    
    def _traditional_retrieval(self, query: str) -> List[Document]:
        """传统检索方法（兼容旧版本）"""
        logger.info(f"🔍 === 使用传统检索方法 ===")
        logger.info(f"📝 查询: {query}")
        
        # 1. 向量检索
        logger.info("🧠 开始向量检索...")
        vector_docs = []
        if self.vectorstore:
            vector_docs = self.vectorstore.similarity_search_with_score(query, k=self.k)
            logger.info(f"📄 向量检索返回 {len(vector_docs)} 个文档")
        else:
            logger.warning("⚠️ 向量存储不可用")

        # 2. BM25检索
        logger.info("📊 开始BM25检索...")
        bm25_docs = []
        if self.enhanced_bm25:
            try:
                results = self.enhanced_bm25.search(query, k=self.k)
                for result, score in results:
                    doc = Document(
                        page_content=result['content'],
                        metadata=result['metadata']
                    )
                    doc.metadata['source_type'] = 'bm25'
                    bm25_docs.append(doc)
                logger.info(f"📊 增强BM25返回 {len(bm25_docs)} 个文档")
            except Exception as e:
                logger.warning(f"增强BM25检索失败: {e}")
        elif self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)[:self.k]
                for doc in bm25_docs:
                    doc.metadata['source_type'] = 'bm25'
                logger.info(f"📊 标准BM25返回 {len(bm25_docs)} 个文档")
            except Exception as e:
                logger.warning(f"BM25检索失败: {e}")
        else:
            logger.warning("⚠️ BM25检索器不可用")

        # 3. 合并和重排序
        logger.info("🔗 开始合并检索结果...")
        
        # 提取向量文档
        vector_only_docs = [doc for doc, score in vector_docs] if vector_docs else []
        for doc in vector_only_docs:
            doc.metadata['source_type'] = 'vector'
        
        # 合并所有文档
        all_docs = vector_only_docs + bm25_docs
        
        # 4. 重排序（如果启用）
        if self.reranker and all_docs:
            try:
                logger.info("🎯 开始重排序...")
                reranked_docs = self.reranker.rerank(query, all_docs, top_k=self.k)
                logger.info(f"✅ 重排序完成，返回 {len(reranked_docs)} 个文档")
                return reranked_docs
            except Exception as e:
                logger.warning(f"重排序失败，使用原始排序: {e}")
        
        # 5. 返回结果
        return all_docs[:self.k]
    
    def _merge_results_enhanced(self, vector_docs: List[Document], bm25_docs: List[Document], 
                              vector_scores: Dict, bm25_scores: Dict, query: str) -> List[Document]:
        """
        增强的结果合并方法，使用真实分数而不是排序位置，增强去重机制
        
        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            vector_scores: 向量检索分数
            bm25_scores: BM25检索分数
            query: 查询文本
            
        Returns:
            合并后的文档列表
        """
        try:
            # 合并计算混合分数 - 使用来源+内容作为唯一键
            all_docs = {}
            
            # 处理向量检索结果
            for doc in vector_docs:
                content = doc.page_content
                source = doc.metadata.get('source', 'unknown')
                
                # 使用来源+内容片段作为唯一键，避免重复文档
                unique_key = f"{source}_{content[:100]}"
                
                score = vector_scores.get(content, 0) * self.vector_weight
                if unique_key in all_docs:
                    all_docs[unique_key]['score'] += score
                    all_docs[unique_key]['sources'].add('vector')
                else:
                    all_docs[unique_key] = {
                        'document': doc,
                        'score': score,
                        'sources': {'vector'},
                        'source': source
                    }
            
            # 处理BM25检索结果
            for doc in bm25_docs:
                content = doc.page_content
                source = doc.metadata.get('source', 'unknown')
                
                # 使用来源+内容片段作为唯一键，避免重复文档
                unique_key = f"{source}_{content[:100]}"
                
                score = bm25_scores.get(content, 0) * self.bm25_weight
                if unique_key in all_docs:
                    all_docs[unique_key]['score'] += score
                    all_docs[unique_key]['sources'].add('bm25')
                else:
                    all_docs[unique_key] = {
                        'document': doc,
                        'score': score,
                        'sources': {'bm25'},
                        'source': source
                    }
            
            # 增强排序：对于定义性查询给予特殊权重
            query_lower = query.lower()
            is_definition_query = any(word in query_lower for word in ['什么', '基本要求', '任务', '要求'])
            
            for key, doc_info in all_docs.items():
                content_lower = doc_info['document'].page_content.lower()
                
                # 定义匹配奖励
                if is_definition_query:
                    definition_bonus = 0.0
                    
                    # 检查任务定义标识符
                    definition_indicators = ['任务一', '任务二', '任务三', '任务四', '任务五', '基本要求', '任务要求', '任务情境']
                    for indicator in definition_indicators:
                        if indicator in content_lower:
                            definition_bonus += 0.2
                    
                    # 特定关键词匹配奖励
                    if '交通信号灯' in query_lower and '交通信号灯' in content_lower:
                        definition_bonus += 0.5
                    elif '信号灯' in query_lower and '信号灯' in content_lower:
                        definition_bonus += 0.3
                    
                    # 应用定义奖励
                    doc_info['score'] *= (1 + definition_bonus)
            
            # 按分数排序
            sorted_results = sorted(all_docs.values(), key=lambda x: x['score'], reverse=True)
            
            # 返回文档列表，确保去重
            return [result['document'] for result in sorted_results[:self.k * 2]]
            
        except Exception as e:
            logger.error(f"结果合并失败: {e}")
            # 降级方案：简单拼接并去重
            seen_sources = set()
            unique_docs = []
            for doc in (vector_docs + bm25_docs):
                source = doc.metadata.get('source', 'unknown')
                content_key = f"{source}_{doc.page_content[:100]}"
                if content_key not in seen_sources:
                    seen_sources.add(content_key)
                    unique_docs.append(doc)
            return unique_docs[:self.k * 2]

    def _merge_results(self, vector_docs: List[Document], bm25_docs: List[Document], query: str) -> List[Document]:
        """
        合并BM25和向量检索结果
        
        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            query: 查询文本
            
        Returns:
            合并后的文档列表
        """
        try:
            # 计算向量相似度分数
            vector_scores = {}
            if self.vectorstore.get_vectorstore():
                try:
                    docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
                    for doc, score in docs_with_scores:
                        vector_scores[doc.page_content] = 1.0 / (1.0 + score)  # 转换为相似度分数
                except Exception as e:
                    logger.warning(f"获取向量分数失败: {e}")
                    # 使用排序位置作为分数
                    for i, doc in enumerate(vector_docs):
                        vector_scores[doc.page_content] = 1.0 / (1.0 + i)
            
            # 计算BM25分数（基于排序位置）
            bm25_scores = {}
            for i, doc in enumerate(bm25_docs):
                bm25_scores[doc.page_content] = 1.0 / (1.0 + i)  # 基于排序位置的分数
            
            # 合并计算混合分数
            all_docs = {}
            
            # 处理向量检索结果
            for doc in vector_docs:
                content = doc.page_content
                vector_score = vector_scores.get(content, 0.0)
                bm25_score = bm25_scores.get(content, 0.0)
                hybrid_score = self.vector_weight * vector_score + self.bm25_weight * bm25_score
                
                if content not in all_docs or hybrid_score > all_docs[content]['score']:
                    all_docs[content] = {
                        'document': doc,
                        'score': hybrid_score,
                        'vector_score': vector_score,
                        'bm25_score': bm25_score
                    }
            
            # 处理BM25检索结果
            for doc in bm25_docs:
                content = doc.page_content
                if content not in all_docs:
                    vector_score = vector_scores.get(content, 0.0)
                    bm25_score = bm25_scores.get(content, 0.0)
                    hybrid_score = self.vector_weight * vector_score + self.bm25_weight * bm25_score
                    
                    all_docs[content] = {
                        'document': doc,
                        'score': hybrid_score,
                        'vector_score': vector_score,
                        'bm25_score': bm25_score
                    }
            
            # 按分数排序
            sorted_results = sorted(all_docs.values(), key=lambda x: x['score'], reverse=True)
            
            # 返回top-k结果
            return [result['document'] for result in sorted_results[:self.k]]
            
        except Exception as e:
            logger.error(f"结果合并失败: {e}")
            # 降级处理：返回向量检索结果
            return vector_docs[:self.k] if vector_docs else bm25_docs[:self.k]
    
    def get_detailed_results(self, query: str) -> List[RetrievalResult]:
        """
        获取详细的检索结果
        
        Args:
            query: 查询文本
            
        Returns:
            详细检索结果列表
        """
        try:
            results = []
            
            # 向量检索
            if self.vectorstore.get_vectorstore():
                try:
                    docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
                    for i, (doc, score) in enumerate(docs_with_scores):
                        results.append(RetrievalResult(
                            document=doc,
                            score=1.0 / (1.0 + score),
                            source='vector',
                            rank=i + 1
                        ))
                except Exception as e:
                    logger.warning(f"向量检索详细结果获取失败: {e}")
                    # 使用基本向量检索
                    vector_docs = self.vectorstore.similarity_search(query, k=self.k)
                    for i, doc in enumerate(vector_docs):
                        results.append(RetrievalResult(
                            document=doc,
                            score=1.0 / (1.0 + i),
                            source='vector',
                            rank=i + 1
                        ))
            
            # BM25检索
            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                for i, doc in enumerate(bm25_docs):
                    # 检查是否已存在
                    existing = next((r for r in results if r.document.page_content == doc.page_content), None)
                    if existing:
                        existing.source = 'hybrid'
                    else:
                        results.append(RetrievalResult(
                            document=doc,
                            score=1.0 / (1.0 + i),
                            source='bm25',
                            rank=i + 1
                        ))
            
            # 按分数排序
            results.sort(key=lambda x: x.score, reverse=True)
            
            # 更新排名
            for i, result in enumerate(results[:self.k]):
                result.rank = i + 1
            
            return results[:self.k]
            
        except Exception as e:
            logger.error(f"详细检索失败: {e}")
            return []
    
    def update_weights(self, bm25_weight: float, vector_weight: float):
        """
        更新检索权重
        
        Args:
            bm25_weight: BM25权重
            vector_weight: 向量检索权重
        """
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        logger.info(f"检索权重已更新，BM25: {bm25_weight}, 向量: {vector_weight}")
    
    def add_documents(self, documents: List[Document]):
        """
        添加文档到检索器
        
        Args:
            documents: 文档列表
        """
        try:
            # 更新向量存储
            self.vectorstore.add_documents(documents)
            
            # 重新初始化BM25检索器
            if BM25Retriever is not None:
                try:
                    self.bm25_retriever = BM25Retriever.from_documents(documents)
                    if hasattr(self.bm25_retriever, 'k'):
                        self.bm25_retriever.k = self.k
                    logger.info("BM25检索器重新初始化成功")
                except Exception as e:
                    logger.warning(f"BM25检索器重新初始化失败: {e}")
                    self.bm25_retriever = SimpleBM25Retriever(documents, self.k)
            else:
                self.bm25_retriever = SimpleBM25Retriever(documents, self.k)
            
            logger.info(f"检索器更新完成，添加 {len(documents)} 个文档")
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise


class LangChainEnsembleRetriever:
    """基于LangChain EnsembleRetriever的实现"""
    
    def __init__(self, 
                 vectorstore: LangChainVectorStore,
                 documents: List[Document] = None,
                 weights: List[float] = [0.5, 0.5]):
        """
        初始化集成检索器
        
        Args:
            vectorstore: 向量存储
            documents: 文档列表
            weights: 检索器权重
        """
        self.vectorstore = vectorstore
        self.weights = weights
        
        # 创建检索器列表
        retrievers = []
        
        # 向量检索器
        if vectorstore.get_vectorstore():
            vector_retriever = vectorstore.get_vectorstore().as_retriever()
            retrievers.append(vector_retriever)
        
        # BM25检索器
        if documents and BM25Retriever is not None:
            try:
                bm25_retriever = BM25Retriever.from_documents(documents)
                retrievers.append(bm25_retriever)
            except Exception as e:
                logger.warning(f"EnsembleRetriever中BM25初始化失败: {e}")
        
        # 创建集成检索器
        if len(retrievers) > 1 and EnsembleRetriever is not None:
            try:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=retrievers,
                    weights=weights
                )
            except Exception as e:
                logger.warning(f"EnsembleRetriever初始化失败: {e}")
                self.ensemble_retriever = retrievers[0] if retrievers else None
        elif len(retrievers) == 1:
            self.ensemble_retriever = retrievers[0]
        else:
            self.ensemble_retriever = None
        
        logger.info(f"集成检索器初始化完成，检索器数量: {len(retrievers)}, 权重: {weights}")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        try:
            if self.ensemble_retriever:
                if hasattr(self.ensemble_retriever, 'k'):
                    self.ensemble_retriever.k = k
                
                if hasattr(self.ensemble_retriever, 'get_relevant_documents'):
                    return self.ensemble_retriever.get_relevant_documents(query)
                elif hasattr(self.ensemble_retriever, 'invoke'):
                    return self.ensemble_retriever.invoke(query)
                else:
                    logger.warning("检索器方法调用失败")
                    return []
            else:
                return []
                
        except Exception as e:
            logger.error(f"集成检索失败: {e}")
            return [] 

    def _apply_competition_filter(self, documents: List[Document], competition_type: str, strategy: Dict) -> List[Document]:
        """
        应用竞赛过滤和加权
        
        Args:
            documents: 原始检索结果
            competition_type: 竞赛类型
            strategy: 检索策略
            
        Returns:
            过滤和加权后的文档列表
        """
        try:
            try:
                from config import Config
            except ImportError:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from config import Config
            competition_mapping = getattr(Config, 'COMPETITION_MAPPING', {})
            
            if competition_type not in competition_mapping:
                logger.warning(f"未知竞赛类型: {competition_type}")
                return documents
            
            comp_info = competition_mapping[competition_type]
            file_pattern = comp_info.get('file_pattern', '')
            boost_keywords = comp_info.get('keywords', [])
            exact_match_boost = comp_info.get('exact_match_boost', 2.0)
            
            logger.info(f"🎯 应用竞赛过滤: {competition_type}")
            logger.info(f"📁 文件模式: {file_pattern}")
            logger.info(f"🔑 关键词: {boost_keywords}")
            
            # 分离匹配和非匹配文档
            matched_docs = []
            other_docs = []
            
            for doc in documents:
                source = doc.metadata.get('source', '')
                source_name = source.split('/')[-1] if source else ''
                
                # 检查文件名匹配
                is_file_match = self._check_file_pattern_match(source_name, file_pattern)
                
                # 检查内容关键词匹配
                is_content_match = any(keyword in doc.page_content for keyword in boost_keywords)
                
                if is_file_match or is_content_match:
                    # 为匹配文档添加加权标记
                    doc.metadata['competition_match'] = True
                    doc.metadata['match_score'] = exact_match_boost
                    matched_docs.append(doc)
                    logger.debug(f"✅ 匹配文档: {source_name} (文件匹配: {is_file_match}, 内容匹配: {is_content_match})")
                else:
                    doc.metadata['competition_match'] = False
                    doc.metadata['match_score'] = 1.0
                    other_docs.append(doc)
            
            logger.info(f"🎯 过滤结果: {len(matched_docs)} 个匹配文档, {len(other_docs)} 个其他文档")
            
            # 重新排序：匹配文档优先，然后是其他文档
            filtered_docs = matched_docs + other_docs[:max(0, self.k - len(matched_docs))]
            
            return filtered_docs[:self.k]
            
        except Exception as e:
            logger.error(f"竞赛过滤失败: {e}")
            return documents
    
    def _check_file_pattern_match(self, filename: str, pattern: str) -> bool:
        """检查文件名是否匹配模式"""
        import fnmatch
        try:
            # 简单的模式匹配
            if '*' in pattern:
                return fnmatch.fnmatch(filename, pattern)
            else:
                return pattern in filename
        except Exception as e:
            logger.warning(f"文件模式匹配失败: {e}")
            return False 