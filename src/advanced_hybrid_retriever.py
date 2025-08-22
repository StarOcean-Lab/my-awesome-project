"""
高级混合检索器
整合增强BM25检索、向量检索和RRF融合
实现关键词+向量混合检索的第一个优化目标
"""

from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger
import os

# 导入自定义模块
try:
    from .enhanced_bm25_retriever import EnhancedBM25Retriever
    from .rrf_fusion import RRFFusion, HybridRetrieverWithRRF
except ImportError:
    from enhanced_bm25_retriever import EnhancedBM25Retriever
    from rrf_fusion import RRFFusion, HybridRetrieverWithRRF

class AdvancedHybridRetriever(BaseRetriever):
    """高级混合检索器 - 实现关键词+向量混合检索优化"""
    
    # Pydantic字段声明
    vectorstore: Optional[object] = None
    documents: List[Document] = []
    vector_weight: float = 0.4
    bm25_weight: float = 0.6
    rrf_k: int = 60
    enable_force_recall: bool = True
    enable_exact_phrase: bool = True
    k: int = 10
    enhanced_bm25: Optional[object] = None
    rrf_fusion: Optional[object] = None
    
    class Config:
        arbitrary_types_allowed = True
    # 父类初始化
    def __init__(self, 
                 vectorstore,
                 documents: List[Document] = None,
                 vector_weight: float = 0.4,  # 降低向量权重
                 bm25_weight: float = 0.6,   # 提高BM25权重
                 rrf_k: int = 60,
                 enable_force_recall: bool = True,
                 enable_exact_phrase: bool = True,
                 k: int = 10):
        """
        初始化高级混合检索器
        
        Args:
            vectorstore: 向量存储
            documents: 文档列表
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
            rrf_k: RRF融合参数
            enable_force_recall: 是否启用强制召回
            enable_exact_phrase: 是否启用精确短语匹配
            k: 默认返回结果数量
        """
        # 调用父类初始化
        super().__init__(
            vectorstore=vectorstore,
            documents=documents if documents else [],
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
            enable_force_recall=enable_force_recall,
            enable_exact_phrase=enable_exact_phrase,
            k=k
        )
        self.vectorstore = vectorstore
        self.documents = documents if documents else []
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.enable_force_recall = enable_force_recall
        self.enable_exact_phrase = enable_exact_phrase
        self.k = k
        
        # 初始化增强BM25检索器
        self.enhanced_bm25 = EnhancedBM25Retriever()
        
        # 初始化RRF融合器
        self.rrf_fusion = RRFFusion(k=rrf_k)
        
        # 构建BM25索引
        self._build_bm25_index()
        
        logger.info(f"高级混合检索器初始化完成")
        logger.info(f"  权重配置: 向量={vector_weight}, BM25={bm25_weight}")
        logger.info(f"  功能配置: 强制召回={enable_force_recall}, 精确短语={enable_exact_phrase}")
        logger.info(f"  文档数量: {len(self.documents)}")
    
    def _build_bm25_index(self):
        """构建增强BM25索引"""
        if not self.documents:
            logger.warning("没有文档用于构建BM25索引")
            return
        
        logger.info("开始构建增强BM25索引...")
        
        # 转换文档格式
        bm25_docs = []
        for doc in self.documents:
            bm25_doc = {
                'content': doc.page_content,
                'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
            }
            bm25_docs.append(bm25_doc)
        
        # 构建索引
        self.enhanced_bm25.build_index(bm25_docs)
        logger.info("增强BM25索引构建完成")
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        获取相关文档 - 使用高级混合检索
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        try:
            logger.info(f"🔍 开始高级混合检索: '{query[:50]}...'")
            
            # 1. 向量检索
            vector_results = self._vector_search(query, k=self.k*2)
            
            # 2. 增强BM25检索
            bm25_results = self._enhanced_bm25_search(query, k=self.k*2)
            
            # 3. RRF融合
            fused_results = self._rrf_fusion(vector_results, bm25_results, query)
            
            # 4. 转换为Document对象
            final_docs = self._convert_to_documents(fused_results[:self.k])
            
            # 5. 记录检索统计
            self._log_retrieval_stats(query, vector_results, bm25_results, final_docs)
            
            return final_docs
            
        except Exception as e:
            logger.error(f"高级混合检索失败: {e}")
            # 降级到基础向量检索
            return self._fallback_vector_search(query)
    
    def _vector_search(self, query: str, k: int) -> List[Tuple]:
        """执行向量检索"""
        vector_results = []
        
        if self.vectorstore:
            try:
                if hasattr(self.vectorstore, 'similarity_search_with_score'):
                    vector_results = self.vectorstore.similarity_search_with_score(query, k=k)
                elif hasattr(self.vectorstore, 'get_vectorstore'):
                    vs = self.vectorstore.get_vectorstore()
                    if vs and hasattr(vs, 'similarity_search_with_score'):
                        vector_results = vs.similarity_search_with_score(query, k=k)
                    elif vs:
                        docs = vs.similarity_search(query, k=k)
                        vector_results = [(doc, 1.0/(i+1)) for i, doc in enumerate(docs)]
                else:
                    docs = self.vectorstore.similarity_search(query, k=k)
                    vector_results = [(doc, 1.0/(i+1)) for i, doc in enumerate(docs)]
                    
                logger.info(f"📊 向量检索返回 {len(vector_results)} 个结果")
                
            except Exception as e:
                logger.warning(f"向量检索失败: {e}")
        
        return vector_results
    
    def _enhanced_bm25_search(self, query: str, k: int) -> List[Tuple]:
        """执行增强BM25检索"""
        bm25_results = []
        
        try:
            # 使用增强BM25检索（包含强制召回和精确短语匹配）
            raw_results = self.enhanced_bm25.search(query, k=k)
            
            # 转换为统一格式
            for doc_dict, score in raw_results:
                doc = Document(
                    page_content=doc_dict.get('content', ''),
                    metadata=doc_dict.get('metadata', {})
                )
                bm25_results.append((doc, score))
            
            logger.info(f"📊 增强BM25检索返回 {len(bm25_results)} 个结果")
            
            # 显示强制召回和精确匹配的文档
            force_recalled_count = 0
            exact_match_count = 0
            for doc, score in bm25_results:
                if doc.metadata.get('force_recalled', False):
                    force_recalled_count += 1
                if doc.metadata.get('exact_match', False):
                    exact_match_count += 1
            
            if force_recalled_count > 0:
                logger.info(f"  🎯 强制召回文档: {force_recalled_count} 个")
            if exact_match_count > 0:
                logger.info(f"  🎯 精确匹配文档: {exact_match_count} 个")
                
        except Exception as e:
            logger.warning(f"增强BM25检索失败: {e}")
        
        return bm25_results
    
    def _rrf_fusion(self, vector_results: List[Tuple], bm25_results: List[Tuple], query: str) -> List:
        """执行RRF融合"""
        try:
            logger.info("🔄 开始RRF融合...")
            
            # 设置权重
            weights = {
                'vector': self.vector_weight,
                'bm25': self.bm25_weight
            }
            
            # 执行RRF融合
            rrf_results = self.rrf_fusion.fuse_results(
                vector_results=vector_results,
                bm25_results=bm25_results,
                weights=weights
            )
            
            logger.info(f"🔄 RRF融合完成，得到 {len(rrf_results)} 个结果")
            
            # 显示融合统计
            if rrf_results:
                top_result = rrf_results[0]
                fusion_sources = list(top_result.ranks.keys())
                logger.info(f"  Top1融合来源: {fusion_sources}")
            
            return rrf_results
            
        except Exception as e:
            logger.error(f"RRF融合失败: {e}")
            # 降级：简单合并去重
            return self._simple_merge(vector_results, bm25_results)
    
    def _simple_merge(self, vector_results: List[Tuple], bm25_results: List[Tuple]) -> List:
        """简单合并（降级方案）"""
        all_docs = {}  # content -> (doc, max_score)
        
        # 处理向量结果
        for doc, score in vector_results:
            content = doc.page_content
            weighted_score = score * self.vector_weight
            if content not in all_docs or weighted_score > all_docs[content][1]:
                all_docs[content] = (doc, weighted_score)
        
        # 处理BM25结果
        for doc, score in bm25_results:
            content = doc.page_content
            weighted_score = score * self.bm25_weight
            if content not in all_docs or weighted_score > all_docs[content][1]:
                all_docs[content] = (doc, weighted_score)
        
        # 排序
        merged_results = list(all_docs.values())
        merged_results.sort(key=lambda x: x[1], reverse=True)
        
        # 转换为RRF结果格式
        from rrf_fusion import RRFResult
        simple_results = []
        for i, (doc, score) in enumerate(merged_results):
            result = RRFResult(
                content=doc.page_content,
                metadata=doc.metadata,
                original_scores={'merged': score},
                ranks={'merged': i+1},
                rrf_score=score,
                final_rank=i+1
            )
            simple_results.append(result)
        
        return simple_results
    
    def _convert_to_documents(self, rrf_results: List) -> List[Document]:
        """将RRF结果转换为Document对象"""
        documents = []
        
        for rrf_result in rrf_results:
            # 创建Document对象
            doc = Document(
                page_content=rrf_result.content,
                metadata=rrf_result.metadata.copy()
            )
            
            # 添加融合信息到metadata
            doc.metadata.update({
                'rrf_score': rrf_result.rrf_score,
                'rrf_rank': rrf_result.final_rank,
                'fusion_sources': list(rrf_result.ranks.keys()),
                'original_ranks': rrf_result.ranks,
                'original_scores': rrf_result.original_scores
            })
            
            documents.append(doc)
        
        return documents
    
    def _fallback_vector_search(self, query: str) -> List[Document]:
        """降级向量检索"""
        try:
            if self.vectorstore:
                if hasattr(self.vectorstore, 'similarity_search'):
                    return self.vectorstore.similarity_search(query, k=self.k)
                elif hasattr(self.vectorstore, 'get_vectorstore'):
                    vs = self.vectorstore.get_vectorstore()
                    if vs:
                        return vs.similarity_search(query, k=self.k)
        except Exception as e:
            logger.error(f"降级向量检索也失败: {e}")
        
        return []
    
    def _log_retrieval_stats(self, query: str, vector_results: List, bm25_results: List, final_docs: List):
        """记录检索统计信息"""
        try:
            logger.info("📊 高级混合检索统计:")
            logger.info(f"  查询: '{query[:50]}...'")
            logger.info(f"  向量检索: {len(vector_results)} 个结果")
            logger.info(f"  BM25检索: {len(bm25_results)} 个结果")
            logger.info(f"  最终结果: {len(final_docs)} 个文档")
            
            # 统计来源分布
            if final_docs:
                from collections import Counter
                sources = []
                for doc in final_docs:
                    fusion_sources = doc.metadata.get('fusion_sources', [])
                    sources.extend(fusion_sources)
                
                source_counter = Counter(sources)
                logger.info(f"  融合来源分布: {dict(source_counter)}")
                
                # 显示Top3结果
                logger.info("  Top3结果:")
                for i, doc in enumerate(final_docs[:3]):
                    source_file = os.path.basename(doc.metadata.get('source', 'unknown'))
                    rrf_score = doc.metadata.get('rrf_score', 0)
                    fusion_sources = doc.metadata.get('fusion_sources', [])
                    preview = doc.page_content[:100].replace('\n', ' ')
                    
                    logger.info(f"    {i+1}. {source_file} (RRF={rrf_score:.4f}, 来源={fusion_sources})")
                    logger.info(f"       内容: {preview}...")
        
        except Exception as e:
            logger.warning(f"记录统计信息失败: {e}")
    
    def get_detailed_results(self, query: str) -> Dict:
        """
        获取详细的检索结果（用于调试和分析）
        
        Args:
            query: 查询文本
            
        Returns:
            包含详细检索信息的字典
        """
        try:
            # 执行各个检索步骤
            vector_results = self._vector_search(query, k=self.k*2)
            bm25_results = self._enhanced_bm25_search(query, k=self.k*2)
            rrf_results = self._rrf_fusion(vector_results, bm25_results, query)
            
            # 整理详细信息
            detailed_info = {
                'query': query,
                'vector_results': [
                    {
                        'content': doc.page_content[:200],
                        'metadata': doc.metadata,
                        'score': score
                    } for doc, score in vector_results[:5]
                ],
                'bm25_results': [
                    {
                        'content': doc.page_content[:200],
                        'metadata': doc.metadata,
                        'score': score
                    } for doc, score in bm25_results[:5]
                ],
                'rrf_results': [
                    {
                        'content': result.content[:200],
                        'metadata': result.metadata,
                        'rrf_score': result.rrf_score,
                        'ranks': result.ranks,
                        'original_scores': result.original_scores
                    } for result in rrf_results[:5]
                ],
                'fusion_weights': {
                    'vector': self.vector_weight,
                    'bm25': self.bm25_weight
                },
                'retrieval_config': {
                    'rrf_k': self.rrf_k,
                    'enable_force_recall': self.enable_force_recall,
                    'enable_exact_phrase': self.enable_exact_phrase
                }
            }
            
            return detailed_info
            
        except Exception as e:
            logger.error(f"获取详细结果失败: {e}")
            return {'error': str(e)} 