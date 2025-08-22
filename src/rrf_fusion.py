"""
RRF (Reciprocal Rank Fusion) 算法实现
用于融合多个检索系统的结果，特别是向量检索和BM25检索
"""

from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger
import math

# 快速定义RRFResult类
@dataclass
class RRFResult:
    """RRF融合结果"""
    content: str
    metadata: Dict
    original_scores: Dict  # 来源 -> 原始分数
    ranks: Dict  # 来源 -> 排名
    rrf_score: float
    final_rank: int

class RRFFusion:
    """RRF (Reciprocal Rank Fusion) 融合器"""
    
    def __init__(self, k: int = 60):
        """
        初始化RRF融合器
        
        Args:
            k: RRF参数，控制排名的平滑程度，一般取60
        """
        self.k = k
        logger.info(f"RRF融合器初始化完成，k={k}")
    
    def fuse_results(self, 
                    vector_results: List[Tuple],
                    bm25_results: List[Tuple],
                    weights: Optional[Dict[str, float]] = None) -> List[RRFResult]:
        """
        使用RRF算法融合向量检索和BM25检索结果
        
        Args:
            vector_results: 向量检索结果列表 [(document, score), ...]
            bm25_results: BM25检索结果列表 [(document, score), ...]
            weights: 权重字典 {'vector': weight, 'bm25': weight}，默认等权重
            
        Returns:
            融合后的结果列表，按RRF分数排序
        """
        if weights is None:
            weights = {'vector': 1.0, 'bm25': 1.0}
        
        logger.info(f"开始RRF融合: 向量结果{len(vector_results)}个, BM25结果{len(bm25_results)}个")
        
        # 构建文档内容到信息的映射
        doc_info = {}  # content -> {metadata, sources: {source: (score, rank)}}
        
        # 处理向量检索结果
        self._process_results(vector_results, 'vector', doc_info, weights.get('vector', 1.0))
        
        # 处理BM25检索结果
        self._process_results(bm25_results, 'bm25', doc_info, weights.get('bm25', 1.0))
        
        # 计算RRF分数
        rrf_results = []
        for content, info in doc_info.items():
            rrf_score = 0.0
            
            # 计算RRF分数：sum(weight / (k + rank))
            for source, (original_score, rank) in info['sources'].items():
                weight = weights.get(source, 1.0)
                rrf_score += weight / (self.k + rank)
            
            # 创建RRF结果对象
            result = RRFResult(
                content=content,
                metadata=info['metadata'],
                original_scores={source: score for source, (score, rank) in info['sources'].items()},
                ranks={source: rank for source, (score, rank) in info['sources'].items()},
                rrf_score=rrf_score,
                final_rank=0  # 将在排序后设置
            )
            rrf_results.append(result)
        
        # 按RRF分数排序
        rrf_results.sort(key=lambda x: x.rrf_score, reverse=True)
        
        # 设置最终排名
        for i, result in enumerate(rrf_results):
            result.final_rank = i + 1
        
        logger.info(f"RRF融合完成，总计{len(rrf_results)}个文档")
        
        # 显示融合统计
        self._log_fusion_stats(rrf_results[:5])
        
        return rrf_results
    
    def _process_results(self, 
                        results: List[Tuple], 
                        source: str, 
                        doc_info: Dict, 
                        weight: float):
        """处理单个检索系统的结果"""
        for rank, (doc, score) in enumerate(results, 1):
            # 获取文档内容
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            elif isinstance(doc, dict):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
            else:
                content = str(doc)
                metadata = {}
            
            # 初始化文档信息
            if content not in doc_info:
                doc_info[content] = {
                    'metadata': metadata.copy(),
                    'sources': {}
                }
            
            # 添加来源信息
            doc_info[content]['sources'][source] = (score, rank)
            
            # 更新metadata（合并来自不同来源的信息）
            doc_info[content]['metadata'].update(metadata)
            doc_info[content]['metadata'][f'{source}_score'] = score
            doc_info[content]['metadata'][f'{source}_rank'] = rank
    
    def _log_fusion_stats(self, top_results: List[RRFResult]):
        """记录融合统计信息"""
        if not top_results:
            return
        
        logger.info("📊 RRF融合Top5结果统计:")
        for i, result in enumerate(top_results):
            sources = list(result.ranks.keys())
            ranks_str = ", ".join([f"{src}#{result.ranks[src]}" for src in sources])
            scores_str = ", ".join([f"{src}:{result.original_scores[src]:.3f}" for src in sources])
            
            logger.info(f"  {i+1}. RRF={result.rrf_score:.4f} | 排名:[{ranks_str}] | 分数:[{scores_str}]")
    
    def fuse_multiple_sources(self, 
                             results_dict: Dict[str, List[Tuple]], 
                             weights: Optional[Dict[str, float]] = None) -> List[RRFResult]:
        """
        融合多个检索源的结果
        
        Args:
            results_dict: 结果字典 {source_name: [(doc, score), ...]}
            weights: 权重字典 {source_name: weight}
            
        Returns:
            融合后的结果列表
        """
        if weights is None:
            weights = {source: 1.0 for source in results_dict.keys()}
        
        logger.info(f"开始多源RRF融合: {list(results_dict.keys())}")
        
        # 构建文档信息映射
        doc_info = {}
        
        # 处理每个来源的结果
        for source, results in results_dict.items():
            if results:
                weight = weights.get(source, 1.0)
                self._process_results(results, source, doc_info, weight)
        
        # 计算RRF分数
        rrf_results = []
        for content, info in doc_info.items():
            rrf_score = 0.0
            
            for source, (original_score, rank) in info['sources'].items():
                weight = weights.get(source, 1.0)
                rrf_score += weight / (self.k + rank)
            
            result = RRFResult(
                content=content,
                metadata=info['metadata'],
                original_scores={source: score for source, (score, rank) in info['sources'].items()},
                ranks={source: rank for source, (score, rank) in info['sources'].items()},
                rrf_score=rrf_score,
                final_rank=0
            )
            rrf_results.append(result)
        
        # 排序并设置排名
        rrf_results.sort(key=lambda x: x.rrf_score, reverse=True)
        for i, result in enumerate(rrf_results):
            result.final_rank = i + 1
        
        logger.info(f"多源RRF融合完成，总计{len(rrf_results)}个文档")
        return rrf_results
    
    def get_fusion_details(self, result: RRFResult) -> Dict:
        """
        获取融合的详细信息
        
        Args:
            result: RRF结果对象
            
        Returns:
            包含融合详细信息的字典
        """
        details = {
            'rrf_score': result.rrf_score,
            'final_rank': result.final_rank,
            'sources': [],
            'fusion_formula': f"RRF = Σ(weight / (k + rank)), k={self.k}"
        }
        
        # 计算每个来源的贡献
        total_contribution = 0
        for source in result.ranks.keys():
            rank = result.ranks[source]
            score = result.original_scores[source]
            contribution = 1.0 / (self.k + rank)  # 假设权重为1
            total_contribution += contribution
            
            details['sources'].append({
                'source': source,
                'original_score': score,
                'rank': rank,
                'contribution': contribution,
                'contribution_percent': 0  # 将在后面计算
            })
        
        # 计算贡献百分比
        for source_info in details['sources']:
            if total_contribution > 0:
                source_info['contribution_percent'] = source_info['contribution'] / total_contribution * 100
        
        return details

class HybridRetrieverWithRRF:
    """集成RRF的混合检索器"""
    
    def __init__(self, 
                 vector_retriever,
                 bm25_retriever, 
                 k: int = 60,
                 vector_weight: float = 0.5,
                 bm25_weight: float = 0.5):
        """
        初始化集成RRF的混合检索器
        
        Args:
            vector_retriever: 向量检索器
            bm25_retriever: BM25检索器
            k: RRF参数
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_fusion = RRFFusion(k=k)
        self.weights = {
            'vector': vector_weight,
            'bm25': bm25_weight
        }
        
        logger.info(f"RRF混合检索器初始化完成")
        logger.info(f"  权重配置: 向量={vector_weight}, BM25={bm25_weight}")
    
    def search(self, query: str, k: int = 10) -> List[Tuple]:
        """
        执行混合检索（带RRF融合）
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            融合后的结果列表 [(document, rrf_score), ...]
        """
        logger.info(f"开始RRF混合检索: '{query[:50]}...'")
        
        try:
            # 1. 获取向量检索结果
            vector_results = []
            if self.vector_retriever:
                try:
                    if hasattr(self.vector_retriever, 'similarity_search_with_score'):
                        vector_results = self.vector_retriever.similarity_search_with_score(query, k=k*2)
                    else:
                        docs = self.vector_retriever.similarity_search(query, k=k*2)
                        vector_results = [(doc, 1.0/(i+1)) for i, doc in enumerate(docs)]
                    logger.info(f"向量检索获得 {len(vector_results)} 个结果")
                except Exception as e:
                    logger.warning(f"向量检索失败: {e}")
            
            # 2. 获取BM25检索结果
            bm25_results = []
            if self.bm25_retriever:
                try:
                    bm25_results = self.bm25_retriever.search(query, k=k*2)
                    logger.info(f"BM25检索获得 {len(bm25_results)} 个结果")
                except Exception as e:
                    logger.warning(f"BM25检索失败: {e}")
            
            # 3. RRF融合
            if vector_results or bm25_results:
                rrf_results = self.rrf_fusion.fuse_results(
                    vector_results=vector_results,
                    bm25_results=bm25_results,
                    weights=self.weights
                )
                
                # 转换为标准格式
                final_results = []
                for rrf_result in rrf_results[:k]:
                    # 创建文档对象
                    from langchain_core.documents import Document
                    doc = Document(
                        page_content=rrf_result.content,
                        metadata=rrf_result.metadata
                    )
                    # 添加RRF信息到metadata
                    doc.metadata.update({
                        'rrf_score': rrf_result.rrf_score,
                        'rrf_rank': rrf_result.final_rank,
                        'fusion_sources': list(rrf_result.ranks.keys())
                    })
                    
                    final_results.append((doc, rrf_result.rrf_score))
                
                logger.info(f"RRF混合检索完成，返回 {len(final_results)} 个结果")
                return final_results
            else:
                logger.warning("没有任何检索结果")
                return []
                
        except Exception as e:
            logger.error(f"RRF混合检索失败: {e}")
            return [] 