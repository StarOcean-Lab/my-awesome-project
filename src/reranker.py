"""
相关性重排模块
使用CrossEncoder模型对检索结果进行重新排序
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger
from langchain_core.documents import Document
import torch

try:
    from sentence_transformers import CrossEncoder
    # 兼容性处理：修复assign参数问题
    import inspect
    if hasattr(torch.nn.Module, 'load_state_dict'):
        original_load_state_dict = torch.nn.Module.load_state_dict
        
        def patched_load_state_dict(self, state_dict, strict=True, assign=None):
            # 检查原函数是否支持assign参数
            sig = inspect.signature(original_load_state_dict)
            if 'assign' in sig.parameters:
                return original_load_state_dict(self, state_dict, strict=strict, assign=assign)
            else:
                # 忽略assign参数
                return original_load_state_dict(self, state_dict, strict=strict)
        
        torch.nn.Module.load_state_dict = patched_load_state_dict
        logger.info("已应用PyTorch兼容性修复")
        
except ImportError:
    CrossEncoder = None
    logger.warning("CrossEncoder未安装，将使用简化的重排方法")
except Exception as e:
    logger.warning(f"CrossEncoder兼容性修复失败: {e}")
    CrossEncoder = None

@dataclass
class RerankResult:
    """重排结果"""
    content: str
    metadata: Dict
    source: str
    original_score: float
    rerank_score: float
    final_score: float
    rank: int

class Reranker:
    """相关性重排器"""
    
    def __init__(self, 
                 model_name: str = './cross-encoder/ms-marco-MiniLM-L6-v2',
                 fallback_enabled: bool = True):
        """
        初始化重排器
        
        Args:
            model_name: CrossEncoder模型名称
            fallback_enabled: 是否启用降级方案
        """
        self.model_name = model_name
        self.fallback_enabled = fallback_enabled
        self.rerank_model = None
        
        # 初始化重排模型
        self._init_rerank_model()
        
        logger.info(f"重排器初始化完成，模型: {model_name}")
    
    def _init_rerank_model(self):
        """初始化重排模型"""
        if CrossEncoder is None:
            logger.warning("CrossEncoder不可用，将使用降级方案")
            return
        
        try:
            # 尝试加载模型
            self.rerank_model = CrossEncoder(self.model_name)
            logger.info(f"CrossEncoder模型加载成功: {self.model_name}")
        except Exception as e:
            logger.warning(f"CrossEncoder模型加载失败: {e}")
            # 尝试离线模式
            try:
                import os
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                self.rerank_model = CrossEncoder(self.model_name, trust_remote_code=False)
                logger.info(f"CrossEncoder离线模式加载成功: {self.model_name}")
            except Exception as e2:
                logger.warning(f"离线模式也失败: {e2}")
                if self.fallback_enabled:
                    logger.info("启用降级重排方案")
                else:
                    raise e
    
    def rerank(self, question: str, docs: List[Document], top_k: int = None) -> List[Document]:
        """
        对文档进行重排序
        
        Args:
            question: 查询问题
            docs: 文档列表
            top_k: 返回的top-k结果数量
        
        Returns:
            List[Document]: 重排后的文档列表
        """
        if not docs:
            return []

        logger.info(f"开始重排序，问题: {question}, 文档数量: {len(docs)}")
        
        # 使用CrossEncoder重排
        if self.rerank_model is not None:
            rerank_results = self._crossencoder_rerank(question, docs)
        else:
            # 降级方案：使用简化的重排方法
            rerank_results = self._fallback_rerank(question, docs)
        
        # 排序
        rerank_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # 将RerankResult转换为Document对象
        reranked_docs = []
        for i, result in enumerate(rerank_results):
            # 创建新的Document对象
            if hasattr(docs[0], 'page_content'):
                # 找到原始Document对象
                original_doc = None
                for doc in docs:
                    if (hasattr(doc, 'page_content') and 
                        doc.page_content == result.content):
                        original_doc = doc
                        break
                
                if original_doc is not None:
                    # 复制原始Document并添加重排序信息
                    new_metadata = original_doc.metadata.copy()
                    new_metadata.update({
                        'rerank_score': result.rerank_score,
                        'original_score': result.original_score,
                        'final_score': result.final_score,
                        'rerank_rank': i + 1
                    })
                    
                    new_doc = Document(
                        page_content=original_doc.page_content,
                        metadata=new_metadata
                    )
                    reranked_docs.append(new_doc)
        
        # 返回top-k结果
        if top_k:
            reranked_docs = reranked_docs[:top_k]
        
        logger.info(f"重排序完成，返回 {len(reranked_docs)} 个结果")
        return reranked_docs
    
    def _crossencoder_rerank(self, question: str, docs: List) -> List[RerankResult]:
        """使用CrossEncoder进行重排"""
        try:
            # 准备输入对
            pairs = []
            for doc in docs:
                # 统一处理Document对象和字典对象
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                else:
                    content = doc.get('content', '')
                pairs.append([question, content])
            
            # 计算重排分数
            rerank_scores = self.rerank_model.predict(pairs)
            
            # 构建结果
            results = []
            for i, (doc, rerank_score) in enumerate(zip(docs, rerank_scores)):
                # 统一处理Document对象和字典对象
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = doc.metadata
                    source = metadata.get('source', '')
                    original_score = metadata.get('score', 0.0)
                else:
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    source = doc.get('source', '')
                    original_score = doc.get('score', 0.0)
                
                # 综合分数：原始分数和重排分数的加权平均
                final_score = 0.6 * float(rerank_score) + 0.4 * original_score
                
                result = RerankResult(
                    content=content,
                    metadata=metadata,
                    source=source,
                    original_score=original_score,
                    rerank_score=float(rerank_score),
                    final_score=final_score,
                    rank=0  # 排名稍后设置
                )
                results.append(result)
            
            logger.debug(f"CrossEncoder重排完成，分数范围: {min(rerank_scores):.3f} - {max(rerank_scores):.3f}")
            return results
            
        except Exception as e:
            logger.error(f"CrossEncoder重排失败: {e}")
            return self._fallback_rerank(question, docs)
    
    def _fallback_rerank(self, question: str, docs: List) -> List[RerankResult]:
        """降级重排方案"""
        logger.info("使用降级重排方案")
        
        results = []
        question_lower = question.lower()
        
        for doc in docs:
            # 统一处理Document对象和字典对象
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                metadata = doc.metadata
                source = metadata.get('source', '')
                original_score = metadata.get('score', 0.0)
            else:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                source = doc.get('source', '')
                original_score = doc.get('score', 0.0)
            
            content_lower = content.lower()
            
            # 简单的相关性评分
            rerank_score = self._calculate_simple_relevance(question_lower, content_lower)
            
            # 综合分数
            final_score = 0.7 * rerank_score + 0.3 * original_score
            
            result = RerankResult(
                content=content,
                metadata=metadata,
                source=source,
                original_score=original_score,
                rerank_score=rerank_score,
                final_score=final_score,
                rank=0
            )
            results.append(result)
        
        return results
    
    def _calculate_simple_relevance(self, question: str, content: str) -> float:
        """计算简单的相关性分数"""
        # 1. 关键词匹配
        question_words = set(question.split())
        content_words = set(content.split())
        
        if not question_words:
            return 0.0
        
        # 交集大小
        intersection = question_words.intersection(content_words)
        keyword_score = len(intersection) / len(question_words)
        
        # 2. 长度惩罚（避免过长或过短的文档）
        content_length = len(content)
        if 100 <= content_length <= 1000:
            length_factor = 1.0
        elif content_length < 100:
            length_factor = 0.5 + (content_length / 100) * 0.5
        else:
            length_factor = max(0.3, 1.0 - (content_length - 1000) / 2000)
        
        # 3. 位置奖励（关键词在文档前部出现的奖励）
        position_bonus = 0.0
        for word in question_words:
            pos = content.find(word)
            if pos != -1 and pos < len(content) * 0.3:  # 前30%位置
                position_bonus += 0.1
        
        # 组合分数
        final_score = keyword_score * length_factor + min(position_bonus, 0.3)
        return min(final_score, 1.0)
    
    def batch_rerank(self, questions_docs: List[Tuple[str, List[Dict]]], 
                    top_k: int = None) -> List[List[RerankResult]]:
        """批量重排"""
        results = []
        for question, docs in questions_docs:
            rerank_result = self.rerank(question, docs, top_k)
            results.append(rerank_result)
        
        return results
    
    def get_rerank_stats(self, results: List[RerankResult]) -> Dict:
        """获取重排统计信息"""
        if not results:
            return {}
        
        original_scores = [r.original_score for r in results]
        rerank_scores = [r.rerank_score for r in results]
        final_scores = [r.final_score for r in results]
        
        return {
            "total_docs": len(results),
            "original_score_avg": np.mean(original_scores),
            "original_score_std": np.std(original_scores),
            "rerank_score_avg": np.mean(rerank_scores),
            "rerank_score_std": np.std(rerank_scores),
            "final_score_avg": np.mean(final_scores),
            "final_score_std": np.std(final_scores),
            "score_improvement": np.mean(final_scores) - np.mean(original_scores)
        }

class MultiStageReranker:
    """多阶段重排器"""
    
    def __init__(self):
        """初始化多阶段重排器"""
        self.base_reranker = Reranker()
        
    def multi_stage_rerank(self, question: str, docs: List, 
                          stage1_k: int = 50, final_k: int = 10) -> List[RerankResult]:
        """
        多阶段重排序
        
        Args:
            question: 查询问题
            docs: 文档列表（支持Document对象或字典）
            stage1_k: 第一阶段保留的文档数量
            final_k: 最终返回的文档数量
        """
        if len(docs) <= final_k:
            return self.base_reranker.rerank(question, docs, final_k)
        
        logger.info(f"开始多阶段重排，初始文档: {len(docs)}, 阶段1保留: {stage1_k}, 最终保留: {final_k}")
        
        # 阶段1: 粗排，快速过滤
        stage1_results = self._stage1_filter(question, docs, stage1_k)
        
        # 阶段2: 精排，使用CrossEncoder
        if stage1_results:
            stage2_docs = [
                {
                    'content': r.content,
                    'metadata': r.metadata,
                    'source': r.source,
                    'score': r.final_score
                }
                for r in stage1_results
            ]
            final_results = self.base_reranker.rerank(question, stage2_docs, final_k)
        else:
            final_results = []
        
        logger.info(f"多阶段重排完成，最终结果: {len(final_results)}")
        return final_results
    
    def _stage1_filter(self, question: str, docs: List, k: int) -> List[RerankResult]:
        """第一阶段：快速过滤"""
        # 使用简化方法快速排序
        results = []
        question_lower = question.lower()
        
        for doc in docs:
            # 统一处理Document对象和字典对象
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                metadata = doc.metadata
                source = metadata.get('source', '')
                original_score = metadata.get('score', 0.0)
            else:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                source = doc.get('source', '')
                original_score = doc.get('score', 0.0)
            
            score = self.base_reranker._calculate_simple_relevance(question_lower, content.lower())
            
            result = RerankResult(
                content=content,
                metadata=metadata,
                source=source,
                original_score=original_score,
                rerank_score=score,
                final_score=score,
                rank=0
            )
            results.append(result)
        
        # 排序并返回top-k
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:k] 