"""
增强的重排序器
优化Cross-Encoder重排序，特别针对竞赛任务需求进行优化
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger
from langchain_core.documents import Document
import torch
import re

try:
    from sentence_transformers import CrossEncoder
    # 兼容性处理：修复assign参数问题
    import inspect
    if hasattr(torch.nn.Module, 'load_state_dict'):
        original_load_state_dict = torch.nn.Module.load_state_dict
        
        def patched_load_state_dict(self, state_dict, strict=True, assign=None):
            sig = inspect.signature(original_load_state_dict)
            if 'assign' in sig.parameters:
                return original_load_state_dict(self, state_dict, strict=strict, assign=assign)
            else:
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
class EnhancedRerankResult:
    """增强重排结果"""
    content: str
    metadata: Dict
    source: str
    original_score: float
    crossencoder_score: float
    entity_bonus: float  # 实体命中奖励
    task_relevance_score: float  # 任务相关性分数
    final_score: float
    rank: int
    matched_entities: List[str]  # 匹配到的实体
    task_indicators: List[str]   # 任务指标

class EnhancedReranker:
    """增强的重排序器 - 专门优化竞赛任务相关性"""
    
    def __init__(self, 
                 model_name: str = './cross-encoder/ms-marco-MiniLM-L6-v2',
                 fallback_enabled: bool = True,
                 entity_bonus_weight: float = 0.3,
                 task_relevance_weight: float = 0.2):
        """
        初始化增强重排器
        
        Args:
            model_name: CrossEncoder模型名称
            fallback_enabled: 是否启用降级方案
            entity_bonus_weight: 实体奖励权重
            task_relevance_weight: 任务相关性权重
        """
        self.model_name = model_name
        self.fallback_enabled = fallback_enabled
        self.entity_bonus_weight = entity_bonus_weight
        self.task_relevance_weight = task_relevance_weight
        self.rerank_model = None
        
        # 重要实体词典（用于实体命中奖励）
        self.important_entities = {
            # 竞赛相关
            "未来校园智能应用专项赛": 2.0,
            "智能交通信号灯": 1.8,
            "泰迪杯": 1.5,
            "数据挖掘挑战赛": 1.3,
            
            # 任务关键词
            "基本要求": 1.6,
            "技术要求": 1.6,
            "任务描述": 1.5,
            "评分标准": 1.4,
            "实现方案": 1.3,
            
            # 技术关键词
            "交通信号灯": 1.7,
            "信号控制": 1.4,
            "智能控制": 1.3,
            "算法设计": 1.2,
            "优化方案": 1.2,
            
            # 评估指标
            "创新性": 1.1,
            "实用性": 1.1,
            "可行性": 1.1,
            "完整性": 1.0
        }
        
        # 任务指标关键词
        self.task_indicators = {
            "任务": 1.5,
            "要求": 1.4,
            "设计": 1.3,
            "实现": 1.2,
            "方案": 1.2,
            "算法": 1.1,
            "优化": 1.1,
            "控制": 1.1,
            "系统": 1.0
        }
        
        # 初始化重排模型
        self._init_rerank_model()
        
        logger.info(f"增强重排器初始化完成")
        logger.info(f"  模型: {model_name}")
        logger.info(f"  权重配置: 实体奖励={entity_bonus_weight}, 任务相关性={task_relevance_weight}")
        logger.info(f"  重要实体数量: {len(self.important_entities)}")
    
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
        对文档进行增强重排序
        
        Args:
            question: 查询问题
            docs: 文档列表
            top_k: 返回的top-k结果数量
        
        Returns:
            List[Document]: 重排后的文档列表
        """
        if not docs:
            return []

        logger.info(f"开始增强重排序，问题: '{question[:50]}...', 文档数量: {len(docs)}")
        
        # 使用增强CrossEncoder重排
        if self.rerank_model is not None:
            rerank_results = self._enhanced_crossencoder_rerank(question, docs)
        else:
            # 降级方案：使用增强的规则重排方法
            rerank_results = self._enhanced_fallback_rerank(question, docs)
        
        # 排序
        rerank_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # 设置最终排名
        for i, result in enumerate(rerank_results):
            result.rank = i + 1
        
        # 将结果转换为Document对象
        reranked_docs = []
        for i, result in enumerate(rerank_results):
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
                    'crossencoder_score': result.crossencoder_score,
                    'entity_bonus': result.entity_bonus,
                    'task_relevance_score': result.task_relevance_score,
                    'final_rerank_score': result.final_score,
                    'rerank_rank': result.rank,
                    'matched_entities': result.matched_entities,
                    'task_indicators': result.task_indicators,
                    'original_score': result.original_score
                })
                
                new_doc = Document(
                    page_content=original_doc.page_content,
                    metadata=new_metadata
                )
                reranked_docs.append(new_doc)
        
        # 返回top-k结果
        if top_k:
            reranked_docs = reranked_docs[:top_k]
        
        # 记录重排统计
        self._log_rerank_stats(question, rerank_results[:5])
        
        logger.info(f"增强重排序完成，返回 {len(reranked_docs)} 个结果")
        return reranked_docs
    
    def _enhanced_crossencoder_rerank(self, question: str, docs: List) -> List[EnhancedRerankResult]:
        """使用增强CrossEncoder进行重排"""
        try:
            # 准备输入对
            pairs = []
            for doc in docs:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                else:
                    content = doc.get('content', '')
                
                # 优化输入对格式，突出任务相关性
                enhanced_question = self._enhance_question_for_crossencoder(question)
                enhanced_content = self._enhance_content_for_crossencoder(content)
                
                pairs.append([enhanced_question, enhanced_content])
            
            # 计算CrossEncoder分数
            crossencoder_scores = self.rerank_model.predict(pairs)
            
            # 构建增强结果
            results = []
            for i, (doc, ce_score) in enumerate(zip(docs, crossencoder_scores)):
                # 获取文档内容和元数据
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
                
                # 计算实体命中奖励
                entity_bonus, matched_entities = self._calculate_entity_bonus(question, content)
                
                # 计算任务相关性分数
                task_score, task_indicators = self._calculate_task_relevance(question, content)
                
                # 计算最终分数
                final_score = (
                    0.5 * float(ce_score) +  # CrossEncoder分数权重
                    0.2 * original_score +   # 原始分数权重
                    self.entity_bonus_weight * entity_bonus +  # 实体奖励
                    self.task_relevance_weight * task_score    # 任务相关性
                )
                
                result = EnhancedRerankResult(
                    content=content,
                    metadata=metadata,
                    source=source,
                    original_score=original_score,
                    crossencoder_score=float(ce_score),
                    entity_bonus=entity_bonus,
                    task_relevance_score=task_score,
                    final_score=final_score,
                    rank=0,
                    matched_entities=matched_entities,
                    task_indicators=task_indicators
                )
                results.append(result)
            
            logger.debug(f"增强CrossEncoder重排完成，CE分数范围: {min(crossencoder_scores):.3f} - {max(crossencoder_scores):.3f}")
            return results
            
        except Exception as e:
            logger.error(f"增强CrossEncoder重排失败: {e}")
            return self._enhanced_fallback_rerank(question, docs)
    
    def _enhance_question_for_crossencoder(self, question: str) -> str:
        """优化问题格式以提高CrossEncoder效果"""
        # 突出关键词
        enhanced = question
        
        # 如果问题中包含重要实体，给予特殊标记
        for entity in self.important_entities:
            if entity in question:
                enhanced = enhanced.replace(entity, f"[重要]{entity}[/重要]")
        
        # 添加任务指标标记
        for indicator in self.task_indicators:
            if indicator in question:
                enhanced = enhanced.replace(indicator, f"[任务]{indicator}[/任务]")
        
        return enhanced
    
    def _enhance_content_for_crossencoder(self, content: str) -> str:
        """优化文档内容格式以提高CrossEncoder效果"""
        # 限制内容长度，保留最相关部分
        if len(content) > 512:
            # 查找包含重要实体的段落
            sentences = content.split('。')
            important_sentences = []
            
            for sentence in sentences:
                for entity in self.important_entities:
                    if entity in sentence:
                        important_sentences.append(sentence)
                        break
            
            if important_sentences:
                # 如果找到重要句子，优先保留
                enhanced = '。'.join(important_sentences[:3]) + '。'
                if len(enhanced) < 400:
                    # 如果重要句子不够，补充其他句子
                    remaining = content.replace(enhanced, '')
                    enhanced += remaining[:400-len(enhanced)]
            else:
                # 否则取前400字符
                enhanced = content[:400]
        else:
            enhanced = content
        
        return enhanced
    
    def _calculate_entity_bonus(self, question: str, content: str) -> Tuple[float, List[str]]:
        """计算实体命中奖励"""
        bonus = 0.0
        matched_entities = []
        
        # 检查问题中的实体在文档中的出现
        for entity, weight in self.important_entities.items():
            if entity in question and entity in content:
                # 计算实体在文档中的出现频率
                count = content.count(entity)
                entity_bonus = weight * min(count * 0.1, 1.0)  # 最大1.0的奖励
                bonus += entity_bonus
                matched_entities.append(entity)
        
        # 归一化奖励分数
        bonus = min(bonus, 2.0)  # 最大2.0的奖励
        
        return bonus, matched_entities
    
    def _calculate_task_relevance(self, question: str, content: str) -> Tuple[float, List[str]]:
        """计算任务相关性分数"""
        relevance = 0.0
        found_indicators = []
        
        # 检查任务指标词
        for indicator, weight in self.task_indicators.items():
            if indicator in question:
                count_in_content = content.count(indicator)
                if count_in_content > 0:
                    indicator_score = weight * min(count_in_content * 0.1, 0.5)
                    relevance += indicator_score
                    found_indicators.append(indicator)
        
        # 特殊奖励：如果文档同时包含多个任务指标
        if len(found_indicators) >= 3:
            relevance += 0.3  # 多指标奖励
        
        # 归一化
        relevance = min(relevance, 1.5)
        
        return relevance, found_indicators
    
    def _enhanced_fallback_rerank(self, question: str, docs: List) -> List[EnhancedRerankResult]:
        """增强的降级重排方案"""
        logger.info("使用增强降级重排方案")
        
        results = []
        question_lower = question.lower()
        
        for doc in docs:
            # 获取文档内容和元数据
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
            
            # 计算基础相关性分数
            base_score = self._calculate_enhanced_relevance(question_lower, content_lower)
            
            # 计算实体命中奖励
            entity_bonus, matched_entities = self._calculate_entity_bonus(question, content)
            
            # 计算任务相关性分数
            task_score, task_indicators = self._calculate_task_relevance(question, content)
            
            # 综合分数
            final_score = (
                0.5 * base_score +
                0.2 * original_score +
                self.entity_bonus_weight * entity_bonus +
                self.task_relevance_weight * task_score
            )
            
            result = EnhancedRerankResult(
                content=content,
                metadata=metadata,
                source=source,
                original_score=original_score,
                crossencoder_score=base_score,  # 在降级模式下使用基础分数
                entity_bonus=entity_bonus,
                task_relevance_score=task_score,
                final_score=final_score,
                rank=0,
                matched_entities=matched_entities,
                task_indicators=task_indicators
            )
            results.append(result)
        
        return results
    
    def _calculate_enhanced_relevance(self, question: str, content: str) -> float:
        """计算增强的相关性分数"""
        score = 0.0
        
        # 1. 关键词匹配
        question_words = question.split()
        for word in question_words:
            if len(word) > 1:
                count = content.count(word)
                score += min(count * 0.1, 0.5)
        
        # 2. 短语匹配
        if len(question) > 10:
            phrases = [question[i:i+10] for i in range(len(question)-9)]
            for phrase in phrases:
                if phrase in content:
                    score += 0.3
        
        # 3. 文档长度归一化
        score = score / (len(content) / 1000 + 1)
        
        return min(score, 2.0)
    
    def _log_rerank_stats(self, question: str, top_results: List[EnhancedRerankResult]):
        """记录重排统计信息"""
        try:
            logger.info("📊 增强重排序Top5统计:")
            logger.info(f"  查询: '{question[:50]}...'")
            
            for i, result in enumerate(top_results):
                source_file = result.source.split('/')[-1] if result.source else 'unknown'
                logger.info(f"  {i+1}. {source_file}")
                logger.info(f"     最终分数: {result.final_score:.4f}")
                logger.info(f"     CE分数: {result.crossencoder_score:.4f}")
                logger.info(f"     实体奖励: {result.entity_bonus:.4f} ({len(result.matched_entities)}个实体)")
                logger.info(f"     任务相关性: {result.task_relevance_score:.4f} ({len(result.task_indicators)}个指标)")
                if result.matched_entities:
                    logger.info(f"     匹配实体: {result.matched_entities}")
                
        except Exception as e:
            logger.warning(f"记录重排统计失败: {e}")
    
    def get_rerank_analysis(self, question: str, docs: List[Document]) -> Dict:
        """
        获取重排分析报告
        
        Args:
            question: 查询问题
            docs: 文档列表
            
        Returns:
            包含重排分析的详细报告
        """
        try:
            # 执行重排但不修改原始文档
            if self.rerank_model is not None:
                rerank_results = self._enhanced_crossencoder_rerank(question, docs)
            else:
                rerank_results = self._enhanced_fallback_rerank(question, docs)
            
            # 排序
            rerank_results.sort(key=lambda x: x.final_score, reverse=True)
            
            # 生成分析报告
            analysis = {
                'query': question,
                'total_docs': len(docs),
                'rerank_method': 'CrossEncoder' if self.rerank_model else 'Enhanced Fallback',
                'top_results': [],
                'entity_analysis': {},
                'task_analysis': {},
                'score_distribution': {}
            }
            
            # Top结果分析
            for i, result in enumerate(rerank_results[:5]):
                analysis['top_results'].append({
                    'rank': i + 1,
                    'source': result.source,
                    'final_score': result.final_score,
                    'crossencoder_score': result.crossencoder_score,
                    'entity_bonus': result.entity_bonus,
                    'task_relevance_score': result.task_relevance_score,
                    'matched_entities': result.matched_entities,
                    'task_indicators': result.task_indicators,
                    'content_preview': result.content[:200]
                })
            
            # 实体分析
            all_entities = set()
            for result in rerank_results:
                all_entities.update(result.matched_entities)
            
            for entity in all_entities:
                count = sum(1 for r in rerank_results if entity in r.matched_entities)
                analysis['entity_analysis'][entity] = {
                    'frequency': count,
                    'weight': self.important_entities.get(entity, 0)
                }
            
            # 任务指标分析
            all_indicators = set()
            for result in rerank_results:
                all_indicators.update(result.task_indicators)
            
            for indicator in all_indicators:
                count = sum(1 for r in rerank_results if indicator in r.task_indicators)
                analysis['task_analysis'][indicator] = {
                    'frequency': count,
                    'weight': self.task_indicators.get(indicator, 0)
                }
            
            # 分数分布
            scores = [r.final_score for r in rerank_results]
            analysis['score_distribution'] = {
                'min': min(scores),
                'max': max(scores),
                'mean': sum(scores) / len(scores),
                'std': np.std(scores)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"生成重排分析失败: {e}")
            return {'error': str(e)} 