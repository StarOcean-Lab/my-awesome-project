#!/usr/bin/env python3
"""
文档层级检索器（Hierarchical Retriever）
通过先定位竞赛类别，再进行精细检索的方式提高检索准确性
"""

import re
import jieba
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
from langchain_core.documents import Document
from loguru import logger

from .langchain_retriever import LangChainHybridRetriever
try:
    from .query_enhancer import get_enhanced_query_optimizer
except ImportError:
    from query_enhancer import get_enhanced_query_optimizer


class HierarchicalRetriever:
    """层级检索器 - 先定位竞赛，再精细检索"""
    
    def __init__(self, 
                 base_retriever: LangChainHybridRetriever,
                 docs: List[Document] = None,
                 similarity_threshold: float = 0.7,
                 enable_general_fallback: bool = True):
        """
        初始化层级检索器
        
        Args:
            base_retriever: 基础检索器
            docs: 文档列表
            similarity_threshold: 竞赛匹配相似度阈值
            enable_general_fallback: 是否启用通用检索降级
        """
        self.base_retriever = base_retriever
        self.docs = docs if docs else []
        self.similarity_threshold = similarity_threshold
        self.enable_general_fallback = enable_general_fallback
        
        # 查询增强器，用于竞赛识别
        self.query_enhancer = get_enhanced_query_optimizer()
        
        # 竞赛索引
        self.competition_index = {}
        self.competition_patterns = {}
        
        # 构建层级结构
        self.create_hierarchy()
        
        logger.info(f"层级检索器初始化完成")
        logger.info(f"  - 竞赛类别数: {len(self.competition_index)}")
        logger.info(f"  - 文档总数: {len(self.docs)}")
        logger.info(f"  - 相似度阈值: {similarity_threshold}")
    
    def create_hierarchy(self):
        """构建文档层级结构"""
        if not self.docs:
            logger.warning("未提供文档，无法构建层级结构")
            return
        
        # 定义竞赛名称模式
        self.competition_patterns = {
            "未来校园": ["未来校园", "智能应用", "校园", "智能校园"],
            "3D编程": ["3D", "编程模型", "三维编程", "三维设计", "立体编程"],
            "编程创作": ["编程创作", "信息学", "编程竞赛", "算法"],
            "机器人工程": ["机器人工程", "机器人设计", "工程设计", "机械"],
            "极地资源": ["极地", "资源勘探", "勘探", "极地探索"],
            "竞技机器人": ["竞技机器人", "机器人竞技", "竞技", "机器人比赛"],
            "开源鸿蒙": ["鸿蒙", "开源", "HarmonyOS", "华为", "鸿蒙系统"],
            "人工智能": ["人工智能", "AI", "机器学习", "深度学习", "智能算法"],
            "三维程序": ["三维程序", "程序创意", "三维设计", "创意设计"],
            "生成式AI": ["生成式", "AIGC", "生成模型", "生成式人工智能"],
            "太空电梯": ["太空电梯", "电梯工程", "太空工程", "电梯设计"],
            "太空探索": ["太空探索", "智能机器人", "太空机器人", "宇航"],
            "无人驾驶": ["无人驾驶", "自动驾驶", "智能车", "自动汽车"],
            "虚拟仿真": ["虚拟仿真", "仿真平台", "虚拟", "仿真"],
            "智慧城市": ["智慧城市", "城市设计", "智慧", "城市规划"],
            "数据采集": ["数据采集", "采集装置", "智能数据", "传感器"],
            "智能芯片": ["智能芯片", "芯片", "计算思维", "微处理器"],
            "泰迪杯": ["泰迪杯", "数据挖掘", "挖掘挑战", "数据分析"]
        }
        
        # 构建索引
        logger.info("开始构建竞赛层级索引...")
        
        for doc in self.docs:
            comp_name = self.extract_competition_name(doc.page_content, doc.metadata)
            
            if comp_name:
                if comp_name not in self.competition_index:
                    self.competition_index[comp_name] = []
                self.competition_index[comp_name].append(doc)
        
        # 打印构建结果
        for comp_name, docs in self.competition_index.items():
            logger.info(f"  {comp_name}: {len(docs)} 个文档")
        
        if not self.competition_index:
            logger.warning("未构建任何竞赛索引，将使用通用检索")
    
    def extract_competition_name(self, content: str, metadata: Dict = None) -> Optional[str]:
        """
        从文档内容或元数据中提取竞赛名称
        
        Args:
            content: 文档内容
            metadata: 文档元数据
            
        Returns:
            竞赛名称或None
        """
        # 1. 首先尝试从文件路径中提取
        if metadata and 'source' in metadata:
            source_path = metadata['source']
            for comp_name, patterns in self.competition_patterns.items():
                for pattern in patterns:
                    if pattern in source_path:
                        return comp_name
        
        # 2. 从内容的前500个字符中提取（通常标题和描述在开头）
        content_head = content[:500]
        
        # 计算每个竞赛的匹配分数
        comp_scores = {}
        for comp_name, patterns in self.competition_patterns.items():
            score = 0
            for pattern in patterns:
                # 计算模式在内容中的出现次数
                count = content_head.lower().count(pattern.lower())
                score += count * len(pattern)  # 长度越长的匹配给更高权重
            
            if score > 0:
                comp_scores[comp_name] = score
        
        # 返回得分最高的竞赛
        if comp_scores:
            best_comp = max(comp_scores.items(), key=lambda x: x[1])
            if best_comp[1] > 0:
                return best_comp[0]
        
        return None
    
    def identify_competition(self, question: str, available_competitions: List[str]) -> Optional[str]:
        """
        从问题中识别竞赛名称
        
        Args:
            question: 用户问题
            available_competitions: 可用的竞赛列表
            
        Returns:
            识别到的竞赛名称或None
        """
        if not available_competitions:
            return None
        
        question_lower = question.lower()
        
        # 1. 直接匹配竞赛名称
        for comp_name in available_competitions:
            if comp_name.lower() in question_lower:
                logger.debug(f"直接匹配到竞赛: {comp_name}")
                return comp_name
        
        # 2. 通过关键词模式匹配
        comp_scores = {}
        for comp_name in available_competitions:
            if comp_name in self.competition_patterns:
                patterns = self.competition_patterns[comp_name]
                score = 0
                
                for pattern in patterns:
                    if pattern.lower() in question_lower:
                        # 根据匹配长度和出现次数计算分数
                        count = question_lower.count(pattern.lower())
                        score += count * len(pattern) * 2  # 问题中的匹配给更高权重
                
                if score > 0:
                    comp_scores[comp_name] = score
        
        # 3. 返回得分最高且超过阈值的竞赛
        if comp_scores:
            best_comp = max(comp_scores.items(), key=lambda x: x[1])
            # 简单阈值：至少要有一个较好的匹配
            if best_comp[1] >= 3:
                logger.debug(f"模式匹配到竞赛: {best_comp[0]} (分数: {best_comp[1]})")
                return best_comp[0]
        
        # 4. 使用查询增强器的竞赛映射
        for comp_key, keywords in self.query_enhancer.competition_analyzer.competition_mapping.items():
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 找到对应的完整竞赛名称
                    for comp_name in available_competitions:
                        if comp_key.lower() in comp_name.lower():
                            logger.debug(f"查询增强器匹配到竞赛: {comp_name}")
                            return comp_name
        
        return None
    
    def fine_grained_retrieval(self, question: str, relevant_docs: List[Document], k: int = None) -> List[Document]:
        """
        在文档子集中进行精细检索
        
        Args:
            question: 查询问题
            relevant_docs: 相关文档子集
            k: 返回结果数量
            
        Returns:
            检索结果
        """
        if not relevant_docs:
            return []
        
        if k is None:
            k = self.base_retriever.k
        
        try:
            # 创建临时向量存储用于子集检索
            logger.debug(f"在 {len(relevant_docs)} 个文档中进行精细检索")
            
            # 1. 使用基础检索器的向量存储进行相似度搜索
            if self.base_retriever.vectorstore and self.base_retriever.vectorstore.get_vectorstore():
                # 获取所有候选文档
                all_candidates = self.base_retriever.vectorstore.similarity_search(question, k=k*3)
                
                # 过滤出属于相关文档子集的结果
                filtered_candidates = []
                relevant_contents = {doc.page_content for doc in relevant_docs}
                
                for candidate in all_candidates:
                    if candidate.page_content in relevant_contents:
                        filtered_candidates.append(candidate)
                        if len(filtered_candidates) >= k:
                            break
                
                if filtered_candidates:
                    logger.debug(f"精细检索完成，从 {len(all_candidates)} 个候选中筛选出 {len(filtered_candidates)} 个结果")
                    return filtered_candidates
            
            # 2. 降级方案：基于关键词的简单匹配
            return self._simple_keyword_match(question, relevant_docs, k)
            
        except Exception as e:
            logger.warning(f"精细检索失败: {e}，使用简单匹配")
            return self._simple_keyword_match(question, relevant_docs, k)
    
    def _simple_keyword_match(self, question: str, docs: List[Document], k: int) -> List[Document]:
        """简单的关键词匹配检索"""
        # 提取问题关键词
        question_words = set(jieba.cut(question.lower()))
        question_words = {word for word in question_words if len(word) > 1}
        
        # 计算每个文档的匹配分数
        doc_scores = []
        for doc in docs:
            content_words = set(jieba.cut(doc.page_content.lower()))
            
            # 计算交集大小作为分数
            intersection = question_words.intersection(content_words)
            score = len(intersection)
            
            if score > 0:
                doc_scores.append((doc, score))
        
        # 按分数排序并返回top-k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores[:k]]
    
    def general_retrieval(self, question: str, stage1_k: int = None, final_k: int = None) -> List[Document]:
        """
        全文档检索（支持多阶段）
        
        Args:
            question: 查询问题
            stage1_k: 第一阶段检索数量
            final_k: 最终返回数量
            
        Returns:
            检索结果
        """
        if stage1_k is None:
            stage1_k = self.base_retriever.k
        if final_k is None:
            final_k = self.base_retriever.k
        
        logger.debug(f"执行全文档检索: 第一阶段{stage1_k}个 → 最终{final_k}个")
        
        # 如果基础检索器支持多阶段检索，直接使用
        if hasattr(self.base_retriever, '_multi_stage_retrieval'):
            from config import Config
            stages = getattr(Config, 'RETRIEVAL_STAGES', {})
            if stages.get('enable_multi_stage', True):
                return self.base_retriever._multi_stage_retrieval(question, stages)
        
        # 否则使用传统方法
        results = self.base_retriever.get_relevant_documents(question)
        return results[:final_k]
    
    def retrieve(self, question: str, k: int = None) -> List[Document]:
        """
        层级检索主方法：先定位竞赛，再检索相关内容
        
        Args:
            question: 用户问题
            k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        if k is None:
            k = self.base_retriever.k
        
        try:
            from config import Config
            
            # 获取多阶段检索配置
            stages = getattr(Config, 'RETRIEVAL_STAGES', {})
            enable_multi_stage = stages.get('enable_multi_stage', True)
            
            if enable_multi_stage:
                # 使用多阶段检索策略
                stage1_k = stages.get('stage1_vector_k', 50)  # 第一阶段检索更多文档
                final_k = stages.get('final_k', k)
                logger.info(f"层级检索使用多阶段策略: 第一阶段{stage1_k}个 → 最终{final_k}个")
            else:
                stage1_k = k
                final_k = k
        
            # 1. 检查是否有可用的竞赛索引
            if not self.competition_index:
                logger.debug("无竞赛索引，直接使用全文档检索")
                return self.general_retrieval(question, stage1_k if enable_multi_stage else k, final_k)
            
            # 2. 识别问题中的竞赛名称
            available_competitions = list(self.competition_index.keys())
            comp_name = self.identify_competition(question, available_competitions)
            
            if comp_name and comp_name in self.competition_index:
                # 3. 在特定竞赛文档中检索
                relevant_docs = self.competition_index[comp_name]
                logger.info(f"识别到竞赛: {comp_name}，在 {len(relevant_docs)} 个相关文档中检索")
                
                results = self.fine_grained_retrieval(question, relevant_docs, stage1_k if enable_multi_stage else k)
                
                # 如果启用多阶段检索，对结果进行重排序
                if enable_multi_stage and len(results) > final_k:
                    if hasattr(self.base_retriever, 'reranker') and self.base_retriever.reranker:
                        try:
                            logger.info(f"层级检索重排序: 从{len(results)}个候选中选择{final_k}个")
                            results = self.base_retriever.reranker.rerank(question, results, top_k=final_k)
                        except Exception as e:
                            logger.warning(f"层级检索重排序失败: {e}")
                            results = results[:final_k]
                    else:
                        results = results[:final_k]
                
                # 如果特定竞赛检索结果不足，可以考虑扩展搜索
                if len(results) < final_k // 2 and self.enable_general_fallback:
                    logger.debug(f"竞赛特定检索结果不足({len(results)})，补充全文档检索")
                    general_results = self.general_retrieval(question, final_k - len(results), final_k - len(results))
                    
                    # 去重合并
                    existing_contents = {doc.page_content for doc in results}
                    for doc in general_results:
                        if doc.page_content not in existing_contents:
                            results.append(doc)
                            if len(results) >= final_k:
                                break
                
                return results[:final_k]
            else:
                # 4. 无法识别特定竞赛，使用全文档检索
                logger.debug("未识别到特定竞赛，使用全文档检索")
                return self.general_retrieval(question, stage1_k if enable_multi_stage else k, final_k)
                
        except Exception as e:
            logger.error(f"层级检索失败: {e}，降级到基础检索")
            return self.base_retriever._get_relevant_documents(question)
    
    def get_competition_stats(self) -> Dict[str, int]:
        """获取竞赛文档统计信息"""
        return {comp: len(docs) for comp, docs in self.competition_index.items()}
    
    def debug_question_analysis(self, question: str) -> Dict[str, Any]:
        """调试问题分析，返回详细的分析结果"""
        available_competitions = list(self.competition_index.keys())
        identified_comp = self.identify_competition(question, available_competitions)
        
        analysis = {
            "question": question,
            "available_competitions": available_competitions,
            "identified_competition": identified_comp,
            "competition_stats": self.get_competition_stats(),
            "question_keywords": list(jieba.cut(question.lower())),
        }
        
        # 分析竞赛匹配分数
        if available_competitions:
            comp_scores = {}
            question_lower = question.lower()
            
            for comp_name in available_competitions:
                if comp_name in self.competition_patterns:
                    patterns = self.competition_patterns[comp_name]
                    score = 0
                    matches = []
                    
                    for pattern in patterns:
                        if pattern.lower() in question_lower:
                            count = question_lower.count(pattern.lower())
                            pattern_score = count * len(pattern) * 2
                            score += pattern_score
                            matches.append({"pattern": pattern, "count": count, "score": pattern_score})
                    
                    if score > 0:
                        comp_scores[comp_name] = {"total_score": score, "matches": matches}
            
            analysis["competition_scores"] = comp_scores
        
        return analysis


def create_hierarchical_retriever(base_retriever: LangChainHybridRetriever, 
                                docs: List[Document] = None) -> HierarchicalRetriever:
    """
    创建层级检索器的便捷函数
    
    Args:
        base_retriever: 基础检索器
        docs: 文档列表
        
    Returns:
        层级检索器实例
    """
    return HierarchicalRetriever(
        base_retriever=base_retriever,
        docs=docs,
        similarity_threshold=0.7,
        enable_general_fallback=True
    ) 