#!/usr/bin/env python3
"""
查询增强和优化模块
增加竞赛类型识别和精确匹配策略
"""

import re
import jieba
from typing import Dict, List, Tuple, Optional
from loguru import logger

class CompetitionQueryAnalyzer:
    """竞赛查询分析器"""
    
    def __init__(self):
        """初始化竞赛分析器"""
        try:
            from config import Config
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from config import Config
        self.competition_mapping = getattr(Config, 'COMPETITION_MAPPING', {})
        self.question_types = getattr(Config, 'QUESTION_TYPES', {})
        
        # 初始化jieba分词
        jieba.initialize()
        
        logger.info("竞赛查询分析器初始化完成")
    
    def identify_competition_type(self, query: str) -> Tuple[Optional[str], float]:
        """
        识别查询中的竞赛类型
        
        Args:
            query: 用户查询
            
        Returns:
            (竞赛类型, 置信度)
        """
        query_lower = query.lower()
        best_match = None
        best_score = 0.0
        
        for comp_name, comp_info in self.competition_mapping.items():
            score = 0.0
            keywords = comp_info.get('keywords', [])
            
            # 计算关键词匹配分数
            for keyword in keywords:
                if keyword in query:
                    # 精确匹配加高分
                    score += 2.0
                elif any(part in query_lower for part in keyword.split()):
                    # 部分匹配加低分
                    score += 0.5
            
            # 特殊规则：如果查询包含竞赛的核心名称
            if comp_name in query:
                score += 3.0
            
            if score > best_score:
                best_score = score
                best_match = comp_name
        
        # 置信度计算
        confidence = min(best_score / 3.0, 1.0) if best_score > 0 else 0.0
        
        logger.info(f"竞赛识别结果: {best_match} (置信度: {confidence:.2f})")
        return best_match, confidence
    
    def get_retrieval_strategy(self, query: str) -> Dict:
        """
        根据查询获取检索策略
        
        Args:
            query: 用户查询
            
        Returns:
            检索策略配置
        """
        competition_type, confidence = self.identify_competition_type(query)
        
        # 默认策略
        strategy = {
            "question_type": "basic",
            "competition_filter": None,
            "boost_keywords": [],
            "alpha": 0.5,
            "vector_k": 20,
            "bm25_k": 30
        }
        
        # 如果成功识别竞赛类型且置信度足够
        if competition_type and confidence > 0.3:
            strategy.update({
                "question_type": "competition",
                "competition_filter": competition_type,
                "boost_keywords": self.competition_mapping[competition_type]["keywords"],
                "alpha": 0.3,  # 偏重精确匹配
                "vector_k": 15,
                "bm25_k": 35
            })
            
            logger.info(f"应用竞赛检索策略: {competition_type}")
        else:
            logger.info("使用默认检索策略")
        
        return strategy
    
    def enhance_query_for_competition(self, query: str, strategy: Dict) -> str:
        """
        根据竞赛类型增强查询
        
        Args:
            query: 原始查询
            strategy: 检索策略
            
        Returns:
            增强后的查询
        """
        enhanced_query = query
        
        competition_filter = strategy.get("competition_filter")
        if competition_filter and competition_filter in self.competition_mapping:
            comp_info = self.competition_mapping[competition_filter]
            
            # 如果查询中没有包含关键的竞赛词汇，添加权重
            main_keyword = comp_info["keywords"][0]
            if main_keyword not in query:
                enhanced_query = f"{main_keyword} {query}"
                logger.info(f"查询增强: 添加关键词 '{main_keyword}'")
        
        return enhanced_query


class EnhancedQueryOptimizer:
    """增强的查询优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.competition_analyzer = CompetitionQueryAnalyzer()
        logger.info("增强查询优化器初始化完成")
    
    def optimize_query_for_retrieval(self, query: str) -> Tuple[str, Dict]:
        """
        为检索优化查询
        
        Args:
            query: 原始查询
            
        Returns:
            (优化后的查询, 检索策略)
        """
        # 1. 获取检索策略
        strategy = self.competition_analyzer.get_retrieval_strategy(query)
        
        # 2. 增强查询
        enhanced_query = self.competition_analyzer.enhance_query_for_competition(query, strategy)
        
        # 3. 进一步优化查询文本
        optimized_query = self._optimize_query_text(enhanced_query)
        
        logger.info(f"查询优化完成: '{query}' -> '{optimized_query}'")
        logger.info(f"检索策略: {strategy['question_type']}, alpha={strategy['alpha']}")
        
        return optimized_query, strategy
    
    def _optimize_query_text(self, query: str) -> str:
        """
        优化查询文本
        
        Args:
            query: 查询文本
            
        Returns:
            优化后的查询
        """
        # 移除冗余词汇
        redundant_words = ["什么", "如何", "怎么", "哪些", "是什么"]
        
        optimized = query
        for word in redundant_words:
            if word in optimized and len(optimized) > len(word) + 5:
                optimized = optimized.replace(word, "").strip()
        
        # 清理多余空格
        optimized = re.sub(r'\s+', ' ', optimized).strip()
        
        return optimized


# 全局实例
_enhanced_optimizer = None

def get_enhanced_query_optimizer():
    """获取增强查询优化器实例"""
    global _enhanced_optimizer
    if _enhanced_optimizer is None:
        _enhanced_optimizer = EnhancedQueryOptimizer()
    return _enhanced_optimizer 