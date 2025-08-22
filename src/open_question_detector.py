#!/usr/bin/env python3
"""
开放性问题检测模块
识别需要创新性回答和合理推理的问题类型
"""

import re
import jieba
from typing import Dict, List, Tuple, Optional, Set
from loguru import logger
from dataclasses import dataclass
from enum import Enum

class QuestionType(Enum):
    """问题类型枚举"""
    FACTUAL = "factual"           # 事实性问题（严格基于文档）
    PROCEDURAL = "procedural"     # 程序性问题（如何做某事）
    ADVISORY = "advisory"         # 建议性问题（寻求建议和指导）
    ANALYTICAL = "analytical"     # 分析性问题（需要分析推理）
    COMPARATIVE = "comparative"   # 比较性问题（比较不同选项）
    PREVENTIVE = "preventive"     # 预防性问题（如何避免/防止）

@dataclass
class QuestionAnalysis:
    """问题分析结果"""
    question_type: QuestionType
    confidence: float
    keywords: List[str]
    reasoning_requirements: List[str]
    creativity_level: str  # low, medium, high
    allow_inference: bool
    response_strategy: str

class OpenQuestionDetector:
    """开放性问题检测器"""
    
    def __init__(self):
        """初始化检测器"""
        # 初始化jieba
        jieba.initialize()
        
        # 定义问题类型关键词和模式
        self.question_patterns = {
            QuestionType.FACTUAL: {
                "keywords": ["什么", "多少", "谁", "哪里", "何时", "是否", "有没有"],
                "patterns": [
                    r".*是什么.*",
                    r".*多少.*",
                    r".*什么时候.*",
                    r".*有哪些.*",
                    r".*是否.*"
                ],
                "creativity": "low",
                "allow_inference": False
            },
            QuestionType.PROCEDURAL: {
                "keywords": ["如何", "怎么", "怎样", "方法", "步骤", "流程"],
                "patterns": [
                    r"如何.*",
                    r"怎么.*",
                    r"怎样.*", 
                    r".*方法.*",
                    r".*步骤.*",
                    r".*流程.*"
                ],
                "creativity": "medium",
                "allow_inference": True
            },
            QuestionType.ADVISORY: {
                "keywords": ["建议", "推荐", "意见", "应该", "最好", "选择"],
                "patterns": [
                    r".*建议.*",
                    r".*推荐.*",
                    r".*应该.*",
                    r".*最好.*",
                    r".*选择.*"
                ],
                "creativity": "high",
                "allow_inference": True
            },
            QuestionType.PREVENTIVE: {
                "keywords": ["避免", "防止", "预防", "杜绝", "确保", "保护"],
                "patterns": [
                    r"如何.*避免.*",
                    r"如何.*防止.*",
                    r"如何.*预防.*",
                    r"如何.*确保.*",
                    r"如何.*保护.*",
                    r".*被.*剽窃.*",
                    r".*知识产权.*"
                ],
                "creativity": "high",
                "allow_inference": True
            },
            QuestionType.ANALYTICAL: {
                "keywords": ["分析", "原因", "为什么", "影响", "优势", "缺点"],
                "patterns": [
                    r"为什么.*",
                    r".*原因.*",
                    r".*影响.*",
                    r".*优势.*",
                    r".*缺点.*"
                ],
                "creativity": "medium",
                "allow_inference": True
            },
            QuestionType.COMPARATIVE: {
                "keywords": ["比较", "对比", "区别", "差异", "哪个更好"],
                "patterns": [
                    r".*比较.*",
                    r".*对比.*",
                    r".*区别.*",
                    r".*差异.*",
                    r".*哪个更.*"
                ],
                "creativity": "medium", 
                "allow_inference": True
            }
        }
        
        # 竞赛相关的特殊关键词
        self.competition_keywords = {
            "intellectual_property": ["知识产权", "剽窃", "抄袭", "原创", "版权", "专利"],
            "protection": ["保护", "防护", "安全", "隐私", "机密"],
            "innovation": ["创新", "独特", "原创性", "新颖"],
            "ethics": ["道德", "伦理", "规范", "准则"]
        }
        
        logger.info("开放性问题检测器初始化完成")
    
    def analyze_question(self, question: str) -> QuestionAnalysis:
        """
        分析问题类型和特征
        
        Args:
            question: 用户问题
            
        Returns:
            QuestionAnalysis: 问题分析结果
        """
        question = question.strip()
        logger.debug(f"分析问题: {question}")
        
        # 分词处理
        words = list(jieba.cut(question.lower()))
        
        # 检测问题类型
        question_type, confidence = self._detect_question_type(question, words)
        
        # 提取关键词
        keywords = self._extract_keywords(question, words)
        
        # 分析推理需求
        reasoning_requirements = self._analyze_reasoning_requirements(question, question_type)
        
        # 确定创新性级别
        creativity_level = self.question_patterns[question_type]["creativity"]
        
        # 是否允许推理
        allow_inference = self.question_patterns[question_type]["allow_inference"]
        
        # 确定回答策略
        response_strategy = self._determine_response_strategy(question_type, question, words)
        
        analysis = QuestionAnalysis(
            question_type=question_type,
            confidence=confidence,
            keywords=keywords,
            reasoning_requirements=reasoning_requirements,
            creativity_level=creativity_level,
            allow_inference=allow_inference,
            response_strategy=response_strategy
        )
        
        logger.debug(f"问题分析结果: {question_type.value}, 创新性: {creativity_level}, 允许推理: {allow_inference}")
        
        return analysis
    
    def _detect_question_type(self, question: str, words: List[str]) -> Tuple[QuestionType, float]:
        """检测问题类型"""
        scores = {}
        
        for q_type, config in self.question_patterns.items():
            score = 0.0
            
            # 关键词匹配
            keyword_matches = sum(1 for keyword in config["keywords"] if keyword in question)
            score += keyword_matches * 0.3
            
            # 模式匹配
            pattern_matches = sum(1 for pattern in config["patterns"] if re.search(pattern, question))
            score += pattern_matches * 0.5
            
            # 特殊处理：知识产权保护相关
            if q_type == QuestionType.PREVENTIVE:
                ip_keywords = self.competition_keywords["intellectual_property"]
                protection_keywords = self.competition_keywords["protection"]
                
                ip_matches = sum(1 for kw in ip_keywords if kw in question)
                protection_matches = sum(1 for kw in protection_keywords if kw in question)
                
                score += (ip_matches + protection_matches) * 0.4
            
            scores[q_type] = score
        
        # 找到得分最高的类型
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # 计算置信度
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.0
        
        # 如果置信度太低，默认为事实性问题
        if confidence < 0.3:
            return QuestionType.FACTUAL, 0.5
        
        return best_type, confidence
    
    def _extract_keywords(self, question: str, words: List[str]) -> List[str]:
        """提取关键词"""
        # 过滤停用词
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        
        keywords = []
        for word in words:
            if len(word) > 1 and word not in stop_words:
                keywords.append(word)
        
        # 添加特殊领域关键词
        for category, kw_list in self.competition_keywords.items():
            for kw in kw_list:
                if kw in question:
                    keywords.append(kw)
        
        return list(set(keywords))
    
    def _analyze_reasoning_requirements(self, question: str, question_type: QuestionType) -> List[str]:
        """分析推理需求"""
        requirements = []
        
        if question_type == QuestionType.PROCEDURAL:
            requirements.extend(["步骤分解", "方法论证", "实践指导"])
        
        elif question_type == QuestionType.ADVISORY:
            requirements.extend(["风险评估", "方案比较", "经验总结"])
        
        elif question_type == QuestionType.PREVENTIVE:
            requirements.extend(["风险识别", "预防策略", "保护措施", "应对方案"])
        
        elif question_type == QuestionType.ANALYTICAL:
            requirements.extend(["因果分析", "逻辑推理", "影响评估"])
        
        elif question_type == QuestionType.COMPARATIVE:
            requirements.extend(["对比分析", "优劣评价", "选择建议"])
        
        # 特殊领域需求
        if any(kw in question for kw in self.competition_keywords["intellectual_property"]):
            requirements.extend(["法律考量", "规范引用", "实用建议"])
        
        return requirements
    
    def _determine_response_strategy(self, question_type: QuestionType, question: str, words: List[str]) -> str:
        """确定回答策略"""
        
        if question_type == QuestionType.FACTUAL:
            return "strict_documentation"  # 严格基于文档
        
        elif question_type == QuestionType.PROCEDURAL:
            return "guided_inference"      # 指导性推理
        
        elif question_type == QuestionType.ADVISORY:
            return "creative_synthesis"    # 创新性综合
        
        elif question_type == QuestionType.PREVENTIVE:
            return "protective_reasoning"  # 保护性推理
        
        elif question_type == QuestionType.ANALYTICAL:
            return "analytical_reasoning"  # 分析性推理
        
        elif question_type == QuestionType.COMPARATIVE:
            return "comparative_analysis"  # 比较分析
        
        else:
            return "balanced_approach"     # 平衡方法
    
    def is_open_question(self, question: str) -> bool:
        """判断是否为开放性问题"""
        analysis = self.analyze_question(question)
        return analysis.allow_inference and analysis.creativity_level in ["medium", "high"]
    
    def get_response_guidelines(self, analysis: QuestionAnalysis) -> Dict[str, any]:
        """获取回答指南"""
        guidelines = {
            "allow_inference": analysis.allow_inference,
            "creativity_level": analysis.creativity_level,
            "reasoning_requirements": analysis.reasoning_requirements,
            "response_strategy": analysis.response_strategy,
            "suggested_structure": self._get_suggested_structure(analysis.question_type),
            "tone": self._get_suggested_tone(analysis.question_type)
        }
        
        return guidelines
    
    def _get_suggested_structure(self, question_type: QuestionType) -> List[str]:
        """获取建议的回答结构"""
        structures = {
            QuestionType.FACTUAL: ["直接回答", "引用文档"],
            QuestionType.PROCEDURAL: ["步骤概述", "详细说明", "注意事项"],
            QuestionType.ADVISORY: ["现状分析", "建议方案", "实施要点"],
            QuestionType.PREVENTIVE: ["风险识别", "预防措施", "应对策略", "相关规范"],
            QuestionType.ANALYTICAL: ["问题分析", "原因探讨", "影响评估"],
            QuestionType.COMPARATIVE: ["选项列举", "对比分析", "推荐建议"]
        }
        
        return structures.get(question_type, ["综合分析", "建议方案"])
    
    def _get_suggested_tone(self, question_type: QuestionType) -> str:
        """获取建议的回答语调"""
        tones = {
            QuestionType.FACTUAL: "客观准确",
            QuestionType.PROCEDURAL: "指导性友好",
            QuestionType.ADVISORY: "建议性专业",
            QuestionType.PREVENTIVE: "谨慎负责",
            QuestionType.ANALYTICAL: "理性分析",
            QuestionType.COMPARATIVE: "平衡客观"
        }
        
        return tones.get(question_type, "专业友好") 