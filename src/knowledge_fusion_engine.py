#!/usr/bin/env python3
"""
知识融合引擎
结合文档内容和常识进行合理推理
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import re
import json

try:
    from .open_question_detector import QuestionType, QuestionAnalysis
except ImportError:
    from open_question_detector import QuestionType, QuestionAnalysis

@dataclass
class KnowledgeSource:
    """知识来源"""
    source_type: str  # "document", "common_sense", "domain_knowledge"
    content: str
    confidence: float
    relevance: float

@dataclass
class FusionResult:
    """融合结果"""
    fused_knowledge: str
    sources_used: List[KnowledgeSource]
    fusion_strategy: str
    confidence_score: float

class KnowledgeFusionEngine:
    """知识融合引擎"""
    
    def __init__(self):
        """初始化知识融合引擎"""
        # 竞赛相关的领域知识库
        self.domain_knowledge = self._build_domain_knowledge()
        
        # 常识知识库
        self.common_sense_knowledge = self._build_common_sense_knowledge()
        
        # 融合策略配置
        self.fusion_strategies = {
            "document_priority": {"doc": 0.7, "domain": 0.2, "common": 0.1},
            "balanced_fusion": {"doc": 0.5, "domain": 0.3, "common": 0.2},
            "creative_synthesis": {"doc": 0.4, "domain": 0.3, "common": 0.3},
            "domain_enhanced": {"doc": 0.6, "domain": 0.4, "common": 0.0}
        }
        
        logger.info("知识融合引擎初始化完成")
    
    def _build_domain_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """构建竞赛领域知识库"""
        return {
            "intellectual_property": {
                "knowledge": {
                    "原创性保护": {
                        "定义": "确保作品完全由参赛者独立创作，不侵犯他人知识产权",
                        "方法": ["时间戳记录", "版本控制", "设计日志", "原创性声明"],
                        "风险": ["无意侵权", "相似作品争议", "团队内部纠纷"],
                        "建议": "建立完整的创作记录，使用正当渠道的开源资源"
                    },
                    "版权保护": {
                        "定义": "保护原创作品不被他人未经授权使用或抄袭",
                        "方法": ["版权登记", "水印技术", "发布时间证明", "第三方托管"],
                        "风险": ["举证困难", "维权成本高", "跨境保护复杂"],
                        "建议": "及时申请版权保护，保留完整的创作证据链"
                    },
                    "商业机密": {
                        "定义": "竞赛过程中产生的具有商业价值的技术和创意",
                        "方法": ["保密协议", "访问控制", "信息分级", "团队约束"],
                        "风险": ["泄露风险", "内部纠纷", "竞争对手获取"],
                        "建议": "建立严格的保密制度，限制信息传播范围"
                    }
                },
                "best_practices": [
                    "建立版本控制系统记录开发历史",
                    "定期备份项目文件和设计文档",
                    "使用正版软件和合法素材",
                    "团队成员签署知识产权协议",
                    "咨询专业律师了解相关法律"
                ]
            },
            "competition_strategy": {
                "knowledge": {
                    "项目管理": {
                        "原则": "合理分工、时间管理、风险控制",
                        "方法": ["甘特图", "里程碑管理", "任务分解", "进度跟踪"],
                        "工具": ["项目管理软件", "版本控制系统", "协作平台"],
                        "建议": "提前规划，预留缓冲时间，建立定期检查机制"
                    },
                    "团队协作": {
                        "原则": "明确分工、有效沟通、共同目标",
                        "方法": ["角色定义", "沟通机制", "决策流程", "冲突解决"],
                        "工具": ["协作平台", "文档共享", "在线会议", "任务管理"],
                        "建议": "建立清晰的沟通渠道，定期团队会议，明确责任分工"
                    }
                }
            },
            "technical_excellence": {
                "knowledge": {
                    "创新性": {
                        "评价维度": ["技术创新", "应用创新", "商业模式创新"],
                        "提升方法": ["技术调研", "需求分析", "创意思维", "原型验证"],
                        "建议": "关注前沿技术，深入了解应用场景，注重用户体验"
                    },
                    "可行性": {
                        "评价维度": ["技术可行性", "经济可行性", "时间可行性"],
                        "评估方法": ["技术验证", "成本分析", "时间估算", "风险评估"],
                        "建议": "充分验证技术方案，合理评估资源需求，制定应急计划"
                    }
                }
            }
        }
    
    def _build_common_sense_knowledge(self) -> Dict[str, List[str]]:
        """构建常识知识库"""
        return {
            "protection_methods": [
                "定期备份重要文件，使用多重存储方式",
                "建立时间戳和版本记录，证明创作时间线",
                "使用可信的第三方平台托管重要代码",
                "与团队成员签署明确的知识产权协议",
                "保留完整的设计思路和开发日志",
                "避免在公共场所讨论核心技术细节",
                "使用加密技术保护敏感信息",
                "定期检查是否有相似作品发布"
            ],
            "legal_considerations": [
                "了解相关法律法规和竞赛规则",
                "咨询专业的知识产权律师",
                "及时申请专利或版权保护",
                "建立完整的法律证据链",
                "购买必要的保险保障",
                "建立应急处理预案"
            ],
            "best_practices": [
                "采用业界标准的开发流程和工具",
                "建立完善的文档记录体系",
                "定期进行安全性评估和检查",
                "保持与行业专家的交流和学习",
                "关注相关技术发展趋势",
                "建立持续改进的机制"
            ],
            "risk_management": [
                "识别潜在的知识产权风险",
                "建立多层次的保护机制",
                "制定风险应对预案",
                "定期评估和更新保护措施",
                "建立内部监督和检查机制",
                "保持与相关机构的沟通"
            ]
        }
    
    def fuse_knowledge(self, 
                      question: str,
                      document_content: str,
                      analysis: QuestionAnalysis) -> FusionResult:
        """
        融合多源知识
        
        Args:
            question: 用户问题
            document_content: 文档内容
            analysis: 问题分析结果
            
        Returns:
            FusionResult: 融合结果
        """
        logger.debug(f"开始知识融合: {question[:50]}...")
        
        # 1. 收集不同来源的知识
        knowledge_sources = []
        
        # 文档知识
        if document_content and len(document_content.strip()) > 20:
            doc_source = KnowledgeSource(
                source_type="document",
                content=document_content,
                confidence=0.9,
                relevance=self._calculate_relevance(question, document_content)
            )
            knowledge_sources.append(doc_source)
        
        # 领域知识
        domain_knowledge = self._extract_domain_knowledge(question, analysis)
        if domain_knowledge:
            domain_source = KnowledgeSource(
                source_type="domain_knowledge",
                content=domain_knowledge,
                confidence=0.8,
                relevance=self._calculate_domain_relevance(question, analysis)
            )
            knowledge_sources.append(domain_source)
        
        # 常识知识
        common_sense = self._extract_common_sense(question, analysis)
        if common_sense:
            common_source = KnowledgeSource(
                source_type="common_sense",
                content=common_sense,
                confidence=0.7,
                relevance=self._calculate_common_sense_relevance(question, analysis)
            )
            knowledge_sources.append(common_source)
        
        # 2. 选择融合策略
        fusion_strategy = self._select_fusion_strategy(analysis)
        
        # 3. 执行知识融合
        fused_knowledge = self._execute_fusion(knowledge_sources, fusion_strategy, analysis)
        
        # 4. 计算置信度
        confidence_score = self._calculate_fusion_confidence(knowledge_sources, fusion_strategy)
        
        result = FusionResult(
            fused_knowledge=fused_knowledge,
            sources_used=knowledge_sources,
            fusion_strategy=fusion_strategy,
            confidence_score=confidence_score
        )
        
        logger.info(f"知识融合完成，策略: {fusion_strategy}, 置信度: {confidence_score:.3f}")
        
        return result
    
    def _extract_domain_knowledge(self, question: str, analysis: QuestionAnalysis) -> str:
        """提取相关的领域知识"""
        relevant_knowledge = []
        question_lower = question.lower()
        
        # 知识产权相关
        if any(kw in question_lower for kw in ["知识产权", "剽窃", "抄袭", "原创", "版权", "保护"]):
            ip_knowledge = self.domain_knowledge.get("intellectual_property", {}).get("knowledge", {})
            
            for topic, info in ip_knowledge.items():
                if any(kw in question_lower for kw in topic.split()):
                    relevant_knowledge.append(f"**{topic}**：{info.get('定义', '')}")
                    if "方法" in info:
                        methods = "、".join(info["方法"])
                        relevant_knowledge.append(f"主要方法：{methods}")
                    if "建议" in info:
                        relevant_knowledge.append(f"专家建议：{info['建议']}")
        
        # 竞赛策略相关
        if any(kw in question_lower for kw in ["策略", "方法", "管理", "团队", "协作"]):
            strategy_knowledge = self.domain_knowledge.get("competition_strategy", {}).get("knowledge", {})
            
            for topic, info in strategy_knowledge.items():
                if any(kw in question_lower for kw in topic.split()):
                    relevant_knowledge.append(f"**{topic}**：{info.get('原则', '')}")
                    if "方法" in info:
                        methods = "、".join(info["方法"])
                        relevant_knowledge.append(f"推荐方法：{methods}")
                    if "建议" in info:
                        relevant_knowledge.append(f"实施建议：{info['建议']}")
        
        # 技术卓越相关
        if any(kw in question_lower for kw in ["创新", "技术", "可行性", "质量"]):
            tech_knowledge = self.domain_knowledge.get("technical_excellence", {}).get("knowledge", {})
            
            for topic, info in tech_knowledge.items():
                if any(kw in question_lower for kw in topic.split()):
                    relevant_knowledge.append(f"**{topic}**：{info.get('评价维度', '')}")
                    if "建议" in info:
                        relevant_knowledge.append(f"提升建议：{info['建议']}")
        
        return "\n".join(relevant_knowledge)
    
    def _extract_common_sense(self, question: str, analysis: QuestionAnalysis) -> str:
        """提取相关的常识知识"""
        relevant_common_sense = []
        question_lower = question.lower()
        
        # 保护方法
        if any(kw in question_lower for kw in ["保护", "防止", "避免", "确保"]):
            protection_methods = self.common_sense_knowledge.get("protection_methods", [])
            relevant_common_sense.extend(protection_methods[:4])  # 取前4条
        
        # 法律考虑
        if any(kw in question_lower for kw in ["法律", "合规", "风险", "规则"]):
            legal_considerations = self.common_sense_knowledge.get("legal_considerations", [])
            relevant_common_sense.extend(legal_considerations[:3])  # 取前3条
        
        # 最佳实践
        if analysis.question_type in [QuestionType.PROCEDURAL, QuestionType.ADVISORY]:
            best_practices = self.common_sense_knowledge.get("best_practices", [])
            relevant_common_sense.extend(best_practices[:3])  # 取前3条
        
        # 风险管理
        if analysis.question_type == QuestionType.PREVENTIVE:
            risk_management = self.common_sense_knowledge.get("risk_management", [])
            relevant_common_sense.extend(risk_management[:4])  # 取前4条
        
        return "\n".join([f"• {item}" for item in relevant_common_sense])
    
    def _select_fusion_strategy(self, analysis: QuestionAnalysis) -> str:
        """选择融合策略"""
        if analysis.question_type == QuestionType.FACTUAL:
            return "document_priority"  # 事实性问题优先文档
        
        elif analysis.question_type in [QuestionType.PREVENTIVE, QuestionType.ADVISORY]:
            return "creative_synthesis"  # 预防性和建议性问题需要创新融合
        
        elif analysis.question_type in [QuestionType.PROCEDURAL, QuestionType.ANALYTICAL]:
            return "domain_enhanced"  # 程序性和分析性问题增强领域知识
        
        else:
            return "balanced_fusion"  # 默认平衡融合
    
    def _execute_fusion(self, 
                       knowledge_sources: List[KnowledgeSource],
                       strategy: str,
                       analysis: QuestionAnalysis) -> str:
        """执行知识融合"""
        if not knowledge_sources:
            return "未找到相关知识信息。"
        
        # 获取策略权重
        weights = self.fusion_strategies.get(strategy, self.fusion_strategies["balanced_fusion"])
        
        # 按来源组织知识
        doc_content = ""
        domain_content = ""
        common_content = ""
        
        for source in knowledge_sources:
            if source.source_type == "document":
                doc_content = source.content
            elif source.source_type == "domain_knowledge":
                domain_content = source.content
            elif source.source_type == "common_sense":
                common_content = source.content
        
        # 构建融合内容
        fused_parts = []
        
        # 文档内容
        if doc_content and weights.get("doc", 0) > 0:
            fused_parts.append(f"📄 **基于文档信息**：\n{doc_content[:600]}...")
        
        # 领域知识
        if domain_content and weights.get("domain", 0) > 0:
            fused_parts.append(f"🎓 **专业知识参考**：\n{domain_content}")
        
        # 常识知识
        if common_content and weights.get("common", 0) > 0:
            fused_parts.append(f"💡 **最佳实践建议**：\n{common_content}")
        
        # 添加综合建议
        if analysis.question_type == QuestionType.PREVENTIVE:
            fused_parts.append(self._generate_protection_synthesis(analysis))
        elif analysis.question_type == QuestionType.ADVISORY:
            fused_parts.append(self._generate_advisory_synthesis(analysis))
        
        return "\n\n".join(fused_parts)
    
    def _generate_protection_synthesis(self, analysis: QuestionAnalysis) -> str:
        """生成保护性综合建议"""
        synthesis = "🛡️ **综合保护策略**：\n"
        synthesis += "1. **立即行动**：建立版本控制，记录创作时间线\n"
        synthesis += "2. **系统保护**：申请相关知识产权保护，签署团队协议\n"
        synthesis += "3. **持续监控**：定期检查相似作品，维护原创性声明\n"
        synthesis += "4. **应急预案**：制定侵权应对流程，保留法律途径"
        return synthesis
    
    def _generate_advisory_synthesis(self, analysis: QuestionAnalysis) -> str:
        """生成建议性综合方案"""
        synthesis = "📋 **综合建议方案**：\n"
        synthesis += "1. **评估现状**：分析当前资源和能力\n"
        synthesis += "2. **制定策略**：基于目标制定多层次方案\n"
        synthesis += "3. **风险控制**：识别潜在风险并制定应对措施\n"
        synthesis += "4. **持续优化**：建立反馈机制，不断改进方案"
        return synthesis
    
    def _calculate_relevance(self, question: str, content: str) -> float:
        """计算文档内容与问题的相关性"""
        question_words = set(question.lower().split())
        content_words = set(content.lower().split())
        
        if not question_words:
            return 0.0
        
        intersection = question_words.intersection(content_words)
        relevance = len(intersection) / len(question_words)
        
        return min(relevance, 1.0)
    
    def _calculate_domain_relevance(self, question: str, analysis: QuestionAnalysis) -> float:
        """计算领域知识的相关性"""
        # 基于问题类型和关键词计算相关性
        base_relevance = 0.6
        
        if analysis.question_type in [QuestionType.PREVENTIVE, QuestionType.ADVISORY]:
            base_relevance = 0.8
        
        # 基于关键词调整
        question_lower = question.lower()
        domain_keywords = ["知识产权", "保护", "策略", "方法", "创新", "技术"]
        
        keyword_matches = sum(1 for kw in domain_keywords if kw in question_lower)
        keyword_boost = min(keyword_matches * 0.1, 0.3)
        
        return min(base_relevance + keyword_boost, 1.0)
    
    def _calculate_common_sense_relevance(self, question: str, analysis: QuestionAnalysis) -> float:
        """计算常识知识的相关性"""
        # 开放性问题的常识相关性更高
        if analysis.allow_inference:
            return 0.7
        else:
            return 0.4
    
    def _calculate_fusion_confidence(self, 
                                   knowledge_sources: List[KnowledgeSource],
                                   strategy: str) -> float:
        """计算融合置信度"""
        if not knowledge_sources:
            return 0.0
        
        weights = self.fusion_strategies.get(strategy, self.fusion_strategies["balanced_fusion"])
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for source in knowledge_sources:
            weight = 0.0
            if source.source_type == "document":
                weight = weights.get("doc", 0)
            elif source.source_type == "domain_knowledge":
                weight = weights.get("domain", 0)
            elif source.source_type == "common_sense":
                weight = weights.get("common", 0)
            
            total_confidence += source.confidence * source.relevance * weight
            total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0 