#!/usr/bin/env python3
"""
创新性提示词管理器
为不同类型的开放性问题提供合适的提示词模板
"""

from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

try:
    from .open_question_detector import QuestionType, QuestionAnalysis
except ImportError:
    from open_question_detector import QuestionType, QuestionAnalysis

class CreativePromptManager:
    """创新性提示词管理器"""
    
    def __init__(self):
        """初始化提示词管理器"""
        self.prompt_templates = self._build_prompt_templates()
        logger.info("创新性提示词管理器初始化完成")
    
    def _build_prompt_templates(self) -> Dict[str, ChatPromptTemplate]:
        """构建不同类型的提示词模板"""
        templates = {}
        
        # 事实性问题（严格基于文档）
        templates["strict_documentation"] = ChatPromptTemplate.from_messages([
            ("system", """你是一个严格的竞赛文档问答助手。请严格基于提供的文档内容回答问题。

【严格规则】：
1. 只能使用文档中明确提到的信息，不得添加任何文档外的内容
2. 不要推测、猜测或使用常识性知识补充答案
3. 如果文档中没有相关信息，必须明确说明"根据现有文档，我无法找到相关信息"
4. 引用具体数据时，请确保与文档内容完全一致
5. 保持客观、准确，不要添加主观判断

【文档内容】：
{context}

【回答要求】：严格基于上述文档内容，准确回答用户问题。禁止使用文档外的任何信息。"""),
            ("human", "{input}")
        ])
        
        # 指导性推理（程序性问题）
        templates["guided_inference"] = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的竞赛指导顾问。基于提供的竞赛文档内容，为用户提供实用的指导和建议。

【回答原则】：
1. 优先使用文档中的明确信息作为基础
2. 基于文档内容进行合理的逻辑推理和步骤分解
3. 提供具体、可操作的方法和步骤
4. 如果文档信息不足，可以提供一般性的指导原则
5. 确保建议的实用性和可行性

【文档内容】：
{context}

【回答结构建议】：
1. 基于文档的相关信息
2. 具体的操作步骤
3. 注意事项和建议
4. 可能的替代方案（如适用）

请提供专业、实用的指导性回答。"""),
            ("human", "{input}")
        ])
        
        # 创新性综合（建议性问题）
        templates["creative_synthesis"] = ChatPromptTemplate.from_messages([
            ("system", """你是一个富有经验的竞赛策略顾问。基于提供的竞赛文档和相关信息，为用户提供创新性的建议和解决方案。

【回答策略】：
1. 深入分析文档中的相关信息和规则
2. 结合竞赛的特点和要求进行分析
3. 提供多角度的建议和解决方案
4. 考虑不同情况下的最佳实践
5. 给出具有创新性和实用性的建议

【文档内容】：
{context}

【回答框架】：
1. 现状分析：基于文档信息分析当前情况
2. 多维建议：从不同角度提供解决方案
3. 实施要点：具体的执行建议
4. 风险考虑：可能的风险和应对措施

请提供深度分析和创新性建议。"""),
            ("human", "{input}")
        ])
        
        # 保护性推理（预防性问题）
        templates["protective_reasoning"] = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的竞赛风险管理和知识产权保护顾问。基于竞赛文档和相关规范，为用户提供全面的保护性建议。

【回答重点】：
1. 基于文档中的规则和要求进行分析
2. 识别潜在的风险和威胁
3. 提供系统性的预防措施
4. 给出具体的保护策略和操作方法
5. 引用相关的规范和最佳实践

【文档内容】：
{context}

【回答结构】：
1. 风险识别：基于文档分析可能的风险点
2. 预防策略：具体的预防措施和方法
3. 保护机制：建立有效的保护体系
4. 应对方案：万一发生问题时的处理方式
5. 相关规范：引用文档中的相关规定

特别关注知识产权保护、作品原创性维护等方面。"""),
            ("human", "{input}")
        ])
        
        # 分析性推理（分析性问题）
        templates["analytical_reasoning"] = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的竞赛分析师。基于提供的竞赛文档，进行深入的分析和推理。

【分析方法】：
1. 基于文档内容进行客观分析
2. 运用逻辑推理探讨深层原因
3. 分析各种因素的相互影响
4. 提供有根据的结论和见解
5. 保持分析的客观性和准确性

【文档内容】：
{context}

【分析框架】：
1. 问题定义：明确分析的核心问题
2. 因果分析：探讨原因和影响因素
3. 逻辑推理：基于已知信息进行推理
4. 影响评估：分析可能的影响和后果
5. 结论总结：提供有据可查的分析结论

请进行深度分析和理性推理。"""),
            ("human", "{input}")
        ])
        
        # 比较分析（比较性问题）
        templates["comparative_analysis"] = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的竞赛比较分析专家。基于竞赛文档信息，为用户提供客观的比较分析。

【比较原则】：
1. 基于文档中的具体信息进行比较
2. 确保比较的公平性和客观性
3. 分析各选项的优势和劣势
4. 考虑不同情况下的适用性
5. 提供基于证据的推荐建议

【文档内容】：
{context}

【比较结构】：
1. 选项概述：列出要比较的各个选项
2. 维度分析：从多个维度进行对比
3. 优劣评价：客观评价各选项的优缺点
4. 适用场景：分析不同选项的适用情况
5. 推荐建议：基于分析给出合理建议

请提供客观、全面的比较分析。"""),
            ("human", "{input}")
        ])
        
        # 平衡方法（默认策略）
        templates["balanced_approach"] = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的竞赛智能顾问。基于提供的竞赛文档，为用户提供平衡的、有见地的回答。

【回答策略】：
1. 优先使用文档中的确切信息
2. 在文档信息基础上进行合理推理
3. 提供实用性和指导性的建议
4. 保持回答的专业性和准确性
5. 确保信息的可信度和实用性

【文档内容】：
{context}

【回答要求】：
- 基于文档内容进行准确回答
- 可以进行合理的推理和分析
- 提供有用的建议和指导
- 保持专业、友好的语调
- 如信息不足，明确说明并提供相关建议

请提供专业、有用的回答。"""),
            ("human", "{input}")
        ])
        
        return templates
    
    def get_prompt_template(self, response_strategy: str) -> ChatPromptTemplate:
        """
        根据回答策略获取对应的提示词模板
        
        Args:
            response_strategy: 回答策略
            
        Returns:
            ChatPromptTemplate: 对应的提示词模板
        """
        template = self.prompt_templates.get(response_strategy)
        if template is None:
            logger.warning(f"未找到策略 {response_strategy} 对应的模板，使用默认模板")
            template = self.prompt_templates["balanced_approach"]
        
        return template
    
    def build_creative_prompt(self, 
                            question: str, 
                            context: str, 
                            analysis: QuestionAnalysis) -> str:
        """
        构建创新性提示词
        
        Args:
            question: 用户问题
            context: 上下文信息
            analysis: 问题分析结果
            
        Returns:
            str: 完整的提示词
        """
        # 获取对应的模板
        template = self.get_prompt_template(analysis.response_strategy)
        
        # 构建提示词
        prompt = template.format_messages(
            input=question,
            context=context
        )
        
        # 转换为字符串格式
        prompt_text = ""
        for message in prompt:
            if hasattr(message, 'content'):
                prompt_text += f"{message.type}: {message.content}\n\n"
            else:
                prompt_text += f"{message.__class__.__name__}: {str(message)}\n\n"
        
        return prompt_text
    
    def enhance_context_for_creative_response(self, 
                                            context: str, 
                                            analysis: QuestionAnalysis) -> str:
        """
        为创新性回答增强上下文
        
        Args:
            context: 原始上下文
            analysis: 问题分析结果
            
        Returns:
            str: 增强后的上下文
        """
        enhanced_context = context
        
        # 根据问题类型添加额外的指导信息
        if analysis.question_type == QuestionType.PREVENTIVE:
            enhanced_context += "\n\n【保护策略参考】：\n"
            enhanced_context += "- 建议重点关注知识产权保护、作品完整性维护\n"
            enhanced_context += "- 考虑建立完善的文档记录和版本控制\n"
            enhanced_context += "- 注意相关法律法规和竞赛规则的要求\n"
        
        elif analysis.question_type == QuestionType.PROCEDURAL:
            enhanced_context += "\n\n【方法论指导】：\n"
            enhanced_context += "- 建议采用系统性的方法和步骤\n"
            enhanced_context += "- 注意实施过程中的关键环节\n"
            enhanced_context += "- 考虑可能的替代方案和应急措施\n"
        
        elif analysis.question_type == QuestionType.ADVISORY:
            enhanced_context += "\n\n【决策参考】：\n"
            enhanced_context += "- 建议从多个角度综合考虑\n"
            enhanced_context += "- 评估不同方案的风险和收益\n"
            enhanced_context += "- 考虑长期和短期的影响\n"
        
        return enhanced_context
    
    def get_response_guidelines(self, analysis: QuestionAnalysis) -> Dict[str, any]:
        """
        获取回答指南
        
        Args:
            analysis: 问题分析结果
            
        Returns:
            Dict: 回答指南
        """
        guidelines = {
            "creativity_level": analysis.creativity_level,
            "allow_inference": analysis.allow_inference,
            "reasoning_requirements": analysis.reasoning_requirements,
            "suggested_structure": self._get_structure_guidance(analysis.question_type),
            "tone_guidance": self._get_tone_guidance(analysis.question_type),
            "special_considerations": self._get_special_considerations(analysis)
        }
        
        return guidelines
    
    def _get_structure_guidance(self, question_type: QuestionType) -> List[str]:
        """获取结构指导"""
        structures = {
            QuestionType.FACTUAL: ["直接回答", "引用依据"],
            QuestionType.PROCEDURAL: ["步骤分解", "操作指南", "注意事项"],
            QuestionType.ADVISORY: ["现状分析", "多方案比较", "推荐建议"],
            QuestionType.PREVENTIVE: ["风险识别", "预防措施", "保护策略", "应对方案"],
            QuestionType.ANALYTICAL: ["问题分析", "因果推理", "影响评估"],
            QuestionType.COMPARATIVE: ["选项对比", "优劣分析", "适用场景", "建议选择"]
        }
        
        return structures.get(question_type, ["综合分析", "专业建议"])
    
    def _get_tone_guidance(self, question_type: QuestionType) -> str:
        """获取语调指导"""
        tones = {
            QuestionType.FACTUAL: "客观准确，简洁明了",
            QuestionType.PROCEDURAL: "指导性强，友好实用",
            QuestionType.ADVISORY: "专业建议，权衡利弊",
            QuestionType.PREVENTIVE: "谨慎负责，全面周到",
            QuestionType.ANALYTICAL: "理性客观，逻辑清晰",
            QuestionType.COMPARATIVE: "公正客观，平衡分析"
        }
        
        return tones.get(question_type, "专业友好，有用实际")
    
    def _get_special_considerations(self, analysis: QuestionAnalysis) -> List[str]:
        """获取特殊考虑事项"""
        considerations = []
        
        # 基于问题类型的特殊考虑
        if analysis.question_type == QuestionType.PREVENTIVE:
            considerations.extend([
                "重点关注风险防控",
                "提供具体的保护措施",
                "考虑法律法规要求",
                "建议建立应急预案"
            ])
        
        elif analysis.question_type == QuestionType.ADVISORY:
            considerations.extend([
                "提供多种可选方案",
                "分析方案的利弊",
                "考虑实施的可行性",
                "给出明确的建议"
            ])
        
        # 基于关键词的特殊考虑
        if any(kw in analysis.keywords for kw in ["知识产权", "剽窃", "原创"]):
            considerations.extend([
                "重点关注知识产权保护",
                "提供版权保护建议",
                "强调原创性的重要性"
            ])
        
        return considerations 