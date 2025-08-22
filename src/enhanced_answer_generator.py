#!/usr/bin/env python3
"""
增强的答案生成器
整合开放性问题检测和创新性提示词功能
"""

import os
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from loguru import logger
from datetime import datetime

try:
    from .open_question_detector import OpenQuestionDetector, QuestionAnalysis, QuestionType
    from .creative_prompt_manager import CreativePromptManager
except ImportError:
    from open_question_detector import OpenQuestionDetector, QuestionAnalysis, QuestionType
    from creative_prompt_manager import CreativePromptManager

class EnhancedAnswerGenerator:
    """增强的答案生成器"""
    
    def __init__(self, llm_client=None):
        """
        初始化增强答案生成器
        
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
        
        # 初始化组件
        self.question_detector = OpenQuestionDetector()
        self.prompt_manager = CreativePromptManager()
        
        # 配置参数
        self.max_context_length = 1500  # 最大上下文长度
        self.enable_creative_mode = True  # 是否启用创新模式
        
        logger.info("增强答案生成器初始化完成")
    
    def generate_answer(self, 
                       question: str, 
                       documents: List[Document],
                       use_creative_mode: bool = None) -> Dict[str, Any]:
        """
        生成增强答案
        
        Args:
            question: 用户问题
            documents: 相关文档
            use_creative_mode: 是否使用创新模式（None时自动判断）
            
        Returns:
            Dict: 包含答案和元数据的字典
        """
        start_time = datetime.now()
        
        try:
            # 1. 分析问题类型
            logger.debug(f"🔍 分析问题: {question[:50]}...")
            analysis = self.question_detector.analyze_question(question)
            
            logger.info(f"📝 问题类型: {analysis.question_type.value}")
            logger.info(f"🎯 创新级别: {analysis.creativity_level}")
            logger.info(f"💡 允许推理: {analysis.allow_inference}")
            logger.info(f"📋 回答策略: {analysis.response_strategy}")
            
            # 2. 确定是否使用创新模式
            if use_creative_mode is None:
                use_creative_mode = self._should_use_creative_mode(analysis)
            
            # 3. 准备上下文
            context = self._prepare_context(documents, analysis)
            
            # 4. 生成答案
            if use_creative_mode and self.enable_creative_mode:
                logger.info("🚀 使用创新模式生成答案")
                answer = self._generate_creative_answer(question, context, analysis)
            else:
                logger.info("📚 使用标准模式生成答案")
                answer = self._generate_standard_answer(question, context)
            
            # 5. 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 6. 构建响应
            response = {
                "answer": answer,
                "question_analysis": {
                    "type": analysis.question_type.value,
                    "creativity_level": analysis.creativity_level,
                    "confidence": analysis.confidence,
                    "allow_inference": analysis.allow_inference,
                    "response_strategy": analysis.response_strategy,
                    "keywords": analysis.keywords,
                    "reasoning_requirements": analysis.reasoning_requirements
                },
                "generation_metadata": {
                    "creative_mode_used": use_creative_mode,
                    "processing_time": processing_time,
                    "context_length": len(context),
                    "documents_used": len(documents)
                },
                "guidelines_used": self.prompt_manager.get_response_guidelines(analysis)
            }
            
            logger.info(f"✅ 答案生成完成，耗时: {processing_time:.2f}s，创新模式: {use_creative_mode}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 答案生成失败: {e}")
            return {
                "answer": f"抱歉，生成答案时发生错误: {e}",
                "question_analysis": None,
                "generation_metadata": {
                    "error": str(e),
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            }
    
    def _should_use_creative_mode(self, analysis: QuestionAnalysis) -> bool:
        """判断是否应该使用创新模式"""
        # 开放性问题使用创新模式
        if analysis.allow_inference and analysis.creativity_level in ["medium", "high"]:
            return True
        
        # 特定类型的问题使用创新模式
        creative_types = [
            QuestionType.PROCEDURAL,
            QuestionType.ADVISORY,
            QuestionType.PREVENTIVE,
            QuestionType.ANALYTICAL,
            QuestionType.COMPARATIVE
        ]
        
        if analysis.question_type in creative_types:
            return True
        
        # 包含特殊关键词的问题使用创新模式
        creative_keywords = ["如何", "建议", "方法", "策略", "保护", "避免", "防止", "确保"]
        question_lower = analysis.keywords if analysis.keywords else []
        
        if any(kw in question_lower for kw in creative_keywords):
            return True
        
        return False
    
    def _prepare_context(self, documents: List[Document], analysis: QuestionAnalysis) -> str:
        """准备上下文信息"""
        if not documents:
            return "未找到相关文档信息。"
        
        # 收集文档内容
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            source = os.path.basename(doc.metadata.get('source', f'文档{i+1}'))
            
            # 添加来源标识
            doc_content = f"【来源：{source}】\n{content}"
            
            # 检查长度限制
            if total_length + len(doc_content) > self.max_context_length:
                # 截断内容
                remaining_length = self.max_context_length - total_length
                if remaining_length > 100:  # 至少保留100字符
                    doc_content = doc_content[:remaining_length] + "..."
                    context_parts.append(doc_content)
                break
            
            context_parts.append(doc_content)
            total_length += len(doc_content)
        
        context = "\n\n".join(context_parts)
        
        # 为创新性回答增强上下文
        if analysis.allow_inference:
            context = self.prompt_manager.enhance_context_for_creative_response(context, analysis)
        
        return context
    
    def _generate_creative_answer(self, 
                                question: str, 
                                context: str, 
                                analysis: QuestionAnalysis) -> str:
        """生成创新性答案"""
        try:
            # 构建创新性提示词
            if hasattr(self.prompt_manager, 'build_creative_prompt'):
                prompt = self.prompt_manager.build_creative_prompt(question, context, analysis)
            else:
                # 获取对应的模板
                template = self.prompt_manager.get_prompt_template(analysis.response_strategy)
                prompt_messages = template.format_messages(input=question, context=context)
                
                # 转换为字符串格式
                prompt = ""
                for message in prompt_messages:
                    if hasattr(message, 'content'):
                        prompt += f"{message.content}\n\n"
                    else:
                        prompt += f"{str(message)}\n\n"
            
            # 使用LLM生成答案
            if self.llm_client:
                answer = self._invoke_llm(prompt)
            else:
                # 后备方案：基于规则的回答
                answer = self._generate_rule_based_answer(question, context, analysis)
            
            # 为答案添加创新性标识
            if analysis.creativity_level == "high":
                answer = f"💡 **创新性建议**\n\n{answer}"
            elif analysis.creativity_level == "medium":
                answer = f"📋 **综合分析**\n\n{answer}"
            
            return answer
            
        except Exception as e:
            logger.error(f"创新性答案生成失败: {e}")
            return self._generate_standard_answer(question, context)
    
    def _generate_standard_answer(self, question: str, context: str) -> str:
        """生成标准答案（严格基于文档）"""
        try:
            # 构建标准提示词
            prompt = f"""基于以下文档内容，严格回答用户问题。

【重要规则】：
1. 只能使用文档中明确提到的信息
2. 如果文档中没有相关信息，请明确说明
3. 不要添加文档外的任何信息
4. 保持回答的准确性和客观性

【文档内容】：
{context}

【用户问题】：{question}

【回答】："""

            # 使用LLM生成答案
            if self.llm_client:
                answer = self._invoke_llm(prompt)
            else:
                # 后备方案：模板回答
                answer = self._generate_template_answer(question, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"标准答案生成失败: {e}")
            return "抱歉，无法生成合适的回答。"
    
    def _invoke_llm(self, prompt: str) -> str:
        """调用LLM生成答案"""
        try:
            if hasattr(self.llm_client, 'invoke'):
                # LangChain LLM
                response = self.llm_client.invoke(prompt)
                if hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
            elif hasattr(self.llm_client, 'generate'):
                # 自定义LLM客户端
                return self.llm_client.generate(prompt)
            elif callable(self.llm_client):
                # 可调用对象
                return self.llm_client(prompt)
            else:
                # 尝试直接调用
                return str(self.llm_client.invoke(prompt))
                
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise e
    
    def _generate_rule_based_answer(self, 
                                  question: str, 
                                  context: str, 
                                  analysis: QuestionAnalysis) -> str:
        """基于规则的答案生成（无LLM时的后备方案）"""
        # 根据问题类型提供不同的回答模板
        if analysis.question_type == QuestionType.PREVENTIVE:
            return self._generate_protection_advice(question, context)
        
        elif analysis.question_type == QuestionType.PROCEDURAL:
            return self._generate_procedural_guidance(question, context)
        
        elif analysis.question_type == QuestionType.ADVISORY:
            return self._generate_advisory_response(question, context)
        
        else:
            return self._generate_template_answer(question, context)
    
    def _generate_protection_advice(self, question: str, context: str) -> str:
        """生成保护性建议"""
        advice = "基于相关文档和最佳实践，以下是保护建议：\n\n"
        
        advice += "🛡️ **风险防控策略**\n"
        advice += "1. **知识产权保护**：\n"
        advice += "   - 及时申请相关专利和版权保护\n"
        advice += "   - 保留完整的设计和开发记录\n"
        advice += "   - 建立版本控制和时间戳机制\n\n"
        
        advice += "2. **作品原创性维护**：\n"
        advice += "   - 确保所有设计和代码的原创性\n"
        advice += "   - 避免使用未经授权的第三方资源\n"
        advice += "   - 建立团队内部的原创性检查机制\n\n"
        
        advice += "3. **文档记录管理**：\n"
        advice += "   - 详细记录设计思路和开发过程\n"
        advice += "   - 保存关键决策的讨论记录\n"
        advice += "   - 建立完整的项目档案\n\n"
        
        if context and len(context.strip()) > 50:
            advice += f"📄 **相关文档信息**：\n{context[:300]}...\n\n"
        
        advice += "⚠️ **注意事项**：\n"
        advice += "- 遵守竞赛规则和相关法律法规\n"
        advice += "- 建议咨询专业的知识产权顾问\n"
        advice += "- 与团队成员签署相关保密协议"
        
        return advice
    
    def _generate_procedural_guidance(self, question: str, context: str) -> str:
        """生成程序性指导"""
        guidance = "根据相关信息，以下是操作指导：\n\n"
        
        guidance += "📋 **基本步骤**：\n"
        guidance += "1. 明确目标和要求\n"
        guidance += "2. 制定详细的实施计划\n"
        guidance += "3. 按步骤执行并记录过程\n"
        guidance += "4. 定期检查和调整策略\n\n"
        
        if context and len(context.strip()) > 50:
            guidance += f"📄 **文档依据**：\n{context[:400]}...\n\n"
        
        guidance += "💡 **实施建议**：\n"
        guidance += "- 建议咨询相关专家或导师\n"
        guidance += "- 参考竞赛规则和要求\n"
        guidance += "- 与团队成员充分沟通协调"
        
        return guidance
    
    def _generate_advisory_response(self, question: str, context: str) -> str:
        """生成建议性回答"""
        response = "基于相关信息，以下是专业建议：\n\n"
        
        response += "🎯 **综合分析**：\n"
        if context and len(context.strip()) > 50:
            response += f"根据文档信息：{context[:300]}...\n\n"
        
        response += "💼 **建议方案**：\n"
        response += "1. **短期策略**：立即采取的措施\n"
        response += "2. **长期规划**：持续改进的方向\n"
        response += "3. **风险管控**：潜在风险的应对\n\n"
        
        response += "⚡ **实施要点**：\n"
        response += "- 根据实际情况调整策略\n"
        response += "- 建立有效的监督机制\n"
        response += "- 定期评估和优化方案"
        
        return response
    
    def _generate_template_answer(self, question: str, context: str) -> str:
        """生成模板回答"""
        if not context or len(context.strip()) < 20:
            return "根据提供的文档，我无法找到足够的相关信息来回答您的问题。建议您提供更多的背景信息或查阅相关的竞赛文档。"
        
        # 简单的关键词匹配
        question_lower = question.lower()
        
        if any(kw in question_lower for kw in ["时间", "什么时候", "开始", "结束"]):
            return f"关于时间安排的信息，根据文档内容：\n\n{context[:400]}..."
        
        elif any(kw in question_lower for kw in ["要求", "条件", "标准"]):
            return f"关于相关要求的信息：\n\n{context[:400]}..."
        
        elif any(kw in question_lower for kw in ["奖项", "奖励", "奖金"]):
            return f"关于奖项设置的信息：\n\n{context[:400]}..."
        
        else:
            return f"根据相关文档，找到以下信息：\n\n{context[:500]}..."
    
    def set_llm_client(self, llm_client):
        """设置LLM客户端"""
        self.llm_client = llm_client
        logger.info("LLM客户端已更新")
    
    def enable_creative_mode(self, enable: bool = True):
        """启用或禁用创新模式"""
        self.enable_creative_mode = enable
        logger.info(f"创新模式已{'启用' if enable else '禁用'}")
    
    def get_question_analysis(self, question: str) -> QuestionAnalysis:
        """获取问题分析结果"""
        return self.question_detector.analyze_question(question) 