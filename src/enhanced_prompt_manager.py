"""
增强的提示词管理器
实现Few-shot重提示优化，特别针对竞赛任务需求进行优化
"""

from typing import Dict, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from loguru import logger

class EnhancedPromptManager:
    """增强的提示词管理器 - 实现Few-shot重提示优化"""
    
    def __init__(self):
        """初始化增强提示词管理器"""
        # 任务关键词检测模式
        self.task_keywords = {
            "交通信号灯": ["智能交通信号灯", "交通信号灯", "信号控制", "信号灯"],
            "未来校园": ["未来校园", "智能校园", "校园应用"],
            "竞赛任务": ["任务描述", "基本要求", "技术要求", "评分标准", "实现方案"],
            "算法设计": ["算法", "设计", "优化", "控制"],
            "系统实现": ["系统", "实现", "架构", "方案"]
        }
        
        # Few-shot示例库
        self.few_shot_examples = self._build_few_shot_examples()
        
        # 构建提示词模板
        self.prompt_templates = self._build_enhanced_prompt_templates()
        
        logger.info("增强提示词管理器初始化完成")
        logger.info(f"  任务类型: {len(self.task_keywords)} 个")
        logger.info(f"  Few-shot示例: {len(self.few_shot_examples)} 个")
        logger.info(f"  提示模板: {len(self.prompt_templates)} 个")
    
    def _build_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """构建Few-shot示例库"""
        examples = {
            "traffic_signal": [
                {
                    "user_question": "智能交通信号灯的基本要求是什么？",
                    "good_context": "【主要章节】未来校园智能应用专项赛 #关键:未来校园智能应用专项赛\n\n三、智能交通信号灯任务\n基本要求：\n1. 设计智能交通信号控制算法\n2. 实现车流量检测功能\n3. 优化信号灯切换时间",
                    "good_answer": "根据未来校园智能应用专项赛的文档，智能交通信号灯的基本要求包括：\n\n1. 设计智能交通信号控制算法\n2. 实现车流量检测功能\n3. 优化信号灯切换时间\n\n这些要求旨在提高交通效率，减少拥堵，实现智能化的交通管理。",
                    "bad_context": "一般信息：交通信号灯是交通管理设备...",
                    "bad_answer": "未找到关于智能交通信号灯任务的具体要求。"
                },
                {
                    "user_question": "交通信号灯优化算法有哪些技术要求？",
                    "good_context": "【子章节】技术要求 #关键:技术要求\n\n算法设计要求：\n1. 采用自适应控制算法\n2. 支持实时数据处理\n3. 具备学习能力",
                    "good_answer": "根据技术要求文档，交通信号灯优化算法的技术要求包括：\n\n1. 采用自适应控制算法 - 能够根据实时交通情况调整信号时序\n2. 支持实时数据处理 - 快速响应交通流量变化\n3. 具备学习能力 - 从历史数据中优化控制策略",
                    "bad_context": "技术文档：算法是计算机程序...",
                    "bad_answer": "未找到关于交通信号灯优化算法的具体技术要求。"
                }
            ],
            "general_task": [
                {
                    "user_question": "竞赛评分标准是什么？",
                    "good_context": "【主要章节】评分标准 #关键:评分标准\n\n评分维度：\n1. 创新性（30%）\n2. 实用性（25%）\n3. 技术水平（25%）\n4. 完整性（20%）",
                    "good_answer": "根据竞赛文档，评分标准包括四个维度：\n\n1. 创新性（30%）- 方案的创新程度和新颖性\n2. 实用性（25%）- 解决方案的实际应用价值\n3. 技术水平（25%）- 技术实现的难度和质量\n4. 完整性（20%）- 方案的完整性和可行性",
                    "bad_context": "一般信息：竞赛是比赛活动...",
                    "bad_answer": "未找到关于竞赛评分标准的具体信息。"
                }
            ]
        }
        
        return examples
    
    def _build_enhanced_prompt_templates(self) -> Dict[str, ChatPromptTemplate]:
        """构建增强的提示词模板"""
        templates = {}
        
        # 交通信号灯专项任务模板
        templates["traffic_signal_task"] = ChatPromptTemplate.from_messages([
            ("system", self._get_traffic_signal_system_prompt()),
            ("human", "{input}")
        ])
        
        # 一般竞赛任务模板
        templates["general_competition"] = ChatPromptTemplate.from_messages([
            ("system", self._get_general_competition_system_prompt()),
            ("human", "{input}")
        ])
        
        # 技术要求专项模板
        templates["technical_requirements"] = ChatPromptTemplate.from_messages([
            ("system", self._get_technical_requirements_system_prompt()),
            ("human", "{input}")
        ])
        
        # 未找到信息的专项模板
        templates["not_found_response"] = ChatPromptTemplate.from_messages([
            ("system", self._get_not_found_system_prompt()),
            ("human", "{input}")
        ])
        
        return templates
    
    def _get_traffic_signal_system_prompt(self) -> str:
        """获取交通信号灯任务的系统提示词"""
        examples = self.few_shot_examples["traffic_signal"]
        
        # 手动构建示例字符串
        example_text = ""
        for i, example in enumerate(examples):
            example_text += f"""
示例{i+1}:
用户问题: {example['user_question']}

良好上下文: {example['good_context']}
期望回答: {example['good_answer']}

不当上下文: {example['bad_context']}
不当回答: {example['bad_answer']}
"""
        
        prompt = f"""你是一个专业的智能交通信号灯竞赛任务助手。请严格基于提供的文档内容回答问题。

【严格规则】：
1. 只能使用文档中明确提到的信息，不得添加任何文档外的内容
2. 如果文档未提及"智能交通信号灯任务"的相关要求，请明确回答"根据现有文档，我无法找到关于智能交通信号灯任务的相关信息"
3. 特别关注包含【主要章节】、【子章节】等标记的内容，这些是重要信息
4. 优先引用包含"#关键:智能交通信号灯"、"#关键:基本要求"等标记的文档片段
5. 引用具体数据时，请确保与文档内容完全一致

【Few-shot示例】：{example_text}

【文档内容】：
{{context}}

【回答要求】：
- 如果文档包含智能交通信号灯任务的相关信息，请详细准确地回答
- 如果文档不包含相关信息，请使用标准回复："根据现有文档，我无法找到关于智能交通信号灯任务的相关信息"
- 保持专业、准确，突出任务的技术要求和实现标准"""
        
        return prompt
    
    def _get_general_competition_system_prompt(self) -> str:
        """获取一般竞赛任务的系统提示词"""
        examples = self.few_shot_examples["general_task"]
        
        # 手动构建示例字符串
        example_text = ""
        if examples:
            example = examples[0]
            example_text = f"""
示例:
用户问题: {example['user_question']}

良好上下文: {example['good_context']}
期望回答: {example['good_answer']}

不当上下文: {example['bad_context']}
不当回答: {example['bad_answer']}
"""
        
        prompt = f"""你是一个专业的竞赛文档分析助手。请严格基于提供的竞赛文档内容回答问题。

【严格规则】：
1. 只能使用文档中明确提到的信息，严禁添加文档外的内容
2. 如果文档未提及相关竞赛要求或标准，请明确回答"根据现有文档，我无法找到相关的竞赛信息"
3. 特别关注章节标题和关键词标记，如【主要章节】、#关键:评分标准等
4. 优先使用包含任务描述、基本要求、技术要求、评分标准等关键信息的文档片段
5. 确保引用的数据和要求与文档完全一致

【Few-shot示例】：{example_text}

【文档内容】：
{{context}}

【回答要求】：
- 基于文档内容准确回答竞赛相关问题
- 如果信息不足，使用标准回复："根据现有文档，我无法找到相关的竞赛信息"
- 保持客观、准确，重点突出竞赛的具体要求和标准"""
        
        return prompt
    
    def _get_technical_requirements_system_prompt(self) -> str:
        """获取技术要求的系统提示词"""
        prompt = """你是一个专业的技术要求分析专家。请严格基于提供的技术文档内容回答问题。

【严格规则】：
1. 只能使用文档中明确提到的技术要求，不得添加常识性技术知识
2. 如果文档未明确说明技术要求，请回答"根据现有文档，我无法找到具体的技术要求信息"
3. 特别关注包含"技术要求"、"算法设计"、"系统架构"等关键词的文档片段
4. 引用技术参数和标准时，确保与文档内容完全一致
5. 避免推测或补充文档中未提及的技术细节

【文档内容】：
{context}

【回答要求】：
- 准确提取和总结文档中的技术要求
- 如果技术要求不明确或缺失，明确说明"根据现有文档，我无法找到具体的技术要求信息"
- 保持技术表述的准确性和专业性"""
        
        return prompt
    
    def _get_not_found_system_prompt(self) -> str:
        """获取未找到信息时的系统提示词"""
        prompt = """你是一个严谨的文档分析助手。当文档中确实没有用户询问的信息时，你需要给出明确的回应。

【严格规则】：
1. 仔细检查文档是否包含用户询问的信息
2. 如果确实没有相关信息，使用以下标准格式回答：
   "根据现有文档，我无法找到关于[具体问题]的相关信息。建议您查阅更详细的文档或联系相关负责人获取准确信息。"
3. 不要猜测、推断或使用常识性知识来补充答案
4. 保持诚实和准确，承认信息的局限性

【文档内容】：
{context}

【回答要求】：
- 诚实地说明文档中缺少相关信息
- 使用标准的"未找到"回复格式
- 建议用户寻找其他信息源"""
        
        return prompt
    
    def detect_question_type(self, question: str) -> str:
        """
        检测问题类型
        
        Args:
            question: 用户问题
            
        Returns:
            问题类型标识
        """
        question_lower = question.lower()
        
        # 检测交通信号灯相关问题
        traffic_keywords = self.task_keywords["交通信号灯"]
        if any(keyword in question for keyword in traffic_keywords):
            logger.debug(f"检测到交通信号灯问题: {question[:50]}...")
            return "traffic_signal_task"
        
        # 检测一般竞赛问题
        competition_keywords = self.task_keywords["竞赛任务"]
        if any(keyword in question for keyword in competition_keywords):
            logger.debug(f"检测到竞赛任务问题: {question[:50]}...")
            return "general_competition"
        
        # 检测技术要求问题
        if "技术" in question or "算法" in question or "设计" in question:
            logger.debug(f"检测到技术要求问题: {question[:50]}...")
            return "technical_requirements"
        
        # 默认使用一般竞赛模板
        logger.debug(f"使用默认竞赛模板: {question[:50]}...")
        return "general_competition"
    
    def check_context_relevance(self, question: str, context: str) -> Tuple[bool, float, List[str]]:
        """
        检查上下文与问题的相关性
        
        Args:
            question: 用户问题
            context: 文档上下文
            
        Returns:
            (是否相关, 相关性分数, 匹配的关键词列表)
        """
        question_lower = question.lower()
        context_lower = context.lower()
        
        # 检测问题类型的关键词
        question_type = self.detect_question_type(question)
        
        matched_keywords = []
        relevance_score = 0.0
        
        # 根据问题类型检查相关性
        if question_type == "traffic_signal_task":
            # 检查交通信号灯相关关键词
            traffic_keywords = ["智能交通信号灯", "交通信号灯", "信号控制", "信号灯", "交通优化"]
            for keyword in traffic_keywords:
                if keyword in context:
                    matched_keywords.append(keyword)
                    relevance_score += 0.3
            
            # 检查任务关键词
            task_keywords = ["基本要求", "技术要求", "任务描述", "算法设计"]
            for keyword in task_keywords:
                if keyword in question and keyword in context:
                    matched_keywords.append(keyword)
                    relevance_score += 0.4
        
        elif question_type == "general_competition":
            # 检查竞赛关键词
            comp_keywords = ["竞赛", "泰迪杯", "专项赛", "评分标准", "基本要求"]
            for keyword in comp_keywords:
                if keyword in context:
                    matched_keywords.append(keyword)
                    relevance_score += 0.25
        
        # 检查章节标记
        if "【主要章节】" in context or "【子章节】" in context:
            relevance_score += 0.2
        
        # 检查关键词标记
        if "#关键:" in context:
            relevance_score += 0.3
        
        # 判断是否相关
        is_relevant = relevance_score >= 0.3 and len(matched_keywords) > 0
        
        logger.debug(f"上下文相关性检查: 分数={relevance_score:.2f}, 相关={is_relevant}, 匹配关键词={matched_keywords}")
        
        return is_relevant, relevance_score, matched_keywords
    
    def get_enhanced_prompt(self, question: str, context: str) -> Tuple[ChatPromptTemplate, Dict]:
        """
        获取增强的提示词
        
        Args:
            question: 用户问题
            context: 文档上下文
            
        Returns:
            (提示词模板, 模板变量)
        """
        # 检测问题类型
        question_type = self.detect_question_type(question)
        
        # 检查上下文相关性
        is_relevant, relevance_score, matched_keywords = self.check_context_relevance(question, context)
        
        # 选择合适的模板
        if not is_relevant or relevance_score < 0.2:
            # 使用未找到信息的模板
            template = self.prompt_templates["not_found_response"]
            variables = {
                "input": question,
                "context": context
            }
            logger.info(f"使用未找到信息模板，相关性分数: {relevance_score:.2f}")
        else:
            # 使用对应类型的模板
            template = self.prompt_templates[question_type]
            variables = {
                "input": question,
                "context": context
            }
            logger.info(f"使用 {question_type} 模板，相关性分数: {relevance_score:.2f}")
        
        return template, variables
    
    def get_task_specific_instructions(self, question: str) -> str:
        """
        获取任务特定的指令
        
        Args:
            question: 用户问题
            
        Returns:
            任务特定指令
        """
        question_type = self.detect_question_type(question)
        
        instructions = {
            "traffic_signal_task": """
特别注意：
1. 如果文档未明确提及"智能交通信号灯任务"的要求，请回答"根据现有文档，我无法找到关于智能交通信号灯任务的相关信息"
2. 优先查找包含"交通信号灯"、"信号控制"等关键词的文档片段
3. 重点关注算法设计、技术参数、实现要求等技术细节
""",
            "general_competition": """
特别注意：
1. 如果文档未明确提及竞赛要求或标准，请回答"根据现有文档，我无法找到相关的竞赛信息"
2. 优先查找包含"评分标准"、"基本要求"等关键词的文档片段
3. 重点关注竞赛规则、评分维度、提交要求等信息
""",
            "technical_requirements": """
特别注意：
1. 如果文档未明确说明技术要求，请回答"根据现有文档，我无法找到具体的技术要求信息"
2. 优先查找包含"技术要求"、"算法设计"等关键词的文档片段
3. 重点关注技术参数、性能指标、实现标准等技术细节
"""
        }
        
        return instructions.get(question_type, "")
    
    def analyze_prompt_effectiveness(self, question: str, context: str, answer: str) -> Dict:
        """
        分析提示词效果
        
        Args:
            question: 用户问题
            context: 文档上下文
            answer: 模型回答
            
        Returns:
            效果分析报告
        """
        try:
            # 检测问题类型
            question_type = self.detect_question_type(question)
            
            # 检查上下文相关性
            is_relevant, relevance_score, matched_keywords = self.check_context_relevance(question, context)
            
            # 分析回答质量
            answer_quality = self._analyze_answer_quality(question, answer, question_type)
            
            # 检查是否正确使用了"未找到"回复
            uses_not_found = "根据现有文档，我无法找到" in answer
            should_use_not_found = not is_relevant or relevance_score < 0.2
            
            correct_not_found_usage = (uses_not_found and should_use_not_found) or (not uses_not_found and not should_use_not_found)
            
            analysis = {
                'question_type': question_type,
                'context_relevance': {
                    'is_relevant': is_relevant,
                    'score': relevance_score,
                    'matched_keywords': matched_keywords
                },
                'answer_analysis': {
                    'uses_not_found_response': uses_not_found,
                    'should_use_not_found': should_use_not_found,
                    'correct_not_found_usage': correct_not_found_usage,
                    'quality_score': answer_quality
                },
                'prompt_effectiveness': {
                    'template_match': question_type,
                    'relevance_detection_accuracy': 1.0 if correct_not_found_usage else 0.0,
                    'overall_score': (relevance_score + answer_quality) / 2
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"提示词效果分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_answer_quality(self, question: str, answer: str, question_type: str) -> float:
        """分析回答质量"""
        quality_score = 0.0
        
        # 检查回答长度
        if len(answer) > 20:
            quality_score += 0.2
        
        # 检查是否包含关键信息
        if question_type == "traffic_signal_task":
            key_terms = ["信号灯", "算法", "控制", "要求"]
        elif question_type == "general_competition":
            key_terms = ["竞赛", "评分", "要求", "标准"]
        else:
            key_terms = ["技术", "要求", "设计", "实现"]
        
        for term in key_terms:
            if term in answer:
                quality_score += 0.2
        
        return min(quality_score, 1.0) 