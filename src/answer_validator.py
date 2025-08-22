"""
答案验证模块
检查答案是否真实来源于上下文，防止模型幻觉
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import re
import jieba
from loguru import logger
from datetime import datetime
import difflib
from collections import Counter

@dataclass
class ValidationResult:
    """验证结果"""
    hallucination: bool
    missing_info: List[str]
    confidence: float
    supported_facts: List[str]
    unsupported_facts: List[str]
    fact_coverage: float
    validation_details: Dict
    # 新增ROUGE/BLEU相关字段
    rouge_score: float = 0.0
    bleu_score: float = 0.0
    text_overlap_ratio: float = 0.0
    is_off_topic: bool = False

class AnswerValidator:
    """答案验证器"""
    
    def __init__(self, 
                 hallucination_threshold: float = 0.2,  # 降低幻觉阈值到0.2，减少对正确答案的误判
                 fact_similarity_threshold: float = 0.6,  # 降低事实相似度阈值，对简短答案更友好
                 # 新增重叠度阈值
                 rouge_threshold: float = 0.3,
                 bleu_threshold: float = 0.2,
                 overlap_threshold: float = 0.25):
        """
        初始化答案验证器
        
        Args:
            hallucination_threshold: 幻觉检测阈值
            fact_similarity_threshold: 事实相似度阈值
            rouge_threshold: ROUGE分数阈值，低于此值认为可能跑题
            bleu_threshold: BLEU分数阈值，低于此值认为可能跑题
            overlap_threshold: 文本重叠度阈值，低于此值认为可能跑题
        """
        self.hallucination_threshold = hallucination_threshold
        self.fact_similarity_threshold = fact_similarity_threshold
        self.rouge_threshold = rouge_threshold
        self.bleu_threshold = bleu_threshold
        self.overlap_threshold = overlap_threshold
        
        # 事实提取模式
        self.fact_patterns = {
            "时间": [
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{4}年\d{1,2}月',
                r'\d{1,2}月\d{1,2}日',
                r'\d{4}-\d{1,2}-\d{1,2}',
                r'截止时间.*?\d',
                r'开始时间.*?\d',
                r'结束时间.*?\d'
            ],
            "数量": [
                r'\d+个',
                r'\d+项',
                r'\d+类',
                r'\d+种',
                r'共\d+',
                r'总计\d+',
                r'总共\d+'
            ],
            "联系方式": [
                r'\d{3,4}-?\d{7,8}',  # 电话
                r'\w+@\w+\.\w+',      # 邮箱
                r'QQ[:：]\s*\d+',     # QQ
                r'微信[:：]\s*\w+'     # 微信
            ],
            "地点": [
                r'\w+省\w+市',
                r'\w+市\w+区',
                r'\w+大学',
                r'\w+学院',
                r'\w+中心'
            ],
            "名称": [
                r'[《""]([^》""]{2,20})[》""]',  # 引号内容
                r'专项赛',
                r'竞赛',
                r'大赛',
                r'挑战赛'
            ]
        }
        
        # 否定词汇
        self.negation_words = {
            '不', '非', '无', '没有', '缺少', '缺乏', '未', '否', '禁止', '拒绝'
        }
        
        logger.info("答案验证器初始化完成")
    
    def calculate_rouge_score(self, answer: str, context: str) -> float:
        """
        计算ROUGE分数（基于1-gram重叠）
        
        Args:
            answer: 生成的答案
            context: 源文档上下文
            
        Returns:
            ROUGE-1分数
        """
        try:
            # 分词
            answer_tokens = set(jieba.lcut(answer.lower()))
            context_tokens = set(jieba.lcut(context.lower()))
            
            # 移除停用词
            stopwords = {'的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到', '说', '要', '去', '会', '着', '看', '好', '这', '那', '与', '或', '但'}
            answer_tokens = answer_tokens - stopwords
            context_tokens = context_tokens - stopwords
            
            if not answer_tokens:
                return 0.0
            
            # 计算重叠
            overlap = answer_tokens.intersection(context_tokens)
            rouge_score = len(overlap) / len(answer_tokens)
            
            return rouge_score
            
        except Exception as e:
            logger.warning(f"ROUGE计算失败: {e}")
            return 0.0
    
    def calculate_bleu_score(self, answer: str, context: str) -> float:
        """
        计算简化的BLEU分数（基于2-gram重叠）
        
        Args:
            answer: 生成的答案
            context: 源文档上下文
            
        Returns:
            简化BLEU分数
        """
        try:
            # 分词
            answer_tokens = jieba.lcut(answer.lower())
            context_tokens = jieba.lcut(context.lower())
            
            # 移除停用词
            stopwords = {'的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到', '说', '要', '去', '会', '着', '看', '好', '这', '那', '与', '或', '但'}
            answer_tokens = [token for token in answer_tokens if token not in stopwords]
            context_tokens = [token for token in context_tokens if token not in stopwords]
            
            if len(answer_tokens) < 2:
                return 0.0
            
            # 生成2-gram
            answer_bigrams = set()
            for i in range(len(answer_tokens) - 1):
                answer_bigrams.add((answer_tokens[i], answer_tokens[i + 1]))
            
            context_bigrams = set()
            for i in range(len(context_tokens) - 1):
                context_bigrams.add((context_tokens[i], context_tokens[i + 1]))
            
            if not answer_bigrams:
                return 0.0
            
            # 计算重叠
            overlap = answer_bigrams.intersection(context_bigrams)
            bleu_score = len(overlap) / len(answer_bigrams)
            
            return bleu_score
            
        except Exception as e:
            logger.warning(f"BLEU计算失败: {e}")
            return 0.0
    
    def calculate_text_overlap_ratio(self, answer: str, context: str) -> float:
        """
        计算文本重叠度比例
        
        Args:
            answer: 生成的答案
            context: 源文档上下文
            
        Returns:
            文本重叠度比例
        """
        try:
            # 使用字符级别的重叠计算
            answer_chars = set(answer.lower().replace(' ', '').replace('\n', ''))
            context_chars = set(context.lower().replace(' ', '').replace('\n', ''))
            
            if not answer_chars:
                return 0.0
            
            overlap = answer_chars.intersection(context_chars)
            overlap_ratio = len(overlap) / len(answer_chars)
            
            return overlap_ratio
            
        except Exception as e:
            logger.warning(f"文本重叠度计算失败: {e}")
            return 0.0

    def validate_answer(self, question: str, answer: str, context: str) -> ValidationResult:
        """
        检查答案是否真实来源于上下文
        
        Args:
            question: 用户问题
            answer: LLM生成的答案
            context: 检索到的上下文
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.info(f"开始验证答案，问题: {question[:50]}...")
        
        # 1. 计算ROUGE/BLEU分数
        rouge_score = self.calculate_rouge_score(answer, context)
        bleu_score = self.calculate_bleu_score(answer, context)
        text_overlap_ratio = self.calculate_text_overlap_ratio(answer, context)
        
        logger.debug(f"ROUGE分数: {rouge_score:.3f}, BLEU分数: {bleu_score:.3f}, 重叠度: {text_overlap_ratio:.3f}")
        
        # 2. 基于重叠度判断是否跑题
        is_off_topic = (rouge_score < self.rouge_threshold and 
                       bleu_score < self.bleu_threshold and 
                       text_overlap_ratio < self.overlap_threshold)
        
        # 3. 提取答案中的关键事实（保留原有逻辑作为补充）
        answer_facts = self._extract_facts(answer)
        context_facts = self._extract_facts(context)
        
        # 4. 检查事实支持度
        supported_facts, unsupported_facts = self._check_fact_support(
            answer_facts, context_facts
        )
        
        # 5. 检查是否有缺失信息
        missing_info = self._check_missing_info(question, answer, context)
        
        # 6. 计算综合置信度（结合重叠度和事实支持度）
        confidence = self._calculate_comprehensive_confidence(
            rouge_score, bleu_score, text_overlap_ratio,
            answer_facts, supported_facts, unsupported_facts
        )
        
        # 7. 判断是否有幻觉（主要基于重叠度）
        hallucination = is_off_topic or confidence < self.hallucination_threshold
        
        # 8. 计算事实覆盖率
        fact_coverage = len(supported_facts) / len(answer_facts) if answer_facts else 1.0
        
        # 9. 构建验证详情
        validation_details = self._build_validation_details(
            answer, context, answer_facts, context_facts, 
            supported_facts, unsupported_facts, rouge_score, bleu_score, text_overlap_ratio
        )
        
        result = ValidationResult(
            hallucination=hallucination,
            missing_info=missing_info,
            confidence=confidence,
            supported_facts=supported_facts,
            unsupported_facts=unsupported_facts,
            fact_coverage=fact_coverage,
            validation_details=validation_details,
            rouge_score=rouge_score,
            bleu_score=bleu_score,
            text_overlap_ratio=text_overlap_ratio,
            is_off_topic=is_off_topic
        )
        
        logger.info(f"验证完成，置信度: {confidence:.3f}, 跑题: {is_off_topic}, 幻觉: {hallucination}")
        return result
    
    def _extract_facts(self, text: str) -> List[str]:
        """提取文本中的关键事实"""
        facts = []
        
        # 使用正则模式提取结构化事实
        for fact_type, patterns in self.fact_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    if match and match not in facts:
                        facts.append(match.strip())
        
        # 提取关键句子（包含关键信息的完整句子）
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if self._is_factual_sentence(sentence):
                facts.append(sentence.strip())
        
        return facts
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        # 使用标点符号分句
        sentences = re.split(r'[。！？\n]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def _is_factual_sentence(self, sentence: str) -> bool:
        """判断是否为事实性句子"""
        # 包含具体信息的句子更可能是事实
        factual_indicators = [
            r'\d+',  # 包含数字
            r'[《""].*?[》""]',  # 包含引用
            r'时间|日期|地点|电话|邮箱',  # 包含关键信息词
            r'专项赛|竞赛|大赛|比赛',  # 包含竞赛相关词
            r'要求|条件|标准|规定'  # 包含规则相关词
        ]
        
        for indicator in factual_indicators:
            if re.search(indicator, sentence):
                return True
        
        return False
    
    def _check_fact_support(self, answer_facts: List[str], 
                           context_facts: List[str]) -> Tuple[List[str], List[str]]:
        """检查事实支持度"""
        supported_facts = []
        unsupported_facts = []
        
        for answer_fact in answer_facts:
            is_supported = False
            
            # 1. 精确匹配
            if answer_fact in context_facts:
                is_supported = True
            else:
                # 2. 模糊匹配
                for context_fact in context_facts:
                    similarity = self._calculate_text_similarity(answer_fact, context_fact)
                    if similarity >= self.fact_similarity_threshold:
                        is_supported = True
                        break
                
                # 3. 部分匹配（对于复合事实）
                if not is_supported:
                    is_supported = self._check_partial_support(answer_fact, context_facts)
            
            if is_supported:
                supported_facts.append(answer_fact)
            else:
                unsupported_facts.append(answer_fact)
        
        return supported_facts, unsupported_facts
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 使用difflib计算序列相似度
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return similarity
    
    def _check_partial_support(self, fact: str, context_facts: List[str]) -> bool:
        """检查部分支持（对于复合事实）"""
        # 将事实分解为关键词
        fact_keywords = self._extract_keywords(fact)
        
        for context_fact in context_facts:
            context_keywords = self._extract_keywords(context_fact)
            
            # 如果关键词有足够的重叠，认为是部分支持
            overlap = set(fact_keywords).intersection(set(context_keywords))
            if len(overlap) >= len(fact_keywords) * 0.6:  # 60%的关键词重叠
                return True
        
        return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 使用jieba分词
        words = jieba.lcut(text)
        
        # 过滤停用词
        stopwords = {'的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到', '说', '要', '去', '会', '着', '看', '好', '这', '那'}
        keywords = [word for word in words if len(word) > 1 and word not in stopwords]
        
        return keywords
    
    def _check_missing_info(self, question: str, answer: str, context: str) -> List[str]:
        """检查缺失信息"""
        missing_info = []
        
        # 分析问题类型，检查是否回答完整
        question_type = self._classify_question_type(question)
        
        if question_type == "时间查询":
            # 检查是否提供了时间信息
            if not re.search(r'\d{4}年|\d{1,2}月|\d{1,2}日|时间', answer):
                if re.search(r'\d{4}年|\d{1,2}月|\d{1,2}日|时间', context):
                    missing_info.append("时间信息")
        
        elif question_type == "统计查询":
            # 检查是否提供了数量信息
            if not re.search(r'\d+个|\d+项|\d+', answer):
                if re.search(r'\d+个|\d+项|\d+', context):
                    missing_info.append("数量统计")
        
        elif question_type == "联系查询":
            # 检查是否提供了联系方式
            contact_patterns = [r'\d{3,4}-?\d{7,8}', r'\w+@\w+\.\w+', r'QQ', r'微信']
            answer_has_contact = any(re.search(pattern, answer) for pattern in contact_patterns)
            context_has_contact = any(re.search(pattern, context) for pattern in contact_patterns)
            
            if not answer_has_contact and context_has_contact:
                missing_info.append("联系方式")
        
        return missing_info
    
    def _classify_question_type(self, question: str) -> str:
        """分类问题类型"""
        if re.search(r'什么时候|何时|时间|日期', question):
            return "时间查询"
        elif re.search(r'多少|几个|数量', question):
            return "统计查询"
        elif re.search(r'联系|电话|邮箱', question):
            return "联系查询"
        elif re.search(r'什么是|是什么|定义', question):
            return "定义查询"
        else:
            return "通用查询"
    
    def _calculate_comprehensive_confidence(self, rouge_score: float, bleu_score: float, 
                                         text_overlap_ratio: float, answer_facts: List[str], 
                                         supported_facts: List[str], unsupported_facts: List[str]) -> float:
        """
        计算综合置信度（结合重叠度和事实支持度）
        
        Args:
            rouge_score: ROUGE分数
            bleu_score: BLEU分数
            text_overlap_ratio: 文本重叠度
            answer_facts: 答案事实
            supported_facts: 支持的事实
            unsupported_facts: 不支持的事实
            
        Returns:
            综合置信度
        """
        # 检查是否为日期类答案（包含明确的日期信息）
        answer_text = " ".join(answer_facts) if answer_facts else ""
        is_date_answer = self._is_date_answer(answer_text)
        
        # 🚀 优化重叠度计算：给文本重叠度更高权重，降低BLEU权重
        # 文本重叠度通常比BLEU更准确，特别是对中文
        weighted_overlap = (rouge_score * 0.3 + bleu_score * 0.2 + text_overlap_ratio * 0.5)
        
        # 🚀 针对日期类答案的特殊处理
        if is_date_answer:
            # 日期类答案：大幅提高文本重叠度权重，降低事实权重
            if text_overlap_ratio >= 0.5:  # 降低阈值
                overlap_score = weighted_overlap * 0.9  # 提高权重
                fact_weight = 0.1  # 大幅降低事实权重
            else:
                overlap_score = weighted_overlap * 0.8
                fact_weight = 0.2
        else:
            # 非日期类答案：使用原有逻辑
            if text_overlap_ratio >= 0.8:
                overlap_score = weighted_overlap * 0.85
                fact_weight = 0.15
            elif weighted_overlap >= 0.6:
                overlap_score = weighted_overlap * 0.8
                fact_weight = 0.2
            else:
                overlap_score = weighted_overlap * 0.65
                fact_weight = 0.35
        
        # 事实支持度得分
        if answer_facts:
            fact_support_ratio = len(supported_facts) / len(answer_facts)
            fact_support_score = fact_support_ratio * fact_weight
        else:
            fact_support_score = fact_weight  # 没有事实时给予满分
        
        # 综合置信度
        base_confidence = overlap_score + fact_support_score
        
        # 🚀 针对日期类答案的惩罚优化
        if is_date_answer:
            # 日期类答案：大幅减少惩罚
            if text_overlap_ratio >= 0.5:
                unsupported_penalty = len(unsupported_facts) * 0.01  # 极小惩罚
            else:
                unsupported_penalty = len(unsupported_facts) * 0.02
        else:
            # 非日期类答案：原有惩罚逻辑
            if text_overlap_ratio >= 0.8:
                unsupported_penalty = len(unsupported_facts) * 0.02
            elif weighted_overlap >= 0.6:
                unsupported_penalty = len(unsupported_facts) * 0.03
            else:
                unsupported_penalty = len(unsupported_facts) * 0.05
            
        confidence = max(0.0, base_confidence - unsupported_penalty)
        
        # 🚀 优化：对于高重叠度给予奖励
        if is_date_answer and text_overlap_ratio >= 0.5:
            confidence = min(1.0, confidence * 1.2)  # 日期类答案20%奖励
        elif text_overlap_ratio >= 0.8:
            confidence = min(1.0, confidence * 1.15)  # 15%奖励
        elif weighted_overlap >= 0.7:
            confidence = min(1.0, confidence * 1.1)   # 10%奖励
        
        return min(1.0, confidence)
    
    def _is_date_answer(self, text: str) -> bool:
        """判断是否为日期类答案"""
        import re
        # 匹配常见的日期格式
        date_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日',  # 2024年4月15日
            r'\d{1,2}月\d{1,2}日',         # 4月15日
            r'\d{4}-\d{1,2}-\d{1,2}',      # 2024-04-15
            r'\d{1,2}/\d{1,2}/\d{4}',      # 04/15/2024
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _build_validation_details(self, answer: str, context: str,
                                answer_facts: List[str], context_facts: List[str],
                                supported_facts: List[str], unsupported_facts: List[str],
                                rouge_score: float, bleu_score: float, text_overlap_ratio: float) -> Dict:
        """构建验证详情"""
        return {
            "answer_length": len(answer),
            "context_length": len(context),
            "total_answer_facts": len(answer_facts),
            "total_context_facts": len(context_facts),
            "supported_count": len(supported_facts),
            "unsupported_count": len(unsupported_facts),
            "validation_timestamp": datetime.now().isoformat(),
            "fact_examples": {
                "supported": supported_facts[:3],  # 前3个支持的事实
                "unsupported": unsupported_facts[:3]  # 前3个不支持的事实
            },
            "rouge_score": rouge_score,
            "bleu_score": bleu_score,
            "text_overlap_ratio": text_overlap_ratio,
            "overlap_thresholds": {
                "rouge_threshold": self.rouge_threshold,
                "bleu_threshold": self.bleu_threshold,
                "overlap_threshold": self.overlap_threshold
            }
        }
    
    def batch_validate(self, questions_answers_contexts: List[Tuple[str, str, str]]) -> List[ValidationResult]:
        """批量验证"""
        results = []
        for question, answer, context in questions_answers_contexts:
            result = self.validate_answer(question, answer, context)
            results.append(result)
        
        return results
    
    def get_validation_stats(self, results: List[ValidationResult]) -> Dict:
        """获取验证统计信息"""
        if not results:
            return {}
        
        hallucination_count = sum(1 for r in results if r.hallucination)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        avg_fact_coverage = sum(r.fact_coverage for r in results) / len(results)
        
        return {
            "total_validations": len(results),
            "hallucination_count": hallucination_count,
            "hallucination_rate": hallucination_count / len(results),
            "avg_confidence": avg_confidence,
            "avg_fact_coverage": avg_fact_coverage,
            "high_confidence_count": sum(1 for r in results if r.confidence >= 0.8),
            "low_confidence_count": sum(1 for r in results if r.confidence < 0.5)
        }

class EnhancedAnswerValidator(AnswerValidator):
    """增强的答案验证器"""
    
    def __init__(self, **kwargs):
        """初始化增强验证器"""
        super().__init__(**kwargs)
        
        # 添加更多验证规则
        self.contradiction_patterns = [
            (r'不是', r'是'),
            (r'没有', r'有'),
            (r'不能', r'能'),
            (r'不允许', r'允许'),
            (r'禁止', r'可以')
        ]
        
        # 时间一致性检查
        self.time_patterns = [
            r'\d{4}年',
            r'\d{1,2}月',
            r'\d{1,2}日'
        ]
    
    def validate_answer(self, question: str, answer: str, context: str) -> ValidationResult:
        """增强的答案验证"""
        # 调用基础验证
        result = super().validate_answer(question, answer, context)
        
        # 添加额外验证
        contradiction_check = self._check_contradictions(answer, context)
        consistency_check = self._check_time_consistency(answer, context)
        completeness_check = self._check_answer_completeness(question, answer)
        
        # 更新验证详情
        result.validation_details.update({
            "contradiction_detected": contradiction_check,
            "time_consistency": consistency_check,
            "answer_completeness": completeness_check
        })
        
        # 如果检测到矛盾或不一致，降低置信度
        if contradiction_check or not consistency_check or not completeness_check:
            result.confidence *= 0.8
            result.hallucination = result.confidence < self.hallucination_threshold or result.is_off_topic
        
        return result
    
    def _check_contradictions(self, answer: str, context: str) -> bool:
        """检查答案与上下文的矛盾"""
        for neg_pattern, pos_pattern in self.contradiction_patterns:
            # 检查答案中的否定表述是否与上下文中的肯定表述矛盾
            if re.search(neg_pattern, answer) and re.search(pos_pattern, context):
                # 进一步检查是否真的矛盾（同一主题）
                answer_words = set(jieba.lcut(answer))
                context_words = set(jieba.lcut(context))
                overlap = answer_words.intersection(context_words)
                
                if len(overlap) > 3:  # 如果有足够的词汇重叠，可能存在矛盾
                    return True
        
        return False
    
    def _check_time_consistency(self, answer: str, context: str) -> bool:
        """检查时间一致性"""
        answer_times = []
        context_times = []
        
        # 提取时间信息
        for pattern in self.time_patterns:
            answer_times.extend(re.findall(pattern, answer))
            context_times.extend(re.findall(pattern, context))
        
        # 检查答案中的时间是否在上下文中出现
        for answer_time in answer_times:
            if answer_time not in context_times:
                return False
        
        return True
    
    def _check_answer_completeness(self, question: str, answer: str) -> bool:
        """检查答案完整性"""
        # 检查答案是否过短
        if len(answer) < 10:
            return False
        
        # 检查是否包含"不知道"、"无法回答"等词汇
        uncertainty_phrases = ['不知道', '无法回答', '不清楚', '没有信息', '无法确定']
        if any(phrase in answer for phrase in uncertainty_phrases):
            return False
        
        # 检查问题的关键词是否在答案中出现
        question_keywords = self._extract_keywords(question)
        answer_keywords = self._extract_keywords(answer)
        
        overlap = set(question_keywords).intersection(set(answer_keywords))
        if len(overlap) < len(question_keywords) * 0.3:  # 至少30%的关键词重叠
            return False
        
        return True 