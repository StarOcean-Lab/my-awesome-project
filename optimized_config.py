"""
优化RAG系统配置文件
集成所有5个优化功能的配置参数
"""

import os

class OptimizedConfig:
    """优化RAG系统配置类"""
    
    # ==================== 基础配置 ====================
    
    # 模型配置
    LLM_MODEL = "deepseek-r1:7b"
    EMBEDDING_MODEL = "./bge-large-zh-v1.5"
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # 向量存储配置
    VECTORSTORE_PATH = "./vectorstore"
    ENABLE_VERSIONING = True
    
    # ==================== 优化1：关键词+向量混合检索 ====================
    
    # 混合检索权重配置（优化：降低向量权重，提高BM25权重）
    VECTOR_WEIGHT = 0.4  # 向量检索权重
    BM25_WEIGHT = 0.6    # BM25检索权重（提高以支持精确匹配）
    
    # RRF融合配置
    RRF_K = 60  # RRF参数，控制排名平滑程度
    
    # BM25强制召回配置
    ENABLE_FORCE_RECALL = True
    FORCE_RECALL_KEYWORDS = {
        "未来校园智能应用专项赛",
        "智能交通信号灯", 
        "基本要求",
        "任务描述",
        "技术要求",
        "评分标准",
        "竞赛通知"
    }
    
    # 精确短语匹配配置
    ENABLE_EXACT_PHRASE = True
    IMPORTANT_PHRASES = [
        "未来校园智能应用专项赛", 
        "智能交通信号灯", 
        "基本要求"
    ]
    
    # ==================== 优化2：Cross-Encoder重排序 ====================
    
    # 重排序配置
    ENABLE_RERANKING = True
    CROSSENCODER_MODEL = './cross-encoder/ms-marco-MiniLM-L6-v2'
    CROSSENCODER_FALLBACK_ENABLED = True
    
    # 重排序权重配置
    CROSSENCODER_WEIGHT = 0.5
    ORIGINAL_SCORE_WEIGHT = 0.2
    
    # ==================== 优化3：实体命中奖励 ====================
    
    # 实体奖励配置
    ENABLE_ENTITY_BONUS = True
    ENTITY_BONUS_WEIGHT = 0.3
    TASK_RELEVANCE_WEIGHT = 0.2
    
    # 重要实体词典
    IMPORTANT_ENTITIES = {
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
    TASK_INDICATORS = {
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
    
    # ==================== 优化4：文档切分+标题增强 ====================
    
    # 文档切分配置
    ENABLE_CHAPTER_SPLITTING = True
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    
    # 章节检测模式
    SECTION_PATTERNS = [
        r'^[一二三四五六七八九十][\s]*[、.．][\s]*(.+)$',  # 一、二、等
        r'^[0-9]+[\s]*[、.．][\s]*(.+)$',                    # 1. 2. 等
        r'^[（(][一二三四五六七八九十][）)][\s]*(.+)$',         # （一）等
        r'^[（(][0-9]+[）)][\s]*(.+)$',                      # （1）等
        r'^第[一二三四五六七八九十]+[章节部分][\s]*(.+)$',      # 第一章等
        r'^[A-Z][\s]*[、.．][\s]*(.+)$',                    # A. B. 等
        r'^\d+\.\d+[\s]*(.+)$',                            # 1.1 1.2 等
        r'^【(.+)】$',                                     # 【标题】
        r'^(?:任务描述|基本要求|技术要求|评分标准|实现方案)[:：]?(.*)$',  # 特定任务关键词
    ]
    
    # 标题增强关键词
    TITLE_ENHANCEMENT_KEYWORDS = {
        "未来校园智能应用专项赛",
        "智能交通信号灯",
        "基本要求",
        "技术要求", 
        "任务描述",
        "评分标准",
        "实现方案",
        "设计要求",
        "算法设计",
        "系统架构"
    }
    
    # ==================== 优化5：Few-shot重提示优化 ====================
    
    # 提示词优化配置
    ENABLE_FEWSHOT_PROMPTS = True
    
    # 任务关键词检测
    TASK_KEYWORDS = {
        "交通信号灯": ["智能交通信号灯", "交通信号灯", "信号控制", "信号灯"],
        "未来校园": ["未来校园", "智能校园", "校园应用"],
        "竞赛任务": ["任务描述", "基本要求", "技术要求", "评分标准", "实现方案"],
        "算法设计": ["算法", "设计", "优化", "控制"],
        "系统实现": ["系统", "实现", "架构", "方案"]
    }
    
    # 上下文相关性阈值
    CONTEXT_RELEVANCE_THRESHOLD = 0.3
    
    # 未找到信息的标准回复模板
    NOT_FOUND_RESPONSE_TEMPLATE = "根据现有文档，我无法找到关于{question_topic}的相关信息。建议您查阅更详细的文档或联系相关负责人获取准确信息。"
    
    # ==================== 检索配置 ====================
    
    # 检索数量配置
    RETRIEVAL_K = 10  # 最终返回文档数量
    STAGE1_VECTOR_K = 20   # 第一阶段向量检索数量
    STAGE1_BM25_K = 30     # 第一阶段BM25检索数量
    STAGE2_CANDIDATE_K = 40 # 第二阶段候选数量
    
    # 多阶段检索配置
    RETRIEVAL_STAGES = {
        'enable_multi_stage': True,
        'stage1_vector_k': STAGE1_VECTOR_K,
        'stage1_bm25_k': STAGE1_BM25_K,
        'stage2_candidate_k': STAGE2_CANDIDATE_K,
        'final_k': RETRIEVAL_K
    }
    
    # ==================== 性能配置 ====================
    
    # 语言模型配置
    LLM_TEMPERATURE = 0.1
    LLM_TIMEOUT = 300
    
    # 批处理配置
    BATCH_SIZE = 5
    MAX_CONCURRENT_REQUESTS = 3
    
    # 缓存配置
    ENABLE_CACHE = True
    CACHE_SIZE = 1000
    
    # ==================== 日志配置 ====================
    
    # 日志级别
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/optimized_rag.log"
    
    # 详细调试配置
    ENABLE_DETAILED_LOGGING = True
    LOG_RETRIEVAL_STATS = True
    LOG_RERANK_STATS = True
    LOG_PROMPT_ANALYSIS = True
    
    # ==================== 输出配置 ====================
    
    # 输出目录
    OUTPUT_DIR = "outputs"
    REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
    ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
    
    # 报告生成配置
    SAVE_DETAILED_REPORTS = True
    SAVE_PERFORMANCE_METRICS = True
    
    # ==================== 验证配置 ====================
    
    # 测试问题（用于系统验证）
    TEST_QUESTIONS = [
        "智能交通信号灯的基本要求是什么？",
        "交通信号灯系统需要实现哪些技术要求？", 
        "智能交通信号灯的优化算法如何设计？",
        "未来校园智能应用专项赛的评分标准是什么？",
        "竞赛任务描述中包含哪些关键要求？",
        "泰迪杯数据挖掘挑战赛的基本要求有哪些？",
        "系统架构设计有什么技术要求？",
        "算法实现需要满足哪些性能指标？"
    ]
    
    # 验证阈值
    VALIDATION_THRESHOLDS = {
        'overall_score_threshold': 0.8,
        'response_time_threshold': 10.0,  # 秒
        'retrieval_accuracy_threshold': 0.7,
        'rerank_effectiveness_threshold': 0.6,
        'prompt_effectiveness_threshold': 0.7
    }
    
    @classmethod
    def get_optimized_retrieval_config(cls) -> dict:
        """获取优化检索配置"""
        return {
            'vector_weight': cls.VECTOR_WEIGHT,
            'bm25_weight': cls.BM25_WEIGHT,
            'rrf_k': cls.RRF_K,
            'enable_force_recall': cls.ENABLE_FORCE_RECALL,
            'enable_exact_phrase': cls.ENABLE_EXACT_PHRASE,
            'retrieval_k': cls.RETRIEVAL_K
        }
    
    @classmethod
    def get_reranking_config(cls) -> dict:
        """获取重排序配置"""
        return {
            'enable_reranking': cls.ENABLE_RERANKING,
            'model_name': cls.CROSSENCODER_MODEL,
            'entity_bonus_weight': cls.ENTITY_BONUS_WEIGHT,
            'task_relevance_weight': cls.TASK_RELEVANCE_WEIGHT,
            'important_entities': cls.IMPORTANT_ENTITIES,
            'task_indicators': cls.TASK_INDICATORS
        }
    
    @classmethod
    def get_document_processing_config(cls) -> dict:
        """获取文档处理配置"""
        return {
            'enable_chapter_splitting': cls.ENABLE_CHAPTER_SPLITTING,
            'chunk_size': cls.CHUNK_SIZE,
            'chunk_overlap': cls.CHUNK_OVERLAP,
            'section_patterns': cls.SECTION_PATTERNS,
            'title_enhancement_keywords': cls.TITLE_ENHANCEMENT_KEYWORDS
        }
    
    @classmethod
    def get_prompt_optimization_config(cls) -> dict:
        """获取提示词优化配置"""
        return {
            'enable_fewshot_prompts': cls.ENABLE_FEWSHOT_PROMPTS,
            'task_keywords': cls.TASK_KEYWORDS,
            'context_relevance_threshold': cls.CONTEXT_RELEVANCE_THRESHOLD,
            'not_found_template': cls.NOT_FOUND_RESPONSE_TEMPLATE
        }
    
    @classmethod
    def validate_config(cls) -> dict:
        """验证配置的完整性"""
        validation_results = {
            'config_valid': True,
            'missing_configs': [],
            'warnings': []
        }
        
        # 检查必需的配置
        required_configs = [
            'LLM_MODEL', 'EMBEDDING_MODEL', 'VECTORSTORE_PATH',
            'VECTOR_WEIGHT', 'BM25_WEIGHT', 'RETRIEVAL_K'
        ]
        
        for config_name in required_configs:
            if not hasattr(cls, config_name):
                validation_results['missing_configs'].append(config_name)
                validation_results['config_valid'] = False
        
        # 检查权重配置
        total_weight = cls.VECTOR_WEIGHT + cls.BM25_WEIGHT
        if abs(total_weight - 1.0) > 0.01:
            validation_results['warnings'].append(f"向量和BM25权重总和不等于1.0: {total_weight}")
        
        # 检查模型路径
        if not os.path.exists(cls.EMBEDDING_MODEL):
            validation_results['warnings'].append(f"嵌入模型路径不存在: {cls.EMBEDDING_MODEL}")
        
        if not os.path.exists(cls.CROSSENCODER_MODEL):
            validation_results['warnings'].append(f"CrossEncoder模型路径不存在: {cls.CROSSENCODER_MODEL}")
        
        return validation_results 

# BGE模型预热配置（解决内存暴涨问题）
BGE_WARMUP_ENABLED = True
BGE_WARMUP_TEXTS = [
    "这是一个预热测试文本",
    "BGE模型初始化测试", 
    "内存分配预热文本",
    "避免中途内存暴涨"
]

# 向量化批处理配置
VECTORIZATION_BATCH_SIZE = 5  # 减小批次大小
DOCUMENT_BATCH_SIZE = 2       # 减小文档批次大小
MEMORY_CLEANUP_THRESHOLD = 2000  # 2GB内存清理阈值
