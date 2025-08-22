"""
修复上下文遗漏的优化配置文件
解决重要信息被过滤的问题
"""

import os

class Config:
    """系统配置类 - 上下文遗漏修复版本"""
    
    # 模型配置
    BGE_MODEL_PATH = "./bge-large-zh-v1.5"
    BGE_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
    VECTOR_DIMENSION = 1024
    
    # CrossEncoder重排模型配置
    CROSSENCODER_MODEL_NAME = './cross-encoder/ms-marco-MiniLM-L6-v2'
    CROSSENCODER_FALLBACK_ENABLED = True
    CROSSENCODER_OFFLINE_MODE = True
    
    # 检索配置
    BM25_K1 = 1.5
    BM25_B = 0.75
    HYBRID_ALPHA = 0.3  # 降低向量权重，增加BM25精确匹配权重
    
    # 检索器配置 - 修复遗漏问题
    RETRIEVAL_CONFIG = {
        # 层级检索配置
        "enable_hierarchical": True,  # 启用层级检索提高精确性
        "hierarchical_similarity_threshold": 0.5,  # 降低阈值，更容易匹配
        "hierarchical_fallback": True,
        
        # 多样性检索配置 - 启用以确保文档来源平衡
        "enable_diversity": True,  
        "diversity_config": {
            "diversity_weight": 0.2,
            "max_docs_per_source": 3,  # 增加每个来源的文档数量
            "reranker_diversity_weight": 0.15
        },
        
        # 文件名感知检索配置
        "enable_filename_aware": True,  # 启用文件名感知检索
    }
    
    # 保持向后兼容
    DIVERSITY_RETRIEVAL_CONFIG = RETRIEVAL_CONFIG["diversity_config"]
    
    # 文件名感知检索配置
    ENABLE_FILENAME_AWARE_RETRIEVAL = True  # 全局启用文件名感知检索
    FILENAME_DETECTION_MIN_CONFIDENCE = 0.6  # 文件名检测的最小置信度
    
    # 文档处理配置 - 关键修复：增加上下文数量
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_CONTEXTS = 8  # 🔥 关键修复：从5增加到8，解决重要信息被过滤问题
    
    # 分数阈值 - 降低阈值，包含更多相关文档
    VECTOR_SCORE_THRESHOLD = 0.4  # 降低阈值
    BM25_SCORE_THRESHOLD = 0.05   # 降低阈值
    HYBRID_SCORE_THRESHOLD = 0.05 # 降低阈值
    
    # 知识库配置
    KNOWLEDGE_BASE_PATH = "knowledge_base"
    INDEX_PATH = "knowledge_base/indexes"
    BACKUP_VERSIONS = 5
    
    # 输出文件
    COMPETITION_INFO_FILE = "result_1.xlsx"
    QA_RESULT_FILE = "result_2.xlsx"
    UPDATE_QA_RESULT_FILE = "result_3.xlsx"
    
    # 日志配置
    LOG_FILE = "chatbot.log"
    LOG_LEVEL = "INFO"
    LOG_ROTATION = "1 MB"
    LOG_RETENTION = "7 days"
    
    # PDF文件模式
    PDF_PATTERNS = [
        "data/*.pdf",
        "0*.pdf", "1*.pdf", "19*.pdf", "20*.pdf", "21*.pdf"
    ]
    
    # 文档监控配置
    WATCH_DIRECTORIES = ["data"]
    WATCH_FILE_PATTERNS = ["*.pdf"]
    WATCH_CHECK_INTERVAL = 30
    WATCH_AUTO_UPDATE = True
    WATCH_MIN_UPDATE_INTERVAL = 60
    WATCH_ENABLE_REALTIME = False
    
    # Web界面配置
    STREAMLIT_THEME = "light"
    PAGE_TITLE = "泰迪杯竞赛智能客服机器人"
    PAGE_ICON = "🤖"
    
    # LLM配置
    USE_LLM = True
    LLM_MODEL = "deepseek-r1:7b"
    OLLAMA_BASE_URL = "http://localhost:11434"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 2000
    LLM_TIMEOUT = 60
    
    # LangChain配置 - 关键修复
    LANGCHAIN_EMBEDDING_MODEL = "./bge-large-zh-v1.5"
    LANGCHAIN_VECTORSTORE_TYPE = "faiss"
    LANGCHAIN_RETRIEVER_K = 15  # 🔥 关键修复：从10增加到15
    LANGCHAIN_CHUNK_SIZE = 800
    LANGCHAIN_CHUNK_OVERLAP = 100
    
    # 检索阶段配置 - 关键修复：增加各阶段文档数量
    RETRIEVAL_STAGES = {
        "stage1_vector_k": 40,      # 🔥 从30增加到40
        "stage1_bm25_k": 40,        # 🔥 从30增加到40  
        "stage2_candidate_k": 60,   # 🔥 从50增加到60
        "final_k": 15,              # 🔥 从10增加到15
        "enable_multi_stage": True,
        "enable_deterministic": True
    }
    
    # 混合检索权重配置 - 关键修复：增加BM25权重用于精确匹配
    LANGCHAIN_HYBRID_ALPHA = 0.3    # 🔥 降低向量权重从0.5到0.3
    LANGCHAIN_BM25_WEIGHT = 0.7     # 🔥 增加BM25权重从0.5到0.7
    LANGCHAIN_VECTOR_WEIGHT = 0.3   # 🔥 对应调整向量权重
    
    # 查询增强配置 - 新增
    QUERY_ENHANCEMENT_CONFIG = {
        "enable_synonym_expansion": True,
        "enable_competition_detection": True,
        "competition_boost_factor": 1.5,  # 竞赛相关文档加权
        "preparation_keywords": [
            "准备工作", "参赛前", "比赛开始前", "检查设备", 
            "登录平台", "下载场景", "15分钟", "竞赛要求"
        ]
    }
    
    # 重排序配置 - 优化
    RERANKING_CONFIG = {
        "enable_reranking": True,
        "preserve_diversity": True,  # 保持多样性，避免过度过滤
        "competition_specific_boost": True,  # 竞赛特定内容加权
        "content_length_penalty": False,  # 取消长度惩罚
        "position_boost_factor": 0.1  # 降低位置影响
    }
    
    # 上下文压缩配置 - 关键修复
    CONTEXT_COMPRESSION_CONFIG = {
        "enable_compression": False,  # 🔥 暂时禁用压缩，避免信息丢失
        "max_compression_ratio": 0.8,
        "preserve_key_information": True,
        "key_phrases": [
            "比赛开始前", "参赛选手", "检查", "登录", "平台", 
            "下载", "场景", "15分钟", "准备工作"
        ]
    }
    
    # 获取PDF文件列表的方法
    @classmethod
    def get_pdf_files(cls):
        """获取PDF文件列表"""
        import glob
        pdf_files = []
        
        for pattern in cls.PDF_PATTERNS:
            files = glob.glob(pattern)
            pdf_files.extend(files)
        
        # 去重并排序
        pdf_files = list(set(pdf_files))
        pdf_files.sort()
        
        return pdf_files
    
    # 上下文遗漏修复方法
    @classmethod
    def apply_context_missing_fixes(cls):
        """应用上下文遗漏修复"""
        
        print("🔧 应用上下文遗漏修复配置...")
        
        # 确保关键参数正确设置
        cls.MAX_CONTEXTS = 8
        cls.LANGCHAIN_RETRIEVER_K = 15
        cls.LANGCHAIN_HYBRID_ALPHA = 0.3
        cls.LANGCHAIN_BM25_WEIGHT = 0.7
        cls.LANGCHAIN_VECTOR_WEIGHT = 0.3
        
        # 优化检索阶段
        cls.RETRIEVAL_STAGES.update({
            "stage1_vector_k": 40,
            "stage1_bm25_k": 40,
            "stage2_candidate_k": 60,
            "final_k": 15
        })
        
        # 启用多样性检索
        cls.RETRIEVAL_CONFIG["enable_diversity"] = True
        cls.RETRIEVAL_CONFIG["diversity_config"]["max_docs_per_source"] = 3
        
        # 降低分数阈值
        cls.VECTOR_SCORE_THRESHOLD = 0.4
        cls.BM25_SCORE_THRESHOLD = 0.05
        cls.HYBRID_SCORE_THRESHOLD = 0.05
        
        print("✅ 上下文遗漏修复配置已应用")
        print(f"  - MAX_CONTEXTS: {cls.MAX_CONTEXTS}")
        print(f"  - RETRIEVER_K: {cls.LANGCHAIN_RETRIEVER_K}")
        print(f"  - BM25权重: {cls.LANGCHAIN_BM25_WEIGHT}")
        print(f"  - 向量权重: {cls.LANGCHAIN_VECTOR_WEIGHT}")
        print(f"  - 最终候选数: {cls.RETRIEVAL_STAGES['final_k']}")
        
        return True
    
    # 集成优化配置
    @classmethod
    def enable_optimized_features(cls):
        """启用优化功能的配置"""
        
        print("🚀 启用优化功能配置...")
        
        # 导入优化配置
        try:
            from optimized_config import OptimizedConfig
            
            # 合并关键优化参数
            cls.USE_OPTIMIZED_RAG = True
            cls.OPTIMIZED_CONFIG = OptimizedConfig()
            
            # 更新检索配置以支持优化
            cls.ENABLE_HYBRID_RETRIEVAL = True
            cls.ENABLE_CROSS_ENCODER_RERANKING = True
            cls.ENABLE_ENTITY_REWARDS = True
            cls.ENABLE_ENHANCED_DOCUMENT_LOADING = True
            cls.ENABLE_FEWSHOT_PROMPTING = True
            
            print("✅ 优化功能配置已启用")
            print("  - 🔄 混合检索：已启用")
            print("  - 🎯 重排序：已启用")
            print("  - 🏆 实体奖励：已启用")
            print("  - 📄 文档增强：已启用")
            print("  - 💡 提示优化：已启用")
            
            return True
            
        except ImportError as e:
            print(f"⚠️ 优化配置导入失败: {e}")
            print("将使用传统RAG配置")
            cls.USE_OPTIMIZED_RAG = False
            return False
    
    # 获取优化状态
    @classmethod
    def is_optimized_enabled(cls):
        """检查优化功能是否启用"""
        return getattr(cls, 'USE_OPTIMIZED_RAG', False)

# 自动应用修复
Config.apply_context_missing_fixes()

# 尝试启用优化功能
Config.enable_optimized_features()
