#!/usr/bin/env python3
"""
基于LangChain+Ollama的智能客服机器人
专为泰迪杯竞赛问答设计
"""

import os
import sys
import glob
import time
import streamlit as st
import pandas as pd
import requests
from typing import List, Dict, Optional, Callable
from loguru import logger
from datetime import datetime
from pathlib import Path
from io import BytesIO

# 添加src目录到路径
sys.path.append('src')

# 导入LangChain组件
from src.langchain_rag import LangChainRAGSystem, RAGResponse
from src.langchain_vectorstore import LangChainVectorStore
from src.langchain_document_loader import LangChainDocumentLoader
from src.langchain_retriever import LangChainHybridRetriever

# 导入优化后的RAG系统
from src.optimized_rag_system import OptimizedRAGSystem

# 导入配置
from config import Config
from optimized_config import OptimizedConfig


def get_ollama_models() -> List[str]:
    """获取Ollama可用模型列表"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            # 过滤出问答模型（排除embedding模型）
            qa_models = []
            for model in models:
                model_name = model['name']
                # 过滤掉embedding模型
                if not any(embed_keyword in model_name.lower() for embed_keyword in 
                          ['embed', 'embedding', 'sentence', 'bge', 'e5']):
                    qa_models.append(model_name)
            return qa_models
        else:
            logger.warning(f"无法获取Ollama模型列表: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取Ollama模型列表失败: {e}")
        return []


def set_page_style():
    """设置页面样式"""
    st.markdown("""
    <style>
    /* 导入Google字体 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 主题色彩 - 简约风格 */
    :root {
        --primary-color: #4A90E2;
        --secondary-color: #6B7280;
        --accent-color: #10B981;
        --background-color: #F8FAFC;
        --surface-color: #FFFFFF;
        --text-primary: #1F2937;
        --text-secondary: #6B7280;
        --border-color: #E5E7EB;
        --hover-color: #F3F4F6;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
    }
    
    /* 基础样式重置 */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {display: none;}
    footer {display: none;}
    
    /* 主容器样式 */
    .main > div {
        padding: 2rem 3rem;
        max-width: 1200px;
        margin: 0 auto;
        background-color: var(--background-color);
        min-height: 100vh;
    }
    
    /* 侧边栏样式 - 恢复可伸缩功能 */
    .stSidebar {
        background-color: var(--surface-color);
        border-right: 1px solid var(--border-color);
    }
    
    .stSidebar > div {
        padding: 1.5rem 1rem;
        background-color: var(--surface-color);
    }
    
    /* 侧边栏内容样式 */
    .stSidebar .stSelectbox label {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    
    .stSidebar .stSelectbox > div > div {
        background-color: var(--surface-color);
        border: 2px solid var(--border-color);
        border-radius: 0.5rem;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    
    .stSidebar .stSelectbox > div > div:hover {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
    }
    
    .stSidebar .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), #357ABD);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        width: 100%;
        margin: 0.25rem 0;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.2);
    }
    
    .stSidebar .stButton > button:hover {
        background: linear-gradient(135deg, #357ABD, #2563EB);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
    }
    
    .stSidebar .stMetric {
        background: linear-gradient(135deg, #F8FAFC, #E5E7EB);
        border: 1px solid var(--border-color);
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stSidebar .stMetric label {
        color: var(--text-secondary);
        font-size: 0.6rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .stSidebar .stMetric [data-testid="metric-value"] {
        color: var(--primary-color);
        font-size: 0.6rem;
        font-weight: 700;
    }
    
    /* 侧边栏子标题样式 */
    .stSidebar h3 {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* 侧边栏状态指示器 */
    .stSidebar .stSuccess {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .stSidebar .stWarning {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .stSidebar .stError {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .stSidebar .stInfo {
        background: linear-gradient(135deg, #3B82F6, #2563EB);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    /* 聊天消息样式 */
    .stChatMessage {
        background-color: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    .stChatMessage:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }
    
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, var(--primary-color), #357ABD);
        color: white;
        border: none;
        margin-left: 3rem;
        box-shadow: 0 4px 16px rgba(74, 144, 226, 0.3);
    }
    
    .stChatMessage[data-testid="assistant"] {
        background: linear-gradient(135deg, var(--surface-color), #F8FAFC);
        color: var(--text-primary);
        margin-right: 3rem;
        border: 1px solid var(--border-color);
    }
    
    /* 聊天输入框样式 - 现代AI网站风格 */
    .stChatInput {
        position: sticky;
        bottom: 0;
        background: var(--surface-color);
        padding: 1rem 0;
        border-top: 1px solid var(--border-color);
    }
    
    .stChatInput > div > div {
        border-radius: 24px;
        border: 2px solid var(--border-color);
        background: var(--surface-color);
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .stChatInput > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 4px 20px rgba(74, 144, 226, 0.15);
    }
    
    /* 模型选择器现代化样式 */
    .model-selector-container {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 16px;
        margin: 20px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease;
    }
    
    .model-selector-container:hover {
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    }
    
    /* 自定义选择框样式 */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        background: var(--surface-color);
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
    }
    
    /* 状态指示器样式 */
    .status-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 40px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .status-indicator:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .status-ready {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 1px solid #bbf7d0;
        color: #166534;
    }
    
    .status-initializing {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid #fde68a;
        color: #92400e;
    }
    
    .status-error {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 1px solid #fecaca;
        color: #991b1b;
    }
    
    /* 主界面容器优化 */
    .main-chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* 聊天消息优化 */
    .stChatMessage {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease;
    }
    
    .stChatMessage:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* 侧边栏优化 - 紧凑布局 */
    .stSidebar .element-container {
        margin-bottom: 0.5rem;
    }
    
    .stSidebar .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 2px solid var(--border-color);
        background: var(--surface-color);
        color: var(--text-primary);
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stSidebar .stButton > button:hover {
        border-color: var(--primary-color);
        background: var(--hover-color);
        transform: translateY(-1px);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background-color: var(--surface-color);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        margin: 0.25rem;
    }
    
    .stButton > button:hover {
        background-color: var(--hover-color);
        border-color: var(--primary-color);
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* 主要按钮样式 */
    .stButton > button[kind="primary"] {
        background-color: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #3B82F6;
        border-color: #3B82F6;
    }
    
    /* 示例问题样式已移除 */
    
    /* 成功/错误/警告消息样式 */
    .stSuccess {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stError {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stWarning {
        background-color: #FFFBEB;
        border: 1px solid #FED7AA;
        color: #92400E;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stInfo {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* 标题样式 */
    h1 {
        color: var(--text-primary);
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -0.025em;
    }
    
    h2 {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: -0.025em;
    }
    
    h3 {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        letter-spacing: -0.025em;
    }
    
    /* 卡片样式 */
    .info-card {
        background-color: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .welcome-card {
        background: linear-gradient(135deg, var(--primary-color), #3B82F6);
        color: white;
        border: none;
        text-align: center;
        padding: 3rem 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(74, 144, 226, 0.2);
        margin: 2rem 0;
    }
    
    .welcome-card h2 {
        color: white;
        margin-bottom: 1rem;
    }
    
    .welcome-card p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    
    /* 分割线样式 */
    .divider {
        border: none;
        height: 1px;
        background-color: var(--border-color);
        margin: 1.5rem 0;
    }
    
    /* 加载动画 */
    .stSpinner {
        color: var(--primary-color);
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
        animation: pulse 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        from { box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3); }
        to { box-shadow: 0 4px 16px rgba(74, 144, 226, 0.5); }
    }
    
    /* 进度状态文本 */
    .progress-status {
        background: linear-gradient(135deg, #F8FAFC, #E5E7EB);
        border: 1px solid var(--border-color);
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-family: 'Inter', monospace;
        font-size: 0.875rem;
        color: var(--text-secondary);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* 进度步骤信息 */
    .progress-step {
        background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
        border: 1px solid #BFDBFE;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: #1E40AF;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    
    /* 文件上传器样式 */
    .stFileUploader {
        background-color: var(--surface-color);
        border: 2px dashed var(--border-color);
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-color);
        background-color: var(--hover-color);
    }
    
    /* 展开器样式 */
    .stExpander {
        background-color: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .stExpander summary {
        background-color: var(--hover-color);
        color: var(--text-primary);
        font-weight: 500;
        padding: 0.75rem;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .main > div {
            padding: 1rem;
        }
        
        .main-chat-container {
            padding: 0 0.5rem;
        }
        
        .model-selector-container {
            margin: 10px 0;
            padding: 12px;
        }
        
        .stChatMessage[data-testid="user"] {
            margin-left: 1rem;
        }
        
        .stChatMessage[data-testid="assistant"] {
            margin-right: 1rem;
        }
        
        /* 响应式示例问题样式已移除 */
    }
    </style>
            """, unsafe_allow_html=True)


class LangChainChatbot:
    """基于LangChain+Ollama的智能客服机器人"""
    
    def __init__(self, 
                 llm_model: str = "deepseek-r1:7b",
                 embedding_model: str = "./bge-large-zh-v1.5",
                 ollama_base_url: str = "http://localhost:11434",
                 use_optimized: bool = True):
        """
        初始化LangChain聊天机器人
        
        Args:
            llm_model: Ollama LLM模型名称
            embedding_model: Ollama embedding模型名称  
            ollama_base_url: Ollama服务地址
            use_optimized: 是否使用优化后的RAG系统
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.use_optimized = use_optimized
        
        # 初始化组件
        self.rag_system = None
        self.document_loader = LangChainDocumentLoader()
        self.vectorstore = None
        self.conversation_history = []
        
        # 配置日志
        logger.add("langchain_chatbot.log", rotation="1 MB", retention="7 days")
        
        # 创建必要目录
        self._create_directories()
        
        # 初始化系统
        self._initialize_system()
    
    def _create_directories(self):
        """创建必要的目录"""
        dirs = [
            "vectorstore",
            "knowledge_base", 
            "logs",
            "outputs"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _initialize_system(self):
        """初始化LangChain RAG系统"""
        try:
            if self.use_optimized:
                logger.info("🚀 初始化优化后的LangChain RAG系统...")
                self.rag_system = OptimizedRAGSystem(
                    llm_model=self.llm_model,
                    base_url=self.ollama_base_url,
                    embedding_model=self.embedding_model
                )
                logger.info("✅ 优化RAG系统初始化完成（包含5项核心优化）")
            else:
                logger.info("初始化传统LangChain RAG系统...")
                self.rag_system = LangChainRAGSystem(
                    model_name=self.llm_model,
                    base_url=self.ollama_base_url,
                    embedding_model=self.embedding_model
                )

            # === 新增：优先自动加载vectorstore下的向量数据库 ===
            auto_loaded = False
            try:
                if hasattr(self.rag_system, 'vectorstore') and self.rag_system.vectorstore:
                    # 优先尝试加载vectorstore
                    if self.rag_system.vectorstore.load_vectorstore():
                        doc_count = self.rag_system.vectorstore.get_document_count()
                        if doc_count > 0:
                            logger.info(f"✅ 已自动加载本地向量数据库，文档数: {doc_count}")
                            auto_loaded = True
                            # === 新增：自动补全RAG链和检索器 ===
                            try:
                                # 优化RAG和传统RAG都兼容
                                rag = self.rag_system
                                # 检查是否有RAG链/检索器
                                rag_chain_ready = False
                                if hasattr(rag, 'rag_chain') and rag.rag_chain:
                                    rag_chain_ready = True
                                if hasattr(rag, 'retriever') and rag.retriever:
                                    rag_chain_ready = True
                                if hasattr(rag, 'advanced_retriever') and rag.advanced_retriever:
                                    rag_chain_ready = True
                                if not rag_chain_ready:
                                    # 优先尝试加载文档缓存
                                    documents = []
                                    if hasattr(rag, '_load_cached_documents'):
                                        documents = rag._load_cached_documents()
                                    if not documents and hasattr(rag, '_reload_source_documents'):
                                        documents = rag._reload_source_documents()
                                    if documents:
                                        # 优化RAG
                                        if hasattr(rag, 'advanced_retriever'):
                                            from src.advanced_hybrid_retriever import AdvancedHybridRetriever
                                            rag.advanced_retriever = AdvancedHybridRetriever(
                                                vectorstore=rag.vectorstore,
                                                documents=documents,
                                                vector_weight=getattr(rag, 'vector_weight', 0.4),
                                                bm25_weight=getattr(rag, 'bm25_weight', 0.6),
                                                enable_force_recall=True,
                                                enable_exact_phrase=True,
                                                k=getattr(rag, 'retrieval_k', 10)
                                            )
                                            if hasattr(rag, '_build_optimized_rag_chain'):
                                                rag._build_optimized_rag_chain()
                                        # 传统RAG
                                        elif hasattr(rag, '_create_enhanced_retriever'):
                                            from src.langchain_retriever import LangChainHybridRetriever
                                            base_retriever = LangChainHybridRetriever(
                                                vectorstore=rag.vectorstore,
                                                documents=documents,
                                                enable_reranking=True
                                            )
                                            rag.retriever = rag._create_enhanced_retriever(base_retriever, documents)
                                            if hasattr(rag, '_build_rag_chain'):
                                                rag._build_rag_chain()
                                        logger.info("✅ 已自动补全RAG链和检索器，系统可直接问答")
                                    else:
                                        logger.warning("自动补全RAG链失败：未能加载文档缓存或PDF")
                            except Exception as e:
                                logger.warning(f"自动补全RAG链失败: {e}")
            except Exception as e:
                logger.warning(f"自动加载本地向量数据库失败: {e}")

            # === 兼容原有自动加载逻辑 ===
            if not auto_loaded:
                self.knowledge_base_auto_loaded = self._check_knowledge_base_status()
            else:
                self.knowledge_base_auto_loaded = True

            if self.knowledge_base_auto_loaded:
                logger.info("✅ 知识库已自动加载，系统就绪")
                try:
                    if self.init_document_watcher():
                        if self.start_document_watching():
                            logger.info("✅ 文档监控已自动启动，系统将自动检测文件变更")
                        else:
                            logger.warning("⚠️ 文档监控启动失败")
                    else:
                        logger.warning("⚠️ 文档监控初始化失败")
                except Exception as e:
                    logger.error(f"文档监控自动启动失败: {e}")
            else:
                logger.info("ℹ️ 知识库未自动加载，需要手动加载")
            logger.info("LangChain智能客服系统初始化完成")
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise
    
    def _check_knowledge_base_status(self) -> bool:
        """检查知识库加载状态"""
        try:
            if self.rag_system and self.rag_system.vectorstore:
                doc_count = self.rag_system.vectorstore.get_document_count()
                return doc_count > 0
            return False
        except Exception as e:
            logger.error(f"检查知识库状态失败: {e}")
            return False
    
    def update_model(self, new_model: str):
        """更新LLM模型"""
        try:
            logger.info(f"更新模型从 {self.llm_model} 到 {new_model}")
            self.llm_model = new_model
            
            # 重新初始化RAG系统
            self.rag_system = LangChainRAGSystem(
                model_name=new_model,
                base_url=self.ollama_base_url,
                embedding_model=self.embedding_model
            )
            
            logger.info(f"模型更新完成: {new_model}")
            return True
            
        except Exception as e:
            logger.error(f"更新模型失败: {e}")
            return False
    
    def load_knowledge_base_incremental(self, pdf_files: List[str] = None, directory: str = None, 
                                       progress_callback: Optional[Callable] = None, force_rebuild: bool = False) -> bool:
        """
        增量加载知识库
        
        Args:
            pdf_files: PDF文件列表
            directory: 包含PDF的目录
            progress_callback: 进度回调函数
            force_rebuild: 是否强制重建
            
        Returns:
            是否成功加载
        """
        try:
            logger.info("开始增量加载知识库...")
            
            # 如果没有指定文件，自动查找data目录下的PDF文件
            if not pdf_files and not directory:
                pdf_files = glob.glob("data/*.pdf")
                if not pdf_files:
                    directory = "data"
            
            # 使用增量加载方法
            if pdf_files:
                # 处理单个文件或文件列表
                success = True
                for pdf_file in pdf_files:
                    file_success = self.rag_system.load_documents_incremental(
                        file_path=pdf_file, 
                        progress_callback=progress_callback,
                        force_rebuild=force_rebuild
                    )
                    success = success and file_success
            elif directory:
                success = self.rag_system.load_documents_incremental(
                    directory_path=directory, 
                    progress_callback=progress_callback,
                    force_rebuild=force_rebuild
                )
            else:
                return False
            
            if success:
                logger.info(f"知识库增量加载成功")
                
                # 自动启动文档监控
                try:
                    if self.init_document_watcher():
                        if self.start_document_watching():
                            logger.info("✅ 文档监控已自动启动，系统将自动检测文件变更")
                        else:
                            logger.warning("⚠️ 文档监控启动失败")
                    else:
                        logger.warning("⚠️ 文档监控初始化失败")
                except Exception as e:
                    logger.error(f"文档监控自动启动失败: {e}")
                
                return True
            else:
                logger.error("知识库增量加载失败")
                return False
                
        except Exception as e:
            logger.error(f"增量加载知识库时出错: {e}")
            return False

    def load_knowledge_base(self, pdf_files: List[str] = None, directory: str = None, progress_callback: Optional[Callable] = None) -> bool:
        """
        加载知识库
        
        Args:
            pdf_files: PDF文件列表
            directory: 包含PDF的目录
            progress_callback: 进度回调函数
            
        Returns:
            是否成功加载
        """
        try:
            logger.info("开始加载知识库...")
            
            # 如果没有指定文件，自动查找data目录下的PDF文件
            if not pdf_files and not directory:
                pdf_files = glob.glob("data/*.pdf")
                if not pdf_files:
                    logger.warning("未找到data目录下的PDF文件")
                    return False
            
            # 加载文档到RAG系统，传递进度回调
            if pdf_files:
                success = all(self.rag_system.load_documents(file_path=pdf, progress_callback=progress_callback) for pdf in pdf_files)
            elif directory:
                success = self.rag_system.load_documents(directory_path=directory, progress_callback=progress_callback)
            else:
                return False
            
            if success:
                logger.info(f"知识库加载成功，包含 {len(pdf_files) if pdf_files else '目录中的'} 个文件")
                
                # 自动启动文档监控
                try:
                    if self.init_document_watcher():
                        if self.start_document_watching():
                            logger.info("✅ 文档监控已自动启动，系统将自动检测文件变更")
                        else:
                            logger.warning("⚠️ 文档监控启动失败")
                    else:
                        logger.warning("⚠️ 文档监控初始化失败")
                except Exception as e:
                    logger.error(f"文档监控自动启动失败: {e}")
                
                return True
            else:
                logger.error("知识库加载失败")
                return False
                
        except Exception as e:
            logger.error(f"加载知识库时出错: {e}")
            return False
    
    def answer_question(self, question: str) -> Dict:
        """
        回答用户问题
        
        Args:
            question: 用户问题
            
        Returns:
            包含答案和相关信息的字典
        """
        try:
            if not question.strip():
                return {
                    "question": "",
                    "answer": "请输入您的问题。",
                    "confidence": 0.0,
                    "sources": [],
                    "retrieval_results": []
                }
            
            logger.info(f"处理问题: {question}")
            
            # 使用LangChain RAG系统回答问题
            rag_response = self.rag_system.answer_question(question)
            
            # 记录对话历史
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "question": question,
                "answer": rag_response.answer,
                "confidence": len(rag_response.source_documents) / 10.0  # 简单的置信度计算
            })
            
            # 构建响应
            result = {
                "question": rag_response.question,
                "answer": rag_response.answer,
                "confidence": len(rag_response.source_documents) / 10.0,
                "sources": [doc.metadata.get('source', '未知来源') for doc in rag_response.source_documents],
                "source_documents": rag_response.source_documents,  # 完整的源文档
                "retrieval_results": [
                    {
                        "content": result.document.page_content,  # 完整内容，不截断
                        "score": result.score,
                        "source": result.source
                    }
                    for result in rag_response.retrieval_results
                ]
            }
            
            logger.info(f"问题回答完成，置信度: {result['confidence']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"回答问题时出错: {e}")
            return {
                "question": question,
                "answer": f"抱歉，处理您的问题时出现错误：{str(e)}",
                "confidence": 0.0,
                "sources": [],
                "retrieval_results": []
            }
    
    def batch_answer_questions(self, questions: List[str], output_file: str = "batch_results.xlsx") -> List[Dict]:
        """
        批量回答问题
        
        Args:
            questions: 问题列表
            output_file: 输出文件路径
            
        Returns:
            结果列表
        """
        try:
            results = []
            
            logger.info(f"开始批量处理 {len(questions)} 个问题")
            
            for i, question in enumerate(questions, 1):
                logger.info(f"处理第 {i}/{len(questions)} 个问题")
                
                answer_result = self.answer_question(question)
                
                result = {
                    '问题编号': f"Q{str(i).zfill(4)}",
                    '问题': question,
                    '回答': answer_result['answer'],
                    '置信度': f"{answer_result['confidence']:.2f}",
                    '来源数量': len(answer_result['sources'])
                }
                
                results.append(result)
            
            # 保存结果到Excel
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False)
            logger.info(f"批量问答结果已保存到: {output_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"批量回答问题时出错: {e}")
            return []
    
    def extract_competition_info(self, output_file: str = "competition_info.xlsx") -> bool:
        """
        提取竞赛信息
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            是否成功提取
        """
        try:
            logger.info("开始提取竞赛信息...")
            
            # 使用信息提取提示模板
            if self.rag_system and hasattr(self.rag_system, 'extract_information'):
                competition_data = self.rag_system.extract_information()
            else:
                # 如果没有专门的提取方法，使用问答方式
                info_questions = [
                    "竞赛名称是什么？",
                    "报名时间是什么时候？", 
                    "比赛时间是什么时候？",
                    "参赛对象有哪些要求？",
                    "奖项设置是怎样的？"
                ]
                
                competition_data = []
                for question in info_questions:
                    result = self.answer_question(question)
                    competition_data.append({
                        "信息类型": question.replace("是什么？", "").replace("是什么时候？", "").replace("有哪些要求？", "").replace("是怎样的？", ""),
                        "详细信息": result['answer']
                    })
            
            # 保存到Excel
            if isinstance(competition_data, list):
                df = pd.DataFrame(competition_data)
                df.to_excel(output_file, index=False)
                logger.info(f"竞赛信息已提取并保存到: {output_file}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"提取竞赛信息时出错: {e}")
            return False
    
    def get_system_status(self) -> Dict:
        """获取系统状态信息"""
        try:
            # 实时检查知识库状态
            kb_loaded = self._check_knowledge_base_status()
            doc_count = 0
            
            if self.rag_system and self.rag_system.vectorstore:
                try:
                    doc_count = self.rag_system.vectorstore.get_document_count()
                except:
                    doc_count = 0
            
            status = {
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model,
                "ollama_url": self.ollama_base_url,
                "conversation_count": len(self.conversation_history),
                "rag_system_ready": self.rag_system is not None,
                "knowledge_base_loaded": kb_loaded,
                "knowledge_base_auto_loaded": getattr(self, 'knowledge_base_auto_loaded', False),
                "document_count": doc_count,
                "last_interaction": self.conversation_history[-1]["timestamp"].isoformat() if self.conversation_history else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"获取系统状态时出错: {e}")
            return {"error": str(e)}
    
    def rebuild_knowledge_base(self, progress_callback: Optional[Callable] = None) -> bool:
        """
        完全重建知识库
        
        Args:
            progress_callback: 进度回调函数
            
        Returns:
            是否重建成功
        """
        try:
            logger.info("开始重建知识库...")
            
            if not self.rag_system:
                logger.error("RAG系统未初始化")
                return False
            
            success = self.rag_system.rebuild_knowledge_base(progress_callback)
            
            if success:
                logger.info("知识库重建完成")
                # 重新启动文档监控
                try:
                    if self.init_document_watcher():
                        if self.start_document_watching():
                            logger.info("✅ 文档监控已重新启动")
                except Exception as e:
                    logger.error(f"重启文档监控失败: {e}")
            else:
                logger.error("知识库重建失败")
            
            return success
            
        except Exception as e:
            logger.error(f"重建知识库时出错: {e}")
            return False
    
    def get_version_statistics(self) -> Dict:
        """获取版本管理统计信息"""
        try:
            if not self.rag_system:
                return {"error": "RAG系统未初始化"}
            
            return self.rag_system.get_version_statistics()
            
        except Exception as e:
            logger.error(f"获取版本统计失败: {e}")
            return {"error": str(e)}
    
    def cleanup_knowledge_base(self) -> Dict:
        """清理知识库"""
        try:
            if not self.rag_system:
                return {"error": "RAG系统未初始化"}
            
            result = self.rag_system.cleanup_knowledge_base()
            logger.info("知识库清理完成")
            return result
            
        except Exception as e:
            logger.error(f"清理知识库失败: {e}")
            return {"error": str(e)}
    
    def detect_pending_updates(self) -> Dict:
        """检测待更新的文档"""
        try:
            from src.incremental_document_loader import IncrementalDocumentLoader
            
            # 创建增量加载器
            loader = IncrementalDocumentLoader()
            
            # 扫描常用目录
            directories = ["./data", "./docs"]
            existing_dirs = [d for d in directories if os.path.exists(d)]
            
            if not existing_dirs:
                return {"message": "未找到文档目录"}
            
            # 获取待更新文档
            pending_updates = loader.get_pending_updates(existing_dirs)
            
            return {
                "new_files": pending_updates['new'],
                "modified_files": pending_updates['modified'],
                "unchanged_files": pending_updates['unchanged'],
                "total_new": len(pending_updates['new']),
                "total_modified": len(pending_updates['modified']),
                "total_unchanged": len(pending_updates['unchanged'])
            }
            
        except Exception as e:
            logger.error(f"检测待更新文档失败: {e}")
            return {"error": str(e)}
    
    def init_document_watcher(self) -> bool:
        """初始化文档监控器"""
        try:
            if self.rag_system is None:
                logger.error("RAG系统未初始化，无法启动文档监控")
                return False
            
            # 使用配置文件中的设置
            from config import Config
            from src.document_watcher import WatchConfig
            
            config = WatchConfig(
                watch_directories=Config.WATCH_DIRECTORIES,
                file_patterns=Config.WATCH_FILE_PATTERNS,
                check_interval=Config.WATCH_CHECK_INTERVAL,
                auto_update=Config.WATCH_AUTO_UPDATE,
                min_update_interval=Config.WATCH_MIN_UPDATE_INTERVAL,
                enable_realtime=Config.WATCH_ENABLE_REALTIME
            )
            
            return self.rag_system.init_document_watcher(config)
            
        except Exception as e:
            logger.error(f"初始化文档监控器失败: {e}")
            return False
    
    def start_document_watching(self) -> bool:
        """启动文档监控"""
        try:
            if self.rag_system is None:
                logger.error("RAG系统未初始化")
                return False
            
            # 如果监控器未初始化，先初始化
            if self.rag_system.document_watcher is None:
                if not self.init_document_watcher():
                    return False
            
            return self.rag_system.start_document_watching()
            
        except Exception as e:
            logger.error(f"启动文档监控失败: {e}")
            return False
    
    def stop_document_watching(self) -> bool:
        """停止文档监控"""
        try:
            if self.rag_system and self.rag_system.document_watcher:
                return self.rag_system.stop_document_watching()
            return True
            
        except Exception as e:
            logger.error(f"停止文档监控失败: {e}")
            return False
    
    def check_documents_now(self) -> Dict[str, str]:
        """立即检查文档变化"""
        try:
            if self.rag_system and self.rag_system.document_watcher:
                return self.rag_system.check_documents_now()
            return {}
            
        except Exception as e:
            logger.error(f"检查文档失败: {e}")
            return {}
    
    def get_watch_status(self) -> Dict:
        """获取文档监控状态"""
        try:
            if self.rag_system and self.rag_system.document_watcher:
                return self.rag_system.get_watch_status()
            return {"enabled": False, "message": "文档监控器未初始化"}
            
        except Exception as e:
            logger.error(f"获取监控状态失败: {e}")
            return {"enabled": False, "error": str(e)}
    
    def get_monitored_files(self) -> List[Dict]:
        """获取监控的文件列表"""
        try:
            if self.rag_system and self.rag_system.document_watcher:
                return self.rag_system.get_monitored_files()
            return []
            
        except Exception as e:
            logger.error(f"获取监控文件列表失败: {e}")
            return []


def create_streamlit_interface():
    """创建Streamlit Web界面"""
    
    # 设置页面配置
    st.set_page_config(
        page_title="🤖 智能客服机器人",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 设置页面样式
    set_page_style()
    
    # 主标题 - 简约风格
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="
            color: #1F2937;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        ">
            🤖 智能客服机器人
        </h1>
        <p style="
            color: #6B7280;
            font-size: 1.1rem;
            font-weight: 400;
            margin: 0;
        ">
            基于 LangChain + Ollama 的智能问答系统
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 获取可用模型
    available_models = get_ollama_models()
    if not available_models:
        available_models = [Config.LLM_MODEL]  # 如果获取失败，使用默认模型
    
    # 初始化session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = available_models[0] if available_models else Config.LLM_MODEL
    if 'use_optimized' not in st.session_state:
        st.session_state.use_optimized = True  # 默认使用优化模式
    if 'auto_init_done' not in st.session_state:
        st.session_state.auto_init_done = False

    # 侧边栏
    with st.sidebar:
        st.title("⚙️ 系统控制")
        
        # RAG系统配置
        st.subheader("🚀 RAG系统配置")
        
        # 优化模式选择
        use_optimized = st.toggle(
            "🎯 启用优化模式",
            value=st.session_state.use_optimized,
            help="启用5项核心优化：混合检索+重排序+实体奖励+文档增强+提示优化"
        )
        
        # 如果优化模式改变了，需要重新初始化
        if use_optimized != st.session_state.use_optimized:
            st.session_state.use_optimized = use_optimized
            st.session_state.chatbot = None
            st.session_state.auto_init_done = False
            st.info("🔄 优化模式已更改，系统将重新初始化")
            st.rerun()
        
        # 优化模式说明
        if st.session_state.use_optimized:
            st.success("✅ 优化模式已启用")
            with st.expander("📋 优化功能详情", expanded=False):
                st.markdown("""
                **🎯 5项核心优化**：
                1. **混合检索**: BM25+向量检索融合
                2. **重排序**: Cross-Encoder智能重排
                3. **实体奖励**: 关键词命中加分
                4. **文档增强**: 章节切分+标题拼接
                5. **提示优化**: Few-shot智能提示
                """)
        else:
            st.info("ℹ️ 使用传统RAG模式")

        # 自动系统初始化
        if st.session_state.chatbot is None and not st.session_state.auto_init_done:
            system_type = "优化后的RAG系统" if st.session_state.use_optimized else "传统RAG系统"
            with st.spinner(f"🚀 正在自动初始化{system_type}..."):
                try:
                    st.session_state.chatbot = LangChainChatbot(
                        llm_model=st.session_state.current_model,
                        embedding_model="./bge-large-zh-v1.5",
                        ollama_base_url=Config.OLLAMA_BASE_URL,
                        use_optimized=st.session_state.use_optimized
                    )
                    st.session_state.auto_init_done = True
                    success_msg = f"✅ {system_type}自动初始化完成！"
                    if st.session_state.use_optimized:
                        success_msg += "\n🎯 5项核心优化已启用"
                    st.success(success_msg)
                    st.rerun()  # 刷新页面状态
                except Exception as e:
                    st.error(f"❌ 自动初始化失败: {e}")
                    st.info("💡 请检查Ollama服务是否正常运行")
        
        # 系统状态显示 - 移出初始化条件，独立显示
        if st.session_state.chatbot:
            # 状态标题和刷新按钮
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("📊 系统状态")
            with col2:
                if st.button("🔄", help="刷新状态", key="refresh_status"):
                    # 强制刷新状态
                    st.session_state.system_status_cache = st.session_state.chatbot.get_system_status()
                    st.session_state.last_status_update = time.time()
                    st.rerun()
            
            # 添加状态缓存，避免频繁调用
            if 'system_status_cache' not in st.session_state:
                st.session_state.system_status_cache = None
                st.session_state.last_status_update = 0
            
            # 每30秒更新一次状态，避免频繁调用
            current_time = time.time()
            if (st.session_state.system_status_cache is None or 
                current_time - st.session_state.last_status_update > 30):
                st.session_state.system_status_cache = st.session_state.chatbot.get_system_status()
                st.session_state.last_status_update = current_time
            
            status = st.session_state.system_status_cache
            
            # 创建状态卡片
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💬 对话次数", status.get("conversation_count", 0))
            with col2:
                st.metric("🤖 当前模型", status.get("llm_model", "未知")[-15:])
            
            # 第二行状态卡片
            col3, col4 = st.columns(2)
            with col3:
                st.metric("📚 文档数量", status.get("document_count", 0))
            with col4:
                auto_loaded = "✅ 自动" if status.get("knowledge_base_auto_loaded") else "⚠️ 手动"
                st.metric("🔄 加载方式", auto_loaded)
            
            # 系统状态指示器
            if status.get("rag_system_ready"):
                st.success("🟢 RAG系统: 就绪")
            else:
                st.warning("🟡 RAG系统: 初始化中...")
            
            # 知识库状态显示
            if status.get("knowledge_base_loaded"):
                if status.get("knowledge_base_auto_loaded"):
                    st.success("🟢 知识库: 已自动加载")
                else:
                    st.success("🟢 知识库: 已手动加载")
            else:
                st.warning("🟡 知识库: 未加载")
        
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            
            # 版本管理统计
            st.subheader("📊 版本管理统计")
            
            # 添加版本统计缓存
            if 'version_stats_cache' not in st.session_state:
                st.session_state.version_stats_cache = None
                st.session_state.last_version_update = 0
            
            # 每60秒更新一次版本统计
            if (st.session_state.version_stats_cache is None or 
                current_time - st.session_state.last_version_update > 60):
                try:
                    st.session_state.version_stats_cache = st.session_state.chatbot.get_version_statistics()
                    st.session_state.last_version_update = current_time
                except Exception as e:
                    logger.debug(f"获取版本统计异常: {e}")
                    st.session_state.version_stats_cache = {"error": str(e)}
            
            version_stats = st.session_state.version_stats_cache
            
            if "error" not in version_stats:
                if version_stats.get("versioning_enabled"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 跟踪文档", version_stats.get("total_documents", 0))
                    with col2:
                        st.metric("🧩 文档片段", version_stats.get("total_chunks", 0))
                    with col3:
                        avg_chunks = version_stats.get("average_chunks_per_doc", 0)
                        st.metric("📊 平均片段", f"{avg_chunks:.1f}")
                    
                    if version_stats.get("latest_update"):
                        st.caption(f"最后更新: {version_stats.get('latest_update')}")
                else:
                    st.info("版本管理功能未启用")
            else:
                st.error(f"获取版本统计失败: {version_stats.get('error', '未知错误')}")
            
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            
            # 知识库管理
            st.subheader("📚 知识库管理")
            
            pdf_files = glob.glob("data/*.pdf")
            if pdf_files:
                st.info(f"📄 发现 {len(pdf_files)} 个PDF文件")
                
                # 根据知识库状态显示不同的状态信息
                if status.get("knowledge_base_loaded"):
                    if status.get("knowledge_base_auto_loaded"):
                        # 已自动加载，显示状态
                        st.success("✅ 知识库已自动加载，系统会自动检测文件变更")
                    else:
                        # 已手动加载，显示状态
                        st.success("✅ 知识库已手动加载完成")
                else:
                    # 未加载，显示加载按钮
                    st.warning("⚠️ 知识库尚未加载，请手动加载")
                
                # 检测待更新文档
                if st.button("🔍 检测文档变化", help="扫描并检测需要更新的文档"):
                    with st.spinner("正在检测文档变化..."):
                        try:
                            pending_updates = st.session_state.chatbot.detect_pending_updates()
                            if "error" not in pending_updates:
                                st.success("✅ 文档变化检测完成")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🆕 新增文件", pending_updates.get("total_new", 0))
                                with col2:
                                    st.metric("📝 修改文件", pending_updates.get("total_modified", 0))
                                with col3:
                                    st.metric("✅ 未变化文件", pending_updates.get("total_unchanged", 0))
                                
                                # 显示详细信息
                                if pending_updates.get("details"):
                                    with st.expander("📋 详细变化信息"):
                                        for file_name, info in pending_updates["details"].items():
                                            if info["status"] != "unchanged":
                                                st.write(f"📄 {file_name}: {info['status']}")
                                                if "reason" in info:
                                                    st.caption(f"   原因: {info['reason']}")
                            else:
                                st.error(f"检测失败: {pending_updates.get('error', '未知错误')}")
                        except Exception as e:
                            st.error(f"检测文档变化时出错: {e}")
                
                st.markdown("---")
                
                # 知识库操作按钮
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 增量更新", help="仅处理新增或修改的文档", key="incremental_update"):
                        with st.spinner("正在进行增量更新..."):
                            def incremental_progress_callback(progress_info):
                                st.write(f"📊 {progress_info.step_name}: {progress_info.description}")
                            
                            success = st.session_state.chatbot.load_knowledge_base_incremental(pdf_files, progress_callback=incremental_progress_callback)
                            if success:
                                st.success("✅ 增量更新成功！")
                                st.rerun()
                            else:
                                st.error("❌ 增量更新失败")
                
                with col2:
                    if st.button("🔨 完全重建", help="清空现有数据并重新构建知识库", key="rebuild_kb"):
                        with st.spinner("正在重建知识库..."):
                            success = st.session_state.chatbot.rebuild_knowledge_base()
                            if success:
                                st.success("✅ 知识库重建成功！")
                                st.rerun()
                            else:
                                st.error("❌ 知识库重建失败")
                
                with col3:
                    if st.button("🧹 清理数据", help="清理孤立的版本信息", key="cleanup_kb"):
                        with st.spinner("正在清理知识库..."):
                            try:
                                result = st.session_state.chatbot.cleanup_knowledge_base()
                                if "error" not in result:
                                    removed_count = result.get("orphaned_removed", 0)
                                    if removed_count > 0:
                                        st.success(f"✅ 清理完成，移除了 {removed_count} 个孤立版本")
                                    else:
                                        st.info("✅ 清理完成，没有发现孤立数据")
                                else:
                                    st.error(f"清理失败: {result.get('error', '未知错误')}")
                            except Exception as e:
                                st.error(f"清理时出错: {e}")
                
                st.markdown("---")
                
                if st.button("📥 传统加载", key="load_kb", help="使用传统方式重新加载所有PDF文件"):
                    if st.session_state.chatbot:
                        # 创建进度条容器
                        progress_container = st.container()
                        
                        with progress_container:
                            # 进度条标题 - 使用empty()容器以便后续清除
                            title_container = st.empty()
                            title_container.markdown("""
                            <div style="
                                background: linear-gradient(135deg, var(--primary-color), #3B82F6);
                                color: white;
                                padding: 1rem;
                                border-radius: 0.5rem;
                                text-align: center;
                                font-weight: 600;
                                margin: 1rem 0;
                                box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
                            ">
                                🚀 正在加载知识库...
    </div>
    """, unsafe_allow_html=True)
        
                            # 初始化进度条
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            step_info = st.empty()
                            
                            # 进度回调函数
                            def progress_callback(progress_info):
                                from src.progress_manager import ProgressInfo
                                
                                # 更新进度条
                                progress_percentage = progress_info.percentage / 100
                                progress_bar.progress(progress_percentage)
                                
                                # 更新状态文本
                                status_text.markdown(f"""
                                <div class="progress-status">
                                    📊 <strong>{progress_info.step_name}</strong> ({progress_info.current_step}/{progress_info.total_steps}) - {progress_info.percentage:.1f}%
        </div>
        """, unsafe_allow_html=True)
        
                                # 更新详细信息
                                step_info.markdown(f"""
                                <div class="progress-step">
                                    🔄 {progress_info.description}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 如果失败，显示错误信息
                                if progress_info.status.value == "failed":
                                    st.error(f"❌ 加载失败: {progress_info.error_message}")
                        
                        # 执行加载
                        success = st.session_state.chatbot.load_knowledge_base(pdf_files, progress_callback=progress_callback)
                        
                        # 清理所有进度显示组件，包括标题
                        title_container.empty()
                        progress_bar.empty()
                        status_text.empty()
                        step_info.empty()
                        
                        if success:
                            st.success(f"✅ 成功加载 {len(pdf_files)} 个PDF文件")
                            # 强制刷新页面状态
                            st.rerun()
                        else:
                            st.error("❌ 知识库加载失败")
                    else:
                        st.warning("⚠️ 请先初始化系统")
            else:
                st.warning("⚠️ 未找到PDF文件")
                st.info("💡 请将PDF文件放入data目录")
            
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            
            # 批量问答
            st.subheader("📝 批量问答")
            
            uploaded_questions = st.file_uploader(
                "上传问题Excel文件",
                type=['xlsx', 'xls'],
                help="Excel文件应包含'问题'列"
            )
            
            if uploaded_questions and st.button("🚀 开始批量问答", key="batch_qa"):
                if st.session_state.chatbot:
                    with st.spinner("正在批量处理问题..."):
                        try:
                            df = pd.read_excel(uploaded_questions)
                            questions = df['问题'].tolist()
                            
                            results = st.session_state.chatbot.batch_answer_questions(
                                questions, 
                                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                            )
                            
                            if results:
                                st.success(f"✅ 成功处理 {len(results)} 个问题")
                                
                                # 创建Excel文件的字节流数据
                                excel_buffer = BytesIO()
                                pd.DataFrame(results).to_excel(excel_buffer, index=False, engine='openpyxl')
                                excel_data = excel_buffer.getvalue()
                                
                                st.download_button(
                                    "📥 下载结果文件",
                                    data=excel_data,
                                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.error("❌ 批量处理失败")
                                
                        except Exception as e:
                            st.error(f"❌ 处理失败: {e}")
                else:
                    st.warning("⚠️ 请先初始化系统")

    # 主界面 - 聊天区域
    # 创建主聊天容器
    main_container = st.container()
    
    with main_container:
        st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
        
        if st.session_state.chatbot is None:
            st.markdown("""
            <div class="welcome-card">
                <h2>🚀 欢迎使用智能客服机器人！</h2>
                <p>系统正在自动初始化，请稍候...</p>
                <p style="opacity: 0.9;">💡 初始化完成后可开始问答</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # 检查知识库状态并显示相应提示
            status = st.session_state.chatbot.get_system_status()
            
            if not status.get("knowledge_base_loaded"):
                st.markdown("""
                <div class="welcome-card">
                    <h2>⚠️ 知识库尚未加载</h2>
                    <p>系统已初始化，但知识库尚未加载</p>
                    <p style="opacity: 0.9;">💡 请在侧边栏中加载知识库以获得准确的竞赛信息</p>
                </div>
                """, unsafe_allow_html=True)
            elif status.get("knowledge_base_auto_loaded"):
                st.markdown("""
                <div class="welcome-card">
                    <h2>✅ 系统就绪！</h2>
                    <p>知识库已自动加载，包含 {doc_count} 个文档</p>
                    <p style="opacity: 0.9;">🎯 您可以开始提问关于竞赛的任何问题</p>
                </div>
                """.format(doc_count=status.get("document_count", 0)), unsafe_allow_html=True)
            
            # 聊天界面标题
            st.markdown("""
            <div class="info-card">
                <h2 style="color: #4A90E2; text-align: center; margin-bottom: 1rem;">
                    💬 智能问答对话
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # 聊天历史容器
            chat_container = st.container()
            
            # 显示聊天历史
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # 创建输入区域的容器 - 现代AI网站风格的输入区域
            input_container = st.container()
            
            with input_container:
                # 模型选择和输入框组合区域
                st.markdown("""
                <div class="model-selector-container">
                """, unsafe_allow_html=True)
                
                # 模型选择行
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.markdown("""
                    <div style="
                        display: flex;
                        align-items: center;
                        height: 40px;
                        font-weight: 500;
                        color: #374151;
                        font-size: 14px;
                    ">
                        🤖 模型选择
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    selected_model = st.selectbox(
                        "选择AI模型",  # 提供有意义的标签
                        available_models,
                        index=available_models.index(st.session_state.current_model) if st.session_state.current_model in available_models else 0,
                        key="model_selector_main",
                        help="选择用于回答问题的AI模型",
                        label_visibility="collapsed"  # 隐藏标签显示
                    )
                
                    # 处理模型切换
                    if selected_model != st.session_state.current_model:
                        if st.session_state.chatbot:
                            with st.spinner(f"正在切换到 {selected_model}..."):
                                success = st.session_state.chatbot.update_model(selected_model)
                                if success:
                                    st.session_state.current_model = selected_model
                                    st.success(f"✅ 已切换到模型: {selected_model}")
                                    st.rerun()
                                else:
                                    st.error("❌ 模型切换失败")
                        else:
                            st.session_state.current_model = selected_model
                
                with col3:
                    # 系统状态指示器
                    if st.session_state.chatbot:
                        status = st.session_state.chatbot.get_system_status()
                        if status.get("rag_system_ready") and status.get("knowledge_base_loaded"):
                            st.markdown("""
                            <div class="status-indicator status-ready">
                                🟢 系统就绪
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="status-indicator status-initializing">
                                🟡 初始化中
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="status-indicator status-error">
                            🔴 未初始化
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # 用户输入 - 现代化的聊天输入框
            if prompt := st.chat_input("💭 请输入您的问题...", key="user_input"):
                # 显示用户消息
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # 生成并显示助手回答
                with st.chat_message("assistant"):
                    # 检查系统状态
                    if not st.session_state.chatbot:
                        st.error("❌ 系统未初始化")
                        st.stop()
                    
                    status = st.session_state.chatbot.get_system_status()
                    if not status.get("knowledge_base_loaded"):
                        st.warning("⚠️ 知识库未加载，回答可能不够准确")
                    
                    with st.spinner("🤔 AI正在思考中..."):
                        result = st.session_state.chatbot.answer_question(prompt)
                    
                    response = result['answer']
                    st.markdown(response)
                    
                    # 显示更详细的上下文来源信息
                    if result.get('source_documents'):
                        with st.expander("📖 查看完整上下文来源", expanded=False):
                            source_docs = result['source_documents']
                            
                            # 统计信息
                            total_docs = len(source_docs)
                            total_chars = sum(len(doc.page_content) for doc in source_docs)
                            avg_length = total_chars / total_docs if total_docs > 0 else 0
                            
                            st.markdown(f"""
                            **📊 上下文统计:**
                            - 📄 文档数量: {total_docs}
                            - 📝 总字符数: {total_chars:,}
                            - 📏 平均长度: {avg_length:.0f} 字符/文档
                            """)
                            
                            st.markdown("---")
                            
                            # 详细文档内容
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f"**📄 文档 {i}: {doc.metadata.get('source', '未知来源').split('/')[-1]}**")
                                
                                if doc.page_content.strip():
                                    with st.container():
                                        st.text_area(
                                            f"文档 {i} 内容",
                                            value=doc.page_content,
                                            height=150,
                                            key=f"doc_content_{i}_{len(st.session_state.messages)}",
                                            help="这是检索到的原始文档内容，大模型基于此内容生成答案"
                                        )
                                else:
                                    st.markdown("*（此文档片段为空）*")
                                
                                # 显示元数据信息
                                if doc.metadata and len(doc.metadata) > 1:  # 除了source还有其他元数据
                                    with st.expander(f"📊 查看文档 {i} 的元数据", expanded=False):
                                        metadata_display = {}
                                        for key, value in doc.metadata.items():
                                            if key != 'source':  # source已经显示了
                                                metadata_display[key] = value
                                        if metadata_display:
                                            st.json(metadata_display)
                                
                                if i < len(result['source_documents']):
                                    st.markdown("---")
                    
                    # 也保留简单的文件名列表供快速查看
                    if result['sources']:
                        with st.expander("📁 来源文件列表（快速查看）", expanded=False):
                            unique_sources = list(set(result['sources']))
                            for i, source in enumerate(unique_sources, 1):
                                source_name = source.split('/')[-1] if '/' in source else source
                                st.text(f"{i}. {source_name}")
            
                # 保存助手回答
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 页脚
    st.markdown("""
    <div style="
        background-color: #F8FAFC;
        border: 1px solid #E5E7EB;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        color: #6B7280;
        margin-top: 3rem;
    ">
        <p style="margin: 0; font-size: 0.9rem;">
            🏆 <strong>泰迪杯竞赛智能客服机器人</strong> | 
            ⚡ 基于 LangChain + Ollama | 
            🚀 版本 2.0
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.8;">
            让AI助力您的竞赛之路 🎯
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Web界面模式
        create_streamlit_interface()
    else:
        # 命令行模式
        print("🤖 LangChain+Ollama 智能客服机器人")
        print("=" * 50)
        
        # 初始化机器人
        try:
            chatbot = LangChainChatbot()
            print("✅ 系统初始化完成")
            
            # 加载知识库
            pdf_files = glob.glob("data/*.pdf")
            if pdf_files:
                print(f"加载 {len(pdf_files)} 个PDF文件...")
                if chatbot.load_knowledge_base(pdf_files):
                    print("✅ 知识库加载完成")
                else:
                    print("❌ 知识库加载失败")
            
            # 交互式问答
            print("\n开始问答（输入'quit'退出）:")
            while True:
                question = input("\n请输入问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    break
                
                if question:
                    result = chatbot.answer_question(question)
                    print(f"\n回答: {result['answer']}")
                    print(f"置信度: {result['confidence']:.2f}")
                    
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出程序")
        except Exception as e:
            print(f"❌ 程序运行出错: {e}")


if __name__ == "__main__":
    main() 