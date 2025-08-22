"""
LangChain RAG系统 - 优化版本
使用LangChain构建检索增强生成系统
"""

from typing import List, Dict, Optional, Any, Callable
import os
import glob
import json
import pickle
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from loguru import logger

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

# 统一导入
from .langchain_vectorstore import LangChainVectorStore
from .langchain_document_loader import LangChainDocumentLoader
from .langchain_retriever import LangChainHybridRetriever, RetrievalResult
from .hierarchical_retriever import create_hierarchical_retriever
from .document_watcher import WatchConfig, create_default_watch_config
from .langchain_watcher import LangChainDocumentWatcher
from .answer_validator import AnswerValidator
from .context_compressor import ContextCompressor
from .query_enhancer import get_enhanced_query_optimizer

# 配置导入
try:
    from config import Config
except ImportError:
    class Config:
        LANGCHAIN_RETRIEVER_K = 5
        RETRIEVAL_CONFIG = {}


@dataclass
class RAGResponse:
    """RAG响应结果"""
    question: str
    answer: str
    source_documents: List[Document]
    retrieval_results: List[RetrievalResult]
    metadata: Dict
    timestamp: datetime
    confidence: float = 0.0


class LangChainRAGSystem:
    """基于LangChain的RAG系统 - 优化版"""
    
    def __init__(self, 
                 model_name: str = "deepseek-r1:7b",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.1,
                 embedding_model: str = "./bge-large-zh-v1.5"):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.embedding_model = embedding_model
        
        # 核心组件
        self.vectorstore = None
        self.document_loader = LangChainDocumentLoader()
        self.retriever = None
        self.hierarchical_retriever = None
        self.llm = None
        self.rag_chain = None
        
        # 增强组件
        self.answer_validator = None
        self.context_compressor = None
        self.document_watcher = None
        
        self._init_system()
        logger.info(f"RAG系统初始化完成: {model_name}")
    
    def _init_system(self):
        """初始化系统"""
        self._init_llm()
        self._build_prompts()
        self._init_enhanced_components()
        self._auto_load_knowledge_base()
    
    def _init_llm(self):
        """初始化LLM"""
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                timeout=60
            )
            logger.info(f"LLM初始化成功: {self.model_name}")
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
    
    def _build_prompts(self):
        """构建提示模板"""
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个严格的竞赛文档问答助手。请严格基于提供的文档内容回答问题。

【严格规则】：
1. 只能使用文档中明确提到的信息，不得添加任何文档外的内容
2. 不要推测、猜测或使用常识性知识补充答案
3. 如果文档中没有相关信息，必须明确说明"根据现有文档，我无法找到相关信息"
4. 引用具体数据时，请确保与文档内容完全一致

【文档内容】：
{context}

【回答要求】：严格基于上述文档内容，准确回答用户问题。"""),
            ("human", "{input}")
        ])
    
    def _init_enhanced_components(self):
        """初始化增强组件"""
        try:
            self.answer_validator = AnswerValidator()
            self.context_compressor = ContextCompressor(max_length=1500)
            logger.info("增强组件初始化成功")
        except Exception as e:
            logger.warning(f"增强组件初始化失败: {e}")
    
    def _create_enhanced_retriever(self, base_retriever: LangChainHybridRetriever, documents: List[Document] = None):
        """创建增强检索器（简化版）"""
        try:
            retrieval_config = getattr(Config, 'RETRIEVAL_CONFIG', {})
            if retrieval_config.get('enable_hierarchical', True) and documents:
                self.hierarchical_retriever = create_hierarchical_retriever(
                    base_retriever=base_retriever,
                    docs=documents
                )
                logger.info("层级检索器创建完成")
                return self.hierarchical_retriever
            return base_retriever
        except Exception as e:
            logger.warning(f"增强检索器创建失败: {e}")
            return base_retriever
    
    def _manage_document_cache(self, action: str, documents: List[Document] = None) -> List[Document]:
        """统一的文档缓存管理"""
        cache_file = os.path.join("vectorstore", "documents_cache.pkl")
        
        if action == "load":
            try:
                if not os.path.exists(cache_file):
                    return []
                
                with open(cache_file, 'rb') as f:
                    docs = pickle.load(f)
                logger.info(f"从缓存加载 {len(docs)} 个文档")
                return docs
                
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")
                return []
        
        elif action == "save" and documents:
            try:
                os.makedirs("vectorstore", exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(documents, f)
                logger.info(f"缓存已保存 {len(documents)} 个文档")
                return True
            except Exception as e:
                logger.error(f"缓存保存失败: {e}")
                return False
        
        return []
    
    def _auto_load_knowledge_base(self) -> bool:
        """自动加载知识库"""
        try:
            vectorstore_path = "./vectorstore"
            if not os.path.exists(vectorstore_path):
                return False
            
            self.vectorstore = LangChainVectorStore(
                model_name=self.embedding_model,
                ollama_base_url=self.base_url,
                persist_path=vectorstore_path
            )
            
            if not self.vectorstore.load_vectorstore():
                return False
            
            documents = self._manage_document_cache("load")
            if not documents:
                documents = self._load_source_documents()
            
            if documents:
                base_retriever = LangChainHybridRetriever(
                    vectorstore=self.vectorstore,
                    documents=documents,
                    enable_reranking=True
                )
                self.retriever = self._create_enhanced_retriever(base_retriever, documents)
                self._build_rag_chain()
                logger.info(f"知识库加载完成，包含 {len(documents)} 个文档")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"知识库加载失败: {e}")
            return False
    
    def _load_source_documents(self) -> List[Document]:
        """加载源文档"""
        try:
            cached_docs = self._manage_document_cache("load")
            if cached_docs:
                return cached_docs
            
            for directory in ["./data", "./docs"]:
                if os.path.exists(directory):
                    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
                    if pdf_files:
                        documents = []
                        for pdf_file in pdf_files:
                            try:
                                docs = self.document_loader.load_pdf(pdf_file)
                                documents.extend(docs)
                            except Exception as e:
                                logger.warning(f"加载失败 {pdf_file}: {e}")
                        
                        self._manage_document_cache("save", documents)
                        return documents
            return []
            
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            return []
    
    def load_documents(self, file_path: str = None, directory_path: str = None) -> bool:
        """加载文档"""
        try:
            files_to_process = []
            if file_path:
                files_to_process = [file_path]
            elif directory_path:
                files_to_process = glob.glob(os.path.join(directory_path, "*.pdf"))
            else:
                logger.error("请提供文件路径或目录路径")
                return False
            
            if not files_to_process:
                logger.error("未找到PDF文件")
                return False
            
            documents = []
            for file_path in files_to_process:
                try:
                    docs = self.document_loader.load_pdf(file_path)
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"加载失败 {file_path}: {e}")
            
            if not documents:
                logger.error("未能加载任何文档")
                return False
            
            if not self.vectorstore:
                self.vectorstore = LangChainVectorStore(
                    model_name=self.embedding_model,
                    ollama_base_url=self.base_url
                )
            
            self.vectorstore.add_documents(documents)
            
            base_retriever = LangChainHybridRetriever(
                vectorstore=self.vectorstore,
                documents=documents,
                enable_reranking=True
            )
            self.retriever = self._create_enhanced_retriever(base_retriever, documents)
            self._build_rag_chain()
            
            self.vectorstore.save_vectorstore()
            self._manage_document_cache("save", documents)
            
            logger.info(f"文档加载完成，共 {len(documents)} 个片段")
            return True
            
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            return False
    
    def _build_rag_chain(self):
        """构建RAG链"""
        try:
            if self.retriever and self.llm:
                document_chain = create_stuff_documents_chain(
                    llm=self.llm,
                    prompt=self.qa_prompt
                )
                
                self.rag_chain = create_retrieval_chain(
                    retriever=self.retriever,
                    combine_docs_chain=document_chain
                )
                logger.info("RAG链构建完成")
        except Exception as e:
            logger.error(f"RAG链构建失败: {e}")
    
    def answer_question(self, question: str, use_hierarchical: bool = True) -> "RAGResponse":
        """回答问题"""
        start_time = datetime.now()
        logger.info(f"回答问题: {question[:50]}...")
        
        try:
            # 查询优化
            query_enhancer = get_enhanced_query_optimizer()
            search_query, _ = query_enhancer.optimize_query_for_retrieval(question)
            
            # 文档检索
            if use_hierarchical and self.hierarchical_retriever:
                source_documents = self.hierarchical_retriever.retrieve(search_query, k=Config.LANGCHAIN_RETRIEVER_K)
            elif self.retriever:
                source_documents = self.retriever.get_relevant_documents(search_query)
            elif self.vectorstore:
                source_documents = self.vectorstore.similarity_search(search_query, k=Config.LANGCHAIN_RETRIEVER_K)
            else:
                return RAGResponse(
                    question=question,
                    answer="知识库未初始化",
                    source_documents=[],
                    retrieval_results=[],
                    metadata={'error': 'knowledge_base_not_loaded'},
                    timestamp=datetime.now()
                )
            
            if not source_documents:
                return self._create_response(
                    question=question,
                    answer="抱歉，没有找到相关信息",
                    source_documents=[],
                    start_time=start_time
                )
            
            # 上下文压缩
            context = None
            if self.context_compressor:
                original_context = "\n\n".join([doc.page_content for doc in source_documents])
                compressed = self.context_compressor.compress_context(question, original_context)
                if hasattr(compressed, 'compressed_text'):
                    context = compressed.compressed_text
            
            # 生成答案
            answer = self._generate_answer(question, source_documents, context)
            
            # 答案验证
            confidence = 0.8
            if self.answer_validator:
                context_text = "\n\n".join([doc.page_content for doc in source_documents])
                validation = self.answer_validator.validate_answer(question, answer, context_text)
                confidence = validation.confidence
            
            return self._create_response(
                question=question,
                answer=answer,
                source_documents=source_documents,
                start_time=start_time,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"回答失败: {e}")
            return RAGResponse(
                question=question,
                answer=f"处理失败: {e}",
                source_documents=[],
                retrieval_results=[],
                metadata={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def _generate_answer(self, question: str, documents: List[Document], context: str = None) -> str:
        """统一答案生成方法"""
        try:
            if not documents and not context:
                return "抱歉，我无法在知识库中找到相关信息来回答您的问题。"
            
            if not context:
                context = "\n\n".join([doc.page_content[:800] for doc in documents[:5]])
            
            if self.rag_chain:
                response = self.rag_chain.invoke({"input": question})
                return response.get("answer", "无法生成答案")
            
            if self.llm:
                prompt = f"""基于以下文档内容回答问题：

{context}

问题：{question}

请严格基于提供的文档内容准确回答。如果文档中没有相关信息，请明确说明"根据文档，我没有找到相关信息"。"""
                
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            
            return f"根据相关文档，找到以下信息：\n\n{context[:500]}..."
                
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"生成答案时出现错误: {e}"
    
    def extract_competition_info(self) -> List[Dict]:
        """提取竞赛信息"""
        try:
            if not self.vectorstore:
                return []
            
            docs = self.vectorstore.similarity_search("竞赛信息", k=10)
            return self.document_loader.extract_competition_info(docs)
            
        except Exception as e:
            logger.error(f"提取失败: {e}")
            return []
    
    def batch_answer(self, questions: List[str]) -> List[RAGResponse]:
        """批量回答问题"""
        responses = []
        for question in questions:
            response = self.answer_question(question)
            responses.append(response)
        return responses
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'vectorstore_initialized': self.vectorstore is not None,
            'retriever_initialized': self.retriever is not None,
            'llm_available': self.llm is not None,
            'rag_chain_ready': self.rag_chain is not None,
            'document_count': self.vectorstore.get_document_count() if self.vectorstore else 0,
            'model_name': self.model_name
        }
    
    def _create_response(self, question: str, answer: str, source_documents: List[Document], 
                        start_time: datetime, confidence: float = 0.8) -> RAGResponse:
        """创建RAG响应对象"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        retrieval_results = [
            RetrievalResult(
                document=doc,
                score=doc.metadata.get('vector_score', 0.0),
                rank=i + 1,
                source=os.path.basename(doc.metadata.get('source', 'unknown'))
            )
            for i, doc in enumerate(source_documents)
        ]
        
        return RAGResponse(
            question=question,
            answer=answer,
            source_documents=source_documents,
            retrieval_results=retrieval_results,
            metadata={
                'model': self.model_name,
                'retrieval_count': len(source_documents),
                'processing_time': processing_time,
                'confidence': confidence
            },
            timestamp=datetime.now(),
            confidence=confidence
        )
