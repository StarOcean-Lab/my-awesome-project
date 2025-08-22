"""
优化的RAG系统
集成所有5个优化模块，实现完整的检索增强生成优化方案
"""

from typing import List, Dict, Optional, Callable, Tuple
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from loguru import logger
import os
import time
from datetime import datetime

# 导入优化模块
try:
    from .advanced_hybrid_retriever import AdvancedHybridRetriever
    from .enhanced_reranker import EnhancedReranker
    from .enhanced_document_loader import EnhancedDocumentLoader
    from .enhanced_prompt_manager import EnhancedPromptManager
    from .langchain_vectorstore import LangChainVectorStore
except ImportError:
    from advanced_hybrid_retriever import AdvancedHybridRetriever
    from enhanced_reranker import EnhancedReranker
    from enhanced_document_loader import EnhancedDocumentLoader
    from enhanced_prompt_manager import EnhancedPromptManager
    from langchain_vectorstore import LangChainVectorStore

class OptimizedRAGResponse:
    """优化RAG响应对象"""
    
    def __init__(self, question: str, answer: str, source_documents: List[Document],
                 retrieval_stats: Dict, rerank_stats: Dict, prompt_analysis: Dict,
                 response_time: float):
        self.question = question
        self.answer = answer
        self.source_documents = source_documents
        self.retrieval_stats = retrieval_stats
        self.rerank_stats = rerank_stats
        self.prompt_analysis = prompt_analysis
        self.response_time = response_time
        self.timestamp = datetime.now()
        
        # 添加兼容性属性 - 基于source_documents构建retrieval_results
        from .langchain_retriever import RetrievalResult
        self.retrieval_results = [
            RetrievalResult(
                document=doc,
                score=1.0 - (i * 0.1),  # 基于排名的模拟分数
                source="hybrid",  # 默认为混合检索
                rank=i + 1
            )
            for i, doc in enumerate(source_documents)
        ]

class OptimizedRAGSystem:
    """优化的RAG系统 - 集成所有优化模块"""
    
    def __init__(self,
                 llm_model: str = "deepseek-r1:7b",
                 embedding_model: str = "./bge-large-zh-v1.5",
                 base_url: str = "http://localhost:11434",
                 vector_weight: float = 0.4, # 向量检索权重
                 bm25_weight: float = 0.6, # BM25检索权重
                 enable_reranking: bool = True, # 是否启用重排序
                 enable_chapter_splitting: bool = True, # 是否启用章节切分
                 retrieval_k: int = 10 # 检索数量
                 ):
        """
        初始化优化RAG系统
        Args:
            llm_model: 语言模型名称
            embedding_model: 嵌入模型路径
            base_url: Ollama服务地址
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
            enable_reranking: 是否启用重排序
            enable_chapter_splitting: 是否启用章节切分
            retrieval_k: 检索返回数量
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.enable_reranking = enable_reranking
        self.enable_chapter_splitting = enable_chapter_splitting
        self.retrieval_k = retrieval_k
        
        # 初始化组件
        self.vectorstore = None
        self.advanced_retriever = None
        self.enhanced_reranker = None
        self.enhanced_document_loader = None
        self.enhanced_prompt_manager = None
        self.llm = None
        self.rag_chain = None
        
        # 初始化各个模块
        self._initialize_components()
        
        logger.info("优化RAG系统初始化完成")
        logger.info(f"  语言模型: {llm_model}")
        logger.info(f"  嵌入模型: {embedding_model}")
        logger.info(f"  检索权重: 向量={vector_weight}, BM25={bm25_weight}")
        logger.info(f"  功能配置: 重排序={enable_reranking}, 章节切分={enable_chapter_splitting}")
    
    def _initialize_components(self):
        """初始化各个组件"""
        logger.info("初始化优化RAG系统组件...")
        
        # 1. 初始化增强文档加载器
        logger.info("初始化增强文档加载器...")
        self.enhanced_document_loader = EnhancedDocumentLoader()
        
        # 2. 初始化向量存储
        logger.info("初始化向量存储...")
        self.vectorstore = LangChainVectorStore(
            model_name=self.embedding_model,
            ollama_base_url=self.base_url,
            enable_versioning=True
        )
        
        # 3. 初始化增强重排序器
        if self.enable_reranking:
            logger.info("初始化增强重排序器...")
            self.enhanced_reranker = EnhancedReranker()
        
        # 4. 初始化增强提示词管理器
        logger.info("初始化增强提示词管理器...")
        self.enhanced_prompt_manager = EnhancedPromptManager()
        
        # 5. 初始化语言模型
        logger.info("初始化语言模型...")
        self.llm = ChatOllama(
            model=self.llm_model,
            base_url=self.base_url,
            temperature=0.1,
            timeout=300
        )
    
    def load_documents(self, file_path: List[str] = None, directory_path: str = None, progress_callback: Optional[Callable] = None) -> bool:
        """
        加载文档（使用增强加载器）- 兼容LangChainRAGSystem接口
        
        Args:
            file_path: 单个文件路径
            directory_path: 目录路径
            progress_callback: 进度回调函数
            
        Returns:
            是否加载成功
        """
        try:
            # 构建文件列表
            file_paths = []
            if file_path:
                file_paths = file_path
            elif directory_path:
                import glob
                file_paths = glob.glob(os.path.join(directory_path, "*.pdf"))
            else:
                logger.error("请提供文件路径或目录路径")
                return False
            
            if not file_paths:
                logger.error("没有找到要加载的文件")
                return False
            
            logger.info(f"开始加载 {len(file_paths)} 个文档...")
            
            all_documents = []
            
            # 使用增强文档加载器逐个处理文件
            for i, file_path in enumerate(file_paths):
                logger.info(f"处理文档 {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
                
                if progress_callback:
                    progress_callback(f"正在处理文档: {os.path.basename(file_path)}")
                
                # 使用章节切分加载
                docs = self.enhanced_document_loader.load_pdf(
                    file_path, 
                    use_chapter_splitting=self.enable_chapter_splitting
                )
                
                if docs:
                    all_documents.extend(docs)
                    logger.info(f"成功加载 {len(docs)} 个文档块")
                else:
                    logger.warning(f"文档加载失败: {file_path}")
            
            if not all_documents:
                logger.error("没有成功加载任何文档")
                return False
            
            logger.info(f"总计加载 {len(all_documents)} 个增强文档块")
            
            # 添加到向量存储
            if progress_callback:  # 预留了一个接口，暂时还用不到
                ("正在构建向量索引...")
            
            self.vectorstore.add_documents(all_documents, progress_callback=progress_callback)

            # 初始化高级混合检索器
            logger.info("初始化高级混合检索器...")
            self.advanced_retriever = AdvancedHybridRetriever(
                vectorstore=self.vectorstore,
                documents=all_documents,
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight,
                enable_force_recall=True,
                enable_exact_phrase=True,
                k=self.retrieval_k
            )
            
            # 构建RAG链
            self._build_optimized_rag_chain()
            
            logger.info("文档加载和系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            return False
    
    def load_documents_by_paths(self, file_paths: List[str], progress_callback: Optional[Callable] = None) -> bool:
        """
        通过文件路径列表加载文档（保留原始接口）
        
        Args:
            file_paths: 文档文件路径列表
            progress_callback: 进度回调函数
            
        Returns:
            是否加载成功
        """
        if len(file_paths) == 1:
            return self.load_documents(file_path=file_paths[0], progress_callback=progress_callback)
        else:
            # 对于多个文件，构造一个临时目录参数
            import os
            if all(os.path.dirname(fp) == os.path.dirname(file_paths[0]) for fp in file_paths):
                # 如果所有文件在同一目录
                return self.load_documents(directory_path=os.path.dirname(file_paths[0]), progress_callback=progress_callback)
            else:
                # 如果文件在不同目录，逐个加载
                all_success = True
                for file_path in file_paths:
                    success = self.load_documents(file_path=file_path, progress_callback=progress_callback)
                    if not success:
                        all_success = False
                return all_success
    
    def _build_optimized_rag_chain(self):
        """构建优化的RAG链"""
        try:
            logger.info("构建优化RAG链...")
            
            def retrieve_and_rerank(inputs):
                """检索和重排序"""
                question = inputs["question"]
                
                # 1. 高级混合检索
                logger.debug("执行高级混合检索...")
                documents = self.advanced_retriever.get_relevant_documents(question)
                
                # 2. 增强重排序
                if self.enable_reranking and self.enhanced_reranker and documents:
                    logger.debug("执行增强重排序...")
                    documents = self.enhanced_reranker.rerank(question, documents, top_k=self.retrieval_k)
                
                return {"question": question, "context": self._format_documents(documents), "documents": documents}
            
            def enhance_prompt(inputs):
                """增强提示词"""
                question = inputs["question"]
                context = inputs["context"]
                
                # 获取增强提示词
                template, variables = self.enhanced_prompt_manager.get_enhanced_prompt(question, context)
                
                # 格式化提示词
                formatted_prompt = template.format(**variables)
                
                return {"question": question, "context": context, "prompt": formatted_prompt, "documents": inputs["documents"]}
            
            # 构建RAG链 - 修复管道操作符兼容性问题
            from langchain_core.runnables import RunnableLambda
            
            # 创建可运行的组件（把普通函数变成 LangChain 可组合的“可运行组件”）
            question_runnable = RunnableLambda(lambda x: {"question": x} if isinstance(x, str) else x)
            retrieve_runnable = RunnableLambda(retrieve_and_rerank)
            enhance_runnable = RunnableLambda(enhance_prompt)
            prompt_runnable = RunnableLambda(lambda x: x["prompt"])
             
            try:
                # 使用管道操作符（新版本LangChain）
                # 前一个的输出是后一个的输入
                self.rag_chain = (
                    question_runnable
                    | retrieve_runnable
                    | enhance_runnable
                    | prompt_runnable
                    | self.llm
                    | StrOutputParser()
                )
            except TypeError:
                # 降级方案：手动链接
                logger.info("使用兼容性链接方式...")
                def combined_chain(question: str):
                    # 1. 准备输入
                    inputs = {"question": question}
                     
                    # 2. 检索和重排序
                    step1_result = retrieve_and_rerank(inputs)
                     
                    # 3. 增强提示词
                    step2_result = enhance_prompt(step1_result)
                     
                    # 4. 提取提示词
                    prompt = step2_result["prompt"]
                      
                    # 5. 调用LLM
                    llm_result = self.llm.invoke(prompt)
                     
                    # 6. 解析输出
                    parser = StrOutputParser()
                    final_result = parser.parse(llm_result)
                      
                    return final_result
                 
                self.rag_chain = RunnableLambda(combined_chain)
            
            logger.info("优化RAG链构建完成")
            
        except Exception as e:
            logger.error(f"RAG链构建失败: {e}")
            raise e
    
    def answer_question(self, question: str) -> OptimizedRAGResponse:
        """
        回答问题（优化版本）
        
        Args:
            question: 用户问题
            
        Returns:
            优化RAG响应对象
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始回答问题: {question[:50]}...")
            
            if not self.rag_chain:
                raise ValueError("RAG链未初始化，请先加载文档")
            
            # 1. 执行检索和重排序
            logger.info("执行高级混合检索...")
            source_documents = self.advanced_retriever.get_relevant_documents(question)
            
            # 获取检索统计
            retrieval_stats = self._get_retrieval_stats(question)
            
            # 2. 执行增强重排序
            rerank_stats = {}
            if self.enable_reranking and self.enhanced_reranker and source_documents:
                logger.info("执行增强重排序...")
                reranked_docs = self.enhanced_reranker.rerank(question, source_documents, top_k=self.retrieval_k)
                
                # 获取重排序统计
                rerank_stats = self._get_rerank_stats(question, source_documents, reranked_docs)
                source_documents = reranked_docs
            
            # 3. 格式化上下文
            context = self._format_documents(source_documents)
            
            # 4. 获取增强提示词
            logger.info("生成增强提示词...")
            template, variables = self.enhanced_prompt_manager.get_enhanced_prompt(question, context)
            
            # 5. 生成答案
            logger.info("生成最终答案...")
            prompt_formatted = template.format(**variables)
            answer = self.llm.invoke(prompt_formatted).content
            
            # 6. 分析提示词效果
            prompt_analysis = self.enhanced_prompt_manager.analyze_prompt_effectiveness(question, context, answer)
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 记录详细统计
            self._log_response_stats(question, answer, source_documents, retrieval_stats, rerank_stats, response_time)
            
            # 创建响应对象
            response = OptimizedRAGResponse(
                question=question,
                answer=answer,
                source_documents=source_documents,
                retrieval_stats=retrieval_stats,
                rerank_stats=rerank_stats,
                prompt_analysis=prompt_analysis,
                response_time=response_time
            )
            
            logger.info(f"问题回答完成，总用时: {response_time:.2f}秒")
            return response
            
        except Exception as e:
            logger.error(f"问题回答失败: {e}")
            
            # 创建错误响应
            error_response = OptimizedRAGResponse(
                question=question,
                answer=f"抱歉，回答过程中发生错误: {str(e)}",
                source_documents=[],
                retrieval_stats={},
                rerank_stats={},
                prompt_analysis={},
                response_time=time.time() - start_time
            )
            return error_response
    
    def _format_documents(self, documents: List[Document]) -> str:
        """格式化文档为上下文字符串"""
        if not documents:
            return ""
        
        formatted_parts = []
        for i, doc in enumerate(documents):
            # 获取文档来源信息
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            chapter_title = doc.metadata.get('chapter_title', '')
            
            # 构建文档标识
            doc_identifier = f"[文档{i+1}] {source}"
            if chapter_title:
                doc_identifier += f" - {chapter_title}"
            
            # 添加文档内容
            content = doc.page_content
            formatted_part = f"{doc_identifier}\n{content}\n"
            formatted_parts.append(formatted_part)
        
        return "\n".join(formatted_parts)
    
    def _get_retrieval_stats(self, question: str) -> Dict:
        """获取检索统计信息"""
        try:
            if self.advanced_retriever:
                return self.advanced_retriever.get_detailed_results(question)
            return {}
        except Exception as e:
            logger.warning(f"获取检索统计失败: {e}")
            return {}
    
    def _get_rerank_stats(self, question: str, original_docs: List[Document], reranked_docs: List[Document]) -> Dict:
        """获取重排序统计信息"""
        try:
            if self.enhanced_reranker:
                return self.enhanced_reranker.get_rerank_analysis(question, original_docs)
            return {}
        except Exception as e:
            logger.warning(f"获取重排序统计失败: {e}")
            return {}
    
    def _log_response_stats(self, question: str, answer: str, documents: List[Document], 
                           retrieval_stats: Dict, rerank_stats: Dict, response_time: float):
        """记录响应统计信息"""
        try:
            logger.info("📊 优化RAG系统响应统计:")
            logger.info(f"  问题: {question[:50]}...")
            logger.info(f"  响应时间: {response_time:.2f}秒")
            logger.info(f"  返回文档数: {len(documents)}")
            
            # 文档来源统计
            if documents:
                from collections import Counter
                sources = [os.path.basename(doc.metadata.get('source', 'unknown')) for doc in documents]
                source_counter = Counter(sources)
                logger.info(f"  文档来源分布: {dict(source_counter)}")
                
                # 显示优化特征
                enhanced_count = sum(1 for doc in documents if doc.metadata.get('enhanced', False))
                rrf_count = sum(1 for doc in documents if 'rrf_score' in doc.metadata)
                rerank_count = sum(1 for doc in documents if 'final_rerank_score' in doc.metadata)
                
                logger.info(f"  优化特征: 增强文档={enhanced_count}, RRF融合={rrf_count}, 重排序={rerank_count}")
            
            # 答案长度和质量指标
            logger.info(f"  答案长度: {len(answer)} 字符")
            
            # 是否使用了"未找到"回复
            uses_not_found = "根据现有文档，我无法找到" in answer
            logger.info(f"  使用未找到回复: {uses_not_found}")
            
        except Exception as e:
            logger.warning(f"记录响应统计失败: {e}")
    
    def get_system_performance_report(self) -> Dict:
        """获取系统性能报告"""
        try:
            report = {
                'system_config': {
                    'llm_model': self.llm_model,
                    'embedding_model': self.embedding_model,
                    'vector_weight': self.vector_weight,
                    'bm25_weight': self.bm25_weight,
                    'enable_reranking': self.enable_reranking,
                    'enable_chapter_splitting': self.enable_chapter_splitting,
                    'retrieval_k': self.retrieval_k
                },
                'components_status': {
                    'vectorstore_loaded': self.vectorstore is not None,
                    'advanced_retriever_ready': self.advanced_retriever is not None,
                    'enhanced_reranker_ready': self.enhanced_reranker is not None,
                    'enhanced_prompt_manager_ready': self.enhanced_prompt_manager is not None,
                    'rag_chain_built': self.rag_chain is not None
                },
                'optimization_features': {
                    'hybrid_retrieval_with_rrf': True,
                    'crossencoder_reranking': self.enable_reranking,
                    'entity_hit_reward': True,
                    'chapter_splitting_with_title_enhancement': self.enable_chapter_splitting,
                    'fewshot_prompt_optimization': True
                }
            }
            
            # 添加文档统计
            if self.vectorstore:
                try:
                    doc_count = self.vectorstore.get_document_count()
                    report['document_stats'] = {
                        'total_documents': doc_count,
                        'vectorstore_ready': doc_count > 0
                    }
                except:
                    report['document_stats'] = {'error': '无法获取文档统计'}
            
            return report
            
        except Exception as e:
            logger.error(f"获取系统性能报告失败: {e}")
            return {'error': str(e)}
    
    def run_optimization_test(self, test_questions: List[str]) -> Dict:
        """
        运行优化测试
        
        Args:
            test_questions: 测试问题列表
            
        Returns:
            测试结果报告
        """
        try:
            logger.info(f"开始运行优化测试，共 {len(test_questions)} 个问题...")
            
            test_results = []
            total_time = 0
            
            for i, question in enumerate(test_questions):
                logger.info(f"测试问题 {i+1}/{len(test_questions)}: {question[:30]}...")
                
                start_time = time.time()
                response = self.answer_question(question)
                end_time = time.time()
                
                question_time = end_time - start_time
                total_time += question_time
                
                # 分析结果
                result = {
                    'question': question,
                    'answer': response.answer,
                    'response_time': question_time,
                    'documents_count': len(response.source_documents),
                    'uses_enhanced_features': self._analyze_enhanced_features(response),
                    'prompt_analysis': response.prompt_analysis
                }
                
                test_results.append(result)
                logger.info(f"  完成，用时: {question_time:.2f}秒")
            
            # 生成测试报告
            report = {
                'test_summary': {
                    'total_questions': len(test_questions),
                    'total_time': total_time,
                    'average_time': total_time / len(test_questions),
                    'fastest_response': min(r['response_time'] for r in test_results),
                    'slowest_response': max(r['response_time'] for r in test_results)
                },
                'optimization_effectiveness': {
                    'questions_using_traffic_signal_template': sum(1 for r in test_results if r['prompt_analysis'].get('question_type') == 'traffic_signal_task'),
                    'questions_using_not_found_response': sum(1 for r in test_results if '根据现有文档，我无法找到' in r['answer']),
                    'average_documents_per_response': sum(r['documents_count'] for r in test_results) / len(test_results),
                    'questions_with_enhanced_documents': sum(1 for r in test_results if r['uses_enhanced_features']['enhanced_documents']),
                    'questions_with_rrf_fusion': sum(1 for r in test_results if r['uses_enhanced_features']['rrf_fusion']),
                    'questions_with_reranking': sum(1 for r in test_results if r['uses_enhanced_features']['reranking'])
                },
                'detailed_results': test_results
            }
            
            logger.info("优化测试完成")
            logger.info(f"  总用时: {total_time:.2f}秒")
            logger.info(f"  平均响应时间: {total_time/len(test_questions):.2f}秒")
            
            return report
            
        except Exception as e:
            logger.error(f"优化测试失败: {e}")
            return {'error': str(e)}
    
    def _analyze_enhanced_features(self, response: OptimizedRAGResponse) -> Dict:
        """分析响应中使用的增强特征"""
        features = {
            'enhanced_documents': False,
            'rrf_fusion': False,
            'reranking': False,
            'chapter_splitting': False,
            'entity_bonuses': False
        }
        
        for doc in response.source_documents:
            metadata = doc.metadata
            
            if metadata.get('enhanced', False):
                features['enhanced_documents'] = True
            
            if 'rrf_score' in metadata:
                features['rrf_fusion'] = True
            
            if 'final_rerank_score' in metadata:
                features['reranking'] = True
            
            if metadata.get('chapter_title'):
                features['chapter_splitting'] = True
            
            if 'entity_bonus' in metadata:
                features['entity_bonuses'] = True
        
        return features 

    # ==================== 兼容性方法 ====================
    # 为了与LangChainRAGSystem保持兼容性，添加以下方法

    def get_version_statistics(self) -> Dict:
        """获取版本管理统计信息"""
        try:
            if not self.vectorstore:
                return {"error": "向量存储未初始化"}
            
            return self.vectorstore.get_version_statistics()
            
        except Exception as e:
            logger.error(f"获取版本统计失败: {e}")
            return {"error": str(e)}
    
    def cleanup_knowledge_base(self) -> Dict:
        """清理知识库（移除孤立的版本信息）"""
        try:
            result = {"orphaned_removed": 0}
            
            if self.vectorstore:
                if hasattr(self.vectorstore, 'cleanup_orphaned_documents'):
                    orphaned_count = self.vectorstore.cleanup_orphaned_documents()
                    result["orphaned_removed"] = orphaned_count
                    
                    if orphaned_count > 0:
                        logger.info(f"清理了 {orphaned_count} 个孤立的文档版本")
            
            return result
            
        except Exception as e:
            logger.error(f"清理知识库失败: {e}")
            return {"error": str(e)}
    
    def rebuild_knowledge_base(self, progress_callback: Optional[Callable] = None) -> bool:
        """完全重建知识库（清空现有数据并重新构建）"""
        try:
            logger.info("开始重建知识库...")
            
            # 清空向量存储
            if self.vectorstore:
                self.vectorstore.clear_vectorstore()
                
                # 清理版本信息
                if hasattr(self.vectorstore, 'version_manager') and self.vectorstore.version_manager:
                    self.vectorstore.version_manager.reset_all()
            
            # 查找所有PDF文件
            pdf_files = []
            for directory in ["./data", "./docs", "./"]:
                if os.path.exists(directory):
                    import glob
                    pattern = os.path.join(directory, "*.pdf")
                    found_files = glob.glob(pattern)
                    if found_files:
                        pdf_files.extend(found_files)
                        logger.info(f"在 {directory} 中找到 {len(found_files)} 个PDF文件")
                        break
            
            if not pdf_files:
                logger.warning("未找到PDF文件进行重建")
                return False
            
            logger.info(f"开始重建，处理 {len(pdf_files)} 个PDF文件")
            
            # 使用目录路径重新加载（因为文件在同一目录）
            directory_path = os.path.dirname(pdf_files[0]) if pdf_files else None
            if directory_path:
                success = self.load_documents(directory_path=directory_path, progress_callback=progress_callback)
            else:
                # 如果文件在不同目录，逐个加载
                success = True
                for pdf_file in pdf_files:
                    file_success = self.load_documents(file_path=pdf_file, progress_callback=progress_callback)
                    if not file_success:
                        success = False
            
            if success:
                logger.info("知识库重建完成")
                return True
            else:
                logger.error("知识库重建失败")
                return False
            
        except Exception as e:
            logger.error(f"重建知识库失败: {e}")
            return False
    
    def _rebuild_retriever(self):
        """重建检索器（重建知识库后调用）"""
        try:
            if self.vectorstore and hasattr(self.vectorstore, 'get_document_count'):
                doc_count = self.vectorstore.get_document_count()
                if doc_count > 0:
                    # 重新初始化高级检索器，但需要文档列表
                    logger.info("重建知识库完成，需要重新加载文档以完成检索器初始化")
                else:
                    logger.warning("重建后的向量存储为空")
        except Exception as e:
            logger.warning(f"重建检索器失败: {e}")
    
    def load_documents_incremental(self, file_path: str = None, 
                                 directory_path: str = None,
                                 progress_callback: Optional[Callable] = None, 
                                 force_rebuild: bool = False) -> bool:
        """增量加载文档（兼容性方法）"""
        try:
            logger.info("执行增量文档加载...")
            
            # 构建要加载的文件列表
            files_to_load = []
            if file_path:
                files_to_load.append(file_path)
            
            if directory_path and os.path.exists(directory_path):
                import glob
                pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
                files_to_load.extend(pdf_files)
            
            if not files_to_load:
                logger.warning("没有找到要加载的文件")
                return False
            
            # 优化RAG系统暂时将增量加载视为完整加载
            # 因为高级混合检索器需要完整的文档集合来构建索引
            if len(files_to_load) == 1:
                return self.load_documents(file_path=files_to_load[0], progress_callback=progress_callback)
            else:
                return self.load_documents(directory_path=directory_path, progress_callback=progress_callback)
            
        except Exception as e:
            logger.error(f"增量加载文档失败: {e}")
            return False
    
    def extract_information(self) -> Dict:
        """提取竞赛信息（兼容性方法）"""
        try:
            logger.info("提取竞赛信息...")
            
            # 基础竞赛信息
            competitions = {}
            
            if self.vectorstore:
                try:
                    # 尝试搜索已知的竞赛文档
                    known_competitions = [
                        "未来校园智能应用专项赛",
                        "人工智能综合创新专项赛", 
                        "智慧城市主题设计专项赛",
                        "生成式人工智能应用专项赛"
                    ]
                    
                    for comp_name in known_competitions:
                        docs = self.vectorstore.similarity_search(comp_name, k=3)
                        if docs:
                            competitions[comp_name] = {
                                "documents_found": len(docs),
                                "sources": list(set(os.path.basename(doc.metadata.get('source', 'unknown')) for doc in docs))
                            }
                    
                    logger.info(f"找到 {len(competitions)} 个竞赛的相关信息")
                    
                except Exception as e:
                    logger.warning(f"提取竞赛信息时出错: {e}")
            
            return {
                "competitions": competitions,
                "total_competitions": len(competitions),
                "extraction_method": "optimized_rag_system"
            }
            
        except Exception as e:
            logger.error(f"提取竞赛信息失败: {e}")
            return {"error": str(e)}
    
    # ==================== 文档监控兼容性方法 ====================
    # 注意：优化RAG系统暂不支持文档监控，返回适当的默认值
    
    @property
    def document_watcher(self):
        """文档监控器属性（兼容性）"""
        return None
    
    def init_document_watcher(self, config=None) -> bool:
        """初始化文档监控器（兼容性方法）"""
        logger.info("优化RAG系统暂不支持文档监控功能")
        return False
    
    def start_document_watching(self) -> bool:
        """开始文档监控（兼容性方法）"""
        logger.info("优化RAG系统暂不支持文档监控功能")
        return False
    
    def stop_document_watching(self) -> bool:
        """停止文档监控（兼容性方法）"""
        logger.info("优化RAG系统暂不支持文档监控功能")
        return False
    
    def check_documents_now(self) -> Dict:
        """立即检查文档变更（兼容性方法）"""
        logger.info("优化RAG系统暂不支持文档监控功能")
        return {"supported": False, "message": "优化RAG系统暂不支持文档监控"}
    
    def get_watch_status(self) -> Dict:
        """获取监控状态（兼容性方法）"""
        return {
            "is_watching": False,
            "supported": False,
            "message": "优化RAG系统暂不支持文档监控"
        }
    
    def get_monitored_files(self) -> List[str]:
        """获取监控文件列表（兼容性方法）"""
        return [] 