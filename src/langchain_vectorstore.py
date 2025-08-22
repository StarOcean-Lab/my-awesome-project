"""
LangChain向量存储系统
使用LangChain FAISS向量存储和Ollama中文embedding模型
"""

import os
import requests
import json
import time
from typing import List, Dict, Optional, Tuple, Callable

# 修复相对导入问题
try:
    from .document_version_manager import DocumentVersionManager
except ImportError:
    try:
        from document_version_manager import DocumentVersionManager
    except ImportError:
        # 如果无法导入，设置为None，后续代码会处理这种情况
        DocumentVersionManager = None

# 修复导入 - 使用兼容的导入方式
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain.vectorstores import FAISS
    except ImportError:
        FAISS = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except ImportError:
            HuggingFaceEmbeddings = None

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

from loguru import logger


class OllamaEmbeddings(Embeddings):
    
    def __init__(self, model: str = "./bge-large-zh-v1.5", base_url: str = "http://localhost:11434"):
        """
        初始化Ollama Embeddings
        
        Args:
            model: 嵌入模型名称，可以是本地路径（如'./bge-large-zh-v1.5'）或Ollama模型名（如'nomic-embed-text:latest'）
            base_url: Ollama服务器地址
        """
        self.model = model
        self.base_url = base_url
        self.embed_url = f"{base_url}/api/embeddings"
        logger.info(f"初始化OllamaEmbeddings，模型: {model}, 服务器: {base_url}")
    
    def _call_ollama_embedding(self, text: str) -> List[float]:
        """调用Ollama API获取单个文本的embedding"""
        try:
            # 确保text是字符串类型
            if not isinstance(text, str):
                text = str(text)
            
            # 清理和验证输入文本
            text = text.strip()
            if not text:
                logger.warning("空文本传入embedding，使用默认文本")
                text = "默认文本"
            
            payload = {
                "model": self.model,
                "prompt": text  # 确保这里是字符串
            }
            
            response = requests.post(
                self.embed_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "embedding" in result:
                    embedding = result["embedding"]
                    # 验证embedding是否为有效的浮点数列表
                    if isinstance(embedding, list) and len(embedding) > 0:
                        return [float(x) for x in embedding]
                    else:
                        logger.error(f"Ollama返回的embedding格式无效: {embedding}")
                        return []
                else:
                    logger.error(f"Ollama响应中没有embedding字段: {result}")
                    return []
            else:
                logger.error(f"Ollama API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API请求异常: {e}")
            return []
        except Exception as e:
            logger.error(f"调用Ollama embedding失败: {e}")
            return []
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        total_texts = len(texts)
        logger.info(f"🚀 开始Ollama向量化：{total_texts}个文档片段")
        
        embeddings = []
        start_time = time.time()
        
        for i, text in enumerate(texts):
            # 每个文档都显示进度（对于大批量很重要）
            if i % 5 == 0 or i == total_texts - 1:  # 每5个文档显示一次进度
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time = elapsed / i
                    remaining = (total_texts - i) * avg_time
                    print(f"\r📊 Ollama向量化进度: {i+1}/{total_texts} ({(i+1)/total_texts*100:.1f}%) "
                          f"⏱️ 已用时: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s", end="", flush=True)
                else:
                    print(f"\r📊 Ollama向量化进度: {i+1}/{total_texts} ({(i+1)/total_texts*100:.1f}%)", end="", flush=True)
            
            embedding = self._call_ollama_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                # 如果某个文档embedding失败，使用空向量作为降级
                logger.warning(f"文档{i+1}的embedding失败，使用空向量")
                if embeddings:  # 如果之前有成功的embedding，使用相同维度的零向量
                    dim = len(embeddings[0])
                    embeddings.append([0.0] * dim)
                else:  # 否则使用默认维度
                    embeddings.append([0.0] * 1024)  # BGE-large-zh-v1.5通常是1024维
        
        total_time = time.time() - start_time
        print()  # 换行
        logger.info(f"✅ 完成Ollama向量化：{total_texts}个文档，总用时: {total_time:.1f}秒，平均: {total_time/total_texts:.2f}秒/文档")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        # 确保text是字符串类型
        if not isinstance(text, str):
            text = str(text)
            
        embedding = self._call_ollama_embedding(text)
        if not embedding:
            logger.warning("查询embedding失败，使用空向量")
            # 使用与文档embedding相同的维度
            return [0.0] * 1024  # BGE-large-zh-v1.5通常是1024维
        
        # 验证embedding维度
        if not isinstance(embedding, list) or len(embedding) == 0:
            logger.warning("无效的embedding格式，使用空向量")
            return [0.0] * 1024
            
        return embedding
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """使embedding对象可调用，兼容FAISS接口"""
        return self.embed_documents(texts)


class SimpleTextSplitter:
    """简单的文本分割器（降级方案）"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # 自己定义的文本分割器，用于将文档分割成更小的块
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        split_docs = []
        
        for doc in documents:
            text = doc.page_content
            
            # 简单分割
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                
                if chunk.strip():
                    new_doc = Document(
                        page_content=chunk.strip(),
                        metadata=doc.metadata.copy()
                    )
                    split_docs.append(new_doc)
                
                start = end - self.chunk_overlap
                
                if start >= len(text):
                    break
        
        return split_docs


class SimpleEmbeddings(Embeddings):
    """简单的embedding（降级方案），兼容FAISS接口"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.bert_model = None
        self._load_model()
    
    def _load_model(self):
        """加载embedding模型"""
        try:
            # 首先尝试sentence-transformers
            from sentence_transformers import SentenceTransformer
            import torch
            
            if os.path.exists(self.model_path):
                # PyTorch 2.7.1 兼容性修复
                logger.info(f"开始加载本地BGE模型: {self.model_path}")
                
                # 设置PyTorch兼容性选项
                original_load_state_dict = torch.nn.Module.load_state_dict
                
                def patched_load_state_dict(self, state_dict, strict=True):
                    """修复PyTorch 2.7.1 meta tensor问题"""
                    try:
                        # 如果遇到meta tensor问题，使用to_empty()方法
                        if hasattr(self, '_meta_registrations'):
                            for name, param in self.named_parameters():
                                if param.is_meta:
                                    param.data = param.to_empty(device='cpu')
                            for name, buffer in self.named_buffers():
                                if buffer.is_meta:
                                    buffer.data = buffer.to_empty(device='cpu')
                        return original_load_state_dict(self, state_dict, strict)
                    except Exception as e:
                        logger.warning(f"Meta tensor修复失败，尝试标准加载: {e}")
                        return original_load_state_dict(self, state_dict, strict)
                
                # 临时替换加载方法
                torch.nn.Module.load_state_dict = patched_load_state_dict
                
                try:
                    # 设置设备为CPU，避免GPU相关问题
                    self.model = SentenceTransformer(
                        self.model_path,
                        device='cpu',  # 强制使用CPU
                        cache_folder=None  # 避免缓存问题
                    )
                    logger.info(f"✅ 成功加载本地BGE模型: {self.model_path}")
                    return
                finally:
                    # 恢复原始加载方法
                    torch.nn.Module.load_state_dict = original_load_state_dict
                    
            else:
                logger.warning(f"本地模型路径不存在: {self.model_path}")
                # 尝试加载预训练模型
                try:
                    self.model = SentenceTransformer(
                        'all-MiniLM-L6-v2',
                        device='cpu'
                    )
                    logger.warning(f"本地模型不存在，使用后备模型: all-MiniLM-L6-v2")
                    return
                except Exception as e:
                    logger.warning(f"无法加载预训练sentence-transformers模型: {e}")
                    
        except Exception as e:
            logger.warning(f"sentence-transformers加载失败: {e}")
        
        # 如果sentence-transformers失败，尝试transformers库
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            logger.info("尝试使用transformers库作为后备...")
            
            # 设置离线模式，避免网络下载
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            # 如果本地有BGE模型配置，尝试使用
            if os.path.exists(self.model_path):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path, 
                        local_files_only=True
                    )
                    self.bert_model = AutoModel.from_pretrained(
                        self.model_path, 
                        local_files_only=True,
                        torch_dtype=torch.float32  # 强制使用float32避免精度问题
                    )
                    logger.info(f"✅ 使用transformers库加载本地BGE模型: {self.model_path}")
                    logger.info(f"📋 模型详情: BGE-large-zh-v1.5 (1024维，通过transformers库加载)")
                    return
                except Exception as e:
                    logger.warning(f"transformers加载本地模型失败: {e}")
            
            # 如果本地模型失败，尝试使用中文BERT作为后备
            self.tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-chinese',
                local_files_only=False  # 允许下载
            )
            self.bert_model = AutoModel.from_pretrained(
                'bert-base-chinese',
                local_files_only=False
            )
            logger.info("使用transformers库的BERT模型作为后备")
        except Exception as e2:
            logger.error(f"transformers库也加载失败: {e2}")
            logger.warning("将使用随机向量作为最终降级方案")
            # 确保所有模型属性都是None
            self.model = None
            self.tokenizer = None
            self.bert_model = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        total_texts = len(texts)
        
        if self.model:
            try:
                logger.info(f"🚀 开始BGE模型向量化：{total_texts}个文档片段")
                start_time = time.time()
                
                # sentence-transformers可以批量处理，但我们添加进度监控
                print(f"📊 BGE向量化进度: 0/{total_texts} (0.0%) ⏱️ 准备中...", end="", flush=True)
                
                embeddings = self.model.encode(texts, show_progress_bar=True)
                
                total_time = time.time() - start_time
                print(f"\r📊 BGE向量化进度: {total_texts}/{total_texts} (100.0%) ⏱️ 完成！                    ")
                logger.info(f"✅ 完成BGE向量化：{total_texts}个文档，总用时: {total_time:.1f}秒，平均: {total_time/total_texts:.3f}秒/文档")
                
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"sentence-transformers编码失败: {e}")
        
        # 降级到transformers
        if hasattr(self, 'tokenizer') and hasattr(self, 'bert_model') and self.tokenizer is not None and self.bert_model is not None:
            try:
                import torch
                # 判断使用的是BGE模型还是BERT模型
                model_info = "BGE-large-zh-v1.5" if self.model_path.endswith("bge-large-zh-v1.5") else "BERT"
                logger.info(f"🚀 开始{model_info}模型向量化：{total_texts}个文档片段 (通过transformers库)")
                
                embeddings = []
                start_time = time.time()
                
                for i, text in enumerate(texts):
                    # 显示详细进度
                    if i % 10 == 0 or i == total_texts - 1:
                        elapsed = time.time() - start_time
                        if i > 0:
                            avg_time = elapsed / i
                            remaining = (total_texts - i) * avg_time
                            print(f"\r📊 {model_info}向量化进度: {i+1}/{total_texts} ({(i+1)/total_texts*100:.1f}%) "
                                  f"⏱️ 已用时: {elapsed:.1f}s, 预计剩余: {remaining:.1f}s", end="", flush=True)
                        else:
                            print(f"\r📊 {model_info}向量化进度: {i+1}/{total_texts} ({(i+1)/total_texts*100:.1f}%)", end="", flush=True)
                    
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                        # 使用[CLS] token的embedding
                        embedding = outputs.last_hidden_state[0][0].numpy().tolist()
                        embeddings.append(embedding)
                
                total_time = time.time() - start_time
                print()  # 换行
                logger.info(f"✅ 完成{model_info}向量化：{total_texts}个文档，总用时: {total_time:.1f}秒")
                return embeddings
            except Exception as e:
                logger.error(f"transformers编码失败: {e}")
        
        # 最终降级方案：返回随机向量
        import random
        dim = 384  # 使用较小的维度
        logger.warning("使用随机向量作为最终降级方案")
        return [[random.random() for _ in range(dim)] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        result = self.embed_documents([text])
        return result[0] if result else []


class LangChainVectorStore:
    """基于LangChain的向量存储系统"""
    
    def __init__(self, 
                 model_name: str = "./bge-large-zh-v1.5", 
                 ollama_base_url: str = "http://localhost:11434",
                 persist_path: str = "./vectorstore",
                 enable_versioning: bool = True):
        """
        初始化LangChain向量存储
        
        Args:
            model_name: 嵌入模型名称（本地路径如'./bge-large-zh-v1.5'或Ollama模型名如'nomic-embed-text:latest'）
            ollama_base_url: Ollama服务器地址（仅在使用Ollama模型时使用）
            persist_path: 向量存储持久化路径
            enable_versioning: 是否启用版本管理
        """
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.persist_path = persist_path
        self.enable_versioning = enable_versioning
        
        # 初始化embedding模型 - 智能选择本地或远程模型
        logger.info(f"初始化embedding模型: {model_name}")
        
        # 检查是否为本地模型路径
        if model_name.startswith('./') or model_name.startswith('/') or os.path.exists(model_name):
            # 使用本地嵌入模型
            logger.info(f"检测到本地模型路径，使用本地嵌入模型: {model_name}")
            try:
                if HuggingFaceEmbeddings is not None:
                    # 优先使用HuggingFaceEmbeddings，添加PyTorch 2.7.1兼容性
                    try:
                        import torch
                        # 检查PyTorch版本，如果是2.x则添加兼容性处理
                        pytorch_version = torch.__version__
                        logger.info(f"检测到PyTorch版本: {pytorch_version}")
                        
                        # 为PyTorch 2.x添加特殊处理
                        if pytorch_version.startswith('2.'):
                            logger.info("检测到PyTorch 2.x，启用兼容性模式")
                            # 设置环境变量避免meta tensor问题
                            os.environ['PYTORCH_DISABLE_LAZY_MODULE'] = '1'
                            
                        self.embeddings = HuggingFaceEmbeddings(
                            model_name=model_name,
                            model_kwargs={
                                'device': 'cpu',  # 使用CPU，避免GPU内存问题
                                'torch_dtype': torch.float32  # 强制使用float32
                            },
                            encode_kwargs={'normalize_embeddings': True}  # 归一化嵌入向量
                        )
                        logger.info("✅ 使用HuggingFaceEmbeddings加载本地BGE模型")
                        
                    except Exception as hf_error:
                        logger.warning(f"HuggingFaceEmbeddings加载失败: {hf_error}")
                        # 降级到SimpleEmbeddings
                        self.embeddings = SimpleEmbeddings(model_path=model_name)
                        logger.info("使用SimpleEmbeddings作为降级方案")
                else:
                    # 降级使用SimpleEmbeddings
                    self.embeddings = SimpleEmbeddings(model_path=model_name)
                    logger.info("HuggingFaceEmbeddings不可用，使用SimpleEmbeddings加载本地模型")
            except Exception as e:
                logger.warning(f"本地模型加载失败: {e}，尝试使用SimpleEmbeddings")
                self.embeddings = SimpleEmbeddings(model_path=model_name)
        else:
            # 使用Ollama远程嵌入模型
            logger.info(f"使用Ollama嵌入模型: {model_name}")
            self.embeddings = OllamaEmbeddings(model=model_name, base_url=ollama_base_url)
        
        # 初始化文本分割器
        if RecursiveCharacterTextSplitter is not None:
            try:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len,
                )
                logger.info("使用RecursiveCharacterTextSplitter")
            except Exception as e:
                logger.warning(f"RecursiveCharacterTextSplitter初始化失败: {e}")
                self.text_splitter = SimpleTextSplitter()
        else:
            self.text_splitter = SimpleTextSplitter()
        
        self.vectorstore = None
        
        # 初始化版本管理器
        if self.enable_versioning and DocumentVersionManager is not None:
            version_file = os.path.join(os.path.dirname(persist_path), "document_versions.json")
            self.version_manager = DocumentVersionManager(version_file)
            logger.info("✅ 文档版本管理器已启用")
        else:
            self.version_manager = None
            if self.enable_versioning and DocumentVersionManager is None:
                logger.warning("⚠️ DocumentVersionManager无法导入，版本管理功能已禁用")
            
        logger.info(f"LangChain向量存储初始化完成，embedding模型: {model_name}")
    
    def load_vectorstore(self) -> bool:
        """加载已保存的向量存储"""
        try:
            if os.path.exists(self.persist_path) and FAISS is not None:
                # 检查FAISS加载方法
                if hasattr(FAISS, 'load_local'):
                    self.vectorstore = FAISS.load_local(
                        self.persist_path, 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"成功加载向量存储，文档数量: {self.get_document_count()}")
                    return True
                else:
                    logger.warning("FAISS.load_local方法不可用")
                    return False
            else:
                logger.warning("向量存储文件不存在或FAISS不可用")
                return False
        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")
            return False
    
    def add_documents(self, documents: List[Document], progress_callback: Optional[Callable] = None, force_update: bool = False) -> None:
        """
        添加文档到向量存储（支持增量更新和去重）
        
        Args:
            documents: LangChain Document对象列表
            progress_callback: 进度回调函数（暂时还用不到，预留的接口）
            force_update: 是否强制更新（忽略版本检查）
        """
        try:
            # 分割文档
            if progress_callback:
                # 模拟进度回调
                from .progress_manager import ProgressInfo, ProgressStatus
                progress_info = ProgressInfo(
                    current_step=1,
                    total_steps=5,
                    step_name="文本分块",
                    description="正在分割文档...",
                    status=ProgressStatus.RUNNING
                )
                progress_callback(progress_info)
            
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档分割完成，chunk数量: {len(split_docs)}")
            
            if not split_docs:
                logger.warning("没有文档片段可添加")
                return
            
            # 版本管理和去重处理
            if self.enable_versioning and self.version_manager and not force_update:
                if progress_callback:
                    progress_info = ProgressInfo(
                        current_step=2,
                        total_steps=5,
                        step_name="版本检查",
                        description="检查文档版本和去重...",
                        status=ProgressStatus.RUNNING
                    )
                    progress_callback(progress_info)
                
                split_docs = self._process_incremental_update(split_docs)
                
                if not split_docs:
                    logger.info("所有文档都是最新版本，无需更新")
                    if progress_callback:
                        progress_info = ProgressInfo(
                            current_step=5,
                            total_steps=5,
                            step_name="完成",
                            description="无需更新",
                            status=ProgressStatus.COMPLETED
                        )
                        progress_callback(progress_info)
                    return
            
            # 向量化处理
            chunk_count = len(split_docs)
            if progress_callback:
                progress_info = ProgressInfo(
                    current_step=3,
                    total_steps=5,
                    step_name="向量化",
                    description=f"正在生成 {chunk_count} 个文档片段的向量...",
                    status=ProgressStatus.RUNNING
                )
                progress_callback(progress_info)
            
            # 显示向量化开始信息
            logger.info(f"🎯 开始向量化处理：{chunk_count} 个文档片段")
            vectorization_start = time.time()
            
            if FAISS is not None:
                # 如果向量存储为空，则创建新的向量存储
                if self.vectorstore is None:
                    # 创建新的向量存储
                    logger.info("📚 创建新的FAISS向量存储...")
                    self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
                    logger.info("✅ 创建新的FAISS向量存储完成")
                else:
                    # 如果向量存储不为空，则添加到现有向量存储
                    if hasattr(self.vectorstore, 'add_documents'):
                        logger.info("📚 添加文档到现有FAISS向量存储...")
                        self.vectorstore.add_documents(split_docs)
                        logger.info("✅ 文档添加到现有向量存储完成")
                    else:
                        # 降级方案：重新创建向量存储
                        logger.warning("使用降级方案重新创建向量存储")
                        existing_docs = []  # 这里应该获取现有文档，但为简化就重新创建
                        all_docs = existing_docs + split_docs
                        self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
                        logger.info("✅ 降级方案向量存储创建完成")
            else:
                logger.error("FAISS不可用，无法创建向量存储")
                raise ImportError("FAISS库不可用")
            
            vectorization_time = time.time() - vectorization_start
            logger.info(f"🎉 向量化处理完成！总用时: {vectorization_time:.1f}秒，处理了 {chunk_count} 个文档片段")
            
            # 更新版本信息
            if progress_callback:
                progress_info = ProgressInfo(
                    current_step=4,
                    total_steps=5,
                    step_name="更新版本",
                    description="更新文档版本信息...",
                    status=ProgressStatus.RUNNING
                )
                progress_callback(progress_info)
            
            # 更新版本管理器
            if self.enable_versioning and self.version_manager:
                self._update_version_info(split_docs)
            
            # 保存向量存储到磁盘
            logger.info("💾 保存向量存储到磁盘...")
            self.save_vectorstore()
            
            # 完成
            if progress_callback:
                progress_info = ProgressInfo(
                    current_step=5,
                    total_steps=5,
                    step_name="完成",
                    description="向量存储完成并已保存",
                    status=ProgressStatus.COMPLETED
                )
                progress_callback(progress_info)
                
        except Exception as e:
            logger.error(f"添加文档失败: {type(e).__name__}: {str(e)}")
            logger.error(f"异常详情: {repr(e)}")
            import traceback
            logger.error(f"堆栈跟踪: {traceback.format_exc()}")
            if progress_callback:
                from .progress_manager import ProgressInfo, ProgressStatus
                progress_info = ProgressInfo(
                    current_step=0,
                    total_steps=3,
                    step_name="错误",
                    description="向量存储失败",
                    status=ProgressStatus.FAILED,
                    error_message=str(e)
                )
                progress_callback(progress_info)
            raise
    
    def save_vectorstore(self) -> None:
        """保存向量存储到磁盘"""
        try:
            if self.vectorstore and hasattr(self.vectorstore, 'save_local'):
                # 确保目录存在
                os.makedirs(os.path.dirname(self.persist_path) if os.path.dirname(self.persist_path) else '.', exist_ok=True)
                self.vectorstore.save_local(self.persist_path)
                logger.info(f"向量存储已保存到: {self.persist_path}")
            else:
                logger.warning("向量存储为空或保存方法不可用，无法保存")
        except Exception as e:
            logger.error(f"保存向量存储失败: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        try:
            if self.vectorstore is None:
                logger.warning("向量存储未初始化")
                return []
                
            docs = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"相似度搜索完成，查询: {query[:50]}..., 返回: {len(docs)}个结果")
            return docs
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        带分数的相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (文档, 分数)元组列表
        """
        try:
            if self.vectorstore is None:
                logger.warning("向量存储未初始化")
                return []
            
            # 确保query是字符串类型
            if not isinstance(query, str):
                query = str(query)
                
            # 预先生成查询向量以验证
            query_embedding = self.embeddings.embed_query(query)
            if not query_embedding or len(query_embedding) == 0:
                logger.warning("查询向量生成失败，使用普通搜索")
                docs = self.similarity_search(query, k)
                return [(doc, 0.0) for doc in docs]
                
            # 检查方法是否存在
            if hasattr(self.vectorstore, 'similarity_search_with_score'):
                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
                logger.info(f"带分数相似度搜索完成，查询: {query[:50]}..., 返回: {len(docs_with_scores)}个结果")
                return docs_with_scores
            else:
                # 降级方案：使用普通搜索
                logger.warning("similarity_search_with_score方法不可用，使用普通搜索")
                docs = self.similarity_search(query, k)
                return [(doc, 0.0) for doc in docs]
            
        except Exception as e:
            logger.error(f"带分数相似度搜索失败: {e}")
            # 降级到普通搜索
            try:
                docs = self.similarity_search(query, k)
                return [(doc, 0.0) for doc in docs]
            except Exception as e2:
                logger.error(f"降级搜索也失败: {e2}")
                return []
    
    def get_vectorstore(self) -> Optional:
        """获取向量存储对象"""
        return self.vectorstore
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        try:
            if self.vectorstore and hasattr(self.vectorstore, 'index') and hasattr(self.vectorstore.index, 'ntotal'):
                return self.vectorstore.index.ntotal
            else:
                # 降级方案：尝试其他方法获取数量
                if self.vectorstore and hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
                    return len(self.vectorstore.docstore._dict)
                else:
                    # 向量存储为空时不记录警告，直接返回0
                    # logger.warning("无法获取确切的文档数量")  # 注释掉频繁的警告
                    return 0
        except Exception as e:
            logger.debug(f"获取文档数量失败: {e}")  # 改为debug级别，避免日志spam
            return 0
    
    def clear_vectorstore(self) -> None:
        """清空向量存储"""
        self.vectorstore = None
        logger.info("向量存储已清空")
    
    def _process_incremental_update(self, documents: List[Document]) -> List[Document]:
        """
        处理增量更新，过滤出需要更新的文档
        
        Args:
            documents: 所有文档列表
            
        Returns:
            需要更新的文档列表
        """
        if not self.version_manager:
            return documents
        
        # 按文件路径分组
        docs_by_file = {}
        for doc in documents:
            file_path = doc.metadata.get('file_path', doc.metadata.get('source', ''))
            if file_path not in docs_by_file:
                docs_by_file[file_path] = []
            docs_by_file[file_path].append(doc)
        
        new_documents = []
        updated_files = []
        
        for file_path, file_docs in docs_by_file.items():
            if not file_path or file_path == 'unknown':
                # 无法识别来源的文档，直接添加
                new_documents.extend(file_docs)
                continue
            
            # 检查文档是否发生变化
            if self.version_manager.is_document_changed(file_path, file_docs):
                # 文档有变化，获取新增的chunks
                new_chunks = self.version_manager.get_new_chunks(file_path, file_docs)
                new_documents.extend(new_chunks)
                updated_files.append(file_path)
            else:
                logger.debug(f"文档未变化，跳过: {file_path}")
        
        if updated_files:
            logger.info(f"检测到 {len(updated_files)} 个文件需要更新: {updated_files}")
        
        return new_documents
    
    def _update_version_info(self, documents: List[Document]):
        """
        更新文档版本信息
        
        Args:
            documents: 文档列表
        """
        if not self.version_manager:
            return
        
        # 按文件路径分组
        docs_by_file = {}
        for doc in documents:
            file_path = doc.metadata.get('file_path', doc.metadata.get('source', ''))
            if file_path and file_path != 'unknown':
                if file_path not in docs_by_file:
                    docs_by_file[file_path] = []
                docs_by_file[file_path].append(doc)
        
        # 更新每个文件的版本信息
        for file_path, file_docs in docs_by_file.items():
            try:
                self.version_manager.update_document_version(file_path, file_docs)
            except Exception as e:
                logger.error(f"更新文档版本失败 {file_path}: {e}")
    
    def rebuild_vectorstore(self, documents: List[Document] = None, progress_callback: Optional[Callable] = None) -> bool:
        """
        重建向量存储（清空现有数据并重新构建）
        
        Args:
            documents: 文档列表（如果为None则重新加载所有文档）
            progress_callback: 进度回调函数
            
        Returns:
            是否成功重建
        """
        try:
            logger.info("开始重建向量存储...")
            
            # 清空现有向量存储
            self.clear_vectorstore()
            
            # 重置版本管理器
            if self.version_manager:
                self.version_manager.reset_all()
                logger.info("版本信息已重置")
            
            # 如果没有提供文档，则需要重新加载
            if documents is None:
                logger.warning("重建向量存储需要提供文档列表")
                return False
            
            # 强制添加所有文档
            self.add_documents(documents, progress_callback=progress_callback, force_update=True)
            
            logger.info("向量存储重建完成")
            return True
            
        except Exception as e:
            logger.error(f"重建向量存储失败: {e}")
            return False
    
    def get_version_statistics(self) -> Dict:
        """获取版本管理统计信息"""
        if not self.version_manager:
            return {"versioning_enabled": False}
        
        stats = self.version_manager.get_statistics()
        stats["versioning_enabled"] = True
        stats["vectorstore_doc_count"] = self.get_document_count()
        
        return stats
    
    def cleanup_orphaned_documents(self) -> int:
        """清理孤立的文档版本信息"""
        if not self.version_manager:
            return 0
        
        return self.version_manager.cleanup_missing_files() 