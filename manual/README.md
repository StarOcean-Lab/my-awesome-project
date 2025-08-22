# LangChain+Ollama 实现泰迪杯竞赛智能客服机器人

## 📑 目录

1. [项目概述](#1-项目概述)
2. [技术架构](#2-技术架构)
3. [快速开始](#3-快速开始)
4. [环境安装](#4-环境安装)
5. [功能使用](#5-功能使用)
6. [Web界面指南](#6-web界面指南)
7. [Ollama集成](#7-ollama集成)
8. [高级配置](#8-高级配置)
9. [故障排除](#9-故障排除)
10. [开发扩展](#10-开发扩展)
11. [技术升级总结](#11-技术升级总结)
12. [附录](#12-附录)

---

## 1. 项目概述

### 🎯 项目简介

本项目是基于 **LangChain + Ollama** 的智能客服机器人，专为**2025年（第13届）"泰迪杯"数据挖掘挑战赛C题**开发。采用最新的检索增强生成（RAG）技术，结合本地化的大语言模型和向量检索，提供准确、实时的竞赛咨询服务。

### 🎭 核心特性

- **📚 智能问答**: 基于优化RAG架构的精确问题回答
- **🔍 混合检索**: BM25关键词 + 向量语义双重检索
- **🤖 本地LLM**: 集成Ollama本地大语言模型
- **📊 批量处理**: 支持Excel批量问答和信息提取
- **🌐 Web界面**: 友好的Streamlit交互界面
- **🔄 实时更新**: 支持知识库动态更新和版本管理

### 🚀 **NEW! 5项核心优化特性**

- **🎯 混合检索增强**: BM25精确短语强制召回 + 向量Top-K的RRF融合
- **🧠 Cross-Encoder重排**: ms-marco-MiniLM-L6-v2智能重排序
- **🏆 实体命中奖励**: 关键词命中在重排阶段额外加分
- **📄 文档切分优化**: 按章节切分PDF并将标题拼接到chunk开头
- **💡 Few-shot提示**: 智能上下文相关性检查和专业回复模板

### 🎯 主要功能

1. **竞赛数据整理** - 自动提取PDF文档中的竞赛基本信息
2. **智能问答系统** - 回答基础信息、统计分析、开放性问题
3. **知识库管理** - 实时更新、版本控制、自动备份
4. **批量处理** - Excel文件批量问答和结果导出

---

## 2. 技术架构

### 🏗️ 系统架构

**传统RAG架构**:
```
用户问题 → 文档加载器 → 向量化存储 → 混合检索器 → Ollama LLM → 生成回答
```

**🚀 优化RAG架构** (5层优化):
```
用户问题 → 增强文档加载器(标题拼接) → 向量化存储 → 
    ↓
混合检索器(BM25+向量+RRF融合) → Cross-Encoder重排 → 实体命中奖励 → 
    ↓  
Few-shot提示管理器 → Ollama LLM → 智能回答生成
```

### 📦 核心技术栈

- **LangChain**: 构建LLM应用的框架
- **Ollama**: 本地LLM部署平台  
- **FAISS**: 高性能向量检索
- **Streamlit**: Web界面框架
- **BGE-large-zh-v1.5**: 中文向量化模型
- **nomic-embed-text**: Ollama embedding模型

### 🔧 核心组件

#### 📄 LangChain文档加载器 (`src/langchain_document_loader.py`)
- 支持PyPDF和pdfplumber两种加载方式
- 自动处理中文PDF文档
- 提取结构化竞赛信息
- 完整的元数据管理

#### 🧮 LangChain向量存储 (`src/langchain_vectorstore.py`)  
- 基于LangChain FAISS实现
- 使用nomic-embed-text:latest进行向量化
- 支持持久化存储
- 自动文档分割和向量化

#### 🔍 LangChain混合检索器 (`src/langchain_retriever.py`)
- 继承LangChain BaseRetriever
- BM25关键词检索 + 向量语义检索
- 自适应权重调整
- 详细的检索结果分析

#### 🧠 LangChain RAG系统 (`src/langchain_rag.py`)
- 完整的LangChain RAG Chain实现
- 支持Ollama LLM集成
- 智能Prompt工程
- 批量处理功能

---

## 3. 快速开始

### ⚡ 一键启动

```bash
# 1. 启动Ollama服务
ollama serve

# 2. 拉取必需模型  
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text

# 3. 安装依赖
pip install -r requirements.txt
# 使用本地Python环境安装包
.\.conda\python.exe -m pip install -r requirements.txt

# 4. 启动系统
python run_langchain.py --mode web
```

**访问地址**: http://localhost:8501

### 🚀 启动优化模式

```bash
# Web界面（推荐）- 支持界面内切换优化模式
python run_langchain.py --mode web

# 命令行优化模式
python run_langchain.py --mode cli --use-optimized

# 优化系统专项测试
python run_langchain.py --mode optimized

# 集成测试（验证优化功能）
python test_integration.py
```

### 🎮 测试验证

```bash
# 运行完整测试
python test_langchain.py

# 检查环境
python run_langchain.py --check

# 测试Ollama连接
python -c "from langchain_chatbot import LangChainChatbot; LangChainChatbot()"
```

---

## 4. 环境安装

### 4.1 系统要求

- **Python**: 3.8+
- **内存**: 16GB+ (推荐32GB)
- **存储**: 20GB可用空间
- **GPU**: 可选，支持CUDA加速

### 4.2 安装Ollama

#### Windows安装
```bash
# 使用winget安装
winget install ollama

# 或访问官网下载：https://ollama.ai/download
```

#### Linux/macOS安装
```bash
# 一键安装
curl -fsSL https://ollama.ai/install.sh | sh
```

### 4.3 Python依赖安装

```bash
# 方式1：使用启动脚本
python run_langchain.py --install

# 方式2：直接安装
pip install -r requirements.txt

# 方式3：使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4.4 模型下载

```bash
# 启动Ollama服务
ollama serve

# 下载LLM模型（推理用）
ollama pull deepseek-r1:7b

# 下载Embedding模型（向量化用）  
ollama pull nomic-embed-text:latest

# 验证模型安装
ollama list
```

### 4.5 环境检查

```bash
# 完整环境检查
python run_langchain.py --check

# 检查Python版本
python --version

# 检查Ollama服务
curl http://localhost:11434/api/tags
```

---

## 5. 功能使用

### 5.1 竞赛数据整理

#### 准备PDF文档
```bash
# 将PDF文档放在项目根目录
01_竞赛1.pdf 至 18_竞赛18.pdf
```

#### 运行数据提取
```python
from langchain_chatbot import LangChainChatbot

chatbot = LangChainChatbot()
pdf_files = glob.glob("*.pdf")
chatbot.load_knowledge_base(pdf_files)

# 提取竞赛信息
chatbot.extract_competition_info("result_1.xlsx")
```

### 5.2 智能问答功能

#### 单问题回答
```python
# 初始化系统
chatbot = LangChainChatbot()
chatbot.load_knowledge_base(pdf_files)

# 问答测试
question = "第七届全国青少年人工智能创新挑战赛的报名时间是什么时候？"
result = chatbot.answer_question(question)

print(f"回答: {result['answer']}")
print(f"置信度: {result['confidence']:.2f}")
print(f"来源: {result['sources']}")
```

#### 问题类型
- **基础信息查询**: 时间、地点、要求等具体信息
- **统计分析查询**: 数量、分类等统计信息  
- **开放性问题**: 准备建议、策略指导等

### 5.3 批量问答处理

#### 准备问题文件
```python
# 创建测试问题
python create_test_questions.py

# 自定义问题Excel文件（需包含"问题"列）
questions = [
    "报名时间是什么时候？",
    "参赛要求是什么？",
    "如何准备竞赛？"
]
df = pd.DataFrame({'问题': questions})
df.to_excel('my_questions.xlsx', index=False)
```

#### 批量处理
```python
# 读取问题文件
df = pd.read_excel('my_questions.xlsx')
questions = df['问题'].tolist()

# 批量回答
results = chatbot.batch_answer_questions(questions, "batch_results.xlsx")
print(f"处理完成，共 {len(results)} 个问题")
```

### 5.4 文档监控与自动更新

系统具备自动文档监控功能，会在后台透明地运行：

#### 自动功能
- **自动检测**: 系统定期扫描PDF文件变更
- **自动更新**: 发现文件变更时自动重新加载知识库
- **透明运行**: 用户无需手动干预，所有更新过程在后台进行

#### 监控特性
- 实时检测文件修改、新增、删除
- 自动备份历史版本
- 支持多目录监控
- 智能去重与增量更新

---

## 6. Web界面指南

### 6.1 界面布局

#### 侧边栏功能
- **📊 系统管理**: 状态监控、参数配置
- **📚 知识库管理**: PDF加载、状态显示  
- **🔧 批量处理**: Excel文件批量问答
- **⚙️ 系统配置**: LLM和embedding模型配置

#### 主界面功能
- **💬 智能问答**: 实时聊天交互界面
- **📝 对话历史**: 完整的对话记录
- **📊 结果展示**: 答案、置信度、来源信息

### 6.2 操作流程

#### 首次使用流程
```
1. 启动系统 → 2. 加载PDF文件 → 3. 开始问答（系统自动检测文件变更）
```

#### 日常使用流程  
```
1. 输入问题 → 2. 查看回答 → 3. 检查来源 → 4. 查看历史
```

#### 批量处理流程
```
1. 上传问题Excel → 2. 设置输出文件名 → 3. 开始处理 → 4. 下载结果
```

### 6.3 界面配置

#### LLM配置
- **模型名称**: deepseek-r1:7b
- **Ollama地址**: http://localhost:11434  
- **温度参数**: 0.1（控制随机性）
- **最大令牌**: 2000

#### Embedding配置
- **模型名称**: nomic-embed-text:latest
- **向量维度**: 768维
- **分块大小**: 1000字符
- **重叠大小**: 200字符

---

## 7. Ollama集成

### 7.1 服务管理

#### 启动和停止
```bash
# 启动Ollama服务
ollama serve

# 后台运行
nohup ollama serve > ollama.log 2>&1 &

# 检查服务状态
ps aux | grep ollama
curl http://localhost:11434/api/tags
```

#### 端口配置
```bash
# 修改默认端口
export OLLAMA_HOST=0.0.0.0:11435
ollama serve

# 或在config.py中修改
OLLAMA_BASE_URL = "http://localhost:11435"
```

### 7.2 模型管理

#### 推荐模型配置
```bash
# 中文友好模型
ollama pull deepseek-r1:7b      # 智能推理（推荐）
ollama pull qwen2:7b            # 阿里千问2
ollama pull chatglm3:6b         # 智谱ChatGLM3

# Embedding模型
ollama pull nomic-embed-text    # 向量化（推荐）
ollama pull mxbai-embed-large   # 替代选择
```

#### 模型切换
```python
# 在代码中切换模型
chatbot = LangChainChatbot(
    llm_model="qwen2:7b",  # 更换LLM模型
    embedding_model="mxbai-embed-large"  # 更换embedding模型
)
```

### 7.3 性能优化

#### GPU加速
```bash
# 检查GPU支持
ollama run deepseek-r1:7b --gpu

# 设置GPU层数
export OLLAMA_NUM_GPU=1
```

#### 内存优化
```python
# config.py中调整参数
LLM_MAX_TOKENS = 1000      # 减少最大令牌数
LLM_TEMPERATURE = 0.1      # 降低随机性
LANGCHAIN_CHUNK_SIZE = 500 # 减少分块大小
```

#### 并发控制
```bash
# 限制并发模型数
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=2
```

---

## 8. 高级配置

### 8.1 系统配置

#### config.py主要配置项
```python
class Config:
    # LLM配置
    LLM_MODEL = "deepseek-r1:7b"
    OLLAMA_BASE_URL = "http://localhost:11434"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 2000
    LLM_TIMEOUT = 60
    
    # LangChain配置  
    LANGCHAIN_EMBEDDING_MODEL = "nomic-embed-text:latest"
    LANGCHAIN_VECTORSTORE_TYPE = "faiss"
    LANGCHAIN_RETRIEVER_K = 5
    LANGCHAIN_CHUNK_SIZE = 1000
    LANGCHAIN_CHUNK_OVERLAP = 200
    
    # 检索配置
    BM25_K1 = 1.5
    BM25_B = 0.75
    HYBRID_ALPHA = 0.5  # 混合检索权重
    
    # 分数阈值
    VECTOR_SCORE_THRESHOLD = 0.5
    BM25_SCORE_THRESHOLD = 0.1
```

### 8.2 检索优化

#### 问题类型策略
```python
QUESTION_TYPES = {
    "basic": {
        "alpha": 0.3,        # 更重视关键词匹配
        "vector_k": 15,
        "bm25_k": 25,
        "description": "基础信息查询"
    },
    "statistical": {
        "alpha": 0.4,        # 平衡策略
        "vector_k": 30,
        "bm25_k": 30,
        "description": "统计分析查询"
    },
    "open": {
        "alpha": 0.7,        # 更重视语义理解
        "vector_k": 25,
        "bm25_k": 15,
        "description": "开放性问题"
    }
}
```

#### 动态权重调整
```python
# 根据问题类型调整检索权重
def adjust_retrieval_weights(question_type):
    config = QUESTION_TYPES.get(question_type, QUESTION_TYPES["basic"])
    return {
        "alpha": config["alpha"],
        "vector_k": config["vector_k"], 
        "bm25_k": config["bm25_k"]
    }
```

### 8.3 提示词工程

#### 自定义提示词模板
```python
def _build_prompt(self, question: str, context: str) -> str:
    system_prompt = """你是一个专业的竞赛智能客服机器人，专门回答关于各类学科竞赛的问题。

请根据提供的竞赛文档内容，准确、详细地回答用户的问题。

回答要求：
1. 基于提供的文档内容进行回答，确保信息准确
2. 如果文档中没有相关信息，请明确说明"根据现有文档无法找到相关信息"
3. 回答要具体、有用，包含关键的时间、地点、要求等信息
4. 对于开放性问题，可以给出合理的建议和指导
5. 保持专业、友好的语调

文档内容：
{context}

用户问题：{question}

请提供准确、有用的回答："""

    return system_prompt.format(context=context, question=question)
```

### 8.4 并发处理

#### 多线程问答
```python
import concurrent.futures

def parallel_qa(questions):
    chatbot = LangChainChatbot()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(chatbot.answer_question, questions)
    
    return list(results)
```

#### 异步处理
```python
import asyncio

async def async_answer_question(chatbot, question):
    return await asyncio.to_thread(chatbot.answer_question, question)

async def batch_async_qa(questions):
    chatbot = LangChainChatbot()
    tasks = [async_answer_question(chatbot, q) for q in questions]
    results = await asyncio.gather(*tasks)
    return results
```

---

## 9. 故障排除

### 9.1 常见问题

#### Q1: Ollama连接失败
**错误**: `Ollama连接异常: Connection refused`

**解决方案**:
```bash
# 1. 检查服务状态
ps aux | grep ollama

# 2. 启动服务
ollama serve

# 3. 检查端口占用
netstat -an | grep 11434

# 4. 重启服务
pkill ollama
ollama serve
```

#### Q2: 模型下载失败
**错误**: `Error pulling model`

**解决方案**:
```bash
# 1. 检查网络连接
ping ollama.ai

# 2. 手动下载模型
ollama pull deepseek-r1:7b --verbose

# 3. 使用代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# 4. 检查磁盘空间
df -h
```

#### Q3: 内存不足
**错误**: `Out of memory` 或模型加载失败

**解决方案**:
```python
# 1. 调整配置参数
CHUNK_SIZE = 300           # 减少分块大小
MAX_CONTEXTS = 3           # 减少检索数量
LLM_MAX_TOKENS = 1000     # 减少令牌数

# 2. 使用更小的模型
LLM_MODEL = "chatglm3:6b"  # 6B替代7B模型

# 3. 限制并发
export OLLAMA_MAX_LOADED_MODELS=1
```

#### Q4: PDF解析失败
**错误**: `PDF processing failed`

**解决方案**:
```python
# 1. 检查PDF文件完整性
import PyPDF2
with open('file.pdf', 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    print(f"页数: {len(reader.pages)}")

# 2. 尝试不同解析器
# 在langchain_document_loader.py中切换解析方法

# 3. 重新生成PDF文件
# 确保PDF文件没有密码保护和损坏
```

#### Q5: 向量检索效果差
**解决方案**:
```python
# 1. 调整分块参数
LANGCHAIN_CHUNK_SIZE = 800      # 调整分块大小
LANGCHAIN_CHUNK_OVERLAP = 150   # 调整重叠大小

# 2. 增加检索数量
LANGCHAIN_RETRIEVER_K = 10      # 增加检索结果数

# 3. 调整混合权重
HYBRID_ALPHA = 0.7              # 更重视语义检索
```

### 9.2 日志调试

#### 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from loguru import logger
logger.add("debug.log", level="DEBUG")
```

#### 查看系统日志
```bash
# 实时查看日志
tail -f langchain_chatbot.log

# 查看错误日志
grep "ERROR" langchain_chatbot.log

# 查看特定组件日志
grep "LLM" langchain_chatbot.log
grep "Vector" langchain_chatbot.log
```

### 9.3 性能监控

#### 系统状态检查
```python
# 获取详细系统状态
chatbot = LangChainChatbot()
status = chatbot.get_system_status()
print(json.dumps(status, indent=2, ensure_ascii=False))
```

#### 资源使用监控
```bash
# 内存使用
htop
free -h

# GPU使用（如果有）
nvidia-smi

# 磁盘空间
df -h

# 网络连接
netstat -an | grep 11434
```

---

## 10. 开发扩展

### 10.1 添加新功能

#### 自定义问题类型
```python
# 在src/rag_system.py中添加新的问题分类逻辑
def classify_question_type(question: str) -> str:
    if "时间" in question or "什么时候" in question:
        return "basic"
    elif "多少" in question or "数量" in question:
        return "statistical"
    elif "如何" in question or "怎么" in question:
        return "open"
    else:
        return "basic"
```

#### 集成外部API
```python
# 添加OpenAI API支持
class OpenAILLMClient:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_answer(self, question: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的竞赛客服助手..."},
                {"role": "user", "content": f"上下文：{context}\n问题：{question}"}
            ]
        )
        return response.choices[0].message.content
```

### 10.2 自定义检索器

#### 实现专门的检索器
```python
from langchain_core.retrievers import BaseRetriever

class CustomRetriever(BaseRetriever):
    def __init__(self, vectorstore, bm25_retriever):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 自定义检索逻辑
        vector_docs = self.vectorstore.similarity_search(query, k=5)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        
        # 自定义融合策略
        return self._merge_results(vector_docs, bm25_docs)
```

### 10.3 扩展数据源

#### 支持更多文档格式
```python
def load_word_document(file_path: str) -> List[Document]:
    from docx import Document as DocxDocument
    
    doc = DocxDocument(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    return [Document(
        page_content=text,
        metadata={"source": file_path, "type": "docx"}
    )]
```

#### 集成在线数据源
```python
def load_web_content(urls: List[str]) -> List[Document]:
    from langchain_community.document_loaders import WebBaseLoader
    
    loader = WebBaseLoader(urls)
    documents = loader.load()
    
    return documents
```

---

## 11. 技术升级总结

### 11.1 Embedding模型升级

#### 升级概述
成功将embedding模型从本地BGE模型升级为Ollama的API调用方式，使用nomic-embed-text:latest模型。

#### 技术特性
- **模型**: nomic-embed-text:latest
- **维度**: 768维向量
- **语言支持**: 中英文
- **调用方式**: Ollama API接口
- **性能**: 优化的向量化处理

#### 主要改进
1. **统一API调用**: 所有模型都通过Ollama管理
2. **更好性能**: nomic-embed-text针对embedding优化
3. **简化部署**: 无需管理本地模型文件
4. **灵活配置**: 可在界面中轻松切换模型

### 11.2 系统架构优化

#### LangChain集成
- 完全基于LangChain框架重构
- 标准化的组件接口
- 更好的可扩展性

#### Ollama深度集成
- LLM和Embedding统一管理
- 简化的模型切换
- 统一的配置方式

---

## 12. 附录

### 12.1 输出文件说明

#### result_1.xlsx - 竞赛基本信息表
```
列名: 赛项名称 | 赛道 | 发布时间 | 报名时间 | 组织单位 | 官网
内容: 从PDF文档提取的18个竞赛基本信息
```

#### result_2.xlsx - 批量问答结果
```
列名: 问题编号 | 问题 | 回答 | 置信度 | 来源数量
内容: 对附件2中问题的回答结果
```

#### result_3.xlsx - 更新后问答结果  
```
列名: 问题编号 | 问题 | 回答 | 置信度 | 来源数量
内容: 知识库更新后的问答结果
```

### 12.2 项目文件结构

```
taidi-langchain/
├── langchain_chatbot.py          # LangChain主程序
├── run_langchain.py              # LangChain启动脚本  
├── test_langchain.py             # LangChain测试脚本
├── config.py                     # 配置文件
├── requirements.txt              # 依赖列表
├── create_test_questions.py      # 测试问题生成
├── src/                          # 核心模块
│   ├── langchain_rag.py         # LangChain RAG系统
│   ├── langchain_vectorstore.py # 向量存储
│   ├── langchain_retriever.py   # 混合检索器
│   ├── langchain_document_loader.py # 文档加载器
│   ├── pdf_processor.py         # PDF处理器
│   ├── vector_store.py          # 传统向量存储
│   ├── bm25_retriever.py        # BM25检索器
│   ├── hybrid_retriever.py      # 混合检索器
│   ├── rag_system.py            # RAG系统
│   ├── knowledge_manager.py     # 知识库管理
│   ├── llm_client.py            # LLM客户端
│   └── __init__.py              # 模块初始化
├── vectorstore/                  # 向量存储目录
├── logs/                        # 日志目录
├── outputs/                     # 输出目录
├── bge-large-zh-v1.5/           # BGE模型目录
├── test_questions.xlsx          # 测试问题文件
├── 附件2.pdf                    # 竞赛文档
├── C题-竞赛智能客服机器人.pdf    # 题目文档
└── *.pdf                        # 竞赛规程文档
```

### 12.3 依赖列表

#### 核心框架
```
langchain>=0.2.0
langchain-community>=0.2.0  
langchain-core>=0.2.0
langchain-ollama>=0.1.1
langchain-text-splitters>=0.2.0
```

#### 文档处理
```
PyPDF2==3.0.1
pdfplumber==0.10.0
python-docx==1.1.0
```

#### 向量存储和检索
```
faiss-cpu==1.7.4
chromadb==0.4.22
sentence-transformers==2.2.2
rank-bm25==0.2.2
```

#### Web界面
```
streamlit>=1.28.0
gradio==4.10.0
```

### 12.4 模型对比

| 模型 | 大小 | 中文能力 | 推理能力 | 内存需求 | 推荐场景 |
|------|------|----------|----------|----------|----------|
| deepseek-r1:7b | 7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 8GB | 智能客服（推荐） |
| qwen2:7b | 7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8GB | 中文对话 |
| chatglm3:6b | 6B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 6GB | 轻量化部署 |
| nomic-embed-text | - | ⭐⭐⭐⭐ | - | 2GB | 向量化（推荐） |

### 12.5 常用命令汇总

#### 系统管理
```bash
# 完整启动流程
ollama serve
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
python run_langchain.py --install
python run_langchain.py --mode web

# 环境检查
python run_langchain.py --check
python test_langchain.py

# 日志查看
tail -f langchain_chatbot.log
grep "ERROR" langchain_chatbot.log
```

#### 模型管理
```bash
# 模型操作
ollama list                    # 查看已安装模型
ollama pull <model>           # 下载模型
ollama rm <model>             # 删除模型
ollama show <model>           # 查看模型信息

# 服务管理
ollama serve                  # 启动服务
ps aux | grep ollama         # 检查服务状态
pkill ollama                 # 停止服务
```

### 12.6 相关链接

- **Ollama官网**: https://ollama.ai/
- **LangChain文档**: https://python.langchain.com/
- **DeepSeek模型**: https://github.com/deepseek-ai/deepseek-coder
- **Streamlit文档**: https://docs.streamlit.io/
- **FAISS文档**: https://faiss.ai/

---

## 📞 技术支持

如遇问题，请检查：

1. **系统日志**: `langchain_chatbot.log`
2. **错误信息**: 终端输出
3. **配置文件**: `config.py`
4. **环境检查**: `python run_langchain.py --check`

**联系方式**: 
- 提交GitHub Issue
- 查看项目文档
- 参与技术交流

---

**🎉 感谢使用LangChain+Ollama智能客服机器人！**

*最后更新时间: 2025年7月* 