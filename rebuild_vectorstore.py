#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

def rebuild_vectorstore():
    """重建向量数据库，使用新的embedding模型"""
    print("=== 重建向量数据库 ===\n")
    
    # 备份现有向量数据库
    vectorstore_dir = "vectorstore"
    backup_dir = "vectorstore_backup"
    
    if os.path.exists(vectorstore_dir):
        print(f"备份现有向量数据库到 {backup_dir}")
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(vectorstore_dir, backup_dir)
        
        # 删除现有向量数据库
        print("删除现有向量数据库")
        shutil.rmtree(vectorstore_dir)
    
    # 初始化新的embedding模型
    print(f"初始化embedding模型: {Config.LANGCHAIN_EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(
        model=Config.LANGCHAIN_EMBEDDING_MODEL,
        base_url=Config.OLLAMA_BASE_URL
    )
    
    # 测试embedding模型
    try:
        test_embedding = embeddings.embed_query("测试")
        print(f"Embedding模型测试成功，向量维度: {len(test_embedding)}")
    except Exception as e:
        print(f"Embedding模型测试失败: {e}")
        return False
    
    # 加载所有PDF文档
    print("\n加载PDF文档...")
    documents = []
    pdf_files = Config.get_pdf_files()
    
    print(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.LANGCHAIN_CHUNK_SIZE,
        chunk_overlap=Config.LANGCHAIN_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    
    for pdf_file in pdf_files:
        try:
            print(f"处理: {pdf_file}")
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            
            # 分割文档
            page_docs = text_splitter.split_documents(pages)
            documents.extend(page_docs)
            
            print(f"  -> {len(pages)} 页，分割为 {len(page_docs)} 个块")
            
        except Exception as e:
            print(f"处理 {pdf_file} 时出错: {e}")
    
    print(f"\n总共加载 {len(documents)} 个文档块")
    
    # 构建向量数据库
    print("\n构建向量数据库...")
    if documents:
        try:
            vectorstore = FAISS.from_documents(documents, embeddings)
            
            # 保存向量数据库
            os.makedirs(vectorstore_dir, exist_ok=True)
            vectorstore.save_local(vectorstore_dir)
            
            print(f"向量数据库构建成功！")
            print(f"文档总数: {len(documents)}")
            print(f"保存位置: {vectorstore_dir}")
            
            # 测试搜索
            print("\n测试向量搜索...")
            test_query = "未来校园智能应用中交通信号灯的技术要求"
            results = vectorstore.similarity_search_with_score(test_query, k=5)
            
            print(f"查询: {test_query}")
            print(f"找到 {len(results)} 个结果:")
            
            for i, (doc, score) in enumerate(results):
                print(f"\n  {i+1}. 相似度分数: {score:.6f}")
                print(f"     来源: {doc.metadata.get('source', 'unknown')}")
                print(f"     内容: {doc.page_content[:150]}...")
                contains_future_campus = '未来校园' in doc.page_content
                print(f"     包含'未来校园': {'是' if contains_future_campus else '否'}")
            
            return True
            
        except Exception as e:
            print(f"构建向量数据库时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("没有找到文档，无法构建向量数据库")
        return False

if __name__ == "__main__":
    success = rebuild_vectorstore()
    if success:
        print("\n✅ 向量数据库重建成功！")
    else:
        print("\n❌ 向量数据库重建失败！") 