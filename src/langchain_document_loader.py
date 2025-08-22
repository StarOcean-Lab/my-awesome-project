"""
LangChain文档加载器
支持PDF文档的加载和处理
"""

import os
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import pdfplumber
from loguru import logger


class LangChainDocumentLoader:
    """基于LangChain的文档加载器"""
    
    def __init__(self):
        """初始化文档加载器"""
        logger.info("LangChain文档加载器初始化完成")
    
    def load_pdf_with_pypdf(self, file_path: str) -> List[Document]:
        """
        使用PyPDF加载PDF文档
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            Document对象列表
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # 添加元数据
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source': os.path.basename(file_path),
                    'file_path': file_path,
                    'page': i + 1,
                    'loader': 'PyPDF'
                })
            
            logger.info(f"PyPDF加载完成: {file_path}, 页数: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"PyPDF加载失败 {file_path}: {e}")
            return []
    
    def load_pdf_with_pdfplumber(self, file_path: str) -> List[Document]:
        """
        使用pdfplumber加载PDF文档（处理复杂布局）
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            Document对象列表
        """
        try:
            documents = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        doc = Document(
                            page_content=text.strip(),
                            metadata={
                                'source': os.path.basename(file_path),
                                'file_path': file_path,
                                'page': i + 1,
                                'loader': 'pdfplumber'
                            }
                        )
                        documents.append(doc)
            
            logger.info(f"pdfplumber加载完成: {file_path}, 页数: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"pdfplumber加载失败 {file_path}: {e}")
            return []
    
    def load_pdf(self, file_path: str, use_plumber: bool = True) -> List[Document]:
        """
        加载PDF文档
        
        Args:
            file_path: PDF文件路径
            use_plumber: 是否使用pdfplumber（默认True）
            
        Returns:
            Document对象列表
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return []
        
        if use_plumber:
            return self.load_pdf_with_pdfplumber(file_path)
        else:
            return self.load_pdf_with_pypdf(file_path)
    
    def load_directory(self, directory_path: str, file_pattern: str = "*.pdf") -> List[Document]:
        """
        加载目录中的所有PDF文档
        
        Args:
            directory_path: 目录路径
            file_pattern: 文件匹配模式
            
        Returns:
            Document对象列表
        """
        try:
            documents = []
            import glob
            
            pattern = os.path.join(directory_path, file_pattern)
            pdf_files = glob.glob(pattern)
            
            logger.info(f"找到 {len(pdf_files)} 个PDF文件")
            
            for pdf_file in pdf_files:
                docs = self.load_pdf(pdf_file)
                documents.extend(docs)
            
            logger.info(f"目录加载完成，总文档数: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"目录加载失败 {directory_path}: {e}")
            return []
    
    def extract_competition_info(self, documents: List[Document]) -> List[Dict]:
        """
        从文档中提取竞赛基本信息
        
        Args:
            documents: Document对象列表
            
        Returns:
            竞赛信息字典列表
        """
        try:
            competition_info = []
            
            for doc in documents:
                # 提取竞赛信息的逻辑
                content = doc.page_content
                info = {
                    'source': doc.metadata.get('source', ''),
                    'content': content[:500] + '...' if len(content) > 500 else content,
                    'page': doc.metadata.get('page', 0),
                    'extracted_time': '',
                    'competition_name': '',
                    'category': '',
                    'registration_time': '',
                    'competition_time': '',
                    'target_audience': '',
                    'requirements': '',
                    'awards': ''
                }
                
                # 基本的信息提取逻辑
                if '报名时间' in content or '注册时间' in content:
                    # 提取报名时间
                    import re
                    time_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2})'
                    matches = re.findall(time_pattern, content)
                    if matches:
                        info['registration_time'] = matches[0]
                
                if '比赛时间' in content or '竞赛时间' in content:
                    # 提取比赛时间
                    import re
                    time_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2})'
                    matches = re.findall(time_pattern, content)
                    if matches:
                        info['competition_time'] = matches[0]
                
                competition_info.append(info)
            
            logger.info(f"竞赛信息提取完成，提取 {len(competition_info)} 条信息")
            return competition_info
            
        except Exception as e:
            logger.error(f"竞赛信息提取失败: {e}")
            return []
    
    def get_document_stats(self, documents: List[Document]) -> Dict:
        """
        获取文档统计信息
        
        Args:
            documents: Document对象列表
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_documents': len(documents),
            'total_pages': 0,
            'total_characters': 0,
            'files': set(),
            'loaders': set()
        }
        
        for doc in documents:
            stats['total_pages'] += 1
            stats['total_characters'] += len(doc.page_content)
            stats['files'].add(doc.metadata.get('source', ''))
            stats['loaders'].add(doc.metadata.get('loader', ''))
        
        stats['files'] = list(stats['files'])
        stats['loaders'] = list(stats['loaders'])
        
        return stats 