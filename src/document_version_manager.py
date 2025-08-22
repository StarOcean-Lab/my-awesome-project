"""
文档版本管理器
实现文档去重、版本跟踪和增量更新功能
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger
from langchain_core.documents import Document


@dataclass
class DocumentFingerprint:
    """文档指纹信息"""
    file_path: str
    file_hash: str  # 文件内容哈希
    file_size: int
    last_modified: float
    chunk_count: int
    content_hash: str  # 文档内容哈希
    version: int
    created_at: str
    updated_at: str


@dataclass
class DocumentMetadata:
    """文档元数据"""
    source: str
    chunk_index: int
    chunk_hash: str  # 单个chunk的哈希
    page: Optional[int] = None
    total_chunks: Optional[int] = None


class DocumentVersionManager:
    """文档版本管理器"""
    
    def __init__(self, version_file: str = "./knowledge_base/document_versions.json"):
        """
        初始化文档版本管理器
        
        Args:
            version_file: 版本信息存储文件路径
        """
        self.version_file = Path(version_file)
        self.version_data: Dict[str, DocumentFingerprint] = {}
        self.document_chunks: Dict[str, Set[str]] = {}  # file_path -> set of chunk_hashes
        
        # 确保目录存在
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载已有版本信息
        self._load_version_data()
        
        logger.info(f"文档版本管理器初始化完成，已跟踪 {len(self.version_data)} 个文档")
    
    def _load_version_data(self):
        """加载版本数据"""
        try:
            if self.version_file.exists():
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 转换为DocumentFingerprint对象
                for file_path, fingerprint_data in data.get('documents', {}).items():
                    self.version_data[file_path] = DocumentFingerprint(**fingerprint_data)
                
                # 加载chunk信息
                self.document_chunks = {
                    file_path: set(chunks) 
                    for file_path, chunks in data.get('chunks', {}).items()
                }
                
                logger.info(f"加载版本数据成功，包含 {len(self.version_data)} 个文档")
            else:
                logger.info("版本文件不存在，创建新的版本管理")
        except Exception as e:
            logger.error(f"加载版本数据失败: {e}")
            self.version_data = {}
            self.document_chunks = {}
    
    def _save_version_data(self):
        """保存版本数据"""
        try:
            data = {
                'documents': {
                    file_path: asdict(fingerprint) 
                    for file_path, fingerprint in self.version_data.items()
                },
                'chunks': {
                    file_path: list(chunks) 
                    for file_path, chunks in self.document_chunks.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.debug("版本数据保存成功")
        except Exception as e:
            logger.error(f"保存版本数据失败: {e}")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {file_path}: {e}")
            return ""
    
    def calculate_content_hash(self, documents: List[Document]) -> str:
        """计算文档内容哈希"""
        try:
            content = ""
            for doc in documents:
                content += doc.page_content
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"计算内容哈希失败: {e}")
            return ""
    
    def calculate_chunk_hash(self, content: str) -> str:
        """计算单个chunk的哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_file_info(self, file_path: str) -> Tuple[str, int, float]:
        """获取文件基本信息"""
        try:
            stat = os.stat(file_path)
            file_hash = self.calculate_file_hash(file_path)
            return file_hash, stat.st_size, stat.st_mtime
        except Exception as e:
            logger.error(f"获取文件信息失败 {file_path}: {e}")
            return "", 0, 0.0
    
    def is_document_changed(self, file_path: str, documents: List[Document] = None) -> bool:
        """
        检查文档是否发生变化
        
        Args:
            file_path: 文件路径
            documents: 文档列表（可选，用于更精确的检查）
            
        Returns:
            是否发生变化
        """
        try:
            # 获取当前文件信息
            current_hash, current_size, current_mtime = self.get_file_info(file_path)
            
            if not current_hash:
                return True  # 无法读取文件，认为是新文件
            
            # 检查是否是新文件
            if file_path not in self.version_data:
                logger.info(f"检测到新文件: {file_path}")
                return True
            
            # 获取已存储的指纹
            stored_fingerprint = self.version_data[file_path]
            
            # 比较文件基本信息
            if (stored_fingerprint.file_hash != current_hash or 
                stored_fingerprint.file_size != current_size or 
                stored_fingerprint.last_modified != current_mtime):
                logger.info(f"检测到文件变化: {file_path}")
                return True
            
            # 如果提供了文档内容，进行更精确的内容比较
            if documents:
                current_content_hash = self.calculate_content_hash(documents)
                if stored_fingerprint.content_hash != current_content_hash:
                    logger.info(f"检测到文档内容变化: {file_path}")
                    return True
            
            logger.debug(f"文档未变化: {file_path}")
            return False
            
        except Exception as e:
            logger.error(f"检查文档变化失败 {file_path}: {e}")
            return True  # 出错时保守处理，认为有变化
    
    def get_new_chunks(self, file_path: str, documents: List[Document]) -> List[Document]:
        """
        获取新增的文档chunks
        
        Args:
            file_path: 文件路径
            documents: 所有文档chunks
            
        Returns:
            新增的文档chunks
        """
        try:
            # 如果文件不存在于版本管理中，所有chunks都是新的
            if file_path not in self.document_chunks:
                logger.info(f"新文件，所有 {len(documents)} 个chunks都是新的: {file_path}")
                return documents
            
            # 获取已存在的chunk哈希集合
            existing_chunks = self.document_chunks[file_path]
            new_docs = []
            
            for doc in documents:
                chunk_hash = self.calculate_chunk_hash(doc.page_content)
                if chunk_hash not in existing_chunks:
                    new_docs.append(doc)
            
            logger.info(f"文件 {file_path} 检测到 {len(new_docs)}/{len(documents)} 个新chunks")
            return new_docs
            
        except Exception as e:
            logger.error(f"获取新chunks失败 {file_path}: {e}")
            return documents  # 出错时返回所有文档
    
    def update_document_version(self, file_path: str, documents: List[Document]) -> DocumentFingerprint:
        """
        更新文档版本信息
        
        Args:
            file_path: 文件路径
            documents: 文档列表
            
        Returns:
            更新后的文档指纹
        """
        try:
            # 获取文件信息
            file_hash, file_size, last_modified = self.get_file_info(file_path)
            content_hash = self.calculate_content_hash(documents)
            
            # 计算chunk哈希集合
            chunk_hashes = set()
            for i, doc in enumerate(documents):
                chunk_hash = self.calculate_chunk_hash(doc.page_content)
                chunk_hashes.add(chunk_hash)
                
                # 更新文档元数据
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata.update({
                    'chunk_hash': chunk_hash,
                    'chunk_index': i,
                    'total_chunks': len(documents),
                    'file_version': 1 if file_path not in self.version_data else self.version_data[file_path].version + 1
                })
            
            # 创建或更新指纹
            current_time = datetime.now().isoformat()
            
            if file_path in self.version_data:
                # 更新现有文档
                old_fingerprint = self.version_data[file_path]
                new_version = old_fingerprint.version + 1
                created_at = old_fingerprint.created_at
            else:
                # 新文档
                new_version = 1
                created_at = current_time
            
            fingerprint = DocumentFingerprint(
                file_path=file_path,
                file_hash=file_hash,
                file_size=file_size,
                last_modified=last_modified,
                chunk_count=len(documents),
                content_hash=content_hash,
                version=new_version,
                created_at=created_at,
                updated_at=current_time
            )
            
            # 更新版本数据
            self.version_data[file_path] = fingerprint
            self.document_chunks[file_path] = chunk_hashes
            
            # 保存到磁盘
            self._save_version_data()
            
            logger.info(f"文档版本已更新: {file_path} -> v{new_version}")
            return fingerprint
            
        except Exception as e:
            logger.error(f"更新文档版本失败 {file_path}: {e}")
            raise
    
    def remove_document(self, file_path: str):
        """移除文档版本信息"""
        try:
            if file_path in self.version_data:
                del self.version_data[file_path]
            if file_path in self.document_chunks:
                del self.document_chunks[file_path]
            
            self._save_version_data()
            logger.info(f"已移除文档版本信息: {file_path}")
        except Exception as e:
            logger.error(f"移除文档版本失败 {file_path}: {e}")
    
    def get_document_info(self, file_path: str) -> Optional[DocumentFingerprint]:
        """获取文档信息"""
        return self.version_data.get(file_path)
    
    def get_all_documents(self) -> Dict[str, DocumentFingerprint]:
        """获取所有文档信息"""
        return self.version_data.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.version_data:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "average_chunks_per_doc": 0,
                "latest_update": None
            }
        
        total_chunks = sum(fp.chunk_count for fp in self.version_data.values())
        latest_update = max(fp.updated_at for fp in self.version_data.values())
        
        return {
            "total_documents": len(self.version_data),
            "total_chunks": total_chunks,
            "average_chunks_per_doc": total_chunks / len(self.version_data),
            "latest_update": latest_update,
            "version_file_size": self.version_file.stat().st_size if self.version_file.exists() else 0
        }
    
    def cleanup_missing_files(self) -> int:
        """清理已删除文件的版本信息"""
        removed_count = 0
        files_to_remove = []
        
        for file_path in self.version_data.keys():
            if not os.path.exists(file_path):
                files_to_remove.append(file_path)
        
        for file_path in files_to_remove:
            self.remove_document(file_path)
            removed_count += 1
        
        if removed_count > 0:
            logger.info(f"清理了 {removed_count} 个已删除文件的版本信息")
        
        return removed_count
    
    def reset_all(self):
        """重置所有版本信息"""
        self.version_data.clear()
        self.document_chunks.clear()
        
        if self.version_file.exists():
            self.version_file.unlink()
        
        logger.info("已重置所有文档版本信息") 