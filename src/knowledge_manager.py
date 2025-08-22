"""
知识库管理器
提供知识库的通用管理接口
"""

from typing import List, Dict, Optional, Any
import os
import glob
from loguru import logger
from datetime import datetime

class KnowledgeManager:
    """知识库管理器"""
    
    def __init__(self, rag_system=None):
        """
        初始化知识库管理器
        
        Args:
            rag_system: RAG系统实例（可选）
        """
        self.rag_system = rag_system
        logger.info("知识库管理器初始化完成")
    
    def update_knowledge_base(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        更新知识库
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            更新结果
        """
        try:
            logger.info(f"开始更新知识库，文件数量: {len(file_paths)}")
            
            # 如果有RAG系统实例，使用它来更新知识库
            if self.rag_system:
                return self._update_with_rag_system(file_paths)
            else:
                return self._basic_update(file_paths)
                
        except Exception as e:
            logger.error(f"更新知识库失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_files": 0,
                "timestamp": datetime.now()
            }
    
    def _update_with_rag_system(self, file_paths: List[str]) -> Dict[str, Any]:
        """使用RAG系统更新知识库"""
        try:
            # 过滤出PDF文件
            pdf_files = [f for f in file_paths if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                logger.warning("没有找到PDF文件")
                return {
                    "success": True,
                    "message": "没有PDF文件需要更新",
                    "updated_files": 0,
                    "timestamp": datetime.now()
                }
            
            # 重新加载文档
            success = self.rag_system.load_documents(directory_path=".")
            
            return {
                "success": success,
                "message": "知识库更新完成" if success else "知识库更新失败",
                "updated_files": len(pdf_files) if success else 0,
                "file_paths": pdf_files,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"RAG系统更新失败: {e}")
            raise
    
    def _basic_update(self, file_paths: List[str]) -> Dict[str, Any]:
        """基础更新方法（当没有RAG系统时）"""
        logger.info("使用基础更新方法")
        
        # 验证文件是否存在
        valid_files = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"文件不存在: {file_path}")
        
        return {
            "success": True,
            "message": f"验证完成，找到 {len(valid_files)} 个有效文件",
            "updated_files": len(valid_files),
            "file_paths": valid_files,
            "timestamp": datetime.now()
        }
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """获取知识库信息"""
        try:
            if self.rag_system and hasattr(self.rag_system, 'vectorstore'):
                # 如果有向量存储，获取文档数量
                if self.rag_system.vectorstore:
                    doc_count = getattr(self.rag_system.vectorstore, 'doc_count', 0)
                else:
                    doc_count = 0
                
                return {
                    "has_rag_system": True,
                    "document_count": doc_count,
                    "vectorstore_ready": self.rag_system.vectorstore is not None,
                    "retriever_ready": getattr(self.rag_system, 'retriever', None) is not None
                }
            else:
                return {
                    "has_rag_system": False,
                    "document_count": 0,
                    "vectorstore_ready": False,
                    "retriever_ready": False
                }
                
        except Exception as e:
            logger.error(f"获取知识库信息失败: {e}")
            return {
                "error": str(e)
            }
    
    def set_rag_system(self, rag_system):
        """设置RAG系统实例"""
        self.rag_system = rag_system
        logger.info("RAG系统实例已设置")


def create_knowledge_manager(rag_system=None) -> KnowledgeManager:
    """
    创建知识库管理器实例
    
    Args:
        rag_system: RAG系统实例（可选）
        
    Returns:
        KnowledgeManager实例
    """
    return KnowledgeManager(rag_system) 