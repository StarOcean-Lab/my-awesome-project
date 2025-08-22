"""
增量文档加载器
智能识别新增和修改的文档，支持批量处理和进度跟踪
"""

import os
import glob
from typing import List, Dict, Optional, Set, Callable, Tuple
from pathlib import Path
from datetime import datetime
from loguru import logger
from langchain_core.documents import Document

from .document_version_manager import DocumentVersionManager
from .langchain_document_loader import LangChainDocumentLoader
from .progress_manager import ProgressManager, ProgressInfo, ProgressStatus


class IncrementalDocumentLoader:
    """增量文档加载器"""
    
    def __init__(self, 
                 document_loader: LangChainDocumentLoader = None,
                 version_manager: DocumentVersionManager = None,
                 auto_detect_changes: bool = True):
        """
        初始化增量文档加载器
        
        Args:
            document_loader: 文档加载器实例
            version_manager: 版本管理器实例
            auto_detect_changes: 是否自动检测文档变化
        """
        self.document_loader = document_loader or LangChainDocumentLoader()
        self.version_manager = version_manager or DocumentVersionManager()
        self.auto_detect_changes = auto_detect_changes
        
        # 支持的文件类型
        self.supported_extensions = {'.pdf'}
        
        logger.info("增量文档加载器初始化完成")
    
    def scan_directory(self, directory: str, recursive: bool = False) -> List[str]:
        """
        扫描目录中的文档文件
        
        Args:
            directory: 目录路径
            recursive: 是否递归扫描子目录
            
        Returns:
            文档文件路径列表
        """
        files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.warning(f"目录不存在: {directory}")
            return files
        
        try:
            if recursive:
                # 递归扫描
                for ext in self.supported_extensions:
                    pattern = f"**/*{ext}"
                    files.extend(directory_path.rglob(pattern))
            else:
                # 仅扫描当前目录
                for ext in self.supported_extensions:
                    pattern = f"*{ext}"
                    files.extend(directory_path.glob(pattern))
            
            # 转换为字符串路径
            file_paths = [str(f) for f in files]
            
            logger.info(f"扫描目录 {directory}，发现 {len(file_paths)} 个文档文件")
            return file_paths
            
        except Exception as e:
            logger.error(f"扫描目录失败 {directory}: {e}")
            return []
    
    def detect_changes(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        检测文档变化
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            变化分类字典 {'new': [...], 'modified': [...], 'unchanged': [...]}
        """
        changes = {
            'new': [],
            'modified': [],
            'unchanged': []
        }
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"文件不存在: {file_path}")
                    continue
                
                if not self.version_manager.get_document_info(file_path):
                    # 新文件
                    changes['new'].append(file_path)
                    logger.debug(f"新文件: {file_path}")
                elif self.version_manager.is_document_changed(file_path):
                    # 修改的文件
                    changes['modified'].append(file_path)
                    logger.debug(f"修改的文件: {file_path}")
                else:
                    # 未变化的文件
                    changes['unchanged'].append(file_path)
                    logger.debug(f"未变化的文件: {file_path}")
                
            except Exception as e:
                logger.error(f"检测文件变化失败 {file_path}: {e}")
                # 出错时保守处理，认为是修改的文件
                changes['modified'].append(file_path)
        
        logger.info(f"变化检测完成: 新增 {len(changes['new'])}, "
                   f"修改 {len(changes['modified'])}, 未变化 {len(changes['unchanged'])}")
        
        return changes
    
    def load_changed_documents(self, 
                             file_paths: List[str], 
                             progress_callback: Optional[Callable] = None,
                             force_reload: bool = False) -> Dict[str, List[Document]]:
        """
        加载变化的文档
        
        Args:
            file_paths: 文件路径列表
            progress_callback: 进度回调函数
            force_reload: 是否强制重新加载所有文档
            
        Returns:
            文档加载结果 {'loaded': [...], 'failed': [...], 'skipped': [...]}
        """
        results = {
            'loaded': [],
            'failed': [],
            'skipped': []
        }
        
        if not file_paths:
            logger.info("没有文件需要处理")
            return results
        
        # 检测变化
        if force_reload:
            changed_files = file_paths
            logger.info(f"强制重新加载 {len(changed_files)} 个文件")
        else:
            changes = self.detect_changes(file_paths)
            changed_files = changes['new'] + changes['modified']
            
            if not changed_files:
                logger.info("没有文件需要加载")
                return results
        
        # 初始化进度管理
        progress_manager = ProgressManager(len(changed_files))
        if progress_callback:
            progress_manager.set_callback(progress_callback)
        
        progress_manager.start()
        
        # 逐个加载文件
        for i, file_path in enumerate(changed_files):
            try:
                progress_manager.update(
                    step=i + 1,
                    step_name="加载文档",
                    description=f"正在处理: {os.path.basename(file_path)}"
                )
                
                logger.info(f"加载文档: {file_path}")
                
                # 加载文档
                documents = self.document_loader.load_pdf(file_path)
                
                if documents:
                    # 更新版本信息
                    self.version_manager.update_document_version(file_path, documents)
                    
                    results['loaded'].extend(documents)
                    logger.info(f"成功加载 {len(documents)} 个文档片段: {file_path}")
                else:
                    logger.warning(f"未加载到任何文档: {file_path}")
                    results['failed'].append(file_path)
                
            except Exception as e:
                logger.error(f"加载文档失败 {file_path}: {e}")
                results['failed'].append(file_path)
        
        progress_manager.complete()
        
        logger.info(f"文档加载完成: 成功 {len(results['loaded'])} 个片段, "
                   f"失败 {len(results['failed'])} 个文件")
        
        return results
    
    def load_directory_incremental(self, 
                                 directory: str, 
                                 recursive: bool = False,
                                 progress_callback: Optional[Callable] = None,
                                 force_reload: bool = False) -> Dict[str, List]:
        """
        增量加载目录中的文档
        
        Args:
            directory: 目录路径
            recursive: 是否递归扫描
            progress_callback: 进度回调函数
            force_reload: 是否强制重新加载
            
        Returns:
            加载结果统计
        """
        try:
            logger.info(f"开始增量加载目录: {directory}")
            
            # 扫描文件
            file_paths = self.scan_directory(directory, recursive)
            
            if not file_paths:
                logger.warning(f"目录中没有找到支持的文档文件: {directory}")
                return {
                    'total_files': 0,
                    'loaded_documents': [],
                    'failed_files': [],
                    'skipped_files': []
                }
            
            # 加载文档
            results = self.load_changed_documents(
                file_paths, 
                progress_callback=progress_callback,
                force_reload=force_reload
            )
            
            return {
                'total_files': len(file_paths),
                'loaded_documents': results['loaded'],
                'failed_files': results['failed'],
                'skipped_files': results['skipped']
            }
            
        except Exception as e:
            logger.error(f"增量加载目录失败 {directory}: {e}")
            raise
    
    def get_pending_updates(self, directories: List[str]) -> Dict[str, List[str]]:
        """
        获取待更新的文档列表
        
        Args:
            directories: 目录列表
            
        Returns:
            待更新文档分类
        """
        all_changes = {
            'new': [],
            'modified': [],
            'unchanged': []
        }
        
        for directory in directories:
            file_paths = self.scan_directory(directory)
            changes = self.detect_changes(file_paths)
            
            for key in all_changes:
                all_changes[key].extend(changes[key])
        
        return all_changes
    
    def batch_process_updates(self, 
                            directories: List[str],
                            progress_callback: Optional[Callable] = None,
                            batch_size: int = 10) -> Dict:
        """
        批量处理更新
        
        Args:
            directories: 目录列表
            progress_callback: 进度回调函数
            batch_size: 批处理大小
            
        Returns:
            处理结果统计
        """
        try:
            logger.info(f"开始批量处理 {len(directories)} 个目录")
            
            # 获取所有待更新文件
            pending_changes = self.get_pending_updates(directories)
            changed_files = pending_changes['new'] + pending_changes['modified']
            
            if not changed_files:
                logger.info("没有文件需要更新")
                return {
                    'total_files': 0,
                    'processed_batches': 0,
                    'loaded_documents': [],
                    'failed_files': []
                }
            
            logger.info(f"发现 {len(changed_files)} 个文件需要处理")
            
            # 分批处理
            all_results = {
                'loaded_documents': [],
                'failed_files': []
            }
            
            total_batches = (len(changed_files) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(changed_files))
                batch_files = changed_files[start_idx:end_idx]
                
                logger.info(f"处理批次 {batch_idx + 1}/{total_batches}, "
                           f"文件数: {len(batch_files)}")
                
                # 处理当前批次
                batch_results = self.load_changed_documents(
                    batch_files,
                    progress_callback=progress_callback
                )
                
                # 合并结果
                all_results['loaded_documents'].extend(batch_results['loaded'])
                all_results['failed_files'].extend(batch_results['failed'])
            
            logger.info(f"批量处理完成: 处理了 {total_batches} 个批次, "
                       f"加载 {len(all_results['loaded_documents'])} 个文档片段")
            
            return {
                'total_files': len(changed_files),
                'processed_batches': total_batches,
                'loaded_documents': all_results['loaded_documents'],
                'failed_files': all_results['failed_files']
            }
            
        except Exception as e:
            logger.error(f"批量处理更新失败: {e}")
            raise
    
    def get_statistics(self) -> Dict:
        """获取加载器统计信息"""
        stats = {
            'loader_type': 'IncrementalDocumentLoader',
            'auto_detect_changes': self.auto_detect_changes,
            'supported_extensions': list(self.supported_extensions)
        }
        
        # 添加版本管理器统计
        if self.version_manager:
            version_stats = self.version_manager.get_statistics()
            stats.update(version_stats)
        
        return stats
    
    def reset_version_tracking(self):
        """重置版本跟踪信息"""
        if self.version_manager:
            self.version_manager.reset_all()
            logger.info("版本跟踪信息已重置")
    
    def cleanup_orphaned_versions(self) -> int:
        """清理孤立的版本信息"""
        if self.version_manager:
            return self.version_manager.cleanup_missing_files()
        return 0 