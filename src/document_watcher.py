"""
文档监控器
自动检测文档变化并触发知识库更新
"""

import os
import time
import threading
import glob
from typing import List, Dict, Optional, Callable, Set
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import hashlib
from dataclasses import dataclass

from .knowledge_manager import KnowledgeManager


@dataclass
class WatchConfig:
    """监控配置"""
    watch_directories: List[str]  # 监控目录列表
    file_patterns: List[str]      # 文件模式
    check_interval: int = 30      # 检查间隔（秒）
    auto_update: bool = True      # 是否自动更新
    enable_realtime: bool = False # 是否启用实时监控
    min_update_interval: int = 60 # 最小更新间隔（秒）


@dataclass
class FileStatus:
    """文件状态"""
    file_path: str
    last_modified: float
    file_hash: str
    size: int


class DocumentWatcher:
    """文档监控器"""
    
    def __init__(self, 
                 knowledge_manager: KnowledgeManager,
                 config: WatchConfig,
                 update_callback: Optional[Callable] = None):
        """
        初始化文档监控器
        
        Args:
            knowledge_manager: 知识库管理器
            config: 监控配置
            update_callback: 更新回调函数
        """
        self.knowledge_manager = knowledge_manager
        self.config = config
        self.update_callback = update_callback
        
        # 监控状态
        self.is_watching = False
        self.watch_thread = None
        self.file_status: Dict[str, FileStatus] = {}
        self.last_update_time = None
        
        # 事件锁
        self.update_lock = threading.Lock()
        
        logger.info("文档监控器初始化完成")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {file_path}: {e}")
            return ""
    
    def _get_file_status(self, file_path: str) -> FileStatus:
        """获取文件状态"""
        try:
            stat = os.stat(file_path)
            return FileStatus(
                file_path=file_path,
                last_modified=stat.st_mtime,
                file_hash=self._calculate_file_hash(file_path),
                size=stat.st_size
            )
        except Exception as e:
            logger.error(f"获取文件状态失败 {file_path}: {e}")
            return None
    
    def _scan_files(self) -> List[str]:
        """扫描所有监控的文件"""
        all_files = []
        
        for directory in self.config.watch_directories:
            if not os.path.exists(directory):
                logger.warning(f"监控目录不存在: {directory}")
                continue
                
            for pattern in self.config.file_patterns:
                search_pattern = os.path.join(directory, pattern)
                files = glob.glob(search_pattern)
                all_files.extend(files)
        
        # 去重并返回
        return list(set(all_files))
    
    def _detect_changes(self) -> Dict[str, str]:
        """检测文件变化"""
        current_files = self._scan_files()
        changes = {}
        
        # 检查新文件和修改的文件
        for file_path in current_files:
            current_status = self._get_file_status(file_path)
            if not current_status:
                continue
                
            if file_path not in self.file_status:
                # 新文件
                changes[file_path] = "新增文件"
                self.file_status[file_path] = current_status
            else:
                # 检查是否修改
                stored_status = self.file_status[file_path]
                if (current_status.file_hash != stored_status.file_hash or
                    current_status.last_modified != stored_status.last_modified):
                    changes[file_path] = "文件已修改"
                    self.file_status[file_path] = current_status
        
        # 检查删除的文件
        stored_files = set(self.file_status.keys())
        current_file_set = set(current_files)
        
        for deleted_file in stored_files - current_file_set:
            if os.path.exists(deleted_file):
                # 文件仍存在但不在扫描结果中，可能是路径变化
                continue
            changes[deleted_file] = "文件已删除"
            del self.file_status[deleted_file]
        
        return changes
    
    def _should_update(self) -> bool:
        """判断是否应该执行更新"""
        if not self.config.auto_update:
            return False
            
        # 检查最小更新间隔
        if self.last_update_time:
            time_since_update = datetime.now() - self.last_update_time
            if time_since_update.total_seconds() < self.config.min_update_interval:
                logger.info(f"距离上次更新时间太短，跳过更新（{time_since_update.total_seconds():.0f}秒）")
                return False
        
        return True
    
    def _trigger_update(self, changes: Dict[str, str]):
        """触发知识库更新"""
        with self.update_lock:
            if not self._should_update():
                return
                
            try:
                logger.info(f"检测到文档变化，开始更新知识库: {changes}")
                
                # 获取当前所有文件
                current_files = self._scan_files()
                
                # 执行更新
                result = self.knowledge_manager.update_knowledge_base(current_files)
                
                # 更新时间
                self.last_update_time = datetime.now()
                
                # 调用回调函数
                if self.update_callback:
                    self.update_callback({
                        "status": "success",
                        "changes": changes,
                        "result": result,
                        "timestamp": self.last_update_time
                    })
                
                logger.info("知识库自动更新完成")
                
            except Exception as e:
                logger.error(f"自动更新知识库失败: {e}")
                if self.update_callback:
                    self.update_callback({
                        "status": "error",
                        "error": str(e),
                        "changes": changes,
                        "timestamp": datetime.now()
                    })
    
    def _watch_loop(self):
        """监控循环"""
        logger.info(f"开始文档监控，检查间隔: {self.config.check_interval}秒")
        
        # 初始扫描
        current_files = self._scan_files()
        for file_path in current_files:
            status = self._get_file_status(file_path)
            if status:
                self.file_status[file_path] = status
        
        logger.info(f"初始扫描完成，发现 {len(current_files)} 个文件")
        
        while self.is_watching:
            try:
                # 检测变化
                changes = self._detect_changes()
                
                if changes:
                    logger.info(f"检测到文档变化: {changes}")
                    self._trigger_update(changes)
                
                # 等待下次检查
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(self.config.check_interval)
    
    def start_watching(self):
        """开始监控"""
        if self.is_watching:
            logger.warning("监控已在运行中")
            return
            
        self.is_watching = True
        self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watch_thread.start()
        
        logger.info("文档监控已启动")
    
    def stop_watching(self):
        """停止监控"""
        if not self.is_watching:
            logger.warning("监控未在运行")
            return
            
        self.is_watching = False
        
        if self.watch_thread and self.watch_thread.is_alive():
            self.watch_thread.join(timeout=5)
        
        logger.info("文档监控已停止")
    
    def check_now(self) -> Dict[str, str]:
        """立即检查一次"""
        logger.info("执行手动文档检查")
        changes = self._detect_changes()
        
        if changes and self.config.auto_update:
            self._trigger_update(changes)
        
        return changes
    
    def get_status(self) -> Dict:
        """获取监控状态"""
        return {
            "is_watching": self.is_watching,
            "monitored_files": len(self.file_status),
            "watch_directories": self.config.watch_directories,
            "file_patterns": self.config.file_patterns,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "auto_update": self.config.auto_update,
            "check_interval": self.config.check_interval
        }
    
    def get_file_list(self) -> List[Dict]:
        """获取监控文件列表"""
        file_list = []
        for file_path, status in self.file_status.items():
            file_list.append({
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "last_modified": datetime.fromtimestamp(status.last_modified).isoformat(),
                "size": status.size,
                "file_hash": status.file_hash[:8]  # 只显示前8位
            })
        
        return sorted(file_list, key=lambda x: x["last_modified"], reverse=True)
    
    def update_config(self, new_config: WatchConfig):
        """更新监控配置"""
        old_watching = self.is_watching
        
        if old_watching:
            self.stop_watching()
        
        self.config = new_config
        
        # 重新扫描文件
        self.file_status.clear()
        
        if old_watching:
            self.start_watching()
        
        logger.info("监控配置已更新")


def create_default_watch_config() -> WatchConfig:
    """创建默认监控配置"""
    return WatchConfig(
        watch_directories=["data"],
        file_patterns=["*.pdf"],
        check_interval=30,
        auto_update=True,
        enable_realtime=False,
        min_update_interval=60
    ) 