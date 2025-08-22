"""
进度管理器
用于跟踪知识库加载进度
"""

import time
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ProgressStatus(Enum):
    """进度状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProgressInfo:
    """进度信息"""
    current_step: int
    total_steps: int
    step_name: str
    description: str
    status: ProgressStatus
    error_message: str = ""
    
    @property
    def percentage(self) -> float:
        """获取完成百分比"""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100

class ProgressManager:
    """进度管理器"""
    
    def __init__(self, total_steps: int = 100):
        """
        初始化进度管理器
        
        Args:
            total_steps: 总步数
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.step_name = ""
        self.description = ""
        self.status = ProgressStatus.PENDING
        self.error_message = ""
        
        # 回调函数
        self.progress_callback: Optional[Callable[[ProgressInfo], None]] = None
        
        # 预定义的加载步骤
        self.loading_steps = [
            ("初始化", "正在初始化系统组件..."),
            ("扫描文件", "正在扫描PDF文件..."),
            ("处理文档", "正在处理PDF文档..."),
            ("文本分块", "正在分割文档文本..."),
            ("向量化", "正在生成文档向量..."),
            ("构建索引", "正在构建检索索引..."),
            ("保存索引", "正在保存索引文件..."),
            ("完成", "知识库加载完成！")
        ]
    
    def set_callback(self, callback: Callable[[ProgressInfo], None]):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def start(self, total_steps: int = None):
        """开始进度跟踪"""
        if total_steps:
            self.total_steps = total_steps
        
        self.current_step = 0
        self.status = ProgressStatus.RUNNING
        self.error_message = ""
        
        self._notify_progress()
    
    def update(self, step: int = None, step_name: str = None, description: str = None):
        """更新进度"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        if step_name:
            self.step_name = step_name
        if description:
            self.description = description
            
        self._notify_progress()
    
    def update_by_step_name(self, step_name: str, description: str = None):
        """根据步骤名称更新进度"""
        # 查找步骤索引
        for i, (name, default_desc) in enumerate(self.loading_steps):
            if name == step_name:
                self.current_step = i + 1
                self.step_name = step_name
                self.description = description or default_desc
                self._notify_progress()
                return
        
        # 如果未找到预定义步骤，直接更新
        self.step_name = step_name
        self.description = description or step_name
        self._notify_progress()
    
    def complete(self):
        """完成进度"""
        self.current_step = self.total_steps
        self.status = ProgressStatus.COMPLETED
        self.step_name = "完成"
        self.description = "知识库加载完成！"
        self._notify_progress()
    
    def fail(self, error_message: str):
        """标记失败"""
        self.status = ProgressStatus.FAILED
        self.error_message = error_message
        self._notify_progress()
    
    def _notify_progress(self):
        """通知进度更新"""
        if self.progress_callback:
            progress_info = ProgressInfo(
                current_step=self.current_step,
                total_steps=self.total_steps,
                step_name=self.step_name,
                description=self.description,
                status=self.status,
                error_message=self.error_message
            )
            self.progress_callback(progress_info)
    
    def get_current_progress(self) -> ProgressInfo:
        """获取当前进度信息"""
        return ProgressInfo(
            current_step=self.current_step,
            total_steps=self.total_steps,
            step_name=self.step_name,
            description=self.description,
            status=self.status,
            error_message=self.error_message
        )

class KnowledgeBaseProgressManager(ProgressManager):
    """知识库专用进度管理器"""
    
    def __init__(self, pdf_files: list = None):
        """
        初始化知识库进度管理器
        
        Args:
            pdf_files: PDF文件列表
        """
        self.pdf_files = pdf_files or []
        self.processed_files = 0
        self.current_file = ""
        
        # 先定义加载步骤
        self.loading_steps = [
            ("初始化", "正在初始化系统组件..."),
            ("扫描文件", f"发现 {len(self.pdf_files)} 个PDF文件"),
            ("处理文档", "正在处理PDF文档..."),
            ("文本分块", "正在分割文档文本..."),
            ("向量化", "正在生成文档向量..."),
            ("构建索引", "正在构建检索索引..."),
            ("保存索引", "正在保存索引文件..."),
            ("完成", "知识库加载完成！")
        ]
        
        # 根据文件数量调整总步数
        file_count = len(self.pdf_files)
        total_steps = len(self.loading_steps) + file_count * 2  # 每个文件需要处理和向量化两步
        
        super().__init__(total_steps)
    
    def update_file_progress(self, file_path: str, step_type: str):
        """更新文件处理进度"""
        self.current_file = file_path
        filename = file_path.split('/')[-1]
        
        if step_type == "processing":
            self.processed_files += 1
            self.update(
                step_name="处理文档",
                description=f"正在处理文件 {self.processed_files}/{len(self.pdf_files)}: {filename}"
            )
        elif step_type == "vectorizing":
            self.update(
                step_name="向量化",
                description=f"正在向量化文件 {self.processed_files}/{len(self.pdf_files)}: {filename}"
            )
        elif step_type == "indexing":
            self.update(
                step_name="构建索引",
                description=f"正在为文件构建索引: {filename}"
            )
    
    def get_detailed_progress(self) -> Dict[str, Any]:
        """获取详细进度信息"""
        progress = self.get_current_progress()
        
        return {
            "percentage": progress.percentage,
            "current_step": progress.current_step,
            "total_steps": progress.total_steps,
            "step_name": progress.step_name,
            "description": progress.description,
            "status": progress.status.value,
            "error_message": progress.error_message,
            "pdf_files": self.pdf_files,
            "processed_files": self.processed_files,
            "current_file": self.current_file
        } 