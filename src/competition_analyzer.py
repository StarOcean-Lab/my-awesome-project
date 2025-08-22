#!/usr/bin/env python3
"""
竞赛分析器模块
专门用于从文档级别分析竞赛类别、数量和详细信息
"""

import os
import re
import glob
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
from loguru import logger

@dataclass
class CompetitionInfo:
    """竞赛信息数据类"""
    name: str                    # 竞赛名称
    category: str                # 竞赛类别
    file_path: str              # 文件路径
    description: str            # 竞赛描述
    registration_time: str      # 报名时间
    competition_time: str       # 比赛时间
    target_audience: str        # 目标受众
    team_size: str             # 团队规模
    requirements: List[str]     # 竞赛要求
    awards: List[str]          # 奖项设置
    keywords: Set[str]         # 关键词

class CompetitionAnalyzer:
    """竞赛分析器"""
    
    def __init__(self):
        """初始化竞赛分析器"""
        # 定义竞赛类别映射
        self.competition_categories = {
            "机器人": {
                "keywords": ["机器人", "机械", "自动化", "控制", "工程"],
                "patterns": [
                    r"机器人.*专项赛",
                    r"机器人.*竞赛", 
                    r"机器人.*设计",
                    r"机器人.*工程",
                    r"竞技.*机器人",
                    r"智能.*机器人",
                    r"太空.*机器人",
                    r"开源.*机器人"
                ],
                "description": "机器人相关竞赛"
            },
            "人工智能": {
                "keywords": ["人工智能", "AI", "机器学习", "深度学习", "智能算法"],
                "patterns": [
                    r"人工智能.*专项赛",
                    r"AI.*竞赛",
                    r"智能.*应用",
                    r"生成式.*AI",
                    r"智能.*算法"
                ],
                "description": "人工智能相关竞赛"
            },
            "编程设计": {
                "keywords": ["编程", "算法", "代码", "软件开发", "程序设计"],
                "patterns": [
                    r"编程.*专项赛",
                    r"3D.*编程",
                    r"程序.*设计",
                    r"算法.*竞赛",
                    r"代码.*竞赛"
                ],
                "description": "编程和算法相关竞赛"
            },
            "数据科学": {
                "keywords": ["数据", "挖掘", "分析", "统计", "大数据"],
                "patterns": [
                    r"数据.*专项赛",
                    r"挖掘.*竞赛",
                    r"数据.*分析",
                    r"智能.*数据"
                ],
                "description": "数据科学相关竞赛"
            },
            "芯片技术": {
                "keywords": ["芯片", "微处理器", "集成电路", "计算思维"],
                "patterns": [
                    r"芯片.*专项赛",
                    r"智能.*芯片",
                    r"计算.*思维"
                ],
                "description": "芯片和计算思维相关竞赛"
            },
            "自动驾驶": {
                "keywords": ["无人驾驶", "自动驾驶", "智能车", "自动汽车"],
                "patterns": [
                    r"无人驾驶.*专项赛",
                    r"自动驾驶.*竞赛",
                    r"智能车.*竞赛"
                ],
                "description": "自动驾驶相关竞赛"
            },
            "虚拟仿真": {
                "keywords": ["虚拟", "仿真", "模拟", "VR", "AR"],
                "patterns": [
                    r"虚拟.*专项赛",
                    r"仿真.*竞赛",
                    r"虚拟.*仿真"
                ],
                "description": "虚拟仿真相关竞赛"
            },
            "智慧城市": {
                "keywords": ["智慧城市", "城市设计", "智慧社区", "城市规划"],
                "patterns": [
                    r"智慧城市.*专项赛",
                    r"城市.*设计",
                    r"智慧.*城市"
                ],
                "description": "智慧城市相关竞赛"
            },
            "太空探索": {
                "keywords": ["太空", "航天", "宇航", "太空电梯", "太空探索"],
                "patterns": [
                    r"太空.*专项赛",
                    r"太空.*竞赛",
                    r"航天.*竞赛"
                ],
                "description": "太空探索相关竞赛"
            },
            "其他": {
                "keywords": ["创新", "设计", "应用", "专项赛"],
                "patterns": [
                    r".*专项赛",
                    r".*竞赛"
                ],
                "description": "其他类型竞赛"
            }
        }
        
        logger.info("竞赛分析器初始化完成")
    
    def analyze_competitions(self, data_directory: str = "data") -> Dict[str, List[CompetitionInfo]]:
        """
        分析指定目录下的所有竞赛文档
        
        Args:
            data_directory: 数据目录路径
            
        Returns:
            按类别分组的竞赛信息字典
        """
        logger.info(f"开始分析竞赛文档: {data_directory}")
        
        # 获取所有PDF文件
        pdf_files = glob.glob(os.path.join(data_directory, "*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        # 分析每个文件
        competitions_by_category = defaultdict(list)
        
        for pdf_file in pdf_files:
            competition_info = self._analyze_single_competition(pdf_file)
            if competition_info:
                category = competition_info.category
                competitions_by_category[category].append(competition_info)
                logger.info(f"分析完成: {competition_info.name} -> {category}")
        
        # 统计结果
        total_competitions = sum(len(comps) for comps in competitions_by_category.values())
        logger.info(f"竞赛分析完成，共发现 {total_competitions} 个竞赛")
        
        for category, comps in competitions_by_category.items():
            logger.info(f"  {category}: {len(comps)} 个")
        
        return dict(competitions_by_category)
    
    def _analyze_single_competition(self, file_path: str) -> Optional[CompetitionInfo]:
        """
        分析单个竞赛文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            竞赛信息对象
        """
        try:
            # 从文件名提取基本信息
            filename = os.path.basename(file_path)
            name = self._extract_competition_name(filename)
            
            if not name:
                logger.warning(f"无法从文件名提取竞赛名称: {filename}")
                return None
            
            # 确定竞赛类别
            category = self._classify_competition(name, filename)
            
            # 提取竞赛描述（从文件内容的前1000个字符）
            description = self._extract_description(file_path)
            
            # 构建竞赛信息对象
            competition_info = CompetitionInfo(
                name=name,
                category=category,
                file_path=file_path,
                description=description,
                registration_time="",
                competition_time="",
                target_audience="",
                team_size="",
                requirements=[],
                awards=[],
                keywords=self._extract_keywords(name, description)
            )
            
            return competition_info
            
        except Exception as e:
            logger.error(f"分析竞赛文件失败 {file_path}: {e}")
            return None
    
    def _extract_competition_name(self, filename: str) -> str:
        """从文件名提取竞赛名称"""
        # 移除扩展名
        name = os.path.splitext(filename)[0]
        
        # 移除编号前缀（如 "01_", "02_" 等）
        name = re.sub(r'^\d+_', '', name)
        
        return name
    
    def _classify_competition(self, name: str, filename: str) -> str:
        """对竞赛进行分类"""
        text_to_analyze = f"{name} {filename}".lower()
        
        best_category = "其他"
        best_score = 0
        
        for category, config in self.competition_categories.items():
            score = 0
            
            # 关键词匹配
            for keyword in config["keywords"]:
                if keyword.lower() in text_to_analyze:
                    score += 1
            
            # 模式匹配
            for pattern in config["patterns"]:
                if re.search(pattern, text_to_analyze):
                    score += 2  # 模式匹配权重更高
            
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    def _extract_description(self, file_path: str) -> str:
        """提取竞赛描述"""
        try:
            # 这里可以集成PDF文本提取功能
            # 暂时返回空字符串，后续可以扩展
            return ""
        except Exception as e:
            logger.warning(f"提取描述失败 {file_path}: {e}")
            return ""
    
    def _extract_keywords(self, name: str, description: str) -> Set[str]:
        """提取关键词"""
        keywords = set()
        
        # 从名称中提取关键词
        for category, config in self.competition_categories.items():
            for keyword in config["keywords"]:
                if keyword.lower() in name.lower():
                    keywords.add(keyword)
        
        # 从描述中提取关键词
        if description:
            for category, config in self.competition_categories.items():
                for keyword in config["keywords"]:
                    if keyword.lower() in description.lower():
                        keywords.add(keyword)
        
        return keywords
    
    def get_competition_statistics(self, competitions_by_category: Dict[str, List[CompetitionInfo]]) -> Dict:
        """获取竞赛统计信息"""
        stats = {
            "total_competitions": 0,
            "categories": {},
            "robot_competitions": 0,
            "ai_competitions": 0,
            "programming_competitions": 0,
            "other_competitions": 0
        }
        
        for category, competitions in competitions_by_category.items():
            count = len(competitions)
            stats["total_competitions"] += count
            stats["categories"][category] = count
            
            # 统计特定类别
            if category == "机器人":
                stats["robot_competitions"] = count
            elif category == "人工智能":
                stats["ai_competitions"] = count
            elif category == "编程设计":
                stats["programming_competitions"] = count
            else:
                stats["other_competitions"] += count
        
        return stats
    
    def answer_competition_question(self, question: str, competitions_by_category: Dict[str, List[CompetitionInfo]]) -> str:
        """回答竞赛相关问题"""
        question_lower = question.lower()
        
        # 统计机器人竞赛数量
        if "机器人" in question and ("多少个" in question or "数量" in question):
            robot_count = len(competitions_by_category.get("机器人", []))
            return f"根据分析，第七届全国青少年人工智能创新挑战赛中共有 {robot_count} 个机器人类别的竞赛。"
        
        # 统计所有竞赛数量
        elif "多少个" in question or "数量" in question:
            total_count = sum(len(comps) for comps in competitions_by_category.values())
            return f"根据分析，第七届全国青少年人工智能创新挑战赛中共有 {total_count} 个竞赛类别。"
        
        # 列出所有竞赛类别
        elif "哪些" in question or "类别" in question:
            categories = list(competitions_by_category.keys())
            return f"竞赛包括以下类别：{', '.join(categories)}"
        
        # 默认回答
        else:
            return "请提供更具体的竞赛相关问题。" 