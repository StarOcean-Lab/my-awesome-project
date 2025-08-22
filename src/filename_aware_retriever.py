"""
基于文件名感知的检索器
从用户问题中检测文件名，实现定向文件检索
"""

import re
import os
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger
import jieba

try:
    from .langchain_retriever import LangChainHybridRetriever
except ImportError:
    from langchain_retriever import LangChainHybridRetriever


class FilenameAwareRetriever(BaseRetriever):
    """基于文件名感知的检索器"""
    
    # Pydantic字段定义
    base_retriever: Any
    documents: List[Document] = []
    enable_filename_detection: bool = True
    fallback_to_global: bool = True
    min_confidence: float = 0.6
    k: int = 10
    file_index: Dict[str, List[Document]] = {}
    filename_keywords: Dict[str, List[str]] = {}
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, 
                 base_retriever: LangChainHybridRetriever,
                 documents: List[Document] = None,
                 enable_filename_detection: bool = True,
                 fallback_to_global: bool = True,
                 min_confidence: float = 0.6,
                 k: int = 10):
        """
        初始化文件名感知检索器
        
        Args:
            base_retriever: 基础检索器
            documents: 文档列表
            enable_filename_detection: 是否启用文件名检测
            fallback_to_global: 当没有检测到文件名时是否降级到全局检索
            min_confidence: 文件名匹配的最小置信度
            k: 默认返回结果数量
        """
        # 构建文件索引和关键词映射
        file_index = self._build_file_index_static(documents if documents else [])
        filename_keywords = self._build_filename_keywords_static(file_index)
        
        super().__init__(
            base_retriever=base_retriever,
            documents=documents if documents else [],
            enable_filename_detection=enable_filename_detection,
            fallback_to_global=fallback_to_global,
            min_confidence=min_confidence,
            k=k,
            file_index=file_index,
            filename_keywords=filename_keywords
        )
        
        logger.info(f"文件名感知检索器初始化完成")
        logger.info(f"  - 文档总数: {len(self.documents)}")
        logger.info(f"  - 文件数量: {len(self.file_index)}")
        logger.info(f"  - 关键词数量: {len(self.filename_keywords)}")
    
    @staticmethod
    def _build_file_index_static(documents: List[Document]) -> Dict[str, List[Document]]:
        """构建文件索引，按文件路径分组文档（静态方法）"""
        file_index = defaultdict(list)
        
        for doc in documents:
            # 尝试多种方式获取文件路径
            file_path = (
                doc.metadata.get('file_path') or 
                doc.metadata.get('source') or 
                doc.metadata.get('filename', '')
            )
            
            if file_path and file_path != 'unknown':
                # 标准化文件路径
                normalized_path = os.path.normpath(file_path)
                file_index[normalized_path].append(doc)
        
        logger.info(f"构建文件索引完成，包含 {len(file_index)} 个文件:")
        for file_path, docs in file_index.items():
            filename = os.path.basename(file_path)
            logger.info(f"  📄 {filename}: {len(docs)} 个文档块")
        
        return dict(file_index)
    
    def _build_file_index(self) -> Dict[str, List[Document]]:
        """构建文件索引，按文件路径分组文档"""
        return self._build_file_index_static(self.documents)
    
    @staticmethod  
    def _build_filename_keywords_static(file_index: Dict[str, List[Document]]) -> Dict[str, List[str]]:
        """构建文件名关键词映射（静态方法）"""
        keywords_map = {}
        
        for file_path in file_index.keys():
            filename = os.path.basename(file_path)
            
            # 提取文件名中的关键词
            keywords = FilenameAwareRetriever._extract_keywords_from_filename_static(filename)
            
            if keywords:
                keywords_map[file_path] = keywords
                logger.debug(f"文件 {filename} 关键词: {keywords}")
        
        return keywords_map
    
    def _build_filename_keywords(self) -> Dict[str, List[str]]:
        """构建文件名关键词映射"""
        return self._build_filename_keywords_static(self.file_index)
    
    @staticmethod
    def _extract_keywords_from_filename_static(filename: str) -> List[str]:
        """从文件名中提取关键词（静态方法）"""
        keywords = []
        
        # 移除文件扩展名
        name_without_ext = os.path.splitext(filename)[0]
        
        # 1. 提取编号模式（如 "01_", "02_"）
        number_match = re.match(r'^(\d+)_', name_without_ext)
        if number_match:
            keywords.append(number_match.group(1))
        
        # 2. 提取中文内容
        chinese_parts = re.findall(r'[一-龥]+', name_without_ext)
        keywords.extend(chinese_parts)
        
        # 3. 使用jieba分词提取关键词
        try:
            words = jieba.lcut(name_without_ext)
            for word in words:
                if len(word) >= 2 and re.match(r'[一-龥]', word):
                    keywords.append(word)
        except:
            pass
        
        # 4. 提取特殊模式（如年份、赛事名称等）
        special_patterns = [
            r'(\d{4}年)',  # 年份
            r'(第\d+届)',  # 届数
            r'([一-龥]*专项赛)',  # 专项赛
            r'([一-龥]*挑战赛)',  # 挑战赛
            r'([一-龥]*竞赛)',    # 竞赛
            r'([一-龥]*设计)',    # 设计
            r'([一-龥]*应用)',    # 应用
            r'([一-龥]*智能[一-龥]*)',  # 智能相关
        ]
        
        for pattern in special_patterns:
            matches = re.findall(pattern, name_without_ext)
            keywords.extend(matches)
        
        # 5. 去重并过滤短词
        keywords = list(set(keywords))
        keywords = [kw for kw in keywords if len(kw) >= 2]
        
        return keywords
    
    def _extract_keywords_from_filename(self, filename: str) -> List[str]:
        """从文件名中提取关键词"""
        return self._extract_keywords_from_filename_static(filename)
    
    def detect_target_files(self, query: str) -> Tuple[List[str], float]:
        """
        从查询中检测目标文件
        
        Args:
            query: 用户查询
            
        Returns:
            (匹配的文件路径列表, 置信度分数)
        """
        if not self.enable_filename_detection:
            return [], 0.0
        
        # 对查询进行预处理
        query_clean = re.sub(r'[^\w\s一-龥]', ' ', query)
        query_words = set(jieba.lcut(query_clean))
        
        # 收集所有匹配结果
        file_scores = {}
        
        for file_path, keywords in self.filename_keywords.items():
            score = 0.0
            matched_keywords = []
            
            for keyword in keywords:
                # 直接匹配
                if keyword in query:
                    score += len(keyword) * 2  # 直接匹配给更高分
                    matched_keywords.append(keyword)
                
                # 分词匹配
                elif keyword in query_words:
                    score += len(keyword)
                    matched_keywords.append(keyword)
                
                # 模糊匹配
                elif any(keyword in word or word in keyword for word in query_words):
                    score += len(keyword) * 0.5
                    matched_keywords.append(keyword)
            
            if score > 0:
                # 计算相对分数
                max_possible_score = sum(len(kw) * 2 for kw in keywords)
                relative_score = score / max_possible_score if max_possible_score > 0 else 0
                
                file_scores[file_path] = {
                    'score': relative_score,
                    'matched_keywords': matched_keywords,
                    'absolute_score': score
                }
        
        # 排序并选择最佳匹配
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # 选择置信度足够的文件
        target_files = []
        max_confidence = 0.0
        
        for file_path, info in sorted_files:
            confidence = info['score']
            if confidence >= self.min_confidence:
                target_files.append(file_path)
                max_confidence = max(max_confidence, confidence)
                
                filename = os.path.basename(file_path)
                logger.info(f"🎯 检测到目标文件: {filename} (置信度: {confidence:.2f})")
                logger.info(f"   匹配关键词: {info['matched_keywords']}")
        
        return target_files, max_confidence
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        获取相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        try:
            logger.info(f"🔍 文件名感知检索: '{query[:50]}...'")
            
            # 1. 检测目标文件
            target_files, confidence = self.detect_target_files(query)
            
            if target_files and confidence > 0:
                logger.info(f"📁 定向检索模式: 在 {len(target_files)} 个文件中检索")
                return self._targeted_retrieval(query, target_files)
            
            elif self.fallback_to_global:
                logger.info("🌍 降级到全局检索模式")
                return self._global_retrieval(query)
            
            else:
                logger.warning("❌ 未检测到目标文件且不允许全局检索")
                return []
                
        except Exception as e:
            logger.error(f"文件名感知检索失败: {e}")
            if self.fallback_to_global:
                return self._global_retrieval(query)
            return []
    
    def _targeted_retrieval(self, query: str, target_files: List[str]) -> List[Document]:
        """
        在指定文件中进行定向检索
        
        Args:
            query: 查询文本
            target_files: 目标文件路径列表
            
        Returns:
            检索结果
        """
        # 收集目标文件的所有文档
        target_documents = []
        for file_path in target_files:
            if file_path in self.file_index:
                target_documents.extend(self.file_index[file_path])
        
        if not target_documents:
            logger.warning("目标文件中没有找到文档")
            return []
        
        logger.info(f"📚 在 {len(target_documents)} 个目标文档中检索")
        
        # 为定向检索创建临时检索器
        return self._search_in_documents(query, target_documents)
    
    def _search_in_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        在指定文档集合中进行检索
        
        Args:
            query: 查询文本
            documents: 文档集合
            
        Returns:
            检索结果
        """
        try:
            # 方法1: 使用基础检索器的向量存储进行过滤检索
            if hasattr(self.base_retriever, 'vectorstore') and self.base_retriever.vectorstore:
                # 获取所有候选结果
                all_candidates = self.base_retriever.vectorstore.similarity_search(query, k=self.k * 3)
                
                # 过滤出属于目标文档的结果
                target_contents = {doc.page_content for doc in documents}
                filtered_results = []
                
                for candidate in all_candidates:
                    if candidate.page_content in target_contents:
                        filtered_results.append(candidate)
                        if len(filtered_results) >= self.k:
                            break
                
                if filtered_results:
                    logger.info(f"✅ 定向检索成功，返回 {len(filtered_results)} 个结果")
                    return filtered_results
            
            # 方法2: 基于关键词的简单匹配
            return self._keyword_based_search(query, documents)
            
        except Exception as e:
            logger.warning(f"定向检索失败: {e}，使用关键词匹配")
            return self._keyword_based_search(query, documents)
    
    def _keyword_based_search(self, query: str, documents: List[Document]) -> List[Document]:
        """
        基于关键词的简单搜索
        
        Args:
            query: 查询文本
            documents: 文档集合
            
        Returns:
            搜索结果
        """
        # 提取查询关键词
        query_keywords = set(jieba.lcut(query))
        query_keywords = {kw for kw in query_keywords if len(kw) >= 2}
        
        # 计算每个文档的相关性分数
        scored_docs = []
        
        for doc in documents:
            content = doc.page_content
            score = 0
            
            # 直接匹配
            for keyword in query_keywords:
                if keyword in content:
                    score += content.count(keyword) * len(keyword)
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # 排序并返回
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        results = [doc for doc, score in scored_docs[:self.k]]
        
        logger.info(f"🔍 关键词搜索返回 {len(results)} 个结果")
        return results
    
    def _global_retrieval(self, query: str) -> List[Document]:
        """
        全局检索
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果
        """
        return self.base_retriever.get_relevant_documents(query)
    
    def get_file_statistics(self) -> Dict:
        """获取文件统计信息"""
        stats = {
            'total_files': len(self.file_index),
            'total_documents': len(self.documents),
            'files_with_keywords': len(self.filename_keywords),
            'file_details': []
        }
        
        for file_path, docs in self.file_index.items():
            filename = os.path.basename(file_path)
            keywords = self.filename_keywords.get(file_path, [])
            
            stats['file_details'].append({
                'filename': filename,
                'file_path': file_path,
                'document_count': len(docs),
                'keywords': keywords
            })
        
        return stats 