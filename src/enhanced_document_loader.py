"""
增强的文档加载器
支持按章节切分PDF文档，并将标题信息拼接到chunk开头以提高检索效果
"""

import os
import re
from typing import List, Dict, Optional, Tuple, Set
from langchain_core.documents import Document  # 导入Document类
from langchain_community.document_loaders import PyPDFLoader  # 导入PyPDFLoader类
import pdfplumber  # 导入pdfplumber库
from loguru import logger  # 导入loguru库

class EnhancedDocumentLoader:
    """增强的文档加载器 - 支持章节切分和标题增强"""
    
    def __init__(self):
        """初始化增强文档加载器"""
        # 章节标题模式
        self.section_patterns = [
            r'^[一二三四五六七八九十][\s]*[、.．][\s]*(.+)$',  # 一、二、等
            r'^[0-9]+[\s]*[、.．][\s]*(.+)$',                    # 1. 2. 等
            r'^[（(][一二三四五六七八九十][）)][\s]*(.+)$',         # （一）等
            r'^[（(][0-9]+[）)][\s]*(.+)$',                      # （1）等
            r'^第[一二三四五六七八九十]+[章节部分][\s]*(.+)$',      # 第一章等
            r'^[A-Z][\s]*[、.．][\s]*(.+)$',                    # A. B. 等
            r'^\d+\.\d+[\s]*(.+)$',                            # 1.1 1.2 等
            r'^【(.+)】$',                                     # 【标题】
            r'^(?:任务描述|基本要求|技术要求|评分标准|实现方案)[:：]?(.*)$',  # 特定任务关键词
        ]
        
        # 重要关键词（用于增强标题）
        self.important_keywords = {
            "未来校园智能应用专项赛",
            "智能交通信号灯",
            "基本要求",
            "技术要求", 
            "任务描述",
            "评分标准",
            "实现方案",
            "设计要求",
            "算法设计",
            "系统架构"
        }
        
        logger.info("增强文档加载器初始化完成")
    
    def _extract_chinese_entities_from_filename(self, file_path: str) -> Set[str]:
        """
        从文件名中提取中文实体
        
        Args:
            file_path: 文件路径
            
        Returns:
            提取到的中文实体集合
        """
        try:
            # 获取文件名（不包含路径和扩展名）
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            # 中文字符正则模式（连续的中文字符）
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
            
            # 提取中文字符串
            chinese_matches = chinese_pattern.findall(filename)
            
            chinese_entities = set() # 创建一个集合来存储提取到的中文实体
            
            # 添加完整的中文字符串
            for match in chinese_matches:
                if len(match) >= 2:  # 至少2个字符才作为实体
                    chinese_entities.add(match)
            
            # 尝试识别常见的实体模式
            entity_patterns = [
                r'(\w*竞赛\w*)',
                r'(\w*专项赛\w*)', 
                r'(\w*智能\w*)',
                r'(\w*应用\w*)',
                r'(\w*设计\w*)',
                r'(\w*算法\w*)',
                r'(\w*系统\w*)',
                r'(\w*平台\w*)',
                r'(\w*创新\w*)',
                r'(\w*机器人\w*)',
                r'(\w*数据\w*)',
                r'(\w*挖掘\w*)',
                r'(\w*分析\w*)',
                r'(\w*模型\w*)',
                r'(\w*优化\w*)',
                r'(\w*控制\w*)',
                r'(\w*检测\w*)',
                r'(\w*识别\w*)',
                r'(\w*处理\w*)'
            ]
            
            for pattern in entity_patterns:
                matches = re.findall(pattern, filename)
                for match in matches:
                    if len(match) >= 2:
                        chinese_entities.add(match)
            
            # 特殊处理：提取数字+中文的组合（如"2024年"、"第12届"等）
            number_chinese_pattern = r'(\d+[年届期回轮次])'
            number_matches = re.findall(number_chinese_pattern, filename)
            for match in number_matches:
                chinese_entities.add(match)
            
            # 提取题目编号（如"C题"、"A题"等）
            topic_pattern = r'([A-Z]题)'
            topic_matches = re.findall(topic_pattern, filename)
            for match in topic_matches:
                chinese_entities.add(match)
            
            logger.debug(f"从文件名 '{filename}' 中提取到中文实体: {chinese_entities}")
            return chinese_entities
            
        except Exception as e:
            logger.warning(f"提取文件名中文实体失败: {e}")
            return set()
    
    def load_pdf_with_chapter_splitting(self, file_path: str) -> List[Document]:
        """
        加载PDF并按章节切分
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            按章节切分的Document对象列表
        """
        try:
            logger.info(f"开始加载PDF并进行章节切分: {file_path}")
            
            # 提取文件名中的中文实体
            filename_entities = self._extract_chinese_entities_from_filename(file_path)
            logger.info(f"从文件名提取到 {len(filename_entities)} 个中文实体: {filename_entities}")
            
            # 首先获取原始文档
            raw_documents = self._load_raw_pdf(file_path)
            
            if not raw_documents:
                logger.warning(f"无法加载PDF文件: {file_path}")
                return []
            
            # 合并所有页面内容
            full_text = ""
            page_boundaries = []
            current_pos = 0
            
            for doc in raw_documents:
                page_content = doc.page_content
                full_text += page_content + "\n"
                current_pos += len(page_content) + 1
                page_boundaries.append(current_pos)
            
            # 检测章节结构
            sections = self._detect_sections(full_text, file_path)
            
            if not sections:
                logger.info("未检测到章节结构，使用段落切分")
                return self._split_by_paragraphs_with_title_enhancement(raw_documents, file_path, filename_entities)
            
            # 按章节切分文档
            chapter_documents = self._create_chapter_documents(sections, file_path, filename_entities)
            
            logger.info(f"章节切分完成，共生成 {len(chapter_documents)} 个章节文档")
            
            # 进一步细分每个章节
            final_documents = []
            for chapter_doc in chapter_documents:
                sub_docs = self._split_chapter_into_chunks(chapter_doc)
                final_documents.extend(sub_docs)
            
            logger.info(f"最终生成 {len(final_documents)} 个文档块")
            return final_documents
            
        except Exception as e:
            logger.error(f"章节切分加载失败 {file_path}: {e}")
            # 降级到普通加载
            return self._load_raw_pdf(file_path)
    
    def _load_raw_pdf(self, file_path: str) -> List[Document]:
        """加载原始PDF文档"""
        try:
            # 优先使用pdfplumber
            documents = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()  # 提取页面文本
                    if text and text.strip():
                        doc = Document(
                            page_content=text.strip(),
                            metadata={
                                'source': os.path.basename(file_path),
                                'file_path': file_path,
                                'page': i + 1,
                                'loader': 'pdfplumber_enhanced'
                            }
                        )
                        documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.warning(f"pdfplumber加载失败，尝试PyPDF: {e}")
            # 降级到PyPDF
            try:
                loader = PyPDFLoader(file_path)  # 创建PyPDFLoader对象
                documents = loader.load()  # 加载文档
                
                for i, doc in enumerate(documents):
                    doc.metadata.update({
                        'source': os.path.basename(file_path),
                        'file_path': file_path,
                        'page': i + 1,
                        'loader': 'pypdf_enhanced'
                    })
                
                return documents
            except Exception as e2:
                logger.error(f"PyPDF加载也失败: {e2}")
                return []
    
    def _detect_sections(self, text: str, file_path: str) -> List[Dict]:
        """
        检测文档中的章节结构
        
        Args:
            text: 完整文档文本
            file_path: 文件路径
            
        Returns:
            章节信息列表
        """
        sections = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 检查是否匹配章节模式
            section_title = self._match_section_pattern(line)
            if section_title:
                # 计算在原文中的位置
                text_before = '\n'.join(lines[:i])
                start_pos = len(text_before)
                
                section_info = {
                    'title': section_title,
                    'original_line': line,
                    'start_pos': start_pos,
                    'line_number': i,
                    'level': self._determine_section_level(line)
                }
                sections.append(section_info)
        
        # 设置每个章节的结束位置
        for i, section in enumerate(sections):
            if i < len(sections) - 1:
                section['end_pos'] = sections[i + 1]['start_pos']
            else:
                section['end_pos'] = len(text)
        
        # 过滤太短的章节
        valid_sections = []
        for section in sections:
            content_length = section['end_pos'] - section['start_pos']
            if content_length > 50:  # 至少50个字符
                valid_sections.append(section)
        
        logger.info(f"检测到 {len(valid_sections)} 个有效章节")
        for i, section in enumerate(valid_sections[:5]):  # 显示前5个
            logger.debug(f"  章节{i+1}: {section['title'][:30]}... (长度: {section['end_pos'] - section['start_pos']})")
        
        return valid_sections
    
    def _match_section_pattern(self, line: str) -> Optional[str]:
        """匹配章节标题模式"""
        for pattern in self.section_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                if match.groups():
                    return match.group(1).strip()
                else:
                    return line.strip()
        
        # 检查是否包含重要关键词
        for keyword in self.important_keywords:
            if keyword in line and len(line) < 100:  # 标题不应该太长
                return line.strip()
        
        return None
    
    def _determine_section_level(self, line: str) -> int:
        """确定章节级别"""
        # 第一级：主要章节
        if re.match(r'^[一二三四五六七八九十][\s]*[、.．]', line):
            return 1
        if re.match(r'^第[一二三四五六七八九十]+[章节部分]', line):
            return 1
        
        # 第二级：子章节
        if re.match(r'^[（(][一二三四五六七八九十][）)]', line):
            return 2
        if re.match(r'^[0-9]+[\s]*[、.．]', line):
            return 2
        if re.match(r'^\d+\.\d+', line):
            return 2
        
        # 第三级：细分
        if re.match(r'^[（(][0-9]+[）)]', line):
            return 3
        if re.match(r'^[A-Z][\s]*[、.．]', line):
            return 3
        
        # 默认级别
        return 2
    
    def _create_chapter_documents(self, sections: List[Dict], file_path: str, filename_entities: Set[str] = None) -> List[Document]:
        """根据章节信息创建文档"""
        chapter_documents = []
        
        if filename_entities is None:
            filename_entities = set()
        
        # 获取完整文本
        with open(file_path, 'rb') as f:
            try:
                with pdfplumber.open(f) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
            except:
                # 降级方案
                full_text = ""
        
        for i, section in enumerate(sections):
            try:
                # 提取章节内容
                start_pos = section['start_pos']
                end_pos = section['end_pos']
                section_content = full_text[start_pos:end_pos].strip()
                
                if not section_content:
                    continue
                
                # 创建增强的章节文档
                enhanced_content = self._enhance_content_with_title(
                    section['title'], 
                    section_content,
                    section['level'],
                    filename_entities
                )
                
                doc = Document(
                    page_content=enhanced_content,
                    metadata={
                        'source': os.path.basename(file_path),
                        'file_path': file_path,
                        'chapter_title': section['title'],
                        'chapter_level': section['level'],
                        'chapter_index': i,
                        'loader': 'enhanced_chapter',
                        'original_content_length': len(section_content),
                        'enhanced': True,
                        'filename_entities': list(filename_entities),  # 将文件名实体添加到元数据
                        'entity_count': len(filename_entities)
                    }
                )
                chapter_documents.append(doc)
                
            except Exception as e:
                logger.warning(f"创建章节文档失败: {e}")
                continue
        
        return chapter_documents
    
    def _enhance_content_with_title(self, title: str, content: str, level: int, filename_entities: Set[str] = None) -> str:
        """
        将标题信息增强到内容中
        
        Args:
            title: 章节标题
            content: 原始内容
            level: 章节级别
            filename_entities: 从文件名提取的中文实体
            
        Returns:
            增强后的内容
        """
        if filename_entities is None:
            filename_entities = set()
        
        # 根据级别添加不同的标记
        level_prefix = {
            1: "【主要章节】",
            2: "【子章节】", 
            3: "【细分章节】"
        }.get(level, "【章节】")
        
        # 构建增强的标题
        enhanced_title = f"{level_prefix}{title}"
        
        # 检查标题中是否包含重要关键词，如果有则特别标记
        important_in_title = []
        for keyword in self.important_keywords:
            if keyword in title:
                important_in_title.append(keyword)
        
        if important_in_title:
            keyword_tags = " ".join([f"#关键:{kw}" for kw in important_in_title])
            enhanced_title += f" {keyword_tags}"
        
        # 添加文件名实体标记
        if filename_entities:
            filename_tags = " ".join([f"#文件实体:{entity}" for entity in filename_entities])
            enhanced_title += f" {filename_tags}"
        
        # 检查内容中是否包含文件名实体，如果有则额外标记
        content_entity_matches = []
        for entity in filename_entities:
            if entity in content and entity not in title:
                content_entity_matches.append(entity)
        
        if content_entity_matches:
            entity_match_tags = " ".join([f"#实体匹配:{entity}" for entity in content_entity_matches])
            enhanced_title += f" {entity_match_tags}"
        
        # 构建最终内容
        enhanced_content = f"{enhanced_title}\n\n{content}"
        
        return enhanced_content
    
    def _split_chapter_into_chunks(self, chapter_doc: Document, chunk_size: int = 800, overlap: int = 100) -> List[Document]:
        """
        将章节文档进一步切分为较小的chunks
        
        Args:
            chapter_doc: 章节文档
            chunk_size: chunk大小
            overlap: 重叠大小
            
        Returns:
            切分后的文档列表
        """
        content = chapter_doc.page_content
        chapter_title = chapter_doc.metadata.get('chapter_title', '')
        
        # 如果内容不够长，直接返回
        if len(content) <= chunk_size:
            return [chapter_doc]
        
        chunks = []
        start = 0
        chunk_index = 0
        max_iterations = len(content) // (chunk_size // 2) + 10  # 添加最大迭代次数保护
        iteration_count = 0
        
        while start < len(content) and iteration_count < max_iterations:
            iteration_count += 1
            end = start + chunk_size
            
            # 寻找合适的切分点
            if end < len(content):
                # 尝试在句子边界切分
                sentence_end = content.rfind('。', start, end)
                if sentence_end == -1:
                    sentence_end = content.rfind('！', start, end)
                if sentence_end == -1:
                    sentence_end = content.rfind('？', start, end)
                if sentence_end == -1:
                    sentence_end = content.rfind('\n', start, end)
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                # 为每个chunk保留章节标题信息
                if chunk_index == 0:
                    # 第一个chunk保持原样（已包含增强标题）
                    final_content = chunk_content
                else:
                    # 后续chunk添加简化的章节引用
                    title_ref = f"【续：{chapter_title}】\n\n{chunk_content}"
                    final_content = title_ref
                
                # 创建chunk文档
                chunk_metadata = chapter_doc.metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'chunk_start': start,
                    'chunk_end': end,
                    'is_continuation': chunk_index > 0
                })
                
                chunk_doc = Document(
                    page_content=final_content,
                    metadata=chunk_metadata
                )
                chunks.append(chunk_doc)
                
                chunk_index += 1
            
            # 计算下一个chunk的起始位置 - 修复死循环问题
            next_start = end - overlap
            
            # 确保能够向前推进，防止死循环
            if next_start <= start:
                # 如果计算出的下一个起始位置没有前进，强制向前移动
                next_start = start + max(1, chunk_size // 4)  # 至少前进chunk_size的1/4
                logger.warning(f"强制推进chunk切分位置: {start} -> {next_start}")
            
            start = next_start
            
            # 安全检查：如果剩余内容太少，直接处理完
            if start >= len(content) - 50:  # 如果剩余内容少于50字符，直接结束
                break
        
        if iteration_count >= max_iterations:
            logger.error(f"章节切分达到最大迭代次数 {max_iterations}，强制退出循环")
        
        logger.debug(f"章节 '{chapter_title}' 切分为 {len(chunks)} 个chunks (迭代次数: {iteration_count})")
        return chunks
    
    def _split_by_paragraphs_with_title_enhancement(self, documents: List[Document], file_path: str, filename_entities: Set[str] = None) -> List[Document]:
        """
        按段落切分并进行标题增强（降级方案）
        
        Args:
            documents: 原始文档列表
            file_path: 文件路径
            filename_entities: 从文件名提取的中文实体
            
        Returns:
            增强后的文档列表
        """
        if filename_entities is None:
            filename_entities = set()
            
        enhanced_documents = []
        
        for doc in documents: # 遍历文档列表
            content = doc.page_content # 获取文档内容
            
            # 按段落分割
            paragraphs = re.split(r'\n\s*\n', content) # 按段落分割（\n\s*\n表示两个或多个换行符，中间可以有任意数量的空格）
            
            for i, paragraph in enumerate(paragraphs): # 遍历段落列表
                paragraph = paragraph.strip() # 去除段落两端的空白字符
                if not paragraph or len(paragraph) < 20:
                    continue
                
                # 检测段落是否包含重要信息
                importance_tags = []
                for keyword in self.important_keywords:
                    if keyword in paragraph:
                        importance_tags.append(f"#重要:{keyword}")
                
                # 检测段落是否包含文件名实体
                entity_tags = []
                for entity in filename_entities:
                    if entity in paragraph:
                        entity_tags.append(f"#文件实体:{entity}")
                
                # 增强段落内容
                all_tags = importance_tags + entity_tags
                if all_tags:
                    enhanced_paragraph = f"【重要段落】{' '.join(all_tags)}\n\n{paragraph}"
                else:
                    enhanced_paragraph = paragraph
                
                # 创建增强文档
                enhanced_metadata = doc.metadata.copy() # 复制文档元数据
                enhanced_metadata.update({
                    'paragraph_index': i,
                    'enhanced': True,
                    'importance_tags': importance_tags,
                    'entity_tags': entity_tags,
                    'filename_entities': list(filename_entities),
                    'entity_count': len(filename_entities),
                    'loader': 'enhanced_paragraph'
                })
                
                enhanced_doc = Document(
                    page_content=enhanced_paragraph,
                    metadata=enhanced_metadata
                )
                enhanced_documents.append(enhanced_doc)
        
        logger.info(f"段落增强完成，生成 {len(enhanced_documents)} 个增强文档")
        return enhanced_documents
    
    def load_pdf(self, file_path: str, use_chapter_splitting: bool = True) -> List[Document]:
        """
        加载PDF文档（主入口）
        
        Args:
            file_path: PDF文件路径
            use_chapter_splitting: 是否使用章节切分
            
        Returns:
            Document对象列表
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return [] # 返回空列表
        
        if use_chapter_splitting:
            return self.load_pdf_with_chapter_splitting(file_path) # 使用章节切分加载
        else:
            # 普通加载但仍进行标题增强和文件名实体提取
            filename_entities = self._extract_chinese_entities_from_filename(file_path)
            raw_docs = self._load_raw_pdf(file_path)  # Document对象列表
            return self._split_by_paragraphs_with_title_enhancement(raw_docs, file_path, filename_entities)
    
    def get_document_structure_analysis(self, file_path: str) -> Dict:
        """
        分析文档结构
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            文档结构分析结果
        """
        try:
            # 加载原始文档
            raw_documents = self._load_raw_pdf(file_path)
            if not raw_documents:
                return {'error': '无法加载文档'}
            
            # 合并文本
            full_text = '\n'.join([doc.page_content for doc in raw_documents])
            
            # 检测章节
            sections = self._detect_sections(full_text, file_path)
            
            # 提取文件名实体
            filename_entities = self._extract_chinese_entities_from_filename(file_path)
            
            # 分析重要关键词分布
            keyword_distribution = {}
            for keyword in self.important_keywords:
                count = full_text.count(keyword)
                if count > 0:
                    keyword_distribution[keyword] = count
            
            # 分析文件名实体在文档中的分布
            entity_distribution = {}
            for entity in filename_entities:
                count = full_text.count(entity)
                entity_distribution[entity] = count
            
            analysis = {
                'file_path': file_path,
                'total_pages': len(raw_documents),
                'total_characters': len(full_text),
                'sections_detected': len(sections),
                'sections': [
                    {
                        'title': s['title'],
                        'level': s['level'],
                        'length': s['end_pos'] - s['start_pos']
                    } for s in sections
                ],
                'keyword_distribution': keyword_distribution,
                'filename_entities': list(filename_entities),
                'entity_distribution': entity_distribution,
                'entity_match_rate': len([e for e in filename_entities if e in full_text]) / max(len(filename_entities), 1),
                'structure_quality': 'good' if len(sections) > 3 else 'poor'
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"文档结构分析失败: {e}")
            return {'error': str(e)} 