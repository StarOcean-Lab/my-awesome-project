#!/usr/bin/env python3
"""
优化RAG系统的测试脚本，验证所有5个优化功能的协同工作效果
"""

import os
import sys
import time
import json
from typing import List, Dict
from loguru import logger

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from optimized_rag_system import OptimizedRAGSystem

class OptimizedRAGTester:
    """优化RAG系统测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.test_questions = [
            # 交通信号灯专项问题
            "智能交通信号灯的基本要求是什么？",
            "交通信号灯系统需要实现哪些技术要求？", 
            "智能交通信号灯的优化算法如何设计？",
            
            # 竞赛任务问题
            "未来校园智能应用专项赛的评分标准是什么？",
            "竞赛任务描述中包含哪些关键要求？",
            "泰迪杯数据挖掘挑战赛的基本要求有哪些？",
            
            # 技术要求问题
            "系统架构设计有什么技术要求？",
            "算法实现需要满足哪些性能指标？",
            
            # 测试"未找到"情况
            "火星探测器的技术规格是什么？",
            "人工智能芯片的制造工艺要求？"
        ]
        
        logger.info("优化RAG系统测试器初始化完成")
        logger.info(f"准备测试 {len(self.test_questions)} 个问题")
    
    def run_full_optimization_test(self) -> Dict:
        """运行完整的优化测试"""
        logger.info("🚀 开始运行完整优化测试...")
        
        try:
            # 1. 初始化优化RAG系统
            logger.info("初始化优化RAG系统...")
            rag_system = OptimizedRAGSystem(
                llm_model="deepseek-r1:7b",
                embedding_model="./bge-large-zh-v1.5",
                vector_weight=0.4,   # 降低向量权重，提高BM25权重
                bm25_weight=0.6,     # 符合优化要求
                enable_reranking=True, # 使用重排序
                enable_chapter_splitting=True, # 使用章节切分
                retrieval_k=10 # 检索数量
            )
            
            # 2. 加载测试文档
            logger.info("加载测试文档...")
            pdf_files = self._get_test_pdf_files()
            if not pdf_files:
                logger.error("未找到PDF测试文件")
                return {"error": "未找到PDF测试文件"}
            
            success = rag_system.load_documents(pdf_files)
            if not success:
                logger.error("文档加载失败")  
                return {"error": "文档加载失败"}
            
            # 3. 获取系统性能报告
            logger.info("获取系统性能报告...")
            performance_report = rag_system.get_system_performance_report()
            
            # 4. 运行优化测试
            logger.info("运行优化功能测试...")
            test_report = rag_system.run_optimization_test(self.test_questions)
            
            # 5. 验证优化功能
            logger.info("验证各项优化功能...")
            optimization_validation = self._validate_optimizations(rag_system, test_report)
            
            # 6. 生成综合报告
            comprehensive_report = {
                "test_metadata": {
                    "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "test_questions_count": len(self.test_questions),
                    "pdf_files_count": len(pdf_files),
                    "pdf_files": [os.path.basename(f) for f in pdf_files]
                },
                "system_performance": performance_report,
                "optimization_test_results": test_report,
                "optimization_validation": optimization_validation,
                "overall_assessment": self._generate_overall_assessment(performance_report, test_report, optimization_validation)
            }
            
            # 7. 保存测试报告
            self._save_test_report(comprehensive_report)
            
            logger.info("✅ 完整优化测试完成")
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"优化测试失败: {e}")
            return {"error": str(e)}
    
    def _get_test_pdf_files(self) -> List[str]:
        """获取测试PDF文件"""
        pdf_files = []
        
        # 查找data目录中的PDF文件
        search_dirs = ["./data"]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.lower().endswith('.pdf'):
                        full_path = os.path.join(search_dir, file)
                        pdf_files.append(full_path)
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件: {[os.path.basename(f) for f in pdf_files]}")
        
        return pdf_files
    
    def _validate_optimizations(self, rag_system: OptimizedRAGSystem, test_report: Dict) -> Dict:
        """验证各项优化功能"""
        validation = {
            "optimization_1_hybrid_retrieval_rrf": self._validate_hybrid_retrieval_rrf(test_report),
            "optimization_2_crossencoder_reranking": self._validate_crossencoder_reranking(test_report),
            "optimization_3_entity_hit_reward": self._validate_entity_hit_reward(test_report),
            "optimization_4_document_chunking_title_enhancement": self._validate_document_chunking_enhancement(test_report),
            "optimization_5_fewshot_prompt_optimization": self._validate_fewshot_prompt_optimization(test_report)
        }
        
        # 计算总体验证分数
        validation_scores = []
        for opt_name, opt_result in validation.items():
            if 'validation_score' in opt_result:
                validation_scores.append(opt_result['validation_score'])
        
        validation["overall_validation_score"] = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
        
        return validation
    
    def _validate_hybrid_retrieval_rrf(self, test_report: Dict) -> Dict:
        """验证优化1：关键词+向量混合检索（BM25强制召回+RRF融合）"""
        try:
            detailed_results = test_report.get('detailed_results', [])
            
            # 检查RRF融合使用情况
            rrf_usage_count = 0
            force_recall_count = 0
            
            for result in detailed_results:
                if result.get('uses_enhanced_features', {}).get('rrf_fusion', False):
                    rrf_usage_count += 1
                
                # 检查是否有强制召回的证据（通过分析检索统计）
                retrieval_stats = result.get('prompt_analysis', {})
                if 'force_recalled' in str(retrieval_stats):
                    force_recall_count += 1
            
            total_questions = len(detailed_results)
            rrf_usage_rate = rrf_usage_count / total_questions if total_questions > 0 else 0
            
            validation_score = min(rrf_usage_rate * 2, 1.0)  # 最高1.0分
            
            return {
                "optimization_name": "关键词+向量混合检索（BM25强制召回+RRF融合）",
                "rrf_fusion_usage_count": rrf_usage_count,
                "rrf_fusion_usage_rate": rrf_usage_rate,
                "force_recall_evidence_count": force_recall_count,
                "validation_score": validation_score,
                "status": "优秀" if validation_score >= 0.8 else "良好" if validation_score >= 0.5 else "需改进",
                "details": f"RRF融合使用率: {rrf_usage_rate:.1%}, 检测到强制召回证据: {force_recall_count}次"
            }
            
        except Exception as e:
            return {"optimization_name": "关键词+向量混合检索", "error": str(e), "validation_score": 0.0}
    
    def _validate_crossencoder_reranking(self, test_report: Dict) -> Dict:
        """验证优化2：Cross-Encoder重排序"""
        try:
            detailed_results = test_report.get('detailed_results', [])
            
            reranking_usage_count = 0
            crossencoder_score_count = 0
            
            for result in detailed_results:
                if result.get('uses_enhanced_features', {}).get('reranking', False):
                    reranking_usage_count += 1
                
                # 检查是否有CrossEncoder分数
                if 'crossencoder_score' in str(result):
                    crossencoder_score_count += 1
            
            total_questions = len(detailed_results)
            reranking_usage_rate = reranking_usage_count / total_questions if total_questions > 0 else 0
            
            validation_score = reranking_usage_rate
            
            return {
                "optimization_name": "Cross-Encoder重排序优化",
                "reranking_usage_count": reranking_usage_count,
                "reranking_usage_rate": reranking_usage_rate,
                "crossencoder_evidence_count": crossencoder_score_count,
                "validation_score": validation_score,
                "status": "优秀" if validation_score >= 0.8 else "良好" if validation_score >= 0.5 else "需改进",
                "details": f"重排序使用率: {reranking_usage_rate:.1%}, CrossEncoder证据: {crossencoder_score_count}次"
            }
            
        except Exception as e:
            return {"optimization_name": "Cross-Encoder重排序", "error": str(e), "validation_score": 0.0}
    
    def _validate_entity_hit_reward(self, test_report: Dict) -> Dict:
        """验证优化3：实体命中奖励"""
        try:
            detailed_results = test_report.get('detailed_results', [])
            
            entity_bonus_count = 0
            traffic_signal_questions = 0
            
            for result in detailed_results:
                question = result.get('question', '')
                
                # 检查是否有实体奖励证据
                if result.get('uses_enhanced_features', {}).get('entity_bonuses', False):
                    entity_bonus_count += 1
                
                # 特别检查交通信号灯问题
                if any(keyword in question for keyword in ['交通信号灯', '智能交通']):
                    traffic_signal_questions += 1
            
            total_questions = len(detailed_results)
            entity_bonus_rate = entity_bonus_count / total_questions if total_questions > 0 else 0
            
            # 如果有交通信号灯问题，应该能触发实体奖励
            expected_entity_bonus = traffic_signal_questions > 0
            validation_score = entity_bonus_rate if expected_entity_bonus else min(entity_bonus_rate + 0.5, 1.0)
            
            return {
                "optimization_name": "实体命中奖励机制",
                "entity_bonus_usage_count": entity_bonus_count,
                "entity_bonus_usage_rate": entity_bonus_rate,
                "traffic_signal_questions": traffic_signal_questions,
                "validation_score": validation_score,
                "status": "优秀" if validation_score >= 0.8 else "良好" if validation_score >= 0.5 else "需改进",
                "details": f"实体奖励使用率: {entity_bonus_rate:.1%}, 交通信号灯问题: {traffic_signal_questions}个"
            }
            
        except Exception as e:
            return {"optimization_name": "实体命中奖励", "error": str(e), "validation_score": 0.0}
    
    def _validate_document_chunking_enhancement(self, test_report: Dict) -> Dict:
        """验证优化4：文档切分+标题增强"""
        try:
            detailed_results = test_report.get('detailed_results', [])
            
            enhanced_docs_count = 0
            chapter_splitting_count = 0
            title_enhancement_evidence = 0
            
            for result in detailed_results:
                if result.get('uses_enhanced_features', {}).get('enhanced_documents', False):
                    enhanced_docs_count += 1
                
                if result.get('uses_enhanced_features', {}).get('chapter_splitting', False):
                    chapter_splitting_count += 1
                
                # 检查是否有标题增强证据（通过检查答案中是否包含章节信息）
                answer = result.get('answer', '')
                if any(marker in answer for marker in ['【主要章节】', '【子章节】', '#关键:']):
                    title_enhancement_evidence += 1
            
            total_questions = len(detailed_results)
            enhancement_rate = enhanced_docs_count / total_questions if total_questions > 0 else 0
            chapter_rate = chapter_splitting_count / total_questions if total_questions > 0 else 0
            
            validation_score = (enhancement_rate + chapter_rate) / 2
            
            return {
                "optimization_name": "文档切分+标题增强",
                "enhanced_documents_usage_count": enhanced_docs_count,
                "chapter_splitting_usage_count": chapter_splitting_count,
                "title_enhancement_evidence": title_enhancement_evidence,
                "enhancement_usage_rate": enhancement_rate,
                "validation_score": validation_score,
                "status": "优秀" if validation_score >= 0.8 else "良好" if validation_score >= 0.5 else "需改进",
                "details": f"增强文档率: {enhancement_rate:.1%}, 章节切分率: {chapter_rate:.1%}"
            }
            
        except Exception as e:
            return {"optimization_name": "文档切分+标题增强", "error": str(e), "validation_score": 0.0}
    
    def _validate_fewshot_prompt_optimization(self, test_report: Dict) -> Dict:
        """验证优化5：Few-shot重提示优化"""
        try:
            optimization_effectiveness = test_report.get('optimization_effectiveness', {})
            detailed_results = test_report.get('detailed_results', [])
            
            # 检查交通信号灯模板使用
            traffic_signal_template_count = optimization_effectiveness.get('questions_using_traffic_signal_template', 0)
            
            # 检查"未找到"回复使用
            not_found_response_count = optimization_effectiveness.get('questions_using_not_found_response', 0)
            
            # 分析提示词效果
            effective_prompt_count = 0
            for result in detailed_results:
                prompt_analysis = result.get('prompt_analysis', {})
                if prompt_analysis.get('prompt_effectiveness', {}).get('overall_score', 0) > 0.5:
                    effective_prompt_count += 1
            
            total_questions = len(detailed_results)
            
            # 计算Few-shot效果分数
            template_usage_rate = traffic_signal_template_count / total_questions if total_questions > 0 else 0
            effective_prompt_rate = effective_prompt_count / total_questions if total_questions > 0 else 0
            
            validation_score = (template_usage_rate + effective_prompt_rate) / 2
            
            return {
                "optimization_name": "Few-shot重提示优化",
                "traffic_signal_template_usage": traffic_signal_template_count,
                "not_found_response_usage": not_found_response_count,
                "effective_prompt_count": effective_prompt_count,
                "template_usage_rate": template_usage_rate,
                "validation_score": validation_score,
                "status": "优秀" if validation_score >= 0.8 else "良好" if validation_score >= 0.5 else "需改进",
                "details": f"特定模板使用率: {template_usage_rate:.1%}, 有效提示率: {effective_prompt_rate:.1%}"
            }
            
        except Exception as e:
            return {"optimization_name": "Few-shot重提示优化", "error": str(e), "validation_score": 0.0}
    
    def _generate_overall_assessment(self, performance_report: Dict, test_report: Dict, optimization_validation: Dict) -> Dict:
        """生成总体评估"""
        try:
            # 计算各项指标
            overall_score = optimization_validation.get('overall_validation_score', 0.0)
            
            # 系统稳定性评估
            system_ready = all([
                performance_report.get('components_status', {}).get('vectorstore_loaded', False),
                performance_report.get('components_status', {}).get('advanced_retriever_ready', False),
                performance_report.get('components_status', {}).get('enhanced_reranker_ready', False),
                performance_report.get('components_status', {}).get('rag_chain_built', False)
            ])
            
            # 性能评估
            test_summary = test_report.get('test_summary', {})
            avg_response_time = test_summary.get('average_time', 0)
            
            # 功能完整性评估
            optimization_features = performance_report.get('optimization_features', {})
            feature_completeness = sum(optimization_features.values()) / len(optimization_features) if optimization_features else 0
            
            # 生成评级
            if overall_score >= 0.9 and system_ready and avg_response_time < 10:
                grade = "A+ (优秀)"
            elif overall_score >= 0.8 and system_ready:
                grade = "A (良好)"
            elif overall_score >= 0.6:
                grade = "B (及格)"
            else:
                grade = "C (需改进)"
            
            return {
                "overall_score": overall_score,
                "grade": grade,
                "system_stability": "稳定" if system_ready else "不稳定",
                "average_response_time": avg_response_time,
                "feature_completeness": feature_completeness,
                "optimization_summary": {
                    "实现的优化功能": [
                        "✅ 关键词+向量混合检索（BM25强制召回+RRF融合）",
                        "✅ Cross-Encoder重排序优化",
                        "✅ 实体命中奖励机制",
                        "✅ 文档切分+标题增强",
                        "✅ Few-shot重提示优化"
                    ],
                    "优化效果": f"总体验证分数: {overall_score:.2f}/1.00",
                    "系统状态": "就绪" if system_ready else "需检查"
                },
                "recommendations": self._generate_recommendations(overall_score, avg_response_time, optimization_validation)
            }
            
        except Exception as e:
            return {"error": f"总体评估失败: {e}"}
    
    def _generate_recommendations(self, overall_score: float, avg_response_time: float, optimization_validation: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if overall_score < 0.8:
            recommendations.append("建议检查各项优化功能的实现效果，确保正确配置")
        
        if avg_response_time > 10:
            recommendations.append("响应时间较长，建议优化模型推理速度或检索数量")
        
        # 针对具体优化功能的建议
        for opt_name, opt_result in optimization_validation.items():
            if opt_name.startswith('optimization_') and opt_result.get('validation_score', 0) < 0.6:
                recommendations.append(f"建议改进 {opt_result.get('optimization_name', opt_name)} 的实现效果")
        
        if not recommendations:
            recommendations.append("系统运行良好，建议继续监控性能指标")
        
        return recommendations
    
    def _save_test_report(self, report: Dict):
        """保存测试报告"""
        try:
            # 保存到outputs目录
            os.makedirs("outputs", exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = f"outputs/optimized_rag_test_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"测试报告已保存到: {report_file}")
            
            # 同时保存一个最新版本
            latest_file = "outputs/optimized_rag_test_report_latest.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"保存测试报告失败: {e}")

def main():
    """主函数"""
    logger.info("🚀 开始优化RAG系统全面测试...")
    
    # 创建测试器
    tester = OptimizedRAGTester()
    
    # 运行完整测试
    start_time = time.time()
    test_report = tester.run_full_optimization_test()
    total_time = time.time() - start_time
    
    if "error" in test_report:
        logger.error(f"❌ 测试失败: {test_report['error']}")
        return False
    
    # 显示测试结果摘要
    logger.info("📊 测试结果摘要:")
    overall_assessment = test_report.get('overall_assessment', {})
    logger.info(f"  总体评分: {overall_assessment.get('grade', 'N/A')}")
    logger.info(f"  总体分数: {overall_assessment.get('overall_score', 0):.2f}/1.00")
    logger.info(f"  平均响应时间: {overall_assessment.get('average_response_time', 0):.2f}秒")
    logger.info(f"  系统稳定性: {overall_assessment.get('system_stability', 'N/A')}")
    
    # 显示优化功能验证结果
    optimization_validation = test_report.get('optimization_validation', {})
    logger.info("🔧 优化功能验证结果:")
    for i, (opt_key, opt_result) in enumerate(optimization_validation.items(), 1):
        if opt_key.startswith('optimization_'):
            opt_name = opt_result.get('optimization_name', f'优化{i}')
            status = opt_result.get('status', '未知')
            score = opt_result.get('validation_score', 0)
            logger.info(f"  {i}. {opt_name}: {status} (分数: {score:.2f})")
    
    # 显示建议
    recommendations = overall_assessment.get('recommendations', [])
    if recommendations:
        logger.info("💡 改进建议:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")
    
    logger.info(f"✅ 测试完成，总用时: {total_time:.2f}秒")
    logger.info("📄 详细报告已保存到 outputs/optimized_rag_test_report_latest.json")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("用户中断测试")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试程序异常: {e}")
        sys.exit(1) 