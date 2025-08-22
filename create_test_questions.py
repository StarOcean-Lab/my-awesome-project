#!/usr/bin/env python3
"""
创建测试问题Excel文件
"""

import pandas as pd

def create_test_questions_excel():
    """创建测试问题Excel文件"""
    
    # 测试问题数据
    test_questions = [
        "第七届全国青少年人工智能创新挑战赛的报名时间是什么时候？",
        '"未来校园智能应用专项赛"中智能交通信号灯任务的基本要求是什么？',
        '3D编程模型创新设计专项赛中"伞"设计需要考虑哪些方面？',
        '"人工智能综合创新专项赛"参赛作品的字数要求是什么？',
        '"未来校园智能应用专项赛"中哪些任务涉及到自动化控制？',
        "如何确保我的竞赛作品不被剽窃？",
        '"未来校园智能应用专项赛"参赛前有哪些准备工作？',
        "在3D编程模型创新设计专项赛中，提交的任务分数不显示或出现错误，怎么办？",
        "第七届全国青少年人工智能创新挑战赛中有多少个机器人类别的竞赛？",
        '我是一名学生家长，孩子现在是高中一年级，他的编程能力很强，但动手制作能力相对较弱，我想问一下，参加"第七届全国青少年人工智能创新挑战赛"中哪一项（或哪一类）的竞赛比较合适。'
    ]
    
    # 创建DataFrame
    df = pd.DataFrame({
        '序号': range(1, len(test_questions) + 1),
        '问题': test_questions
    })
    
    # 保存为Excel文件
    filename = 'test_questions.xlsx'
    df.to_excel(filename, index=False)
    print(f"测试问题文件已创建: {filename}")
    print(f"包含 {len(test_questions)} 个问题")
    
    return filename

if __name__ == "__main__":
    create_test_questions_excel() 