#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 查询优化策略配置
QUERY_OPTIMIZATION = {
    # 未来校园相关查询的同义词扩展
    "future_campus_synonyms": [
        "元宇宙智能应用",
        "智慧学校",
        "数字孪生技术",
        "校园智能应用场景",
        "虚拟仿真平台"
    ],
    
    # 交通信号灯相关查询的扩展
    "traffic_light_synonyms": [
        "交通管理",
        "信号控制",
        "路口管理",
        "智能交通系统",
        "交通信号控制"
    ],
    
    # 查询增强规则
    "query_enhancement_rules": {
        "未来校园.*交通": "智慧学校 交通管理",
        "未来校园.*信号灯": "元宇宙智能应用 交通信号",
        "交通信号灯.*技术要求": "智慧学校 交通管理 信号控制",
        "智能交通.*未来校园": "元宇宙智能应用 交通管理"
    }
}

def optimize_query(query):
    """优化查询策略"""
    import re
    
    # 应用增强规则
    for pattern, replacement in QUERY_OPTIMIZATION["query_enhancement_rules"].items():
        if re.search(pattern, query):
            return replacement
    
    # 关键词替换
    if "未来校园" in query:
        if any(keyword in query for keyword in ["交通", "信号灯"]):
            return "智慧学校 交通管理"
        else:
            return "元宇宙智能应用"
    
    return query 