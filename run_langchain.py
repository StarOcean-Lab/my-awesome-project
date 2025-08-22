#!/usr/bin/env python3
"""
LangChain+Ollama智能客服机器人启动脚本
"""

import os
import sys
import argparse
import subprocess
import requests
from pathlib import Path

def check_ollama_service():
    """检查Ollama服务是否运行"""
    print("🔍 检查Ollama服务...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            print(f"✅ Ollama服务运行正常，可用模型: {model_names}")
            return True, model_names
        else:
            print(f"❌ Ollama服务响应异常: {response.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ Ollama服务不可用: {e}")
        print("请先启动Ollama服务: ollama serve")
        return False, []

def check_required_models():
    """检查所需模型是否已下载"""
    print("🔍 检查必需模型...")
    
    required_models = [
        "deepseek-r1:7b",
        "mxbai-embed-large:latest"
    ]
    
    success, available_models = check_ollama_service()
    if not success:
        return False
    
    missing_models = []
    for model in required_models:
        if model not in available_models:
            missing_models.append(model)
    
    if missing_models:
        print(f"❌ 缺少模型: {missing_models}")
        print("请先下载模型:")
        for model in missing_models:
            print(f"  执行：ollama pull {model}")
        return False
    
    print("✅ 所有必需模型已就绪")
    return True

def check_dependencies():
    """检查Python依赖"""
    print("🔍 检查Python依赖...")
    
    # 包名映射：pip包名 -> 实际导入名
    package_mapping = {
        'langchain': 'langchain',
        'langchain-ollama': 'langchain_ollama', 
        'langchain-community': 'langchain_community',
        'streamlit': 'streamlit',
        'faiss-cpu': 'faiss',
        'sentence-transformers': 'sentence_transformers',
        'pandas': 'pandas',
        'openpyxl': 'openpyxl',
        'loguru': 'loguru'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖检查通过！")
    return True

def check_pdf_files():
    """检查PDF文件"""
    print("🔍 检查PDF文件...")
    
    import glob
    pdf_files = glob.glob("data/*.pdf")
    
    if pdf_files:
        print(f"✅ 找到 {len(pdf_files)} 个PDF文件:")
        for pdf_file in pdf_files[:5]:
            print(f"  - {pdf_file}")
        if len(pdf_files) > 5:
            print(f"  ... 还有 {len(pdf_files) - 5} 个文件")
        return True
    else:
        print("⚠️  未找到PDF文件")
        print("建议将竞赛PDF文档放在data目录下")
        return False

def setup_environment():
    """设置环境"""
    print("🛠️  设置环境...")
    
    # 创建必要目录
    dirs = [
        "logs",
        "outputs",
        "vectorstore"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ 目录结构创建完成")

# 启动Web界面
def run_web_interface():
    """启动Web界面"""
    print("🚀 启动LangChain+Ollama Web界面...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "langchain_chatbot.py",
            "web",
            "--server.address", "127.0.0.1",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出程序")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

# 启动命令行模式
def run_command_line():
    """运行命令行模式"""
    print("💻 启动LangChain+Ollama命令行模式...")
    
    try:
        subprocess.run([sys.executable, "langchain_chatbot.py"])
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出程序")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

# 运行系统测试
def run_test():
    print("🧪 运行LangChain+Ollama系统测试...")
    
    try:
        # 简单测试
        from langchain_chatbot import LangChainChatbot
        
        print("初始化系统...")
        chatbot = LangChainChatbot()
        
        print("测试基础功能...")
        result = chatbot.answer_question("测试问题")
        print(f"测试结果: {result['answer'][:100]}...")
        
        print("✅ 系统测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def run_optimized_test():
    print("🚀 运行优化RAG系统测试...")
    
    try:
        # 测试优化系统
        from test_optimized_rag_system import main as test_main
        
        print("启动优化RAG系统测试...")
        test_main()
        print("✅ 优化系统测试完成")
        
    except Exception as e:
        print(f"❌ 优化系统测试失败: {e}")
        print("请确保已正确安装所有依赖和模型")

def run_command_line_with_options(use_optimized=True):
    mode_name = "优化模式" if use_optimized else "传统模式"
    print(f"💻 启动LangChain+Ollama命令行模式 ({mode_name})...")
    
    try:
        from langchain_chatbot import LangChainChatbot
        
        # 初始化机器人
        chatbot = LangChainChatbot(use_optimized=use_optimized)
        print(f"✅ {mode_name}系统初始化完成")
        
        # 加载知识库
        import glob
        pdf_files = glob.glob("data/*.pdf")
        if pdf_files:
            print(f"加载 {len(pdf_files)} 个PDF文件...")
            if chatbot.load_knowledge_base(pdf_files):
                print("✅ 知识库加载完成")
            else:
                print("❌ 知识库加载失败")
        
        # 交互式问答
        print(f"\n开始问答 ({mode_name}) - 输入'quit'退出:")
        while True:
            question = input("\n请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                break
            
            if question:
                result = chatbot.answer_question(question)
                print(f"\n回答: {result['answer']}")
                print(f"置信度: {result['confidence']:.2f}")
                if use_optimized and 'optimization_info' in result:
                    print(f"优化信息: {result['optimization_info']}")
                
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出程序")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")

def install_dependencies():
    """安装依赖"""
    print("📦 安装LangChain+Ollama依赖包...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ 依赖安装完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False
    return True

def pull_ollama_models():
    """下载Ollama模型"""
    print("📥 下载Ollama模型...")
    
    models = ["deepseek-r1:7b", "Qwen2.5-7B-Instruct:latest"]
    
    for model in models:
        print(f"下载模型: {model}")
        try:
            subprocess.run(["ollama", "pull", model], check=True)
            print(f"✅ {model} 下载完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ {model} 下载失败: {e}")
        except FileNotFoundError:
            print("❌ 未找到ollama命令，请先安装Ollama")
            return False
    
    return True

# 可支持命令行参数的启动
def main():
    parser = argparse.ArgumentParser(description="LangChain+Ollama智能客服机器人")
    parser.add_argument("--mode", choices=["web", "cli", "test", "optimized"], default="web",
                       help="可选运行模式: web(Web界面), cli(命令行), test(测试), optimized(优化模式测试)")
    parser.add_argument("--install", action="store_true", help="安装Python依赖")
    parser.add_argument("--pull-models", action="store_true", help="下载Ollama模型")
    parser.add_argument("--check", action="store_true", help="仅检查环境")
    parser.add_argument("--skip-check", action="store_true", help="跳过环境检查")
    parser.add_argument("--use-optimized", action="store_true", help="强制使用优化RAG系统")
    parser.add_argument("--disable-optimized", action="store_true", help="禁用优化RAG系统")
    
    args = parser.parse_args()
    
    print("🤖 LangChain+Ollama 泰迪杯竞赛智能客服机器人")
    print("=" * 60)
    
    # 安装依赖
    if args.install:
        if install_dependencies():
            print("✅ 依赖安装完成，可以开始使用系统")
        return
    
    # 下载模型
    if args.pull_models:
        pull_ollama_models()
        return
    
    # 环境检查
    if not args.skip_check:
        checks_passed = True
        
        # 检查Ollama服务
        if not check_ollama_service()[0]:
            checks_passed = False
        
        # 检查模型
        if not check_required_models():
            checks_passed = False
        
        # 检查Python依赖
        if not check_dependencies():
            checks_passed = False
        
        # 检查PDF文件
        check_pdf_files()  # 不是必须的，只是警告
        
        if not checks_passed:
            print("\n❌ 环境检查未通过，请先解决上述问题")
            print("\n💡 建议操作:")
            print("1. 启动Ollama服务: ollama serve")
            print("2. 下载模型: python run_langchain.py --pull-models")
            print("3. 安装依赖: python run_langchain.py --install")
            return
        
        if args.check:
            print("\n✅ 环境检查通过，系统可以正常运行")
            return
    
    # 设置环境
    setup_environment()
    
    # 检查优化模式参数
    use_optimized = True  # 默认使用优化模式
    if args.disable_optimized:
        use_optimized = False
        print("ℹ️ 优化模式已禁用，将使用传统RAG系统")
    elif args.use_optimized:
        use_optimized = True
        print("🚀 强制启用优化模式")
    
    # 根据模式运行
    if args.mode == "web":
        # Web界面会在内部处理优化模式选择
        run_web_interface()
    elif args.mode == "cli":
        run_command_line_with_options(use_optimized=use_optimized)
    elif args.mode == "test":
        run_test()
    elif args.mode == "optimized":
        run_optimized_test()

if __name__ == "__main__":
    main() 