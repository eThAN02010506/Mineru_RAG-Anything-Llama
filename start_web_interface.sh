#!/bin/bash
# 启动RAG-Anything Web界面的脚本

echo "====================================================="
echo "  RAG-Anything Web界面启动脚本"
echo "====================================================="

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3。请安装 Python3 后再试。"
    exit 1
fi

# 检查 virtualenv 是否存在
if [ ! -d ".venv" ]; then
    echo "错误: 未找到 .venv 虚拟环境。请先创建虚拟环境并安装依赖："
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 激活虚拟环境
source .venv/bin/activate

# 检查是否存在 Web 界面脚本
if [ ! -f "rag_web_interface_complete.py" ]; then
    echo "错误: 未找到 rag_web_interface_complete.py 文件。"
    echo "请确保该文件与此脚本在同一目录中。"
    deactivate
    exit 1
fi

# 检查 data 目录
if [ ! -d "data" ]; then
    echo "警告: 未找到 data 目录，正在创建..."
    mkdir -p data
fi

# 检查 data 中是否有 PDF 文件
pdf_count=$(find data -name "*.pdf" | wc -l)
if [ "$pdf_count" -eq 0 ]; then
    echo "警告: data 目录中没有 PDF 文件。"
    echo "请将 PDF 文件放入 data 目录后再使用系统。"
fi

# 启动 Web 界面
echo "正在启动 Web 界面..."
echo "将自动打开浏览器..."
echo "按 Ctrl+C 停止服务器"
echo "====================================================="

python3 rag_web_interface_complete.py

# 如果脚本异常退出，显示错误信息
if [ $? -ne 0 ]; then
    echo "====================================================="
    echo "错误: Web 界面启动失败。"
    echo "请检查错误信息并确保所有依赖已安装。"
    echo "====================================================="
    deactivate
    exit 1
fi

# 退出虚拟环境
deactivate
