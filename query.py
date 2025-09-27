import json
import os

import faiss
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer


def query(question, index_dir, top_k=3):
    """
    完全离线的 RAG 查询，返回比较详细的答案
    """
    # 索引与映射文件路径
    faiss_index_path = os.path.join(index_dir, "faiss_index.index")
    mapping_path = os.path.join(index_dir, "mapping.json")

    # 模型文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    llama_model_path = os.path.join(script_dir, "models", "llama", "mistral-7b.gguf")
    embed_model_path = os.path.join(script_dir, "models", "all-MiniLM-L6-v2")

    # 检查所有必需文件
    for file_path in (
        faiss_index_path,
        mapping_path,
        llama_model_path,
        embed_model_path,
    ):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"错误: 必需文件未找到: {file_path}")

    # 加载 embedding 模型
    print("正在加载离线 embedding 模型...")
    embedder = SentenceTransformer(embed_model_path, device="cpu")

    # 加载 FAISS 索引和文本片段
    print("正在加载向量索引和文本片段映射...")
    index = faiss.read_index(faiss_index_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"索引信息: {index.ntotal} 个向量, {len(chunks)} 个文本片段")

    # 生成查询向量并搜索最相关片段
    print("正在生成查询向量并搜索最相关片段...")
    q_emb = embedder.encode([question], device="cpu").astype("float32")
    k = min(top_k, len(chunks), index.ntotal)
    distances, indices = index.search(q_emb, k=k)
    print(f"找到 {k} 个相关片段：")
    relevant_chunks = []
    for i, idx in enumerate(indices[0]):
        dist = float(distances[0][i])
        txt = chunks[idx]
        print(f"  片段{i+1}: 距离 {dist:.4f}")
        relevant_chunks.append(txt)

    # 构建上下文字符串
    context = "\n\n".join(
        f"[片段{i+1}]\n{txt}" for i, txt in enumerate(relevant_chunks)
    )

    # —— 改成“详细回答” Prompt，并抑制标题格式
    prompt = f"""
    你是一个知识助手，你的任务是根据提供的文档内容回答用户问题。

    请严格遵守以下规则：
    1. 仅根据下面提供的文档片段回答，避免编造。
    2. 如果文档中没有相关信息，请明确回答：“未在文档中找到相关内容。”
    3. 回答时保持条理清晰，可以分点说明。
    4. 当涉及示例、注意事项、步骤时，请尽量详细展开。
    5. 使用用户问题的语言（中文或英文）回答。

    以下是与问题最相关的文档片段：
    {context}

    用户问题：
    {question}

    请给出一个完整、详细且结构化的回答：
    """

    # 调试：打印 Prompt
    print("🔍 PROMPT =====")
    print(prompt)
    print("🔍 END PROMPT =====")

    # 调用 LLaMA 生成，去掉 stop 以获取完整正文
    print("正在加载离线 LLaMA 模型...")
    llm = Llama(
        model_path=llama_model_path,
        n_ctx=4096,
        n_threads=4,
        temperature=0.2,
        top_p=0.95,
        top_k=50,
        repeat_penalty=1.1,
        verbose=False,
    )

    print("正在生成回答...")
    response = llm(prompt, max_tokens=256, echo=False)

    # 调试：打印原始返回结果
    print("🔍 RAW RESPONSE =====")
    print(response)
    print("🔍 END RAW RESPONSE =====")

    # 提取并返回答案，加上来源信息
    answer = response["choices"][0]["text"].strip()
    return f"{answer}\n\n回答基于 {len(relevant_chunks)} 个文档片段"


def interactive_query(index_dir):
    """
    交互式查询模式
    """
    print("进入交互式模式，输入 'quit' 或 'exit' 退出。")
    while True:
        question = input("请输入问题: ").strip()
        if question.lower() in ["quit", "exit", "q", "退出"]:
            print("再见！")
            break
        if not question:
            continue
        print("=" * 40)
        print("回答:\n" + query(question, index_dir))
        print("=" * 40)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python query.py <index_dir> [问题]")
        sys.exit(1)
    idx_dir = sys.argv[1]
    if len(sys.argv) > 2:
        print("回答:\n" + query(" ".join(sys.argv[2:]), idx_dir))
    else:
        interactive_query(idx_dir)
