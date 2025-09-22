import json
import os
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def embed_chunks(
    jsonl_path, faiss_index_path="faiss_index.index", mapping_path="mapping.json"
):
    print("\u6b63\u5728\u52a0\u8f7d\u79bb\u7ebf embedding \u6a21\u578b...")

    # \u4f7f\u7528\u5b9e\u9645\u7684\u6a21\u578b\u8def\u5f84
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "all-MiniLM-L6-v2")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "\u9519\u8bef: \u627e\u4e0d\u5230\u79bb\u7ebf\u6a21\u578b: {}".format(
                model_path
            )
        )

    # \u8bbe\u7f6e\u73af\u5883\u53d8\u91cf\uff0c\u5f3a\u5236\u79bb\u7ebf\u6a21\u5f0f
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # \u79bb\u7ebf\u52a0\u8f7d\u6a21\u578b - \u4f7f\u7528\u672c\u5730\u8def\u5f84\uff0c\u4e0d\u662f\u5728\u7ebf\u8def\u5f84
    try:
        model = SentenceTransformer(model_path, device="cpu")
    except Exception as e:
        print("\u9519\u8bef: \u52a0\u8f7d\u6a21\u578b\u5931\u8d25: {}".format(e))
        print("\u5c1d\u8bd5\u4f7f\u7528\u7edd\u5bf9\u8def\u5f84...")
        try:
            abs_model_path = os.path.abspath(model_path)
            print("\u4f7f\u7528\u7edd\u5bf9\u8def\u5f84: {}".format(abs_model_path))
            model = SentenceTransformer(abs_model_path, device="cpu")
        except Exception as e2:
            print(
                "\u9519\u8bef: \u4f7f\u7528\u7edd\u5bf9\u8def\u5f84\u52a0\u8f7d\u6a21\u578b\u4e5f\u5931\u8d25: {}".format(
                    e2
                )
            )
            raise

    texts = []
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            "\u9519\u8bef: \u627e\u4e0d\u5230\u6587\u672c\u7247\u6bb5\u6587\u4ef6: {}".format(
                jsonl_path
            )
        )

    print("\u6b63\u5728\u8bfb\u53d6\u6587\u672c\u7247\u6bb5: {}".format(jsonl_path))

    # \u68c0\u67e5\u6587\u4ef6\u5927\u5c0f
    file_size = os.path.getsize(jsonl_path)
    if file_size == 0:
        raise ValueError(
            "\u9519\u8bef: \u6587\u4ef6 {} \u662f\u7a7a\u7684".format(jsonl_path)
        )

    # \u68c0\u67e5\u6587\u4ef6\u5185\u5bb9
    print("\u68c0\u67e5JSONL\u6587\u4ef6\u5185\u5bb9...")
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            first_few_lines = []
            for i, line in enumerate(f):
                if i < 5:  # \u53ea\u663e\u793a\u524d5\u884c
                    first_few_lines.append(line.strip())
                if line.strip():
                    try:
                        data = json.loads(line)
                        text = data.get("text", "").strip()
                        if (
                            text and len(text) > 10
                        ):  # \u8fc7\u6ee4\u592a\u77ed\u7684\u6587\u672c
                            texts.append(text)
                    except json.JSONDecodeError as e:
                        print(
                            "\u8b66\u544a: \u7b2c{}\u884cJSON\u89e3\u6790\u9519\u8bef: {}".format(
                                i + 1, e
                            )
                        )
                        print(
                            "   \u95ee\u9898\u884c\u5185\u5bb9: {}".format(
                                line[:100] + "..." if len(line) > 100 else line
                            )
                        )
                        continue

            # \u663e\u793a\u6587\u4ef6\u524d\u51e0\u884c\u7528\u4e8e\u8c03\u8bd5
            if first_few_lines:
                print("\u6587\u4ef6\u524d\u51e0\u884c\u5185\u5bb9:")
                for i, line in enumerate(first_few_lines):
                    print(
                        "  \u884c {}: {}".format(
                            i + 1, line[:100] + "..." if len(line) > 100 else line
                        )
                    )
            else:
                print(
                    "\u8b66\u544a: \u6587\u4ef6\u4f3c\u4e4e\u662f\u7a7a\u7684\u6216\u683c\u5f0f\u4e0d\u6b63\u786e"
                )
    except Exception as e:
        print("\u9519\u8bef: \u8bfb\u53d6\u6587\u4ef6\u65f6\u51fa\u9519: {}".format(e))
        raise

    print(
        "\u6b63\u5728\u751f\u6210 {} \u4e2a\u6587\u672c\u7247\u6bb5\u7684 embedding...".format(
            len(texts)
        )
    )

    if len(texts) == 0:
        # \u5c1d\u8bd5\u4fee\u590dJSONL\u6587\u4ef6
        try:
            fix_jsonl_file(jsonl_path)
            # \u91cd\u65b0\u5c1d\u8bd5\u8bfb\u53d6
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            text = data.get("text", "").strip()
                            if text and len(text) > 10:
                                texts.append(text)
                        except:
                            continue

            if len(texts) == 0:
                raise ValueError(
                    "\u9519\u8bef: \u6ca1\u6709\u6709\u6548\u7684\u6587\u672c\u7247\u6bb5\uff0c\u65e0\u6cd5\u751f\u6210 embedding"
                )
        except Exception as e:
            print(
                "\u9519\u8bef: \u5c1d\u8bd5\u4fee\u590dJSONL\u6587\u4ef6\u5931\u8d25: {}".format(
                    e
                )
            )
            raise ValueError(
                "\u9519\u8bef: \u6ca1\u6709\u6709\u6548\u7684\u6587\u672c\u7247\u6bb5\uff0c\u65e0\u6cd5\u751f\u6210 embedding"
            )

    # \u5206\u6279\u5904\u7406\uff0c\u907f\u514d\u5185\u5b58\u95ee\u9898
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        print(
            "\u5904\u7406\u6279\u6b21 {}/{}".format(
                i // batch_size + 1, (len(texts) - 1) // batch_size + 1
            )
        )
        batch_embeddings = model.encode(
            batch_texts, show_progress_bar=True, convert_to_numpy=True, device="cpu"
        )
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings).astype("float32")
    dimension = embeddings.shape[1]

    print(
        "\u6b63\u5728\u521b\u5efa FAISS \u5411\u91cf\u7d22\u5f15 (\u7ef4\u5ea6: {})...".format(
            dimension
        )
    )
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # \u4fdd\u5b58\u7d22\u5f15
    faiss.write_index(index, faiss_index_path)
    print(
        "\u6210\u529f: \u5411\u91cf\u7d22\u5f15\u5df2\u4fdd\u5b58\u81f3 {}".format(
            faiss_index_path
        )
    )

    # \u4fdd\u5b58\u6587\u672c\u6620\u5c04
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(
        "\u6210\u529f: \u6587\u672c\u6620\u5c04\u5df2\u4fdd\u5b58\u81f3 {} ({} \u4e2a\u7247\u6bb5)".format(
            mapping_path, len(texts)
        )
    )


def fix_jsonl_file(jsonl_path):
    """尝试修复JSONL文件"""
    print(f"尝试修复JSONL文件: {jsonl_path}")

    backup_path = jsonl_path + ".bak"
    fixed_path = jsonl_path + ".fixed"

    # 创建备份
    import shutil

    shutil.copy2(jsonl_path, backup_path)
    print(f"已创建备份: {backup_path}")

    fixed_lines = []
    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 尝试解析JSON
            try:
                data = json.loads(line)
                # 确保有text字段
                if "text" not in data or not data["text"]:
                    continue
                fixed_lines.append(line)
            except json.JSONDecodeError:
                # 尝试修复常见JSON错误
                try:
                    # 尝试添加缺失的引号
                    if line.startswith("{") and line.endswith("}"):
                        # 替换单引号为双引号
                        line = line.replace("'", '"')
                        # 尝试解析
                        data = json.loads(line)
                        if "text" in data and data["text"]:
                            fixed_lines.append(line)
                except:
                    # 如果还是失败，尝试提取文本并创建新的JSON
                    try:
                        # 查找可能的文本内容
                        import re

                        text_match = re.search(r'"text"\s*:\s*"([^"]+)"', line)
                        if text_match:
                            text = text_match.group(1)
                            if len(text) > 10:
                                new_json = json.dumps({"text": text})
                                fixed_lines.append(new_json)
                    except:
                        continue

    # 如果没有有效行，尝试更激进的修复
    if not fixed_lines:
        print("警告: 没有找到有效的JSON行，尝试更激进的修复...")
        with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

            # 尝试提取文本块
            import re

            text_blocks = re.findall(r'[A-Za-z0-9\s,.;:\'"_-]{50,}', content)

            for block in text_blocks:
                block = block.strip()
                if len(block) > 100:  # 只保留较长的文本块
                    fixed_lines.append(json.dumps({"text": block}))

    # 写入修复后的文件
    with open(fixed_path, "w", encoding="utf-8") as f:
        for line in fixed_lines:
            f.write(line + "\n")

    # 如果修复成功，替换原文件
    if fixed_lines:
        shutil.move(fixed_path, jsonl_path)
        print(f"成功: 成功修复了 {len(fixed_lines)} 行数据")
        return True
    else:
        print("错误: 修复失败，没有找到有效数据")
        return False


if __name__ == "__main__":
    import sys

    # \u8bbe\u7f6e\u73af\u5883\u53d8\u91cf\uff0c\u5f3a\u5236\u79bb\u7ebf\u6a21\u5f0f
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if len(sys.argv) > 1:
        try:
            embed_chunks(sys.argv[1])
        except Exception as e:
            print("\u9519\u8bef: {}".format(e))
            sys.exit(1)
    else:
        print("\u7528\u6cd5: python embed.py <jsonl\u6587\u4ef6\u8def\u5f84>")
        sys.exit(1)
