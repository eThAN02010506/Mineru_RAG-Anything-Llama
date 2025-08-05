import os
from mineru.utils.enum_class import ModelPath

def get_local_models_dir():
    """
    返回本地模型根目录配置字典。
    请根据你本地缓存路径实际情况修改下面路径。
    """
    return {
        'pipeline': '/Users/ethanjiang/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1___0',
        'vlm': '/Users/ethanjiang/.cache/modelscope/hub/models/OpenDataLab/OtherModel'  # 如果没用vlm可以不管
    }

def auto_download_and_get_model_root_path(relative_path: str, repo_mode='pipeline') -> str:
    """
    纯本地模式，不联网下载，直接返回本地模型根路径。
    - relative_path参数可以忽略，因为不联网。
    - repo_mode区分pipeline/vlm模型根路径。
    """
    local_models_config = get_local_models_dir()
    root_path = local_models_config.get(repo_mode)
    if not root_path:
        raise ValueError(f"本地模型路径未配置，repo_mode={repo_mode}")
    
    # 返回本地根路径，不拼接relative_path，因为代码里会拼接
    return root_path


if __name__ == '__main__':
    # 测试打印pipeline模型根路径
    path1 = "models/README.md"
    root = auto_download_and_get_model_root_path(path1)
    print("本地模型根目录:", root)
    print("完整路径示例:", os.path.join(root, path1))
