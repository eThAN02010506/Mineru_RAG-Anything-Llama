
#!/usr/bin/env python3
"""
\u68c0\u67e5\u6a21\u578b\u76ee\u5f55\u7ed3\u6784\uff0c\u786e\u4fdd\u5b83\u5305\u542b\u6240\u6709\u5fc5\u8981\u7684\u6587\u4ef6
"""
import os
import sys
import json

def check_model_directory(model_dir):
    """\u68c0\u67e5\u6a21\u578b\u76ee\u5f55\u7ed3\u6784"""
    print("=" * 60)
    print("\u68c0\u67e5\u6a21\u578b\u76ee\u5f55: {}".format(model_dir))
    print("=" * 60)
    
    if not os.path.exists(model_dir):
        print("\u9519\u8bef: \u6a21\u578b\u76ee\u5f55\u4e0d\u5b58\u5728: {}".format(model_dir))
        return False
    
    # \u68c0\u67e5\u76ee\u5f55\u4e2d\u7684\u6587\u4ef6
    files = os.listdir(model_dir)
    print("\u6a21\u578b\u76ee\u5f55\u5305\u542b {} \u4e2a\u6587\u4ef6/\u76ee\u5f55:".format(len(files)))
    for file in files:
        file_path = os.path.join(model_dir, file)
        if os.path.isdir(file_path):
            print("  \u76ee\u5f55: {}".format(file))
        else:
            print("  \u6587\u4ef6: {} ({:.2f} MB)".format(file, os.path.getsize(file_path) / (1024 * 1024)))
    
    # \u68c0\u67e5\u5fc5\u8981\u7684\u6587\u4ef6
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "vocab.txt"
    ]
    
    missing_files = []
    for req_file in required_files:
        if not any(f == req_file for f in files):
            missing_files.append(req_file)
    
    if missing_files:
        print("\
\u8b66\u544a: \u7f3a\u5c11\u4ee5\u4e0b\u5fc5\u8981\u6587\u4ef6:")
        for file in missing_files:
            print("  - {}".format(file))
        print("\
\u8fd9\u53ef\u80fd\u5bfc\u81f4\u6a21\u578b\u65e0\u6cd5\u6b63\u5e38\u52a0\u8f7d\u3002")
    else:
        print("\
\u6210\u529f: \u627e\u5230\u6240\u6709\u5fc5\u8981\u7684\u6587\u4ef6\u3002")
    
    # \u68c0\u67e5config.json
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            print("\
config.json\u5185\u5bb9\u6458\u8981:")
            for key in ["model_type", "hidden_size", "num_hidden_layers", "vocab_size"]:
                if key in config:
                    print("  {}: {}".format(key, config[key]))
        except Exception as e:
            print("\
\u8b66\u544a: \u65e0\u6cd5\u8bfb\u53d6config.json: {}".format(e))
    
    return True

def main():
    # \u9ed8\u8ba4\u68c0\u67e5models/all-MiniLM-L6-v2\u76ee\u5f55
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, "models", "all-MiniLM-L6-v2")
    
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = default_model_dir
    
    check_model_directory(model_dir)

if __name__ == "__main__":
    main()
