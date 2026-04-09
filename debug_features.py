import json
import sys
import pickle
import os

# 添加 src 目录到路径
sys.path.append("src")

from src.features import FeatureBuilder


def debug_feature_extraction():
    """调试特征提取过程，观察URL解析范式和参数对向量"""
    
    # 方式一：如果已有训练好的 feature_builder，加载它
    feature_builder = None
    model_dir = "artifacts/models"
    feature_builder_path = os.path.join(model_dir, "feature_builder.pkl")
    
    if os.path.exists(feature_builder_path):
        print(f"Loading trained FeatureBuilder from {feature_builder_path}")
        with open(feature_builder_path, "rb") as f:
            feature_builder = pickle.load(f)
    else:
        print("No trained FeatureBuilder found, creating a new one (without fitting)")
        # 创建新的 FeatureBuilder（不会进行 TF-IDF 拟合，仅用于观察中间文本）
        from src.features import FeatureBuilder as FB
        feature_builder = FB(feature_cfg={})
    
    # 选择日志文件进行分析
    log_files = [
        "waf_classified_attacks_get.json",
        "waf_classified_normal_post.json",
    ]
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"File not found: {log_file}")
            continue
        
        print("\n" + "="*80)
        print(f"Analyzing: {log_file}")
        print("="*80)
        
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
        
        # 取前 5 条进行分析
        sample_logs = logs[:5]
        
        # 获取中间表示（现在返回6个值）
        texts_param_kv, texts_param_key, texts_path, texts_http, texts_full_url, stats = feature_builder._prepare_parts(sample_logs)
        
        # 1. 打印模板化后的完整URL（新增）
        print("\n--- 模板化后的完整URL (full_url) ---")
        for i, full_url in enumerate(texts_full_url):
            print(f"  [{i}] {full_url}")
        
        # 2. 打印模板化后的路径
        print("\n--- 模板化后的路径 (path) ---")
        for i, path in enumerate(texts_path):
            print(f"  [{i}] {path}")
        
        # 3. 打印参数对文本
        print("\n--- 参数对文本 (param_kv) ---")
        for i, kv in enumerate(texts_param_kv):
            preview = kv[:200] + "..." if len(kv) > 200 else kv
            print(f"  [{i}] {preview}")
        
        # 4. 打印参数键文本
        print("\n--- 参数键文本 (param_key) ---")
        for i, keys in enumerate(texts_param_key):
            preview = keys[:200] + "..." if len(keys) > 200 else keys
            print(f"  [{i}] {preview}")
        
        # 5. 打印手工特征（前10维作为示例）
        print("\n--- 手工特征 (前10维) ---")
        for i in range(min(5, len(stats))):
            preview = [round(x, 4) for x in stats[i][:10]]
            print(f"  [{i}] {preview} ... (共{len(stats[i])}维)")
        
        # 6. 如果 feature_builder 已拟合，打印 TF-IDF 特征信息
        if hasattr(feature_builder, "param_kv_vectorizer"):
            if hasattr(feature_builder.param_kv_vectorizer, "vocabulary_"):
                vocab_size = len(feature_builder.param_kv_vectorizer.vocabulary_)
                print(f"\n--- TF-IDF 词表大小: param_kv={vocab_size}")
        
        print("\n" + "-"*40)


def main():
    debug_feature_extraction()


if __name__ == "__main__":
    main()