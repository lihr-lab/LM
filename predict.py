import argparse
import os
import numpy as np

from src.data_loader import load_normal_logs
from src.evaluate import summarize_predictions
from src.infer import run_inference
from src.model_ocsvm import OCSVMMethodModel
from src.utils.io_utils import ensure_dir, load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict anomalies with GET/POST OCSVM.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--get_path", default=None, help="Override GET json path.")
    parser.add_argument("--post_path", default=None, help="Override POST json path.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    get_path = args.get_path or paths["normal_get"]
    post_path = args.post_path or paths["normal_post"]
    model_dir = paths["model_dir"]
    output_dir = paths["output_dir"]
    ensure_dir(output_dir)

    # 加载模型（使用 pickle 格式）
    models = {
        "GET": OCSVMMethodModel.load(os.path.join(model_dir, "get_model.pkl")),
        "POST": OCSVMMethodModel.load(os.path.join(model_dir, "post_model.pkl")),
    }

    # ========== 诊断：检查 POST 特征维度 ==========
    print("\n" + "="*60)
    print("特征诊断（取第一条 POST 日志）")
    print("="*60)
    
    # 加载测试数据（只取前几条用于诊断）
    data_diagnostic = load_normal_logs(get_path, post_path)
    if data_diagnostic.get("POST"):
        sample_post = data_diagnostic["POST"][:1]
        # 提取特征
        X_post = models["POST"].feature_builder.transform(sample_post)
        print(f"POST 特征矩阵形状: {X_post.shape}")
        print(f"非零元素数量: {X_post.nnz}")
        print(f"特征维度（列数）: {X_post.shape[1]}")
        
        # 尝试打印各特征块的维度（需要访问 feature_builder 的内部向量化器）
        fb = models["POST"].feature_builder
        print("\n各特征块维度：")
        if hasattr(fb, 'http_vectorizer') and hasattr(fb.http_vectorizer, 'vocabulary_'):
            print(f"  - http_vectorizer: {len(fb.http_vectorizer.vocabulary_)}")
        if hasattr(fb, 'param_kv_vectorizer') and hasattr(fb.param_kv_vectorizer, 'vocabulary_'):
            print(f"  - param_kv_vectorizer: {len(fb.param_kv_vectorizer.vocabulary_)}")
        if hasattr(fb, 'param_key_vectorizer') and hasattr(fb.param_key_vectorizer, 'vocabulary_'):
            print(f"  - param_key_vectorizer: {len(fb.param_key_vectorizer.vocabulary_)}")
        if hasattr(fb, 'path_vectorizer') and hasattr(fb.path_vectorizer, 'vocabulary_'):
            print(f"  - path_vectorizer: {len(fb.path_vectorizer.vocabulary_)}")
        # POST 新增特征
        if hasattr(fb, 'post_json_path_vectorizer') and hasattr(fb.post_json_path_vectorizer, 'vocabulary_'):
            print(f"  - post_json_path_vectorizer: {len(fb.post_json_path_vectorizer.vocabulary_)}")
        if hasattr(fb, 'post_body_vectorizer') and hasattr(fb.post_body_vectorizer, 'vocabulary_'):
            print(f"  - post_body_vectorizer: {len(fb.post_body_vectorizer.vocabulary_)}")
        if hasattr(fb, 'post_param_key_vectorizer') and hasattr(fb.post_param_key_vectorizer, 'vocabulary_'):
            print(f"  - post_param_key_vectorizer: {len(fb.post_param_key_vectorizer.vocabulary_)}")
        if hasattr(fb, 'post_stats_scaler') and hasattr(fb.post_stats_scaler, 'scale_'):
            print(f"  - post_stats_scaler: {fb.post_stats_scaler.scale_.shape[0]} 维")
        
        # 检查 JSON 解析成功率（可选：遍历所有 POST 日志）
        print("\n检查 POST 请求中 JSON 解析成功率：")
        total_post = len(data_diagnostic.get("POST", []))
        json_success = 0
        for rec in data_diagnostic.get("POST", [])[:100]:  # 只检查前100条
            http_raw = rec.get("http", "")
            http_decoded = http_raw.replace("\\/", "/").replace("\\r\\n", "\r\n")
            if "\r\n\r\n" in http_decoded:
                _, body_section = http_decoded.split("\r\n\r\n", 1)
                body = body_section.split("\r\n")[0] if body_section else ""
                if body.strip().startswith(("{", "[")):
                    try:
                        import json
                        json.loads(body)
                        json_success += 1
                    except:
                        pass
        print(f"  前100条中 JSON 格式成功解析: {json_success}/100 = {json_success}%")
    else:
        print("没有 POST 数据用于诊断")
    
    print("="*60 + "\n")

    # 正式预测
    data = load_normal_logs(get_path, post_path)
    predictions = run_inference(models, data)
    summary = summarize_predictions(predictions)

    save_json(os.path.join(output_dir, "predictions.json"), predictions)
    save_json(os.path.join(output_dir, "summary.json"), summary)
    print("Prediction completed.")
    print(summary)


if __name__ == "__main__":
    main()