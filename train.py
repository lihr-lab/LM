import argparse
import os
import pickle

from src.data_loader import load_normal_logs
from src.model_ocsvm import OCSVMMethodModel
from src.utils.io_utils import ensure_dir, load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GET/POST One-Class SVM models.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    feature_cfg = cfg["feature"]
    model_cfg = cfg["model"]

    # 加载正常日志（用于训练）
    data = load_normal_logs(paths["normal_get"], paths["normal_post"])
    
    # 可选：加载验证集（用于确定阈值）
    val_data = None
    if "val_get" in paths and "val_post" in paths:
        val_data = load_normal_logs(paths["val_get"], paths["val_post"])
    
    model_dir = paths["model_dir"]
    ensure_dir(model_dir)

    trained_paths = {}
    for method in ("GET", "POST"):
        logs = data.get(method, [])
        if not logs:
            print(f"[{method}] No training data, skipping")
            continue
        
        val_logs = val_data.get(method, []) if val_data else None
        
        model = OCSVMMethodModel(method, feature_cfg, model_cfg)
        model.fit(logs, validation_logs=val_logs)
        
        model_path = model.save(model_dir)
        trained_paths[method] = {
            "model_path": model_path,
            "train_count": len(logs),
            "val_count": len(val_logs) if val_logs else 0,
            "threshold": model.threshold_,
        }
        print(f"[{method}] trained on {len(logs)} logs, threshold={model.threshold_:.6f}")

    # 保存训练元信息
    save_json(os.path.join(model_dir, "train_meta.json"), trained_paths)
    
    # 单独保存 FeatureBuilder（作为备份）
    if trained_paths:
        first_method = next(iter(trained_paths.keys()))
        with open(os.path.join(model_dir, "feature_builder.pkl"), "wb") as f:
            # 从第一个模型中获取 feature_builder
            with open(trained_paths[first_method]["model_path"], "rb") as mf:
                model_data = pickle.load(mf)
                pickle.dump(model_data["feature_builder"], f)
    
    print(f"Saved artifacts to: {model_dir}")


if __name__ == "__main__":
    main()