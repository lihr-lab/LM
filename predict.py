import argparse
import os

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
    get_path = args.get_path or paths.get("predict_get") or paths.get("normal_get")
    post_path = args.post_path or paths.get("predict_post") or paths.get("normal_post")
    model_dir = paths["model_dir"]
    output_dir = paths["output_dir"]
    ensure_dir(output_dir)

    # 加载模型 - 使用 pickle 格式
    models = {
        "GET": OCSVMMethodModel.load(os.path.join(model_dir, "get_model.pkl")),
        "POST": OCSVMMethodModel.load(os.path.join(model_dir, "post_model.pkl")),
    }
    
    data = load_normal_logs(get_path, post_path)
    predictions = run_inference(models, data)
    summary = summarize_predictions(predictions)

    save_json(os.path.join(output_dir, "predictions.json"), predictions)
    save_json(os.path.join(output_dir, "summary.json"), summary)
    print("Prediction completed.")
    print(summary)


if __name__ == "__main__":
    main()