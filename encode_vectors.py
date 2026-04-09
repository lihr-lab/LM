import argparse
import os

from scipy.sparse import save_npz

from src.data_loader import load_normal_logs
from src.features import FeatureBuilder
from src.utils.io_utils import ensure_dir, load_yaml, save_json


def _build_vectors_for_method(method: str, logs, feature_cfg: dict, output_dir: str) -> None:
    builder = FeatureBuilder(feature_cfg)
    x = builder.fit_transform(logs)

    matrix_path = os.path.join(output_dir, f"{method.lower()}_vectors.npz")
    meta_path = os.path.join(output_dir, f"{method.lower()}_vectors_meta.json")
    save_npz(matrix_path, x)
    save_json(
        meta_path,
        [
            {
                "row_id": idx,
                "method": method,
                "uri": rec.get("uri", ""),
                "raw_client_ip": rec.get("raw_client_ip", ""),
                "stat_time": rec.get("stat_time", ""),
            }
            for idx, rec in enumerate(logs)
        ],
    )
    print(f"[{method}] vectors: {x.shape} -> {matrix_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode GET/POST logs into per-request vectors.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output_dir", default="artifacts/output", help="Directory to store vectors.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    feature_cfg = cfg["feature"]
    data = load_normal_logs(paths["normal_get"], paths["normal_post"])

    ensure_dir(args.output_dir)
    _build_vectors_for_method("GET", data.get("GET", []), feature_cfg, args.output_dir)
    _build_vectors_for_method("POST", data.get("POST", []), feature_cfg, args.output_dir)


if __name__ == "__main__":
    main()
