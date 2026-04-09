from typing import Dict, List


def summarize_predictions(predictions: Dict[str, List[Dict]]) -> Dict:
    summary = {"methods": {}}
    for method, rows in predictions.items():
        total = len(rows)
        anomalies = sum(1 for x in rows if x["is_anomaly"])
        summary["methods"][method] = {
            "total": total,
            "anomaly_count": anomalies,
            "anomaly_ratio": float(anomalies) / float(total) if total else 0.0,
        }
    return summary
