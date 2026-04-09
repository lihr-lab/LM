from typing import Dict, List


def run_inference(models: Dict, data: Dict) -> Dict:
    """
    运行推理
    
    Args:
        models: {"GET": OCSVMMethodModel, "POST": OCSVMMethodModel}
        data: {"GET": List[Dict], "POST": List[Dict]}
    
    Returns:
        {"GET": List[Dict], "POST": List[Dict]} 每个元素包含 is_anomaly 和 score
    """
    predictions = {"GET": [], "POST": []}
    
    for method, logs in data.items():
        if not logs:
            continue
        
        # predict 返回 (labels, scores) 元组
        # labels: 1=正常, -1=异常
        labels, scores = models[method].predict(logs)
        
        # 构建结果列表，使用 is_anomaly 键名
        for log, label, score in zip(logs, labels, scores):
            predictions[method].append({
                "is_anomaly": bool(label == -1),  # -1 表示异常，转换为 True/False
                "score": float(score)
            })
    
    return predictions