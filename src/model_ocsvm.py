import pickle
import os
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from scipy.sparse import csr_matrix, hstack

from src.features import FeatureBuilder


class OCSVMMethodModel:
    """
    One-Class SVM 异常检测模型（针对单个HTTP方法）
    集成了 FeatureBuilder 特征提取和 RobustScaler
    """
    
    def __init__(self, method: str, feature_cfg: dict, model_cfg: dict):
        self.method = method
        self.feature_cfg = feature_cfg
        self.model_cfg = model_cfg
        
        # 特征提取器（内部使用 RobustScaler）
        self.feature_builder = FeatureBuilder(feature_cfg)
        
        # One-Class SVM 模型
        contamination = model_cfg.get("contamination", "auto")
        if contamination == "auto":
            self.ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
        else:
            self.ocsvm = OneClassSVM(
                nu=contamination,
                kernel=model_cfg.get("kernel", "rbf"),
                gamma=model_cfg.get("gamma", "scale")
            )
        
        self.threshold_ = None
        self.is_fitted = False
    
    def fit(self, logs: list, validation_logs: list = None):
        """
        训练模型
        
        Args:
            logs: 训练日志列表（应为正常样本）
            validation_logs: 可选，验证集日志（用于确定阈值）
        """
        # 1. 提取特征
        X_train = self.feature_builder.fit_transform(logs)
        
        # 2. 训练 One-Class SVM
        self.ocsvm.fit(X_train)
        
        # 3. 计算训练集的异常分数
        train_scores = self.ocsvm.decision_function(X_train)
        
        # 4. 确定阈值
        if validation_logs:
            # 如果有验证集，使用验证集确定阈值（取指定分位数）
            X_val = self.feature_builder.transform(validation_logs)
            val_scores = self.ocsvm.decision_function(X_val)
            # 默认取验证集分数的 1% 分位数作为阈值（即期望误报率 1%）
            percentile = self.model_cfg.get("threshold_percentile", 1)
            self.threshold_ = np.percentile(val_scores, percentile)
        else:
            # 无验证集时，使用训练集分数的 5% 分位数（更保守）
            percentile = self.model_cfg.get("train_threshold_percentile", 5)
            self.threshold_ = np.percentile(train_scores, percentile)
        
        self.is_fitted = True
        return self
    
    def predict(self, logs: list) -> tuple:
        """
        预测日志是否为异常
        
        Returns:
            (predictions, scores)
            predictions: list of int, 1=正常, -1=异常
            scores: list of float, 异常分数（越高越正常）
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = self.feature_builder.transform(logs)
        scores = self.ocsvm.decision_function(X)
        predictions = [1 if s >= self.threshold_ else -1 for s in scores]
        return predictions, scores
    
    def save(self, model_dir: str) -> str:
        """保存模型到指定目录"""
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.method.lower()}_model.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump({
                "method": self.method,
                "feature_cfg": self.feature_cfg,
                "model_cfg": self.model_cfg,
                "ocsvm": self.ocsvm,
                "feature_builder": self.feature_builder,
                "threshold_": self.threshold_,
                "is_fitted": self.is_fitted,
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, model_path: str) -> "OCSVMMethodModel":
        """从文件加载模型"""
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        
        model = cls(data["method"], data["feature_cfg"], data["model_cfg"])
        model.ocsvm = data["ocsvm"]
        model.feature_builder = data["feature_builder"]
        model.threshold_ = data["threshold_"]
        model.is_fitted = data["is_fitted"]
        return model