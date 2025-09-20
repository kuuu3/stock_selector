"""
模型訓練模組
訓練 Logistic Regression 和 XGBoost 模型
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import subprocess
import platform

from ..config import (
    MODEL_CONFIG,
    PROCESSED_FEATURES_FILE,
    PROCESSED_LABELS_FILE,
    OUTPUTS_DIR
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型訓練器"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.use_gpu = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """
        檢查 GPU 可用性
        
        Returns:
            是否可以使用 GPU
        """
        try:
            # 檢查 NVIDIA GPU
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("檢測到 NVIDIA GPU，將使用 GPU 加速訓練")
                return True
        except FileNotFoundError:
            pass
        
        # 檢查 CUDA 是否可用
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("檢測到 CUDA 可用，將使用 GPU 加速訓練")
                return True
        except ImportError:
            pass
        
        logger.info("未檢測到 GPU，將使用 CPU 訓練")
        return False
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        載入特徵和標籤數據
        
        Returns:
            (features, labels) 元組
        """
        try:
            features = np.load(PROCESSED_FEATURES_FILE)
            labels = np.load(PROCESSED_LABELS_FILE)
            
            logger.info(f"載入數據: 特徵形狀 {features.shape}, 標籤形狀 {labels.shape}")
            return features, labels
            
        except FileNotFoundError as e:
            logger.error(f"找不到數據檔案: {e}")
            raise
    
    def prepare_data(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        準備訓練數據
        
        Args:
            features: 特徵矩陣
            labels: 標籤矩陣
            
        Returns:
            包含訓練和測試數據的字典
        """
        # 確保數據類型正確
        X = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        
        # 分離特徵和標籤
        y_classification = labels[:, 0]  # 分類標籤 (label_1w)
        y_regression = labels[:, 2] if labels.shape[1] > 2 else labels[:, 0]  # 回歸標籤
        
        # 將分類標籤從 [-1, 0, 1] 轉換為 [0, 1, 2] 以符合 XGBoost 要求
        y_classification = y_classification + 1  # [-1, 0, 1] -> [0, 1, 2]
        
        # 移除 NaN 值
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_classification) | np.isnan(y_regression))
        X = X[valid_mask]
        y_classification = y_classification[valid_mask]
        y_regression = y_regression[valid_mask]
        
        logger.info(f"清理後數據: X形狀 {X.shape}, y分類 {y_classification.shape}, y回歸 {y_regression.shape}")
        
        # 時間序列分割（避免未來資訊洩漏）
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 使用最後一個分割作為測試集
        train_indices = []
        test_indices = []
        
        for train_idx, test_idx in tscv.split(X):
            train_indices = train_idx
            test_indices = test_idx
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train_cls, y_test_cls = y_classification[train_indices], y_classification[test_indices]
        y_train_reg, y_test_reg = y_regression[train_indices], y_regression[test_indices]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train_cls': y_train_cls,
            'y_test_cls': y_test_cls,
            'y_train_reg': y_train_reg,
            'y_test_reg': y_test_reg
        }
    
    def train_logistic_regression(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        訓練 Logistic Regression 模型
        
        Args:
            data: 訓練數據字典
            
        Returns:
            模型評估結果
        """
        logger.info("開始訓練 Logistic Regression 模型...")
        
        # 數據標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(data['X_train'])
        X_test_scaled = scaler.transform(data['X_test'])
        
        # 訓練模型
        model = LogisticRegression(
            random_state=MODEL_CONFIG["LOGISTIC_REGRESSION"]["random_state"],
            max_iter=MODEL_CONFIG["LOGISTIC_REGRESSION"]["max_iter"]
        )
        
        model.fit(X_train_scaled, data['y_train_cls'])
        
        # 預測
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # 評估
        accuracy = accuracy_score(data['y_test_cls'], y_pred)
        
        logger.info(f"Logistic Regression 準確率: {accuracy:.4f}")
        
        # 保存模型
        self.models['logistic_regression'] = model
        self.scalers['logistic_regression'] = scaler
        
        return {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(data['y_test_cls'], y_pred)
        }
    
    def train_xgboost_classifier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        訓練 XGBoost 分類器
        
        Args:
            data: 訓練數據字典
            
        Returns:
            模型評估結果
        """
        logger.info("開始訓練 XGBoost Classifier 模型...")
        
        # 訓練模型
        model_params = {
            'n_estimators': MODEL_CONFIG["XGBOOST_CLASSIFIER"]["n_estimators"],
            'max_depth': MODEL_CONFIG["XGBOOST_CLASSIFIER"]["max_depth"],
            'learning_rate': MODEL_CONFIG["XGBOOST_CLASSIFIER"]["learning_rate"],
            'random_state': MODEL_CONFIG["XGBOOST_CLASSIFIER"]["random_state"]
        }
        
        # 根據 GPU 可用性添加 GPU 參數
        if self.use_gpu:
            model_params.update({
                'tree_method': 'gpu_hist',
                'device': 'gpu:0'
            })
            logger.info("使用 GPU 加速訓練 XGBoost Classifier")
        else:
            logger.info("使用 CPU 訓練 XGBoost Classifier")
        
        model = xgb.XGBClassifier(**model_params)
        
        model.fit(data['X_train'], data['y_train_cls'])
        
        # 預測
        y_pred = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test'])
        
        # 評估
        accuracy = accuracy_score(data['y_test_cls'], y_pred)
        
        logger.info(f"XGBoost Classifier 準確率: {accuracy:.4f}")
        
        # 保存模型
        self.models['xgboost_classifier'] = model
        
        return {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(data['y_test_cls'], y_pred),
            'feature_importance': model.feature_importances_
        }
    
    def train_xgboost_regressor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        訓練 XGBoost 回歸器
        
        Args:
            data: 訓練數據字典
            
        Returns:
            模型評估結果
        """
        logger.info("開始訓練 XGBoost Regressor 模型...")
        
        # 訓練模型
        model_params = {
            'n_estimators': MODEL_CONFIG["XGBOOST_REGRESSOR"]["n_estimators"],
            'max_depth': MODEL_CONFIG["XGBOOST_REGRESSOR"]["max_depth"],
            'learning_rate': MODEL_CONFIG["XGBOOST_REGRESSOR"]["learning_rate"],
            'random_state': MODEL_CONFIG["XGBOOST_REGRESSOR"]["random_state"]
        }
        
        # 根據 GPU 可用性添加 GPU 參數
        if self.use_gpu:
            model_params.update({
                'tree_method': 'gpu_hist',
                'device': 'gpu:0'
            })
            logger.info("使用 GPU 加速訓練 XGBoost Regressor")
        else:
            logger.info("使用 CPU 訓練 XGBoost Regressor")
        
        model = xgb.XGBRegressor(**model_params)
        
        model.fit(data['X_train'], data['y_train_reg'])
        
        # 預測
        y_pred = model.predict(data['X_test'])
        
        # 評估
        mse = np.mean((data['y_test_reg'] - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(data['y_test_reg'] - y_pred))
        
        logger.info(f"XGBoost Regressor RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # 保存模型
        self.models['xgboost_regressor'] = model
        
        return {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred,
            'feature_importance': model.feature_importances_
        }
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        訓練所有模型
        
        Returns:
            所有模型的評估結果
        """
        logger.info("=== 開始訓練所有模型 ===")
        
        # 載入數據
        features, labels = self.load_data()
        
        # 準備數據
        data = self.prepare_data(features, labels)
        
        # 訓練各個模型
        results = {}
        
        try:
            results['logistic_regression'] = self.train_logistic_regression(data)
            results['xgboost_classifier'] = self.train_xgboost_classifier(data)
            results['xgboost_regressor'] = self.train_xgboost_regressor(data)
            
            logger.info("=== 所有模型訓練完成 ===")
            
            # 保存模型
            self.save_models()
            
            return results
            
        except Exception as e:
            logger.error(f"模型訓練過程中發生錯誤: {e}")
            raise
    
    def train_with_data(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        使用提供的數據訓練所有模型
        
        Args:
            features: 特徵矩陣
            labels: 標籤矩陣
            
        Returns:
            所有模型的評估結果
        """
        logger.info("=== 開始使用提供數據訓練所有模型 ===")
        logger.info(f"特徵形狀: {features.shape}, 標籤形狀: {labels.shape}")
        
        # 準備數據
        data = self.prepare_data(features, labels)
        
        # 訓練各個模型
        results = {}
        
        try:
            results['logistic_regression'] = self.train_logistic_regression(data)
            results['xgboost_classifier'] = self.train_xgboost_classifier(data)
            results['xgboost_regressor'] = self.train_xgboost_regressor(data)
            
            logger.info("=== 所有模型訓練完成 ===")
            
            # 保存模型
            self.save_models()
            
            return results
            
        except Exception as e:
            logger.error(f"模型訓練過程中發生錯誤: {e}")
            raise
    
    def save_models(self):
        """保存訓練好的模型"""
        models_dir = OUTPUTS_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = models_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"模型已保存: {model_path}")
        
        # 保存標準化器
        for scaler_name, scaler in self.scalers.items():
            scaler_path = models_dir / f"{scaler_name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"標準化器已保存: {scaler_path}")
    
    def load_models(self):
        """載入已訓練的模型"""
        models_dir = OUTPUTS_DIR / "models"
        
        if not models_dir.exists():
            logger.warning("模型目錄不存在")
            return
        
        # 載入模型
        for model_file in models_dir.glob("*.joblib"):
            if "_scaler" not in model_file.name:
                model_name = model_file.stem
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"模型已載入: {model_name}")
        
        # 載入標準化器
        for scaler_file in models_dir.glob("*_scaler.joblib"):
            scaler_name = scaler_file.stem.replace("_scaler", "")
            self.scalers[scaler_name] = joblib.load(scaler_file)
            logger.info(f"標準化器已載入: {scaler_name}")


def main():
    """主函數 - 用於測試模型訓練"""
    trainer = ModelTrainer()
    
    try:
        # 訓練所有模型
        results = trainer.train_all_models()
        
        # 顯示結果摘要
        logger.info("=== 模型訓練結果摘要 ===")
        for model_name, result in results.items():
            if 'accuracy' in result:
                logger.info(f"{model_name}: 準確率 = {result['accuracy']:.4f}")
            elif 'rmse' in result:
                logger.info(f"{model_name}: RMSE = {result['rmse']:.4f}")
        
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}")


if __name__ == "__main__":
    main()
