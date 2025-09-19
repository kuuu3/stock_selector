"""
模型預測模組
使用訓練好的模型進行股票預測
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

from ..config import (
    OUTPUTS_DIR,
    SELECTION_CONFIG
)

logger = logging.getLogger(__name__)


class StockPredictor:
    """股票預測器"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        
    def load_models(self):
        """載入訓練好的模型"""
        models_dir = OUTPUTS_DIR / "models"
        
        if not models_dir.exists():
            logger.error("模型目錄不存在，請先訓練模型")
            return False
        
        try:
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
            
            return True
            
        except Exception as e:
            logger.error(f"載入模型時發生錯誤: {e}")
            return False
    
    def predict_classification(self, features: np.ndarray) -> Dict[str, Any]:
        """
        使用分類模型進行預測
        
        Args:
            features: 特徵矩陣
            
        Returns:
            預測結果字典
        """
        results = {}
        
        # Logistic Regression 預測
        if 'logistic_regression' in self.models and 'logistic_regression' in self.scalers:
            scaler = self.scalers['logistic_regression']
            model = self.models['logistic_regression']
            
            features_scaled = scaler.transform(features)
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
            
            results['logistic_regression'] = {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': np.max(probabilities, axis=1)
            }
        
        # XGBoost Classifier 預測
        if 'xgboost_classifier' in self.models:
            model = self.models['xgboost_classifier']
            
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            
            results['xgboost_classifier'] = {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': np.max(probabilities, axis=1)
            }
        
        return results
    
    def predict_regression(self, features: np.ndarray) -> Dict[str, Any]:
        """
        使用回歸模型進行預測
        
        Args:
            features: 特徵矩陣
            
        Returns:
            預測結果字典
        """
        results = {}
        
        # XGBoost Regressor 預測
        if 'xgboost_regressor' in self.models:
            model = self.models['xgboost_regressor']
            
            predictions = model.predict(features)
            
            results['xgboost_regressor'] = {
                'predictions': predictions,
                'feature_importance': model.feature_importances_
            }
        
        return results
    
    def ensemble_predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        集成預測（結合多個模型的結果）
        
        Args:
            features: 特徵矩陣
            
        Returns:
            集成預測結果
        """
        classification_results = self.predict_classification(features)
        regression_results = self.predict_regression(features)
        
        ensemble_results = {}
        
        # 集成分類預測（簡單平均）
        if classification_results:
            all_probabilities = []
            for model_name, result in classification_results.items():
                all_probabilities.append(result['probabilities'])
            
            if all_probabilities:
                avg_probabilities = np.mean(all_probabilities, axis=0)
                ensemble_predictions = np.argmax(avg_probabilities, axis=1) - 1  # 轉換為 -1, 0, 1
                ensemble_confidence = np.max(avg_probabilities, axis=1)
                
                ensemble_results['classification'] = {
                    'predictions': ensemble_predictions,
                    'probabilities': avg_probabilities,
                    'confidence': ensemble_confidence
                }
        
        # 集成回歸預測
        if regression_results:
            all_regression = []
            for model_name, result in regression_results.items():
                all_regression.append(result['predictions'])
            
            if all_regression:
                avg_regression = np.mean(all_regression, axis=0)
                ensemble_results['regression'] = {
                    'predictions': avg_regression
                }
        
        return ensemble_results
    
    def calculate_stock_scores(self, features: np.ndarray, stock_codes: List[str]) -> pd.DataFrame:
        """
        計算股票評分
        
        Args:
            features: 特徵矩陣
            stock_codes: 股票代碼列表
            
        Returns:
            包含股票評分的 DataFrame
        """
        if not self.models:
            logger.error("沒有載入的模型")
            return pd.DataFrame()
        
        # 進行集成預測
        ensemble_results = self.ensemble_predict(features)
        
        if not ensemble_results:
            logger.error("無法進行預測")
            return pd.DataFrame()
        
        # 創建結果 DataFrame
        results_data = []
        
        for i, stock_code in enumerate(stock_codes):
            stock_result = {
                'stock_code': stock_code,
                'rank': i + 1
            }
            
            # 分類結果
            if 'classification' in ensemble_results:
                cls_result = ensemble_results['classification']
                stock_result.update({
                    'prediction_class': cls_result['predictions'][i],
                    'confidence': cls_result['confidence'][i],
                    'prob_up': cls_result['probabilities'][i][2] if len(cls_result['probabilities'][i]) > 2 else 0,
                    'prob_down': cls_result['probabilities'][i][0] if len(cls_result['probabilities'][i]) > 0 else 0,
                    'prob_neutral': cls_result['probabilities'][i][1] if len(cls_result['probabilities'][i]) > 1 else 0
                })
            
            # 回歸結果
            if 'regression' in ensemble_results:
                reg_result = ensemble_results['regression']
                stock_result['expected_return'] = reg_result['predictions'][i]
            
            # 計算綜合評分
            stock_result['final_score'] = self._calculate_final_score(stock_result)
            
            results_data.append(stock_result)
        
        df = pd.DataFrame(results_data)
        
        # 按綜合評分排序
        df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _calculate_final_score(self, stock_result: Dict[str, Any]) -> float:
        """
        計算最終評分
        
        Args:
            stock_result: 單一股票的預測結果
            
        Returns:
            最終評分
        """
        score = 0.0
        
        # 基於分類結果的評分
        if 'prediction_class' in stock_result:
            prediction_class = stock_result['prediction_class']
            confidence = stock_result.get('confidence', 0)
            
            # 上漲(+1)給正分，下跌(-1)給負分，平盤(0)給0分
            score += prediction_class * confidence * 0.6
        
        # 基於回歸結果的評分
        if 'expected_return' in stock_result:
            expected_return = stock_result['expected_return']
            # 預期報酬率越高，評分越高
            score += expected_return * 0.4
        
        return score
    
    def predict_stocks(self, features: np.ndarray, stock_codes: List[str], 
                      top_n: int = None) -> pd.DataFrame:
        """
        預測股票並返回排名
        
        Args:
            features: 特徵矩陣
            stock_codes: 股票代碼列表
            top_n: 返回前N名股票
            
        Returns:
            股票排名 DataFrame
        """
        if not self.load_models():
            return pd.DataFrame()
        
        # 計算股票評分
        scores_df = self.calculate_stock_scores(features, stock_codes)
        
        if scores_df.empty:
            return pd.DataFrame()
        
        # 返回前N名
        if top_n:
            scores_df = scores_df.head(top_n)
        
        logger.info(f"預測完成，共 {len(scores_df)} 支股票")
        
        return scores_df
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        獲取特徵重要性
        
        Returns:
            各模型的特徵重要性
        """
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
        
        return importance_dict


def main():
    """主函數 - 用於測試預測功能"""
    predictor = StockPredictor()
    
    # 創建測試數據
    n_stocks = 10
    n_features = 20
    
    test_features = np.random.randn(n_stocks, n_features)
    test_stock_codes = [f"233{i:02d}" for i in range(n_stocks)]
    
    # 進行預測
    results = predictor.predict_stocks(test_features, test_stock_codes, top_n=5)
    
    if not results.empty:
        logger.info("預測結果:")
        logger.info(results[['stock_code', 'final_score', 'prediction_class', 'confidence']].to_string())


if __name__ == "__main__":
    main()

