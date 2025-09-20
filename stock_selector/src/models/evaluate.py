"""
模型評估模組
評估模型性能和生成評估報告
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from ..config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型評估器"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray = None, 
                              model_name: str = "Model") -> Dict[str, Any]:
        """
        評估分類模型
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            y_pred_proba: 預測機率（可選）
            model_name: 模型名稱
            
        Returns:
            評估結果字典
        """
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }
        
        # 如果有預測機率，計算 AUC
        if y_pred_proba is not None:
            try:
                # 對於多類別問題，使用 macro average
                results['auc_score'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', average='macro'
                )
            except:
                results['auc_score'] = None
        
        logger.info(f"{model_name} 評估結果:")
        logger.info(f"  準確率: {results['accuracy']:.4f}")
        logger.info(f"  精確率: {results['precision']:.4f}")
        logger.info(f"  召回率: {results['recall']:.4f}")
        logger.info(f"  F1分數: {results['f1_score']:.4f}")
        
        return results
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          model_name: str = "Model") -> Dict[str, Any]:
        """
        評估回歸模型
        
        Args:
            y_true: 真實值
            y_pred: 預測值
            model_name: 模型名稱
            
        Returns:
            評估結果字典
        """
        results = {
            'model_name': model_name,
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
        
        logger.info(f"{model_name} 評估結果:")
        logger.info(f"  MSE: {results['mse']:.4f}")
        logger.info(f"  RMSE: {results['rmse']:.4f}")
        logger.info(f"  MAE: {results['mae']:.4f}")
        logger.info(f"  R²: {results['r2_score']:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, confusion_mat: np.ndarray, 
                            class_names: List[str] = None, 
                            model_name: str = "Model"):
        """
        繪製混淆矩陣
        
        Args:
            confusion_mat: 混淆矩陣
            class_names: 類別名稱
            model_name: 模型名稱
        """
        if class_names is None:
            class_names = ['下跌', '平盤', '上漲']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - 混淆矩陣')
        plt.xlabel('預測標籤')
        plt.ylabel('真實標籤')
        
        # 保存圖片
        output_dir = OUTPUTS_DIR / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"混淆矩陣已保存: {model_name}_confusion_matrix.png")
    
    def plot_feature_importance(self, feature_importance: np.ndarray, 
                              feature_names: List[str] = None,
                              model_name: str = "Model", top_n: int = 20):
        """
        繪製特徵重要性
        
        Args:
            feature_importance: 特徵重要性數組
            feature_names: 特徵名稱列表
            model_name: 模型名稱
            top_n: 顯示前N個重要特徵
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
        
        # 排序特徵重要性
        indices = np.argsort(feature_importance)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'{model_name} - 特徵重要性 (Top {top_n})')
        plt.barh(range(top_n), feature_importance[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('重要性')
        plt.gca().invert_yaxis()
        
        # 保存圖片
        output_dir = OUTPUTS_DIR / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{model_name}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"特徵重要性圖已保存: {model_name}_feature_importance.png")
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str = "Model"):
        """
        繪製預測分布
        
        Args:
            y_true: 真實值
            y_pred: 預測值
            model_name: 模型名稱
        """
        plt.figure(figsize=(12, 5))
        
        # 真實值分布
        plt.subplot(1, 2, 1)
        plt.hist(y_true, bins=20, alpha=0.7, label='真實值')
        plt.title(f'{model_name} - 真實值分布')
        plt.xlabel('值')
        plt.ylabel('頻率')
        plt.legend()
        
        # 預測值分布
        plt.subplot(1, 2, 2)
        plt.hist(y_pred, bins=20, alpha=0.7, label='預測值', color='orange')
        plt.title(f'{model_name} - 預測值分布')
        plt.xlabel('值')
        plt.ylabel('頻率')
        plt.legend()
        
        # 保存圖片
        output_dir = OUTPUTS_DIR / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{model_name}_prediction_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"預測分布圖已保存: {model_name}_prediction_distribution.png")
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        生成評估報告
        
        Args:
            evaluation_results: 評估結果字典
            
        Returns:
            格式化的評估報告
        """
        report = "# 模型評估報告\n\n"
        
        for model_name, results in evaluation_results.items():
            report += f"## {model_name}\n\n"
            
            if 'accuracy' in results:  # 分類模型
                report += f"### 分類性能指標\n\n"
                report += f"- **準確率 (Accuracy)**: {results['accuracy']:.4f}\n"
                report += f"- **精確率 (Precision)**: {results['precision']:.4f}\n"
                report += f"- **召回率 (Recall)**: {results['recall']:.4f}\n"
                report += f"- **F1分數**: {results['f1_score']:.4f}\n"
                
                if 'auc_score' in results and results['auc_score'] is not None:
                    report += f"- **AUC分數**: {results['auc_score']:.4f}\n"
                
                report += "\n### 詳細分類報告\n\n"
                report += "```\n"
                report += results['classification_report']
                report += "\n```\n\n"
            
            if 'rmse' in results:  # 回歸模型
                report += f"### 回歸性能指標\n\n"
                report += f"- **均方誤差 (MSE)**: {results['mse']:.4f}\n"
                report += f"- **均方根誤差 (RMSE)**: {results['rmse']:.4f}\n"
                report += f"- **平均絕對誤差 (MAE)**: {results['mae']:.4f}\n"
                report += f"- **R²分數**: {results['r2_score']:.4f}\n\n"
        
        return report
    
    def save_evaluation_report(self, evaluation_results: Dict[str, Any], 
                             filename: str = "evaluation_report.md"):
        """
        保存評估報告
        
        Args:
            evaluation_results: 評估結果字典
            filename: 檔案名稱
        """
        report = self.generate_evaluation_report(evaluation_results)
        
        output_file = OUTPUTS_DIR / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"評估報告已保存: {output_file}")
    
    def compare_models(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """
        比較多個模型的性能
        
        Args:
            evaluation_results: 評估結果字典
            
        Returns:
            模型比較 DataFrame
        """
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            model_data = {'Model': model_name}
            
            # 分類指標
            if 'accuracy' in results:
                model_data.update({
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score']
                })
                
                if 'auc_score' in results and results['auc_score'] is not None:
                    model_data['AUC'] = results['auc_score']
            
            # 回歸指標
            if 'rmse' in results:
                model_data.update({
                    'RMSE': results['rmse'],
                    'MAE': results['mae'],
                    'R²': results['r2_score']
                })
            
            comparison_data.append(model_data)
        
        df = pd.DataFrame(comparison_data)
        return df


def main():
    """主函數 - 用於測試評估功能"""
    evaluator = ModelEvaluator()
    
    # 創建測試數據
    np.random.seed(42)
    n_samples = 1000
    
    # 分類測試
    y_true_cls = np.random.choice([-1, 0, 1], size=n_samples)
    y_pred_cls = np.random.choice([-1, 0, 1], size=n_samples)
    y_pred_proba_cls = np.random.rand(n_samples, 3)
    y_pred_proba_cls = y_pred_proba_cls / y_pred_proba_cls.sum(axis=1, keepdims=True)
    
    # 回歸測試
    y_true_reg = np.random.randn(n_samples)
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 0.1
    
    # 評估分類模型
    cls_results = evaluator.evaluate_classification(
        y_true_cls, y_pred_cls, y_pred_proba_cls, "Test_Classifier"
    )
    
    # 評估回歸模型
    reg_results = evaluator.evaluate_regression(
        y_true_reg, y_pred_reg, "Test_Regressor"
    )
    
    # 繪製圖表
    evaluator.plot_confusion_matrix(cls_results['confusion_matrix'], "Test_Classifier")
    
    # 生成報告
    all_results = {"Test_Classifier": cls_results, "Test_Regressor": reg_results}
    evaluator.save_evaluation_report(all_results)
    
    # 模型比較
    comparison_df = evaluator.compare_models(all_results)
    logger.info("模型比較結果:")
    logger.info(comparison_df.to_string())


if __name__ == "__main__":
    main()


