"""
繼續訓練腳本 - 基於現有模型進行增量訓練
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.preprocessing import FeatureEngineer
from src.models import ModelTrainer
from src.config import get_data_file_path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinueTrainer:
    """繼續訓練器 - 基於現有模型進行增量訓練"""
    
    def __init__(self, checkpoint_name=None):
        self.models_dir = Path("outputs/models")
        self.checkpoints_dir = Path("outputs/checkpoints")
        self.training_logs_dir = Path("outputs/training_logs")
        self.checkpoint_name = checkpoint_name
        
        # 創建必要的目錄
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.training_logs_dir.mkdir(parents=True, exist_ok=True)
        
    def backup_current_models(self):
        """備份當前模型"""
        if not self.models_dir.exists():
            logger.error("找不到現有模型，無法進行繼續訓練")
            return False
        
        # 創建時間戳目錄
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_dir = self.checkpoints_dir / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 複製所有模型文件
        model_files = list(self.models_dir.glob("*.joblib"))
        if not model_files:
            logger.error("沒有找到模型文件")
            return False
        
        for model_file in model_files:
            backup_file = backup_dir / model_file.name
            joblib.dump(joblib.load(model_file), backup_file)
        
        logger.info(f"模型已備份到: {backup_dir}")
        return True
    
    def get_available_checkpoints(self):
        """獲取可用的 checkpoint 列表"""
        if not self.checkpoints_dir.exists():
            return []
        
        checkpoints = []
        for checkpoint_dir in self.checkpoints_dir.iterdir():
            if checkpoint_dir.is_dir():
                checkpoints.append({
                    'name': checkpoint_dir.name,
                    'path': checkpoint_dir,
                    'modified': checkpoint_dir.stat().st_mtime
                })
        
        # 按修改時間排序，最新的在前
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        return checkpoints
    
    def load_existing_models(self):
        """載入現有模型（從 checkpoint 或當前模型）"""
        models = {}
        scalers = {}
        
        try:
            # 決定載入來源
            if self.checkpoint_name:
                # 從指定 checkpoint 載入
                checkpoint_path = self.checkpoints_dir / self.checkpoint_name
                if not checkpoint_path.exists():
                    logger.error(f"找不到指定的 checkpoint: {self.checkpoint_name}")
                    return None, None
                logger.info(f"從 checkpoint 載入模型: {self.checkpoint_name}")
                source_dir = checkpoint_path
            else:
                # 從當前模型載入（最新）
                logger.info("從當前模型載入（最新）")
                source_dir = self.models_dir
            
            if not source_dir.exists():
                logger.error(f"找不到模型目錄: {source_dir}")
                return None, None
            
            # 載入模型
            for model_file in source_dir.glob("*.joblib"):
                if "_scaler" not in model_file.name:
                    model_name = model_file.stem
                    models[model_name] = joblib.load(model_file)
                    logger.info(f"載入模型: {model_name}")
            
            # 載入標準化器
            for scaler_file in source_dir.glob("*_scaler.joblib"):
                scaler_name = scaler_file.stem.replace("_scaler", "")
                scalers[scaler_name] = joblib.load(scaler_file)
                logger.info(f"載入標準化器: {scaler_name}")
            
            return models, scalers
            
        except Exception as e:
            logger.error(f"載入模型時發生錯誤: {e}")
            return None, None
    
    def continue_train_models(self, features, labels):
        """繼續訓練模型"""
        logger.info("開始繼續訓練...")
        
        # 載入現有模型
        existing_models, existing_scalers = self.load_existing_models()
        if existing_models is None:
            logger.error("無法載入現有模型")
            return False
        
        results = {}
        
        # 準備數據
        X = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        
        y_classification = labels[:, 0] + 1  # 轉換為 [0,1,2]
        y_regression = labels[:, 2] if labels.shape[1] > 2 else labels[:, 0]
        
        # 移除 NaN 值
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_classification) | np.isnan(y_regression))
        X = X[valid_mask]
        y_classification = y_classification[valid_mask]
        y_regression = y_regression[valid_mask]
        
        logger.info(f"清理後數據: X形狀 {X.shape}, y分類 {y_classification.shape}, y回歸 {y_regression.shape}")
        
        # 時間序列分割
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)  # 使用較少的分割以加快訓練
        
        train_indices = []
        test_indices = []
        
        for train_idx, test_idx in tscv.split(X):
            train_indices = train_idx
            test_indices = test_idx
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train_cls, y_test_cls = y_classification[train_indices], y_classification[test_indices]
        y_train_reg, y_test_reg = y_regression[train_indices], y_regression[test_indices]
        
        # 繼續訓練 Logistic Regression
        if 'logistic_regression' in existing_models and 'logistic_regression' in existing_scalers:
            logger.info("繼續訓練 Logistic Regression...")
            
            model = existing_models['logistic_regression']
            scaler = existing_scalers['logistic_regression']
            
            # 使用較小的迭代次數進行微調
            model.set_params(max_iter=100)  # Logistic Regression 使用 max_iter 而不是 learning_rate
            
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train_cls)
            
            # 評估
            train_acc = model.score(X_train_scaled, y_train_cls)
            test_acc = model.score(X_test_scaled, y_test_cls)
            
            results['logistic_regression'] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'model': model
            }
            
            logger.info(f"Logistic Regression - 訓練準確率: {train_acc:.4f}, 測試準確率: {test_acc:.4f}")
        
        # 繼續訓練 XGBoost Classifier
        if 'xgboost_classifier' in existing_models:
            logger.info("繼續訓練 XGBoost Classifier...")
            
            model = existing_models['xgboost_classifier']
            
            # 使用較小的學習率和較少的樹進行微調
            model.set_params(learning_rate=0.01, n_estimators=20)
            
            model.fit(X_train, y_train_cls)
            
            # 評估
            train_acc = model.score(X_train, y_train_cls)
            test_acc = model.score(X_test, y_test_cls)
            
            results['xgboost_classifier'] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'model': model
            }
            
            logger.info(f"XGBoost Classifier - 訓練準確率: {train_acc:.4f}, 測試準確率: {test_acc:.4f}")
        
        # 繼續訓練 XGBoost Regressor
        if 'xgboost_regressor' in existing_models:
            logger.info("繼續訓練 XGBoost Regressor...")
            
            model = existing_models['xgboost_regressor']
            
            # 使用較小的學習率和較少的樹進行微調
            model.set_params(learning_rate=0.01, n_estimators=20)
            
            model.fit(X_train, y_train_reg)
            
            # 評估
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred))
            train_mae = mean_absolute_error(y_train_reg, y_train_pred)
            test_mae = mean_absolute_error(y_test_reg, y_test_pred)
            
            results['xgboost_regressor'] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'model': model
            }
            
            logger.info(f"XGBoost Regressor - 訓練RMSE: {train_rmse:.4f}, 測試RMSE: {test_rmse:.4f}")
        
        return results
    
    def save_updated_models(self, results):
        """保存更新後的模型"""
        logger.info("保存更新後的模型...")
        
        for model_name, result in results.items():
            model = result['model']
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"模型已保存: {model_path}")
    
    def log_training_history(self, results):
        """記錄訓練歷史"""
        # 只保存數值結果，不保存模型對象
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'train_accuracy': result.get('train_accuracy'),
                'test_accuracy': result.get('test_accuracy'),
                'train_rmse': result.get('train_rmse'),
                'test_rmse': result.get('test_rmse'),
                'train_mae': result.get('train_mae'),
                'test_mae': result.get('test_mae')
            }
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'results': serializable_results
        }
        
        log_file = self.training_logs_dir / "continue_training_log.json"
        
        # 讀取現有日誌
        import json
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, ValueError):
                logger.warning("無法讀取現有日誌文件，將創建新的日誌")
                history = []
        else:
            history = []
        
        # 添加新記錄
        history.append(log_data)
        
        # 保存日誌（只保留最近10次記錄）
        if len(history) > 10:
            history = history[-10:]
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"訓練記錄已保存到: {log_file}")


def main():
    """主函數"""
    import time
    import argparse
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='繼續訓練模型')
    parser.add_argument('--checkpoint', type=str, help='指定要使用的 checkpoint 名稱（如：2024-05-31_14-30-25）')
    parser.add_argument('--list', action='store_true', help='列出所有可用的 checkpoint')
    args = parser.parse_args()
    
    logger.info("=== 開始繼續訓練流程 ===")
    start_time = time.time()
    
    try:
        # 如果要求列出 checkpoint
        if args.list:
            temp_trainer = ContinueTrainer()
            checkpoints = temp_trainer.get_available_checkpoints()
            if checkpoints:
                logger.info("可用的 checkpoint:")
                for i, cp in enumerate(checkpoints):
                    logger.info(f"  {i+1}. {cp['name']}")
            else:
                logger.info("沒有找到任何 checkpoint")
            return
        
        # 初始化繼續訓練器
        continue_trainer = ContinueTrainer(checkpoint_name=args.checkpoint)
        
        # 顯示載入來源
        if args.checkpoint:
            logger.info(f"將使用指定的 checkpoint: {args.checkpoint}")
        else:
            logger.info("將使用最新的模型（未指定 checkpoint）")
        
        # 步驟1: 備份現有模型
        logger.info("步驟1: 備份現有模型...")
        if not continue_trainer.backup_current_models():
            return
        
        # 步驟2: 載入數據
        logger.info("步驟2: 載入數據...")
        price_csv_path = get_data_file_path("raw/prices.csv")
        if not price_csv_path.exists():
            logger.error("找不到股價數據文件，請先運行 fetch_data.py")
            return
        
        price_df = pd.read_csv(price_csv_path)
        price_df['date'] = pd.to_datetime(price_df['date'])
        
        # 步驟3: 特徵工程
        logger.info("步驟3: 進行特徵工程...")
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(price_df)
        
        if features_df.empty:
            logger.error("特徵工程失敗")
            return
        
        # 準備訓練數據
        feature_columns = [col for col in features_df.columns 
                          if not col.startswith('label_') and 
                          col not in ['future_return_1w', 'future_return_1m', 'date', 'stock_code']]
        
        numeric_features = features_df[feature_columns].select_dtypes(include=[np.number])
        features = numeric_features.values
        
        label_columns = ['label_1w', 'label_1m', 'future_return_1w', 'future_return_1m']
        available_labels = [col for col in label_columns if col in features_df.columns]
        labels = features_df[available_labels].values
        
        logger.info(f"使用特徵: {features.shape}, 標籤: {labels.shape}")
        
        # 步驟4: 繼續訓練
        logger.info("步驟4: 繼續訓練模型...")
        results = continue_trainer.continue_train_models(features, labels)
        
        if not results:
            logger.error("繼續訓練失敗")
            return
        
        # 步驟5: 保存模型
        logger.info("步驟5: 保存更新後的模型...")
        continue_trainer.save_updated_models(results)
        
        # 步驟6: 記錄訓練歷史
        logger.info("步驟6: 記錄訓練歷史...")
        continue_trainer.log_training_history(results)
        
        total_time = time.time() - start_time
        logger.info("=== 繼續訓練流程完成 ===")
        logger.info(f"總訓練時間: {total_time:.1f}秒 ({total_time/60:.1f}分鐘)")
        
        # 顯示結果摘要
        logger.info("=== 訓練結果摘要 ===")
        for model_name, result in results.items():
            if 'test_accuracy' in result:
                logger.info(f"  {model_name}: 測試準確率 = {result['test_accuracy']:.4f}")
            elif 'test_rmse' in result:
                logger.info(f"  {model_name}: 測試RMSE = {result['test_rmse']:.4f}")
        
    except Exception as e:
        logger.error(f"繼續訓練過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序執行失敗: {e}")
        import traceback
        traceback.print_exc()
