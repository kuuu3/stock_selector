"""
統一訓練腳本
整合完整訓練和增量訓練功能
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import joblib
import shutil
from datetime import datetime
import argparse
import time

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_collection import PriceFetcher
from src.preprocessing import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.config import get_data_file_path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """統一訓練器 - 支持完整訓練和增量訓練"""
    
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.feature_engineer = FeatureEngineer()
    
    def full_train(self):
        """完整訓練模式"""
        logger.info("=== 開始完整訓練模式 ===")
        
        # 記錄完整訓練開始時間
        full_train_start = time.time()
        
        # 步驟1: 載入股價數據
        step_start = time.time()
        logger.info("步驟1: 載入股價數據...")
        
        price_csv_path = get_data_file_path("raw/prices.csv")
        if not price_csv_path.exists():
            logger.error("找不到股價數據文件，請先運行: python fetch_all_data.py")
            return False
        
        try:
            price_df = pd.read_csv(price_csv_path)
            price_df['date'] = pd.to_datetime(price_df['date'])
            price_df['stock_code'] = price_df['stock_code'].astype(str)
            logger.info(f"載入股價數據: {len(price_df)} 筆")
        except Exception as e:
            logger.error(f"載入股價數據失敗: {e}")
            return False
        
        step_time = time.time() - step_start
        logger.info(f"步驟1完成，耗時: {step_time:.2f}秒")
        
        # 步驟2: 載入新聞數據
        step_start = time.time()
        logger.info("步驟2: 載入新聞數據...")
        
        news_df = None
        news_csv_path = get_data_file_path("processed/news_with_sentiment.csv")
        if news_csv_path.exists():
            try:
                news_df = pd.read_csv(news_csv_path)
                news_df['analyzed_time'] = pd.to_datetime(news_df['analyzed_time'])
                logger.info(f"載入新聞數據: {len(news_df)} 筆")
                
                sentiment_counts = news_df['sentiment'].value_counts()
                logger.info(f"情感分布: 正面 {sentiment_counts.get('positive', 0)}, "
                           f"負面 {sentiment_counts.get('negative', 0)}, "
                           f"中性 {sentiment_counts.get('neutral', 0)}")
            except Exception as e:
                logger.error(f"載入新聞數據時發生錯誤: {e}")
                news_df = None
        else:
            logger.info("找不到新聞情感分析數據，將只使用股價數據進行訓練")
        
        step_time = time.time() - step_start
        logger.info(f"步驟2完成，耗時: {step_time:.2f}秒")
        
        # 步驟3: 特徵工程
        step_start = time.time()
        logger.info("步驟3: 進行特徵工程...")
        
        try:
            features_df = self.feature_engineer.create_features(price_df, news_df)
            
            if features_df.empty:
                logger.error("特徵工程失敗")
                return False
            
            logger.info(f"特徵工程完成，生成 {len(features_df)} 個樣本")
            
            # 顯示新聞特徵統計
            news_feature_columns = [col for col in features_df.columns if 'news' in col.lower() or 'sentiment' in col.lower()]
            if news_feature_columns:
                logger.info(f"新聞相關特徵: {news_feature_columns}")
            else:
                logger.info("沒有生成新聞相關特徵")
                
        except Exception as e:
            logger.error(f"特徵工程失敗: {e}")
            return False
        
        step_time = time.time() - step_start
        logger.info(f"步驟3完成，耗時: {step_time:.2f}秒")
        
        # 步驟4: 訓練模型
        step_start = time.time()
        logger.info("步驟4: 開始訓練模型...")
        
        try:
            # 創建標籤數據
            labels_df = self.feature_engineer.create_labels(price_df)
            
            # 使用1週標籤進行訓練
            valid_mask = ~(labels_df['label_1w'].isna())
            features_clean = features_df[valid_mask].copy()
            labels_clean = labels_df[valid_mask].copy()
            
            # 移除日期和股票代碼欄位，只保留數值特徵（排除標籤欄位）
            label_columns = ['future_return_1w', 'future_return_1m', 'label_1w', 'label_1m']
            feature_columns = [col for col in features_clean.columns 
                             if col not in ['stock_code', 'date'] + label_columns and 
                             features_clean[col].dtype in ['float64', 'int64']]
            features_clean = features_clean[feature_columns]
            
            logger.info(f"清理後數據: {len(features_clean)} 筆")
            
            # 轉換為 numpy 數組
            features_array = features_clean.values
            # 準備標籤數組：包含分類標籤和回歸標籤
            labels_array = np.column_stack([
                labels_clean['label_1w'].values,  # 分類標籤 (1週)
                labels_clean['future_return_1w'].values,  # 回歸標籤 (1週)
                labels_clean['future_return_1m'].values   # 回歸標籤 (1個月)
            ])
            
            # 訓練模型
            trained_models = self.model_trainer.train_with_data(features_array, labels_array)
            
            if not trained_models:
                logger.error("模型訓練失敗")
                return False
            
            logger.info("模型訓練完成")
            
        except Exception as e:
            logger.error(f"模型訓練失敗: {e}")
            return False
        
        step_time = time.time() - step_start
        logger.info(f"步驟4完成，耗時: {step_time:.2f}秒")
        
        # 步驟5: 保存模型
        step_start = time.time()
        logger.info("步驟5: 保存模型...")
        
        try:
            # 創建模型保存目錄
            models_dir = Path("outputs/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存模型（從評估字典中提取實際模型）
            for model_name, model_result in trained_models.items():
                if isinstance(model_result, dict) and 'model' in model_result:
                    # 從評估字典中提取實際模型
                    actual_model = model_result['model']
                    model_path = models_dir / f"{model_name}_model.pkl"
                    joblib.dump(actual_model, model_path)
                    logger.info(f"保存模型: {model_path}")
                else:
                    # 如果直接是模型對象
                    model_path = models_dir / f"{model_name}_model.pkl"
                    joblib.dump(model_result, model_path)
                    logger.info(f"保存模型: {model_path}")
            
            # 保存特徵列名
            feature_columns_path = models_dir / "feature_columns.pkl"
            joblib.dump(feature_columns, feature_columns_path)
            logger.info(f"保存特徵列名: {feature_columns_path}")
            
            # 保存模型元數據
            metadata = {
                'training_date': datetime.now().isoformat(),
                'data_samples': len(features_clean),
                'feature_count': len(feature_columns),
                'models': list(trained_models.keys()),
                'has_news_data': news_df is not None
            }
            metadata_path = models_dir / "training_metadata.pkl"
            joblib.dump(metadata, metadata_path)
            logger.info(f"保存訓練元數據: {metadata_path}")
            
        except Exception as e:
            logger.error(f"保存模型失敗: {e}")
            return False
        
        step_time = time.time() - step_start
        logger.info(f"步驟5完成，耗時: {step_time:.2f}秒")
        
        # 計算完整的端到端訓練時間
        total_time = time.time() - full_train_start
        logger.info(f"=== 完整訓練完成，總耗時: {total_time:.2f}秒 ===")
        return True
    
    def continue_train(self, checkpoint_name=None):
        """增量訓練模式"""
        logger.info("=== 開始增量訓練模式 ===")
        
        # 步驟1: 查找最新的 checkpoint
        checkpoints_dir = Path("outputs/checkpoints")
        if not checkpoints_dir.exists():
            logger.error("找不到 checkpoints 目錄，請先進行完整訓練")
            return False
        
        if checkpoint_name:
            checkpoint_path = checkpoints_dir / checkpoint_name
            if not checkpoint_path.exists():
                logger.error(f"找不到指定的 checkpoint: {checkpoint_name}")
                return False
        else:
            # 查找最新的 checkpoint
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
            if not checkpoint_dirs:
                logger.error("找不到任何 checkpoint，請先進行完整訓練")
                return False
            
            checkpoint_path = max(checkpoint_dirs, key=lambda x: x.name)
            logger.info(f"使用最新的 checkpoint: {checkpoint_path.name}")
        
        # 步驟2: 載入現有模型
        logger.info("步驟2: 載入現有模型...")
        
        try:
            model_path = checkpoint_path / "models"
            if not model_path.exists():
                logger.error(f"checkpoint 中找不到模型文件: {model_path}")
                return False
            
            # 載入模型
            models = {}
            for model_file in model_path.glob("*.pkl"):
                if model_file.name != "feature_columns.pkl":
                    model_name = model_file.stem.replace("_model", "")
                    models[model_name] = joblib.load(model_file)
                    logger.info(f"載入模型: {model_name}")
            
            if not models:
                logger.error("沒有找到任何模型文件")
                return False
            
            # 載入特徵列名
            feature_columns_path = model_path / "feature_columns.pkl"
            if feature_columns_path.exists():
                feature_columns = joblib.load(feature_columns_path)
                logger.info(f"載入特徵列名: {len(feature_columns)} 個")
            else:
                logger.error("找不到特徵列名文件")
                return False
            
        except Exception as e:
            logger.error(f"載入現有模型失敗: {e}")
            return False
        
        # 步驟3: 載入新數據
        logger.info("步驟3: 載入新數據...")
        
        try:
            price_csv_path = get_data_file_path("raw/prices.csv")
            if not price_csv_path.exists():
                logger.error("找不到股價數據文件")
                return False
            
            price_df = pd.read_csv(price_csv_path)
            price_df['date'] = pd.to_datetime(price_df['date'])
            price_df['stock_code'] = price_df['stock_code'].astype(str)
            logger.info(f"載入股價數據: {len(price_df)} 筆")
            
            # 載入新聞數據
            news_df = None
            news_csv_path = get_data_file_path("processed/news_with_sentiment.csv")
            if news_csv_path.exists():
                try:
                    news_df = pd.read_csv(news_csv_path)
                    news_df['analyzed_time'] = pd.to_datetime(news_df['analyzed_time'])
                    logger.info(f"載入新聞數據: {len(news_df)} 筆")
                except Exception as e:
                    logger.warning(f"載入新聞數據時發生錯誤: {e}")
                    news_df = None
            
        except Exception as e:
            logger.error(f"載入數據失敗: {e}")
            return False
        
        # 步驟4: 特徵工程
        logger.info("步驟4: 進行特徵工程...")
        
        try:
            features_df = self.feature_engineer.create_features(price_df, news_df)
            
            if features_df.empty:
                logger.error("特徵工程失敗")
                return False
            
            logger.info(f"特徵工程完成，生成 {len(features_df)} 個樣本")
            
        except Exception as e:
            logger.error(f"特徵工程失敗: {e}")
            return False
        
        # 步驟5: 增量訓練
        logger.info("步驟5: 進行增量訓練...")
        
        try:
            # 創建標籤數據
            labels_df = self.feature_engineer.create_labels(price_df)
            
            # 使用1週標籤進行訓練
            valid_mask = ~(labels_df['label_1w'].isna())
            features_clean = features_df[valid_mask].copy()
            labels_clean = labels_df[valid_mask].copy()
            
            # 只保留數值特徵（排除標籤欄位）
            label_columns = ['future_return_1w', 'future_return_1m', 'label_1w', 'label_1m']
            feature_columns = [col for col in features_clean.columns 
                             if col not in ['stock_code', 'date'] + label_columns and 
                             features_clean[col].dtype in ['float64', 'int64']]
            features_clean = features_clean[feature_columns]
            
            # 轉換為 numpy 數組
            features_array = features_clean.values
            
            # 準備分類和回歸標籤
            y_classification = labels_clean['label_1w'].values
            y_regression_1w = labels_clean['future_return_1w'].values  # 1週回歸標籤
            y_regression_1m = labels_clean['future_return_1m'].values  # 1個月回歸標籤
            
            # 將分類標籤從 [-1, 0, 1] 轉換為 [0, 1, 2] 以符合 XGBoost 要求
            y_classification_adjusted = y_classification + 1  # [-1, 0, 1] -> [0, 1, 2]
            
            # 增量訓練
            updated_models = {}
            for model_name, model in models.items():
                try:
                    # 使用較小的學習率進行增量訓練
                    if hasattr(model, 'learning_rate'):
                        original_lr = model.learning_rate
                        model.learning_rate = original_lr * 0.1  # 降低學習率
                    
                    # 根據模型類型使用正確的標籤
                    if hasattr(model, 'fit'):
                        if 'regressor' in model_name:
                            # 回歸模型使用1個月回歸標籤
                            model.fit(features_array, y_regression_1m)
                        else:
                            # 分類模型使用分類標籤
                            model.fit(features_array, y_classification_adjusted)
                        
                        updated_models[model_name] = model
                        logger.info(f"增量訓練完成: {model_name}")
                    else:
                        logger.warning(f"模型 {model_name} 不支持增量訓練")
                        updated_models[model_name] = model
                        
                except Exception as e:
                    logger.error(f"增量訓練失敗 {model_name}: {e}")
                    updated_models[model_name] = model
            
            if not updated_models:
                logger.error("所有模型增量訓練失敗")
                return False
            
        except Exception as e:
            logger.error(f"增量訓練失敗: {e}")
            return False
        
        # 步驟6: 保存更新後的模型
        logger.info("步驟6: 保存更新後的模型...")
        
        try:
            # 創建新的 checkpoint
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_checkpoint_path = checkpoints_dir / f"continue_{timestamp}"
            new_checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            models_path = new_checkpoint_path / "models"
            models_path.mkdir(parents=True, exist_ok=True)
            
            for model_name, model_result in updated_models.items():
                if isinstance(model_result, dict) and 'model' in model_result:
                    # 從評估字典中提取實際模型
                    actual_model = model_result['model']
                    model_file = models_path / f"{model_name}_model.pkl"
                    joblib.dump(actual_model, model_file)
                    logger.info(f"保存更新後的模型: {model_file}")
                else:
                    # 如果直接是模型對象
                    model_file = models_path / f"{model_name}_model.pkl"
                    joblib.dump(model_result, model_file)
                    logger.info(f"保存更新後的模型: {model_file}")
            
            # 保存特徵列名
            feature_columns_path = models_path / "feature_columns.pkl"
            joblib.dump(feature_columns, feature_columns_path)
            
            # 保存元數據
            metadata = {
                'training_date': datetime.now().isoformat(),
                'training_type': 'continue',
                'base_checkpoint': checkpoint_path.name,
                'data_samples': len(features_clean),
                'feature_count': len(feature_columns),
                'models': list(updated_models.keys())
            }
            metadata_path = new_checkpoint_path / "training_metadata.pkl"
            joblib.dump(metadata, metadata_path)
            
            logger.info(f"增量訓練完成，新 checkpoint: {new_checkpoint_path.name}")
            
        except Exception as e:
            logger.error(f"保存更新後的模型失敗: {e}")
            return False
        
        logger.info("=== 增量訓練完成 ===")
        return True
    
    def list_checkpoints(self):
        """列出所有可用的 checkpoint"""
        checkpoints_dir = Path("outputs/checkpoints")
        if not checkpoints_dir.exists():
            logger.info("沒有找到任何 checkpoint")
            return
        
        checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
        if not checkpoint_dirs:
            logger.info("沒有找到任何 checkpoint")
            return
        
        logger.info("可用的 checkpoint:")
        for checkpoint_dir in sorted(checkpoint_dirs, key=lambda x: x.name, reverse=True):
            metadata_path = checkpoint_dir / "training_metadata.pkl"
            if metadata_path.exists():
                try:
                    metadata = joblib.load(metadata_path)
                    training_date = metadata.get('training_date', 'Unknown')
                    training_type = metadata.get('training_type', 'full')
                    data_samples = metadata.get('data_samples', 0)
                    logger.info(f"  {checkpoint_dir.name} - {training_type} - {data_samples} samples - {training_date}")
                except:
                    logger.info(f"  {checkpoint_dir.name} - 元數據讀取失敗")
            else:
                logger.info(f"  {checkpoint_dir.name} - 無元數據")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='統一訓練腳本')
    parser.add_argument('--mode', choices=['full', 'continue'], default='full',
                       help='訓練模式: full (完整訓練) 或 continue (增量訓練)')
    parser.add_argument('--checkpoint', type=str,
                       help='指定 checkpoint 名稱 (增量訓練時使用)')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用的 checkpoint')
    
    args = parser.parse_args()
    
    trainer = UnifiedTrainer()
    
    if args.list:
        trainer.list_checkpoints()
        return
    
    if args.mode == 'full':
        success = trainer.full_train()
    elif args.mode == 'continue':
        success = trainer.continue_train(args.checkpoint)
    
    if success:
        logger.info("訓練完成")
    else:
        logger.error("訓練失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
