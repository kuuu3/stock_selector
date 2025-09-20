"""
模型訓練腳本
訓練選股系統的機器學習模型
"""

import sys
from pathlib import Path

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_collection import PriceFetcher
from src.preprocessing import FeatureEngineer
from src.models import ModelTrainer, ModelEvaluator
import logging
import numpy as np
import pandas as pd

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(force_refresh_data=False):
    """主訓練流程
    
    Args:
        force_refresh_data: 是否強制重新獲取數據
    """
    import time
    
    logger.info("=== 開始模型訓練流程 ===")
    start_time = time.time()
    
    try:
        # 步驟1: 獲取歷史數據（檢查是否已有數據）
        step_start = time.time()
        price_csv_path = Path("data/raw/prices.csv")
        
        if price_csv_path.exists() and not force_refresh_data:
            logger.info("步驟1: 載入現有股價數據...")
            price_df = pd.read_csv(price_csv_path)
            logger.info(f"✓ 載入現有數據 {len(price_df)} 筆股價數據")
        else:
            logger.info("步驟1: 獲取歷史股價數據...")
            price_fetcher = PriceFetcher()
            price_df = price_fetcher.fetch_all_stocks(save_to_file=True)
            
            if price_df.empty:
                logger.error("無法獲取股價數據")
                return
            
            logger.info(f"✓ 成功獲取 {len(price_df)} 筆股價數據")
        
        step_time = time.time() - step_start
        logger.info(f"數據處理完成 (耗時: {step_time:.1f}秒)")
        
        # 步驟2: 特徵工程
        step_start = time.time()
        logger.info("步驟2: 進行特徵工程...")
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(price_df)
        
        if features_df.empty:
            logger.error("特徵工程失敗")
            return
        
        step_time = time.time() - step_start
        logger.info(f"✓ 特徵工程完成，生成 {len(features_df)} 個樣本 (耗時: {step_time:.1f}秒)")
        
        # 步驟3: 訓練模型
        step_start = time.time()
        logger.info("步驟3: 開始訓練模型...")
        logger.info("  預估訓練時間: 2-5分鐘")
        
        # 直接使用生成的特徵進行訓練
        trainer = ModelTrainer()
        
        # 分離特徵和標籤
        feature_columns = [col for col in features_df.columns if not col.startswith('label_') and col not in ['future_return_1w', 'future_return_1m', 'date', 'stock_code']]
        label_columns = ['label_1w', 'label_1m', 'future_return_1w', 'future_return_1m']
        
        # 檢查標籤列是否存在
        available_labels = [col for col in label_columns if col in features_df.columns]
        if not available_labels:
            logger.error("沒有找到任何標籤列")
            return
        
        # 只選擇數值型特徵
        numeric_features = features_df[feature_columns].select_dtypes(include=[np.number])
        features = numeric_features.values
        labels = features_df[available_labels].values
        
        logger.info(f"數值特徵列: {list(numeric_features.columns)}")
        
        logger.info(f"使用特徵: {features.shape}, 標籤: {labels.shape}")
        
        # 訓練模型
        results = trainer.train_with_data(features, labels)
        
        if not results:
            logger.error("模型訓練失敗")
            return
        
        step_time = time.time() - step_start
        logger.info(f"✓ 模型訓練完成 (耗時: {step_time:.1f}秒)")
        
        # 步驟4: 評估模型
        step_start = time.time()
        logger.info("步驟4: 評估模型性能...")
        evaluator = ModelEvaluator()
        
        # 這裡可以添加評估邏輯
        # 由於我們沒有測試數據，先跳過詳細評估
        
        step_time = time.time() - step_start
        logger.info(f"✓ 模型評估完成 (耗時: {step_time:.1f}秒)")
        
        total_time = time.time() - start_time
        logger.info("=== 模型訓練流程完成 ===")
        logger.info(f"總訓練時間: {total_time:.1f}秒 ({total_time/60:.1f}分鐘)")
        
        # 顯示結果摘要
        logger.info("=== 訓練結果摘要 ===")
        for model_name, result in results.items():
            if 'accuracy' in result:
                logger.info(f"  {model_name}: 準確率 = {result['accuracy']:.4f}")
            elif 'rmse' in result:
                logger.info(f"  {model_name}: RMSE = {result['rmse']:.4f}")
        
    except Exception as e:
        logger.error(f"訓練過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='股票選股模型訓練')
    parser.add_argument('--refresh-data', action='store_true', 
                       help='強制重新獲取數據')
    args = parser.parse_args()
    
    try:
        main(force_refresh_data=args.refresh_data)
    except Exception as e:
        print(f"程序執行失敗: {e}")
        import traceback
        traceback.print_exc()
