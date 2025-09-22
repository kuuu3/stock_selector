"""
快速預測腳本 - 使用現有數據進行預測
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.preprocessing import FeatureEngineer
from src.models import StockPredictor
from src.config import get_data_file_path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """快速預測流程"""
    import time
    
    logger.info("=== 快速預測模式 ===")
    logger.info("提示: 使用現有數據進行預測，如需更新數據請先運行 fetch_data.py")
    
    start_time = time.time()
    
    try:
        # 步驟1: 檢查數據文件
        price_csv_path = get_data_file_path("raw/prices.csv")
        if not price_csv_path.exists():
            logger.error("找不到數據文件 data/raw/prices.csv")
            logger.error("請先運行: python fetch_data.py")
            return
        
        # 步驟2: 載入數據
        logger.info("步驟1: 載入股價數據...")
        price_df = pd.read_csv(price_csv_path)
        price_df['date'] = pd.to_datetime(price_df['date'])
        
        logger.info(f"載入 {len(price_df)} 筆數據")
        logger.info(f"數據日期: {price_df['date'].max().strftime('%Y-%m-%d')}")
        
        # 步驟3: 特徵工程
        logger.info("步驟2: 進行特徵工程...")
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(price_df)
        
        if features_df.empty:
            logger.error("特徵工程失敗")
            return
        
        logger.info(f"生成 {len(features_df)} 個樣本")
        
        # 步驟4: 載入模型並預測
        logger.info("步驟3: 載入模型並預測...")
        predictor = StockPredictor()
        if not predictor.load_models():
            logger.error("無法載入模型，請先運行 train_model.py")
            return
        
        # 準備預測數據
        feature_columns = [col for col in features_df.columns 
                          if not col.startswith('label_') and 
                          col not in ['future_return_1w', 'future_return_1m', 'date', 'stock_code']]
        
        numeric_features = features_df[feature_columns].select_dtypes(include=[np.number])
        prediction_data = numeric_features.values
        
        # 進行預測
        classification_results = predictor.predict_classification(prediction_data)
        regression_results = predictor.predict_regression(prediction_data)
        
        logger.info("預測完成")
        
        # 步驟5: 生成結果
        logger.info("步驟4: 生成選股結果...")
        
        # 創建結果DataFrame
        results_df = features_df[['date', 'stock_code']].copy()
        
        # 添加預測結果
        if 'logistic_regression' in classification_results:
            lr_probs = classification_results['logistic_regression']['probabilities']
            results_df['logistic_regression_prob'] = lr_probs[:, 2]
        
        if 'xgboost_classifier' in classification_results:
            xgb_probs = classification_results['xgboost_classifier']['probabilities']
            results_df['xgboost_classifier_prob'] = xgb_probs[:, 2]
        
        if 'xgboost_regressor' in regression_results:
            results_df['xgboost_regressor_return'] = regression_results['xgboost_regressor']['predictions']
        
        # 計算綜合評分
        score_components = []
        if 'logistic_regression_prob' in results_df.columns:
            score_components.append(results_df['logistic_regression_prob'] * 0.3)
        if 'xgboost_classifier_prob' in results_df.columns:
            score_components.append(results_df['xgboost_classifier_prob'] * 0.5)
        if 'xgboost_regressor_return' in results_df.columns:
            score_components.append(results_df['xgboost_regressor_return'] * 100 * 0.2)
        
        if score_components:
            results_df['composite_score'] = sum(score_components)
        else:
            logger.error("沒有可用的預測結果")
            return
        
        # 按評分排序並選擇最新日期的股票
        results_df = results_df.sort_values('composite_score', ascending=False)
        latest_date = results_df['date'].max()
        latest_stocks = results_df[results_df['date'] == latest_date].head(10)
        
        # 顯示結果
        total_time = time.time() - start_time
        logger.info("=== 預測結果 ===")
        logger.info(f"預測日期: {latest_date.strftime('%Y-%m-%d')}")
        logger.info(f"預測時間: {total_time:.1f}秒")
        logger.info(f"Top 10 推薦股票:")
        
        for i, (_, stock) in enumerate(latest_stocks.iterrows(), 1):
            logger.info(f"   {i:2d}. {stock['stock_code']} - 評分: {stock['composite_score']:.4f}")
        
        # 保存結果
        output_file = Path("outputs/quick_predictions.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        latest_stocks.to_csv(output_file, index=False)
        logger.info(f"結果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"預測過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序執行失敗: {e}")
        import traceback
        traceback.print_exc()
