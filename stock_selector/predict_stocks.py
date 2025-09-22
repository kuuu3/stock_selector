"""
股票預測腳本
使用訓練好的模型進行股票預測和選股
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_collection import PriceFetcher
from src.preprocessing import FeatureEngineer
from src.models import StockPredictor
from src.selection import StockSelector
from src.config import get_data_file_path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主預測流程"""
    import time
    
    logger.info("=== 開始股票預測流程 ===")
    start_time = time.time()
    
    try:
        # 步驟1: 載入股價數據
        step_start = time.time()
        logger.info("步驟1: 載入股價數據...")
        
        price_csv_path = get_data_file_path("raw/prices.csv")
        if not price_csv_path.exists():
            logger.error("找不到股價數據文件，請先運行 fetch_data.py 獲取數據")
            return
        
        price_df = pd.read_csv(price_csv_path)
        price_df['date'] = pd.to_datetime(price_df['date'])
        
        step_time = time.time() - step_start
        logger.info(f"成功載入 {len(price_df)} 筆股價數據 (耗時: {step_time:.1f}秒)")
        logger.info(f"數據日期範圍: {price_df['date'].min().strftime('%Y-%m-%d')} 到 {price_df['date'].max().strftime('%Y-%m-%d')}")
        
        # 步驟2: 特徵工程
        step_start = time.time()
        logger.info("步驟2: 進行特徵工程...")
        
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(price_df)
        
        if features_df.empty:
            logger.error("特徵工程失敗")
            return
        
        step_time = time.time() - step_start
        logger.info(f"特徵工程完成，生成 {len(features_df)} 個樣本 (耗時: {step_time:.1f}秒)")
        
        # 步驟3: 模型預測
        step_start = time.time()
        logger.info("步驟3: 載入模型並進行預測...")
        
        predictor = StockPredictor()
        if not predictor.load_models():
            logger.error("無法載入模型，請先運行 train_model.py")
            return
        
        # 準備預測數據
        feature_columns = [col for col in features_df.columns 
                          if not col.startswith('label_') and 
                          col not in ['future_return_1w', 'future_return_1m', 'date', 'stock_code']]
        
        # 只選擇數值型特徵
        numeric_features = features_df[feature_columns].select_dtypes(include=[np.number])
        prediction_data = numeric_features.values
        
        logger.info(f"預測數據形狀: {prediction_data.shape}")
        
        # 進行預測
        classification_results = predictor.predict_classification(prediction_data)
        regression_results = predictor.predict_regression(prediction_data)
        
        step_time = time.time() - step_start
        logger.info(f"模型預測完成 (耗時: {step_time:.1f}秒)")
        
        # 步驟4: 選股和排序
        step_start = time.time()
        logger.info("步驟4: 進行股票選擇和排序...")
        
        # 創建包含預測結果的DataFrame
        results_df = features_df[['date', 'stock_code']].copy()
        
        # 添加分類預測結果
        if 'logistic_regression' in classification_results:
            lr_probs = classification_results['logistic_regression']['probabilities']
            results_df['logistic_regression_prob'] = lr_probs[:, 2]  # 取上漲概率 (標籤2)
        
        if 'xgboost_classifier' in classification_results:
            xgb_probs = classification_results['xgboost_classifier']['probabilities']
            results_df['xgboost_classifier_prob'] = xgb_probs[:, 2]  # 取上漲概率 (標籤2)
        
        # 添加回歸預測結果
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
            logger.warning("沒有可用的預測結果")
            return
        
        # 按評分排序
        results_df = results_df.sort_values('composite_score', ascending=False)
        
        # 獲取最新日期的股票
        latest_date = results_df['date'].max()
        latest_stocks = results_df[results_df['date'] == latest_date].copy()
        
        # 選擇Top20
        top20_stocks = latest_stocks.head(20)
        
        step_time = time.time() - step_start
        logger.info(f"股票選擇完成 (耗時: {step_time:.1f}秒)")
        
        # 步驟5: 輸出結果
        logger.info("=== 預測結果 ===")
        logger.info(f"預測日期: {latest_date}")
        logger.info(f"Top 20 推薦股票:")
        
        for i, (_, stock) in enumerate(top20_stocks.iterrows(), 1):
            logger.info(f"{i:2d}. {stock['stock_code']} - 評分: {stock['composite_score']:.4f}")
        
        # 保存結果
        output_file = Path("outputs/top20_predictions.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        top20_stocks.to_csv(output_file, index=False)
        logger.info(f"結果已保存到: {output_file}")
        
        total_time = time.time() - start_time
        logger.info("=== 股票預測流程完成 ===")
        logger.info(f"總預測時間: {total_time:.1f}秒 ({total_time/60:.1f}分鐘)")
        
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
