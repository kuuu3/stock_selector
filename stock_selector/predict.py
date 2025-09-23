"""
統一預測腳本
整合完整預測流程和快速預測功能
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import argparse
import time

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_collection import PriceFetcher
from src.preprocessing import FeatureEngineer
from src.models.predict import StockPredictor
from src.selection import StockSelector
from src.config import get_data_file_path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_predict():
    """快速預測 - 使用現有數據"""
    logger.info("=== 快速預測模式 ===")
    
    # 步驟1: 檢查數據文件
    step_start = time.time()
    price_csv_path = get_data_file_path("raw/prices.csv")
    if not price_csv_path.exists():
        logger.error("找不到數據文件 data/raw/prices.csv")
        logger.error("請先運行: python fetch_all_data.py")
        return
    
    logger.info("步驟1: 載入股價數據...")
    try:
        price_df = pd.read_csv(price_csv_path)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['stock_code'] = price_df['stock_code'].astype(str)
        logger.info(f"載入股價數據: {len(price_df)} 筆")
    except Exception as e:
        logger.error(f"載入股價數據失敗: {e}")
        return
    
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
        logger.info("找不到新聞情感分析數據，將只使用股價數據")
    
    step_time = time.time() - step_start
    logger.info(f"步驟2完成，耗時: {step_time:.2f}秒")
    
    # 步驟3: 特徵工程
    step_start = time.time()
    logger.info("步驟3: 進行特徵工程...")
    
    try:
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(price_df, news_df)
        
        if features_df.empty:
            logger.error("特徵工程失敗")
            return
        
        # 確保保留 stock_code 和 date 欄位
        if 'stock_code' not in features_df.columns:
            # 如果特徵工程過程中丟失了 stock_code，從原始數據中恢復
            features_df = features_df.reset_index(drop=True)
            price_df_reset = price_df.reset_index(drop=True)
            features_df['stock_code'] = price_df_reset['stock_code']
            features_df['date'] = price_df_reset['date']
        
        logger.info(f"特徵工程完成，生成 {len(features_df)} 個樣本")
        
        # 顯示新聞特徵統計
        news_feature_columns = [col for col in features_df.columns if 'news' in col.lower() or 'sentiment' in col.lower()]
        if news_feature_columns:
            logger.info(f"新聞相關特徵: {len(news_feature_columns)} 個")
        else:
            logger.info("沒有生成新聞相關特徵")
            
    except Exception as e:
        logger.error(f"特徵工程失敗: {e}")
        return
    
    step_time = time.time() - step_start
    logger.info(f"步驟3完成，耗時: {step_time:.2f}秒")
    
    # 步驟4: 模型預測
    step_start = time.time()
    logger.info("步驟4: 進行模型預測...")
    
    try:
        predictor = StockPredictor()
        predictor.load_models()
        
        # 準備預測數據（排除標籤欄位）
        label_columns = ['future_return_1w', 'future_return_1m', 'label_1w', 'label_1m']
        feature_columns = [col for col in features_df.columns 
                          if col not in ['stock_code', 'date'] + label_columns and 
                          features_df[col].dtype in ['float64', 'int64']]
        features_array = features_df[feature_columns].values
        stock_codes = features_df['stock_code'].tolist()
        
        predictions_df = predictor.predict_stocks(features_array, stock_codes)
        
        if predictions_df.empty:
            logger.error("預測失敗")
            return
        
        logger.info(f"預測完成，共 {len(predictions_df)} 支股票")
        
    except Exception as e:
        logger.error(f"預測失敗: {e}")
        return
    
    step_time = time.time() - step_start
    logger.info(f"步驟4完成，耗時: {step_time:.2f}秒")
    
    # 步驟5: 選股
    step_start = time.time()
    logger.info("步驟5: 進行選股...")
    
    try:
        selector = StockSelector()
        selected_stocks = selector.select_top_stocks(predictions_df)
        
        if selected_stocks.empty:
            logger.error("選股失敗")
            return
        
        logger.info(f"選股完成，選出 {len(selected_stocks)} 支股票")
        
        # 保存結果
        output_path = selector.save_selection_results(selected_stocks)
        logger.info(f"選股結果已保存到: {output_path}")
        
        # 顯示前10支股票
        logger.info("\\n前10支推薦股票:")
        for i, (_, stock) in enumerate(selected_stocks.head(10).iterrows(), 1):
            logger.info(f"{i:2d}. {stock['stock_code']} - 風險調整分數: {stock['risk_adjusted_score']:.4f}")
        
    except Exception as e:
        logger.error(f"選股失敗: {e}")
        return
    
    step_time = time.time() - step_start
    logger.info(f"步驟5完成，耗時: {step_time:.2f}秒")
    
    total_time = time.time() - (time.time() - step_time)
    logger.info(f"=== 快速預測完成，總耗時: {total_time:.2f}秒 ===")


def full_predict():
    """完整預測 - 包含數據獲取"""
    logger.info("=== 完整預測模式 ===")
    
    # 步驟1: 獲取最新數據
    step_start = time.time()
    logger.info("步驟1: 獲取最新數據...")
    
    try:
        price_fetcher = PriceFetcher()
        price_df = price_fetcher.fetch_all_stocks(save_to_file=True)
        
        if price_df.empty:
            logger.error("獲取股價數據失敗")
            return
        
        logger.info(f"獲取股價數據: {len(price_df)} 筆")
        
        # 嘗試獲取新聞數據
        try:
            from src.data_collection.news_scraper import NewsScraper
            news_scraper = NewsScraper()
            news_df = news_scraper.scrape_all_news(pages_per_source=2)
            
            if not news_df.empty:
                logger.info(f"獲取新聞數據: {len(news_df)} 筆")
                
                # 進行情感分析
                logger.info("進行新聞情感分析...")
                # 這裡可以調用情感分析功能
                
        except Exception as e:
            logger.warning(f"獲取新聞數據失敗: {e}")
            news_df = None
        
    except Exception as e:
        logger.error(f"獲取數據失敗: {e}")
        return
    
    step_time = time.time() - step_start
    logger.info(f"步驟1完成，耗時: {step_time:.2f}秒")
    
    # 步驟2-5: 使用快速預測流程
    logger.info("使用現有數據進行預測...")
    quick_predict()


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='統一預測腳本')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='預測模式: quick (使用現有數據) 或 full (獲取最新數據)')
    parser.add_argument('--top-n', type=int, default=20,
                       help='選出的股票數量 (預設: 20)')
    
    args = parser.parse_args()
    
    logger.info(f"=== 開始預測 (模式: {args.mode}) ===")
    
    if args.mode == 'quick':
        quick_predict()
    elif args.mode == 'full':
        full_predict()
    
    logger.info("=== 預測完成 ===")


if __name__ == "__main__":
    main()
