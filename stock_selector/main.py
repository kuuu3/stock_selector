"""
Stock Selector 主程式入口點
"""

import sys
from pathlib import Path

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_collection import PriceFetcher, NewsScraper
from src.preprocessing import FeatureEngineer
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主程式"""
    logger.info("=== Stock Selector 系統啟動 ===")
    
    try:
        # 1. 數據收集
        logger.info("步驟1: 收集股價數據...")
        price_fetcher = PriceFetcher()
        price_df = price_fetcher.get_latest_data()
        
        if price_df.empty:
            logger.error("無法獲取股價數據")
            return
        
        logger.info("步驟2: 收集新聞數據...")
        news_scraper = NewsScraper()
        news_df = news_scraper.scrape_all_news(pages_per_source=3)
        
        # 2. 特徵工程
        logger.info("步驟3: 進行特徵工程...")
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(price_df, news_df)
        
        if features_df.empty:
            logger.error("特徵工程失敗")
            return
        
        # 3. 保存結果
        logger.info("步驟4: 保存處理結果...")
        feature_engineer.save_features(features_df)
        
        logger.info("=== 數據處理完成 ===")
        logger.info(f"處理了 {len(price_df)} 筆股價數據")
        logger.info(f"處理了 {len(news_df)} 則新聞")
        logger.info(f"生成了 {len(features_df)} 個樣本的特徵矩陣")
        
    except Exception as e:
        logger.error(f"程式執行過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    main()

