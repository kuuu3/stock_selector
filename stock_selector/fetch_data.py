"""
獨立數據抓取腳本
專門負責獲取和更新股票數據
"""

import sys
from pathlib import Path
import logging
import pandas as pd
from datetime import datetime, timedelta

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_collection import PriceFetcher

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_existing_data():
    """檢查現有數據"""
    price_file = Path("data/raw/prices.csv")
    
    if not price_file.exists():
        logger.info("沒有找到現有數據文件")
        return None, None
    
    try:
        df = pd.read_csv(price_file)
        df['date'] = pd.to_datetime(df['date'])
        
        latest_date = df['date'].max()
        data_count = len(df)
        stock_count = df['stock_code'].nunique()
        
        logger.info(f"現有數據: {data_count} 筆記錄, {stock_count} 支股票")
        logger.info(f"最新數據日期: {latest_date.strftime('%Y-%m-%d')}")
        
        # 計算數據年齡
        days_old = (datetime.now() - latest_date).days
        logger.info(f"數據年齡: {days_old} 天")
        
        return df, latest_date
        
    except Exception as e:
        logger.error(f"讀取現有數據時發生錯誤: {e}")
        return None, None


def fetch_new_data(force_refresh=False):
    """獲取新數據"""
    logger.info("=== 開始獲取股票數據 ===")
    
    # 檢查現有數據
    existing_df, latest_date = check_existing_data()
    
    if existing_df is not None and not force_refresh:
        days_old = (datetime.now() - latest_date).days
        
        if days_old <= 1:
            logger.info("數據已是最新，無需更新")
            return existing_df
        elif days_old <= 3:
            logger.info(f"數據較舊 ({days_old} 天)，建議更新")
        else:
            logger.info(f"數據過舊 ({days_old} 天)，正在更新...")
    
    # 獲取新數據
    try:
        price_fetcher = PriceFetcher()
        new_df = price_fetcher.fetch_all_stocks(save_to_file=True)
        
        if new_df.empty:
            logger.error("無法獲取新數據")
            return None
        
        new_df['date'] = pd.to_datetime(new_df['date'])
        latest_new = new_df['date'].max()
        
        logger.info(f"成功獲取新數據")
        logger.info(f"數據筆數: {len(new_df)}")
        logger.info(f"股票數量: {new_df['stock_code'].nunique()}")
        logger.info(f"最新日期: {latest_new.strftime('%Y-%m-%d')}")
        
        # 比較數據更新情況
        if existing_df is not None:
            old_latest = existing_df['date'].max()
            if latest_new > old_latest:
                logger.info("數據已更新")
            else:
                logger.info("數據日期相同")
        
        return new_df
        
    except Exception as e:
        logger.error(f"獲取數據時發生錯誤: {e}")
        return None


def show_data_summary(df):
    """顯示數據摘要"""
    if df is None or df.empty:
        logger.warning("沒有數據可顯示")
        return
    
    logger.info("=== 數據摘要 ===")
    logger.info(f"總筆數: {len(df)}")
    logger.info(f"股票數量: {df['stock_code'].nunique()}")
    logger.info(f"日期範圍: {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
    
    # 顯示最新日期的股票價格
    latest_date = df['date'].max()
    latest_data = df[df['date'] == latest_date].copy()
    latest_data = latest_data.sort_values('close', ascending=False)
    
    logger.info(f"\n最新交易日 ({latest_date.strftime('%Y-%m-%d')}) 股票價格:")
    for _, row in latest_data.iterrows():
        logger.info(f"   {row['stock_code']}: ${row['close']:.2f} (成交量: {row['volume']:,})")


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='股票數據抓取工具')
    parser.add_argument('--force', action='store_true', help='強制重新獲取數據')
    parser.add_argument('--check', action='store_true', help='只檢查數據，不獲取新數據')
    args = parser.parse_args()
    
    if args.check:
        # 只檢查現有數據
        df, latest_date = check_existing_data()
        if df is not None:
            show_data_summary(df)
    else:
        # 獲取數據
        df = fetch_new_data(force_refresh=args.force)
        if df is not None:
            show_data_summary(df)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序執行失敗: {e}")
        import traceback
        traceback.print_exc()
