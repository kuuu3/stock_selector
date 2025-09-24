"""
統一數據準備腳本
整合數據抓取和新聞處理功能
"""

import sys
from pathlib import Path
import logging
import pandas as pd
from datetime import datetime, timedelta
import argparse
import time

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_collection import PriceFetcher
from src.data_collection.api_fetcher import APIFetcher
from src.data_collection.news_scraper import NewsScraper
from src.preprocessing.sentiment_analyzer import SentimentAnalyzer
from src.config import RAW_PRICES_FILE, RAW_NEWS_FILE, get_data_file_path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_stock_codes(text: str) -> list:
    """從文本中提取股票代碼"""
    import re
    # 股票代碼模式（4位數字）
    stock_pattern = r'\\b(\\d{4})\\b'
    matches = re.findall(stock_pattern, text)
    
    # 過濾出可能的股票代碼（台股代碼通常在1000-9999之間）
    stock_codes = []
    for match in matches:
        if 1000 <= int(match) <= 9999:
            stock_codes.append(match)
    
    return list(set(stock_codes))  # 去重


def analyze_stock_sentiments(sentiment_df: pd.DataFrame) -> dict:
    """分析各股票的情感"""
    stock_sentiments = {}
    
    for _, row in sentiment_df.iterrows():
        stock_codes_str = row.get('stock_codes', '')
        if not stock_codes_str:
            continue
        
        stock_codes = stock_codes_str.split('|') if stock_codes_str else []
        
        for stock_code in stock_codes:
            if stock_code not in stock_sentiments:
                stock_sentiments[stock_code] = {
                    'news_count': 0,
                    'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'total_score': 0.0,
                    'total_confidence': 0.0
                }
            
            # 更新統計
            sentiment_info = stock_sentiments[stock_code]
            sentiment_info['news_count'] += 1
            sentiment_info['sentiment_distribution'][row['sentiment']] += 1
            sentiment_info['total_score'] += row['sentiment_score']
            sentiment_info['total_confidence'] += row['confidence']
    
    # 計算平均值和主要情感
    for stock_code, sentiment_info in stock_sentiments.items():
        news_count = sentiment_info['news_count']
        sentiment_info['avg_score'] = sentiment_info['total_score'] / news_count
        sentiment_info['avg_confidence'] = sentiment_info['total_confidence'] / news_count
        
        # 確定主要情感
        sentiment_dist = sentiment_info['sentiment_distribution']
        dominant_sentiment = max(sentiment_dist, key=sentiment_dist.get)
        sentiment_info['dominant_sentiment'] = dominant_sentiment
    
    return stock_sentiments


def check_price_data():
    """檢查現有股價數據"""
    price_file = RAW_PRICES_FILE
    if not price_file.exists():
        logger.info("沒有找到現有股價數據文件")
        return None, None
    try:
        df = pd.read_csv(price_file)
        df['date'] = pd.to_datetime(df['date'])
        df['stock_code'] = df['stock_code'].astype(str)
        latest_date = df['date'].max()
        data_count = len(df)
        stock_count = df['stock_code'].nunique()
        days_old = (datetime.now() - latest_date).days
        logger.info(f"現有股價數據: {data_count} 筆記錄, {stock_count} 支股票")
        logger.info(f"最新數據日期: {latest_date.strftime('%Y-%m-%d')}")
        logger.info(f"數據年齡: {days_old} 天")
        return df, latest_date
    except Exception as e:
        logger.error(f"讀取現有股價數據時發生錯誤: {e}")
        return None, None


def fetch_price_data(force_refresh=False, use_api=False):
    """獲取股價數據"""
    logger.info("=== 開始獲取股票數據 ===")
    existing_df, latest_date = check_price_data()
    
    # 檢查是否有缺失的股票（無論數據年齡）
    if existing_df is not None and not force_refresh:
        from src.config import DATA_COLLECTION_CONFIG
        existing_stocks = set(existing_df['stock_code'].astype(str).unique())
        all_stocks = set(DATA_COLLECTION_CONFIG["STOCK_LIST"])
        missing_stocks = all_stocks - existing_stocks
        
        if missing_stocks:
            logger.info(f"發現 {len(missing_stocks)} 支股票缺失數據: {sorted(missing_stocks)}")
        else:
            days_old = (datetime.now() - latest_date).days
            if days_old <= 1:
                logger.info("股價數據已是最新，無需更新")
                return existing_df
            elif days_old <= 3:
                logger.info(f"股價數據較舊 ({days_old} 天)，建議更新")
            else:
                logger.info(f"股價數據過舊 ({days_old} 天)，正在更新...")
    try:
        # 首先嘗試使用 twstock，如果失敗則使用 API
        price_fetcher = PriceFetcher()
        api_fetcher = APIFetcher()
        
        if force_refresh:
            logger.info("強制刷新模式：重新獲取所有股價數據")
            
            if use_api:
                logger.info("使用 API 獲取數據...")
                new_df = api_fetcher.fetch_all_stocks(force_refresh=True)
                if not new_df.empty:
                    logger.info("API 獲取數據成功")
                else:
                    logger.error("API 無法獲取數據")
            else:
                # 先嘗試 twstock
                logger.info("嘗試使用 twstock 獲取數據...")
                new_df = price_fetcher.fetch_all_stocks(save_to_file=False)
                
                if new_df.empty or len(new_df['stock_code'].unique()) < len(price_fetcher.stock_list) * 0.8:
                    logger.warning("twstock 獲取數據不完整，嘗試使用 API...")
                    api_df = api_fetcher.fetch_all_stocks(force_refresh=True)
                    if not api_df.empty:
                        new_df = api_df
                        logger.info("使用 API 成功獲取數據")
                    else:
                        logger.error("API 也無法獲取數據")
                else:
                    logger.info("twstock 獲取數據成功")
        else:
            logger.info("增量更新模式：檢查缺失的股價數據")
            
            if use_api:
                # 使用 API 進行增量更新
                new_df = api_fetcher.fetch_missing_stocks(existing_df)
                if new_df.empty:
                    logger.info("沒有新股價數據需要更新")
                    return existing_df
                if existing_df is None or existing_df.empty:
                    logger.info("Cold-start：使用完整的新股價數據")
                else:
                    new_df = pd.concat([existing_df, new_df], ignore_index=True)
                    new_df = new_df.sort_values(['stock_code', 'date']).reset_index(drop=True)
            else:
                # 使用 twstock 進行增量更新
                new_data = price_fetcher.fetch_incremental_data(existing_df)
                if new_data.empty:
                    logger.info("沒有新股價數據需要更新")
                    return existing_df
                if existing_df is None or existing_df.empty:
                    logger.info("Cold-start：使用完整的新股價數據")
                    new_df = new_data
                else:
                    new_df = pd.concat([existing_df, new_data], ignore_index=True)
                    new_df = new_df.sort_values(['stock_code', 'date']).reset_index(drop=True)
            
        # 保存數據
        if not new_df.empty:
            new_df.to_csv(RAW_PRICES_FILE, index=False)
            logger.info(f"股價數據已保存到: {RAW_PRICES_FILE}")
        if new_df.empty:
            logger.error("無法獲取新股價數據")
            return None
        new_df['date'] = pd.to_datetime(new_df['date'])
        latest_new = new_df['date'].max()
        logger.info(f"成功獲取股價數據")
        logger.info(f"數據筆數: {len(new_df)}")
        logger.info(f"股票數量: {new_df['stock_code'].nunique()}")
        logger.info(f"最新日期: {latest_new.strftime('%Y-%m-%d')}")
        return new_df
    except Exception as e:
        logger.error(f"獲取股價數據時發生錯誤: {e}")
        return None


def check_news_data():
    """檢查現有新聞數據"""
    news_file = RAW_NEWS_FILE
    if not news_file.exists():
        logger.info("沒有找到現有新聞數據文件")
        return None, None
    try:
        df = pd.read_csv(news_file, encoding='utf-8-sig')
        df['scraped_time'] = pd.to_datetime(df['scraped_time'])
        latest_time = df['scraped_time'].max()
        data_count = len(df)
        days_old = (datetime.now() - latest_time).days
        logger.info(f"現有新聞數據: {data_count} 則")
        logger.info(f"最新抓取時間: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"數據年齡: {days_old} 天")
        return df, latest_time
    except Exception as e:
        logger.warning(f"讀取現有新聞數據時發生錯誤: {e}")
        return None, None


def fetch_news_data(force_refresh=False, pages_per_source=3):
    """獲取新聞數據"""
    logger.info("=== 開始獲取新聞數據 ===")
    existing_df, latest_time = check_news_data()
    if existing_df is not None and not force_refresh:
        days_old = (datetime.now() - latest_time).days
        if days_old <= 0:
            logger.info("新聞數據已是最新，無需更新")
            return existing_df
        elif days_old <= 1:
            logger.info(f"新聞數據較舊 ({days_old} 天)，建議更新")
        else:
            logger.info(f"新聞數據過舊 ({days_old} 天)，正在更新...")
    
    try:
        # 使用統一新聞爬蟲
        logger.info("使用統一新聞爬蟲...")
        scraper = NewsScraper()
        news_list = scraper.scrape_all_news(
            cnyes_limit=15,
            yahoo_limit=15,
            cna_limit=10
        )
        
        if news_list:
            df = pd.DataFrame(news_list)
            df['scraped_time'] = datetime.now()
            logger.info(f"統一新聞爬蟲成功獲取 {len(df)} 則新聞")
            return df
        else:
            logger.warning("統一新聞爬蟲未獲取到任何新聞")
            return None
    except Exception as e:
        logger.error(f"獲取新聞數據時發生錯誤: {e}")
        return None


def analyze_news_sentiment(news_df):
    """分析新聞情感"""
    logger.info("=== 開始新聞情感分析 ===")
    
    try:
        analyzer = SentimentAnalyzer()
        
        # 批量分析情感
        logger.info("開始批量情感分析...")
        results = []
        
        for idx, row in news_df.iterrows():
            title = row.get('title', '')
            
            # 分析情感
            sentiment_result = analyzer.analyze_sentiment(title, method='hybrid')
            
            # 提取股票代碼
            stock_codes = extract_stock_codes(title)
            
            # 添加到結果
            results.append({
                'index': idx,
                'title': title,
                'source': row.get('source', ''),
                'sentiment': sentiment_result['sentiment'],
                'sentiment_score': sentiment_result['score'],
                'confidence': sentiment_result['confidence'],
                'positive_words': '|'.join(sentiment_result['positive_words']),
                'negative_words': '|'.join(sentiment_result['negative_words']),
                'stock_codes': '|'.join(stock_codes),
                'analyzed_time': datetime.now()
            })
            
            if (idx + 1) % 100 == 0:
                logger.info(f"已處理 {idx + 1}/{len(news_df)} 則新聞")
        
        # 轉換為 DataFrame
        sentiment_df = pd.DataFrame(results)
        
        # 統計結果
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        logger.info(f"情感分析完成:")
        logger.info(f"  正面: {sentiment_counts.get('positive', 0)} 則")
        logger.info(f"  負面: {sentiment_counts.get('negative', 0)} 則")
        logger.info(f"  中性: {sentiment_counts.get('neutral', 0)} 則")
        logger.info(f"  平均信心度: {sentiment_df['confidence'].mean():.3f}")
        
        # 保存結果
        output_file = get_data_file_path("processed/news_with_sentiment.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        sentiment_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"情感分析結果已保存到: {output_file}")
        
        return sentiment_df
        
    except Exception as e:
        logger.error(f"情感分析時發生錯誤: {e}")
        return None


def analyze_stock_sentiment(sentiment_df):
    """分析各股票的情感"""
    logger.info("=== 分析各股票的情感 ===")
    
    try:
        stock_sentiments = analyze_stock_sentiments(sentiment_df)
        
        if stock_sentiments:
            # 顯示前10個股票的情感分析結果
            logger.info("前10個股票的情感分析:")
            for i, (stock_code, sentiment_info) in enumerate(stock_sentiments.items()):
                if i >= 10:
                    break
                logger.info(f"  {stock_code}: {sentiment_info['dominant_sentiment']} "
                           f"(新聞數: {sentiment_info['news_count']}, "
                           f"平均分數: {sentiment_info['avg_score']:.3f})")
            
            # 保存股票情感分析結果
            stock_sentiment_df = pd.DataFrame([
                {
                    'stock_code': stock_code,
                    'news_count': info['news_count'],
                    'positive_count': info['sentiment_distribution'].get('positive', 0),
                    'negative_count': info['sentiment_distribution'].get('negative', 0),
                    'neutral_count': info['sentiment_distribution'].get('neutral', 0),
                    'avg_sentiment_score': info['avg_score'],
                    'avg_confidence': info['avg_confidence'],
                    'dominant_sentiment': info['dominant_sentiment']
                }
                for stock_code, info in stock_sentiments.items()
            ])
            
            stock_output_file = get_data_file_path("processed/stock_sentiments.csv")
            stock_sentiment_df.to_csv(stock_output_file, index=False, encoding='utf-8-sig')
            logger.info(f"股票情感分析結果已保存到: {stock_output_file}")
        else:
            logger.info("沒有找到包含股票代碼的新聞")
        
    except Exception as e:
        logger.error(f"分析股票情感時發生錯誤: {e}")


def check_system_status():
    """檢查系統狀態和數據完整性"""
    logger.info("🔍 股票選擇系統狀態檢查")
    
    # 檢查數據文件
    logger.info("\n=== 檢查數據文件狀態 ===")
    
    data_files = {
        "股價數據": "data/raw/prices.csv",
        "新聞數據": "data/raw/news.csv", 
        "統一新聞": "data/raw/unified_news.csv",
        "新聞情感分析": "data/processed/news_with_sentiment.csv",
        "TPEX手動數據": "data/processed/tpex_manual_data.csv"
    }
    
    for name, path in data_files.items():
        file_path = Path(path)
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                logger.info(f"✅ {name}: {len(df)} 筆數據")
            except Exception as e:
                logger.warning(f"❌ {name}: 讀取錯誤 - {str(e)[:50]}")
        else:
            logger.warning(f"❌ {name}: 文件不存在")
    
    # 檢查模型文件
    logger.info("\n=== 檢查模型文件狀態 ===")
    
    models_dir = Path("outputs/models")
    model_files = [
        "feature_columns.pkl",
        "training_metadata.pkl", 
        "logistic_regression_model.pkl",
        "xgboost_classifier_model.pkl",
        "xgboost_regressor_model.pkl"
    ]
    
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            size = model_path.stat().st_size
            logger.info(f"✅ {model_file}: {size} 字節")
        else:
            logger.warning(f"❌ {model_file}: 文件不存在")
    
    # 檢查核心腳本
    logger.info("\n=== 檢查核心腳本文件 ===")
    
    scripts = [
        "prepare_data.py",
        "train.py", 
        "predict.py",
        "run_backtest.py",
        "process_news.py",
        "process_manual_tpex.py"
    ]
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            size = script_path.stat().st_size
            logger.info(f"✅ {script}: {size} 字節")
        else:
            logger.warning(f"❌ {script}: 文件不存在")
    
    logger.info("\n=== 系統狀態檢查完成 ===")


def prepare_all_data(force_refresh=False, pages_per_source=3, include_news=True, use_api=False):
    """準備所有數據"""
    logger.info("=== 開始準備所有數據 ===")
    
    # 步驟1: 獲取股價數據
    logger.info("步驟1: 獲取股價數據...")
    price_df = fetch_price_data(force_refresh, use_api)
    if price_df is None:
        logger.error("股價數據獲取失敗")
        return False
    
    # 步驟2: 獲取新聞數據（如果啟用）
    if include_news:
        logger.info("步驟2: 獲取新聞數據...")
        news_df = fetch_news_data(force_refresh, pages_per_source)
        if news_df is not None and not news_df.empty:
            # 步驟3: 新聞情感分析
            logger.info("步驟3: 進行新聞情感分析...")
            sentiment_df = analyze_news_sentiment(news_df)
            if sentiment_df is not None:
                # 步驟4: 股票情感分析
                logger.info("步驟4: 分析各股票的情感...")
                analyze_stock_sentiment(sentiment_df)
        else:
            logger.warning("新聞數據獲取失敗或為空")
    else:
        logger.info("跳過新聞數據處理")
    
    logger.info("=== 數據準備完成 ===")
    return True


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='統一數據準備腳本')
    parser.add_argument('--check', action='store_true', help='檢查系統狀態和數據完整性')
    parser.add_argument('--force', action='store_true', help='強制重新獲取所有數據')
    parser.add_argument('--prices-only', action='store_true', help='只獲取股價數據')
    parser.add_argument('--news-only', action='store_true', help='只獲取新聞數據')
    parser.add_argument('--no-news', action='store_true', help='跳過新聞數據處理')
    parser.add_argument('--use-api', action='store_true', help='強制使用API獲取股價數據')
    parser.add_argument('--pages', type=int, default=3, help='每個新聞來源抓取的頁數')
    
    args = parser.parse_args()
    
    logger.info("=== 統一數據準備開始 ===")
    
    if args.check:
        logger.info("=== 系統狀態檢查 ===")
        check_system_status()
    else:
        if args.prices_only:
            fetch_price_data(args.force, args.use_api)
        elif args.news_only:
            fetch_news_data(args.force, args.pages)
        else:
            # 預設獲取所有數據
            prepare_all_data(
                force_refresh=args.force,
                pages_per_source=args.pages,
                include_news=not args.no_news,
                use_api=args.use_api
            )
    
    logger.info("=== 統一數據準備結束 ===")


if __name__ == "__main__":
    main()
