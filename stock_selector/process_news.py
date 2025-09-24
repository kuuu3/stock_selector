#!/usr/bin/env python3
"""
新聞預處理腳本
專門處理新聞數據的情感分析和預處理
"""

import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.preprocessing.sentiment_analyzer import SentimentAnalyzer
from src.config import get_data_file_path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_stock_codes(text):
    """從文本中提取股票代碼"""
    import re
    # 匹配4位數字的股票代碼
    pattern = r'\b(\d{4})\b'
    matches = re.findall(pattern, text)
    return list(set(matches))


def process_news_sentiment(input_file, output_file=None):
    """
    處理新聞情感分析
    
    Args:
        input_file: 輸入新聞文件路徑
        output_file: 輸出文件路徑（可選）
    """
    logger.info("=== 開始新聞情感分析 ===")
    
    # 載入新聞數據
    try:
        news_df = pd.read_csv(input_file)
        logger.info(f"載入新聞數據: {len(news_df)} 筆")
    except Exception as e:
        logger.error(f"載入新聞數據失敗: {e}")
        return False
    
    # 初始化情感分析器
    try:
        analyzer = SentimentAnalyzer()
        logger.info("情感分析器初始化完成")
    except Exception as e:
        logger.error(f"情感分析器初始化失敗: {e}")
        return False
    
    # 進行情感分析
    logger.info("開始批量情感分析...")
    results = []
    
    for idx, row in news_df.iterrows():
        title = row.get('title', '')
        content = row.get('content', '')
        text = f'{title} {content}'
        
        try:
            # 分析情感
            sentiment_result = analyzer.analyze_sentiment(text, method='hybrid')
            
            # 提取股票代碼
            stock_codes = extract_stock_codes(text)
            
            # 添加到結果
            result = {
                'title': title,
                'content': content,
                'url': row.get('url', ''),
                'date': row.get('date', ''),
                'source': row.get('source', ''),
                'scraped_time': row.get('scraped_time', ''),
                'sentiment': sentiment_result.get('sentiment', 'neutral'),
                'sentiment_score': sentiment_result.get('sentiment_score', 0.0),
                'confidence': sentiment_result.get('confidence', 0.5),
                'positive_words': sentiment_result.get('positive_words', []),
                'negative_words': sentiment_result.get('negative_words', []),
                'stock_codes': stock_codes,
                'analyzed_time': datetime.now()
            }
            results.append(result)
            
            logger.info(f"處理第 {idx+1} 則新聞: {title[:50]}... -> {sentiment_result.get('sentiment', 'neutral')}")
            
        except Exception as e:
            logger.error(f"處理第 {idx+1} 則新聞時發生錯誤: {e}")
            # 添加預設結果
            results.append({
                'title': title,
                'content': content,
                'url': row.get('url', ''),
                'date': row.get('date', ''),
                'source': row.get('source', ''),
                'scraped_time': row.get('scraped_time', ''),
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.5,
                'positive_words': [],
                'negative_words': [],
                'stock_codes': [],
                'analyzed_time': datetime.now()
            })
    
    # 轉換為DataFrame
    result_df = pd.DataFrame(results)
    
    # 設置輸出文件路徑
    if output_file is None:
        output_file = get_data_file_path("processed/news_with_sentiment.csv")
    
    # 保存結果
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        logger.info(f"新聞情感分析結果已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存結果失敗: {e}")
        return False
    
    # 顯示統計信息
    sentiment_counts = result_df['sentiment'].value_counts()
    logger.info("=== 情感分析統計 ===")
    logger.info(f"總新聞數: {len(result_df)}")
    logger.info(f"情感分布: {sentiment_counts.to_dict()}")
    
    # 顯示股票代碼統計
    all_stock_codes = []
    for codes in result_df['stock_codes']:
        all_stock_codes.extend(codes)
    
    if all_stock_codes:
        stock_code_counts = pd.Series(all_stock_codes).value_counts()
        logger.info(f"提及的股票代碼: {stock_code_counts.head(10).to_dict()}")
    else:
        logger.info("沒有檢測到股票代碼")
    
    return True


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='新聞預處理腳本')
    parser.add_argument('--input', '-i', 
                       default='data/raw/news.csv',
                       help='輸入新聞文件路徑 (預設: data/raw/news.csv)')
    parser.add_argument('--output', '-o',
                       help='輸出文件路徑 (預設: data/processed/news_with_sentiment.csv)')
    parser.add_argument('--force', action='store_true',
                       help='強制重新處理，覆蓋現有文件')
    
    args = parser.parse_args()
    
    # 檢查輸入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"輸入文件不存在: {input_path}")
        return
    
    # 檢查輸出文件
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(get_data_file_path("processed/news_with_sentiment.csv"))
    
    if output_path.exists() and not args.force:
        logger.warning(f"輸出文件已存在: {output_path}")
        logger.warning("使用 --force 參數強制覆蓋")
        return
    
    # 處理新聞
    success = process_news_sentiment(args.input, args.output)
    
    if success:
        logger.info("=== 新聞預處理完成 ===")
    else:
        logger.error("=== 新聞預處理失敗 ===")


if __name__ == "__main__":
    main()
