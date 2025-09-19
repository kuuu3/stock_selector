"""
新聞爬蟲模組
爬取鉅亨網、工商時報等財經新聞
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path
import re
import jieba

from ..config import (
    RAW_NEWS_FILE,
    DATA_COLLECTION_CONFIG,
    NEWS_PROCESSING_CONFIG,
    API_CONFIG
)

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsScraper:
    """新聞爬蟲類別"""
    
    def __init__(self):
        self.news_sources = DATA_COLLECTION_CONFIG["NEWS_SOURCES"]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def scrape_cnyes_news(self, pages: int = 5) -> List[Dict]:
        """
        爬取鉅亨網新聞
        
        Args:
            pages: 爬取頁數
            
        Returns:
            新聞數據列表
        """
        news_list = []
        base_url = "https://news.cnyes.com"
        
        try:
            for page in range(1, pages + 1):
                url = f"{base_url}/news/cat/headline?page={page}"
                logger.info(f"正在爬取鉅亨網第 {page} 頁...")
                
                response = self.session.get(url, timeout=API_CONFIG["NEWS_API"]["TIMEOUT"])
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 查找新聞項目
                news_items = soup.find_all('div', class_='_1h45')
                
                for item in news_items:
                    try:
                        # 提取標題和連結
                        title_link = item.find('a')
                        if not title_link:
                            continue
                            
                        title = title_link.get_text(strip=True)
                        link = title_link.get('href')
                        
                        if not title or not link:
                            continue
                            
                        # 補全連結
                        if link.startswith('/'):
                            link = base_url + link
                        
                        # 提取時間
                        time_element = item.find('time')
                        publish_time = None
                        if time_element:
                            time_text = time_element.get_text(strip=True)
                            publish_time = self._parse_time(time_text)
                        
                        news_list.append({
                            'title': title,
                            'link': link,
                            'source': '鉅亨網',
                            'publish_time': publish_time,
                            'scraped_time': datetime.now()
                        })
                        
                    except Exception as e:
                        logger.warning(f"解析新聞項目時發生錯誤: {e}")
                        continue
                
                # 避免請求過於頻繁
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"爬取鉅亨網新聞時發生錯誤: {e}")
        
        logger.info(f"鉅亨網共爬取到 {len(news_list)} 則新聞")
        return news_list
    
    def scrape_ctee_news(self, pages: int = 5) -> List[Dict]:
        """
        爬取工商時報新聞
        
        Args:
            pages: 爬取頁數
            
        Returns:
            新聞數據列表
        """
        news_list = []
        base_url = "https://ctee.com.tw"
        
        try:
            for page in range(1, pages + 1):
                url = f"{base_url}/news/page/{page}/"
                logger.info(f"正在爬取工商時報第 {page} 頁...")
                
                response = self.session.get(url, timeout=API_CONFIG["NEWS_API"]["TIMEOUT"])
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 查找新聞項目
                news_items = soup.find_all('article', class_='post')
                
                for item in news_items:
                    try:
                        # 提取標題和連結
                        title_link = item.find('h2', class_='post-title').find('a')
                        if not title_link:
                            continue
                            
                        title = title_link.get_text(strip=True)
                        link = title_link.get('href')
                        
                        if not title or not link:
                            continue
                        
                        # 補全連結
                        if link.startswith('/'):
                            link = base_url + link
                        
                        # 提取時間
                        time_element = item.find('time', class_='published')
                        publish_time = None
                        if time_element:
                            time_text = time_element.get_text(strip=True)
                            publish_time = self._parse_time(time_text)
                        
                        news_list.append({
                            'title': title,
                            'link': link,
                            'source': '工商時報',
                            'publish_time': publish_time,
                            'scraped_time': datetime.now()
                        })
                        
                    except Exception as e:
                        logger.warning(f"解析新聞項目時發生錯誤: {e}")
                        continue
                
                # 避免請求過於頻繁
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"爬取工商時報新聞時發生錯誤: {e}")
        
        logger.info(f"工商時報共爬取到 {len(news_list)} 則新聞")
        return news_list
    
    def scrape_all_news(self, pages_per_source: int = 5) -> pd.DataFrame:
        """
        爬取所有新聞來源
        
        Args:
            pages_per_source: 每個來源爬取的頁數
            
        Returns:
            合併後的新聞數據 DataFrame
        """
        all_news = []
        
        # 爬取鉅亨網
        if "https://news.cnyes.com/" in self.news_sources:
            cnyes_news = self.scrape_cnyes_news(pages_per_source)
            all_news.extend(cnyes_news)
        
        # 爬取工商時報
        if "https://ctee.com.tw/" in self.news_sources:
            ctee_news = self.scrape_ctee_news(pages_per_source)
            all_news.extend(ctee_news)
        
        if not all_news:
            logger.warning("沒有爬取到任何新聞")
            return pd.DataFrame()
        
        # 轉換為 DataFrame
        df = pd.DataFrame(all_news)
        
        # 去重（基於標題）
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # 按時間排序
        df = df.sort_values('scraped_time', ascending=False).reset_index(drop=True)
        
        logger.info(f"總共爬取到 {len(df)} 則不重複新聞")
        
        # 保存到檔案
        self.save_news(df, RAW_NEWS_FILE)
        
        return df
    
    def get_news_content(self, url: str) -> str:
        """
        獲取新聞內容
        
        Args:
            url: 新聞連結
            
        Returns:
            新聞內容文本
        """
        try:
            response = self.session.get(url, timeout=API_CONFIG["NEWS_API"]["TIMEOUT"])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 根據不同網站提取內容
            if 'cnyes.com' in url:
                content_div = soup.find('div', class_='_1h45')
            elif 'ctee.com.tw' in url:
                content_div = soup.find('div', class_='post-content')
            else:
                # 通用內容提取
                content_div = soup.find('div', class_='content') or soup.find('article')
            
            if content_div:
                # 移除腳本和樣式標籤
                for script in content_div(["script", "style"]):
                    script.decompose()
                
                content = content_div.get_text(separator=' ', strip=True)
                # 清理多餘的空白
                content = re.sub(r'\s+', ' ', content)
                
                return content[:NEWS_PROCESSING_CONFIG["MAX_NEWS_LENGTH"]]
            
        except Exception as e:
            logger.warning(f"獲取新聞內容時發生錯誤 {url}: {e}")
        
        return ""
    
    def extract_stock_mentions(self, text: str) -> List[str]:
        """
        從文本中提取股票代碼
        
        Args:
            text: 文本內容
            
        Returns:
            提到的股票代碼列表
        """
        # 股票代碼模式（4位數字）
        stock_pattern = r'\b\d{4}\b'
        matches = re.findall(stock_pattern, text)
        
        # 過濾出可能的股票代碼
        stock_codes = []
        for match in matches:
            # 簡單的股票代碼驗證（台股代碼通常在1000-9999之間）
            if 1000 <= int(match) <= 9999:
                stock_codes.append(match)
        
        return list(set(stock_codes))  # 去重
    
    def _parse_time(self, time_text: str) -> Optional[datetime]:
        """
        解析時間文本
        
        Args:
            time_text: 時間文本
            
        Returns:
            解析後的 datetime 對象
        """
        try:
            # 處理不同的時間格式
            if '小時前' in time_text:
                hours = int(re.findall(r'(\d+)', time_text)[0])
                return datetime.now() - timedelta(hours=hours)
            elif '分鐘前' in time_text:
                minutes = int(re.findall(r'(\d+)', time_text)[0])
                return datetime.now() - timedelta(minutes=minutes)
            elif '天前' in time_text:
                days = int(re.findall(r'(\d+)', time_text)[0])
                return datetime.now() - timedelta(days=days)
            else:
                # 嘗試解析具體日期時間
                formats = [
                    '%Y-%m-%d %H:%M',
                    '%Y/%m/%d %H:%M',
                    '%Y-%m-%d',
                    '%Y/%m/%d'
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(time_text, fmt)
                    except ValueError:
                        continue
                        
        except Exception as e:
            logger.warning(f"解析時間失敗 {time_text}: {e}")
        
        return None
    
    def save_news(self, df: pd.DataFrame, file_path: Path):
        """保存新聞數據到 CSV 檔案"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        logger.info(f"新聞數據已保存到 {file_path}")
    
    def load_news(self, file_path: Path) -> pd.DataFrame:
        """從 CSV 檔案載入新聞數據"""
        if file_path.exists():
            return pd.read_csv(file_path, encoding='utf-8-sig')
        else:
            logger.warning(f"檔案 {file_path} 不存在")
            return pd.DataFrame()


def main():
    """主函數 - 用於測試新聞爬蟲功能"""
    scraper = NewsScraper()
    
    # 測試爬取新聞
    logger.info("開始測試新聞爬蟲...")
    
    df = scraper.scrape_all_news(pages_per_source=2)
    
    if not df.empty:
        logger.info(f"成功爬取 {len(df)} 則新聞")
        logger.info(f"新聞來源分布: {df['source'].value_counts().to_dict()}")
        
        # 顯示前幾則新聞標題
        logger.info("前5則新聞標題:")
        for i, title in enumerate(df['title'].head()):
            logger.info(f"{i+1}. {title}")
        
        # 測試股票代碼提取
        sample_title = df['title'].iloc[0]
        stock_mentions = scraper.extract_stock_mentions(sample_title)
        if stock_mentions:
            logger.info(f"在標題中發現股票代碼: {stock_mentions}")


if __name__ == "__main__":
    main()

