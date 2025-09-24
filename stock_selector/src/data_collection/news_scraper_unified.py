#!/usr/bin/env python3
"""
統一新聞爬蟲模組 - 專注於最穩定的API和RSS源
整合了所有新聞爬蟲功能，提供最優化的爬取體驗
"""

import requests
import time
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import re
from pathlib import Path

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsScraper:
    """統一新聞爬蟲 - 專注於穩定源"""
    
    def __init__(self):
        self.session = self._setup_session()
        
    def _setup_session(self):
        """設置會話"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-TW,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
        })
        return session
    
    def scrape_cnyes_api(self, limit: int = 20) -> List[Dict]:
        """使用鉅亨網API爬取新聞（最穩定）"""
        logger.info(f"使用鉅亨網API爬取 {limit} 則新聞...")
        
        api_url = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"
        params = {
            'page': 1,
            'limit': limit
        }
        
        headers = {
            'Referer': 'https://www.cnyes.com/',
            'Accept': 'application/json, text/plain, */*',
        }
        
        try:
            response = self.session.get(api_url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'items' in data and data['items']:
                    news_items = []
                    items_data = data['items']
                    
                    # 檢查items結構
                    if isinstance(items_data, dict) and 'data' in items_data:
                        news_list = items_data['data']
                    elif isinstance(items_data, list):
                        news_list = items_data
                    else:
                        logger.warning(f"未知的items結構: {type(items_data)}")
                        news_list = []
                    
                    for item in news_list:
                        if isinstance(item, dict):
                            # 構建新聞URL
                            news_id = item.get('newsId', '')
                            url = f"https://www.cnyes.com/news/id/{news_id}" if news_id else ''
                            
                            # 從content字段提取純文本內容
                            content = item.get('content', '')
                            if content:
                                # 移除HTML標籤
                                soup = BeautifulSoup(content, 'html.parser')
                                content = soup.get_text().strip()
                                # 限制長度
                                content = content[:1500] if len(content) > 1500 else content
                            
                            news_item = {
                                'title': item.get('title', ''),
                                'url': url,
                                'published_time': self._parse_cnyes_time(item.get('publishAt', '')),
                                'source': '鉅亨網',
                                'content': content
                            }
                            news_items.append(news_item)
                    
                    logger.info(f"鉅亨網API成功獲取 {len(news_items)} 則新聞")
                    return news_items
                else:
                    logger.warning("鉅亨網API返回空數據")
                    return []
            else:
                logger.warning(f"鉅亨網API請求失敗: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"鉅亨網API爬取失敗: {e}")
            return []
    
    def scrape_yahoo_rss(self, limit: int = 20) -> List[Dict]:
        """使用Yahoo財經RSS爬取新聞（穩定）"""
        logger.info(f"使用Yahoo財經RSS爬取 {limit} 則新聞...")
        
        rss_url = "https://tw.news.yahoo.com/rss/finance"
        headers = {
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
        }
        
        try:
            response = self.session.get(rss_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # 解析RSS XML
                root = ET.fromstring(response.content)
                
                # 找到所有item
                items = root.findall('.//item')
                news_items = []
                
                for item in items[:limit]:  # 限制數量
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    pub_date_elem = item.find('pubDate')
                    description_elem = item.find('description')
                    
                    if title_elem is not None and link_elem is not None:
                        # 清理標題中的CDATA
                        title = title_elem.text or ''
                        if title.startswith('<![CDATA[') and title.endswith(']]>'):
                            title = title[9:-3]
                        
                        # 從description提取內容
                        content = ''
                        if description_elem is not None and description_elem.text:
                            desc_text = description_elem.text
                            if desc_text.startswith('<![CDATA[') and desc_text.endswith(']]>'):
                                desc_text = desc_text[9:-3]
                            # 移除HTML標籤
                            soup = BeautifulSoup(desc_text, 'html.parser')
                            content = soup.get_text().strip()[:1000]  # 限制長度
                        
                        news_item = {
                            'title': title.strip(),
                            'url': link_elem.text or '',
                            'published_time': self._parse_rss_time(pub_date_elem.text if pub_date_elem is not None else ''),
                            'source': 'Yahoo財經',
                            'content': content
                        }
                        news_items.append(news_item)
                
                logger.info(f"Yahoo RSS成功獲取 {len(news_items)} 則新聞")
                return news_items
            else:
                logger.warning(f"Yahoo RSS請求失敗: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Yahoo RSS爬取失敗: {e}")
            return []
    
    def scrape_cna_fast(self, limit: int = 10) -> List[Dict]:
        """快速爬取中央社新聞（僅標題，不獲取內容）"""
        logger.info(f"快速爬取中央社 {limit} 則新聞...")
        
        base_url = "https://www.cna.com.tw/list/aipl.aspx"
        news_items = []
        
        try:
            response = self.session.get(base_url, timeout=8)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 使用更寬鬆的選擇器
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link.get('href', '')
                    text = link.get_text().strip()
                    
                    # 過濾新聞鏈接
                    if ('/news/' in href or '/aipl/' in href) and len(text) > 10 and len(text) < 200:
                        # 避免重複
                        if not any(item['url'] == href for item in news_items):
                            news_item = {
                                'title': text,
                                'url': href if href.startswith('http') else f"https://www.cna.com.tw{href}",
                                'published_time': datetime.now(),  # 使用當前時間
                                'source': '中央社',
                                'content': ''  # 不獲取內容以節省時間
                            }
                            news_items.append(news_item)
                            
                            if len(news_items) >= limit:
                                break
                
                logger.info(f"中央社快速爬取獲取 {len(news_items)} 則新聞")
                return news_items
            else:
                logger.warning(f"中央社請求失敗: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"中央社爬取失敗: {e}")
            return []
    
    def _parse_cnyes_time(self, time_str: str) -> datetime:
        """解析鉅亨網時間格式"""
        try:
            if time_str:
                if time_str.isdigit():
                    return datetime.fromtimestamp(int(time_str))
                else:
                    return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except:
            pass
        return datetime.now()
    
    def _parse_rss_time(self, time_str: str) -> datetime:
        """解析RSS時間格式"""
        try:
            if time_str:
                from email.utils import parsedate_to_datetime
                return parsedate_to_datetime(time_str)
        except:
            pass
        return datetime.now()
    
    def scrape_all_news(self, cnyes_limit: int = 15, yahoo_limit: int = 15, cna_limit: int = 10) -> List[Dict]:
        """爬取所有新聞源"""
        logger.info("開始爬取所有新聞源...")
        
        all_news = []
        
        # 1. 鉅亨網API（最穩定，有完整內容）
        cnyes_news = self.scrape_cnyes_api(cnyes_limit)
        all_news.extend(cnyes_news)
        
        # 2. Yahoo RSS（穩定，有部分內容）
        yahoo_news = self.scrape_yahoo_rss(yahoo_limit)
        all_news.extend(yahoo_news)
        
        # 3. 中央社（快速模式，僅標題）
        cna_news = self.scrape_cna_fast(cna_limit)
        all_news.extend(cna_news)
        
        # 去重（基於標題）
        unique_news = []
        seen_titles = set()
        
        for news in all_news:
            title = news['title']
            if title not in seen_titles and title:
                seen_titles.add(title)
                unique_news.append(news)
        
        logger.info(f"總共獲取 {len(unique_news)} 則唯一新聞")
        
        # 按來源統計
        sources = {}
        for news in unique_news:
            source = news['source']
            sources[source] = sources.get(source, 0) + 1
        
        logger.info("按來源統計:")
        for source, count in sources.items():
            logger.info(f"  {source}: {count} 則")
        
        return unique_news
    
    def save_to_csv(self, news_list: List[Dict], output_path: str):
        """保存新聞到CSV"""
        if not news_list:
            logger.warning("沒有新聞數據可保存")
            return
        
        df = pd.DataFrame(news_list)
        
        # 添加分析時間
        df['scraped_time'] = datetime.now()
        
        # 保存到CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"新聞數據已保存到: {output_file}")

def main():
    """測試統一新聞爬蟲"""
    scraper = NewsScraper()
    
    # 爬取所有新聞
    news_list = scraper.scrape_all_news(
        cnyes_limit=10,
        yahoo_limit=10, 
        cna_limit=5
    )
    
    # 顯示結果
    print(f"\n=== 統一新聞爬蟲結果 ===")
    print(f"總共獲取 {len(news_list)} 則新聞")
    
    # 顯示前幾則新聞
    print("\n前5則新聞:")
    for i, news in enumerate(news_list[:5], 1):
        print(f"{i}. [{news['source']}] {news['title'][:60]}...")
        print(f"   時間: {news['published_time']}")
        print(f"   內容長度: {len(news['content'])} 字符")
        print()
    
    # 保存到CSV
    scraper.save_to_csv(news_list, "data/raw/unified_news.csv")

if __name__ == "__main__":
    main()
