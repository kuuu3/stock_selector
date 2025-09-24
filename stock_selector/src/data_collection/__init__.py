"""
數據收集模組
包含股價數據和新聞數據的收集功能
"""

# 延遲導入以避免依賴問題
try:
    from .price_fetcher import PriceFetcher
except ImportError:
    PriceFetcher = None

try:
    from .news_scraper import NewsScraper
except ImportError:
    NewsScraper = None

try:
    from .enhanced_news_scraper import EnhancedNewsScraper
except ImportError:
    EnhancedNewsScraper = None

__all__ = ['PriceFetcher', 'NewsScraper', 'EnhancedNewsScraper']




