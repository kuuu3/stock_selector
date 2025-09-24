"""
Stock Selector 專案配置檔案
包含所有全域設定、路徑和參數
"""

import os
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# 確保目錄存在
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 數據檔案路徑
RAW_NEWS_FILE = RAW_DATA_DIR / "news.csv"
RAW_PRICES_FILE = RAW_DATA_DIR / "prices.csv"
PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "features.npy"
PROCESSED_LABELS_FILE = PROCESSED_DATA_DIR / "labels.npy"

def get_data_file_path(filename: str) -> Path:
    """
    獲取數據文件的絕對路徑，支持從不同目錄運行腳本
    
    Args:
        filename: 相對於 data/ 目錄的文件名，如 "raw/prices.csv"
        
    Returns:
        數據文件的絕對路徑
    """
    return PROJECT_ROOT / "data" / filename

# 輸出檔案路徑
TOP20_OUTPUT_FILE = OUTPUTS_DIR / "top20.csv"
BACKTEST_REPORT_FILE = OUTPUTS_DIR / "backtest_report.csv"

# 技術指標參數（調整為適合小數據集）
TECHNICAL_INDICATORS = {
    "MA_SHORT": 3,           # 短期移動平均線（降低要求）
    "MA_LONG": 5,            # 長期移動平均線（降低要求）
    "RSI_PERIOD": 7,         # RSI週期（降低要求）
    "MACD_FAST": 5,          # MACD快線（降低要求）
    "MACD_SLOW": 10,         # MACD慢線（降低要求）
    "MACD_SIGNAL": 3,        # MACD訊號線（降低要求）
    "VOLATILITY_PERIOD": 5,  # 波動率計算週期（降低要求）
}

# 模型參數
MODEL_CONFIG = {
    "LOGISTIC_REGRESSION": {
        "random_state": 42,
        "max_iter": 1000
    },
    "XGBOOST_CLASSIFIER": {
        "n_estimators": 1000,  # 增加樹的數量
        "max_depth": 8,        # 增加深度
        "learning_rate": 0.03, # 降低學習率
        "random_state": 42,
        "subsample": 0.8,      # 添加子採樣
        "colsample_bytree": 0.8 # 添加特徵採樣
    },
    "XGBOOST_REGRESSOR": {
        "n_estimators": 1000,  # 增加樹的數量
        "max_depth": 8,        # 增加深度
        "learning_rate": 0.03, # 降低學習率
        "random_state": 42,
        "subsample": 0.8,      # 添加子採樣
        "colsample_bytree": 0.8 # 添加特徵採樣
    }
}

# 選股參數
SELECTION_CONFIG = {
    "TOP_N_STOCKS": 20,           # 選股數量
    "MIN_VOLUME": 1000000,        # 最低成交量（股）
    "MIN_MARKET_CAP": 1000000000, # 最低市值（台幣）
    "MIN_PRICE": 10,              # 最低股價（台幣）
}

# 回測參數
BACKTEST_CONFIG = {
    "INITIAL_CAPITAL": 1000000,   # 初始資金（台幣）
    "COMMISSION_RATE": 0.001425,  # 手續費率
    "TAX_RATE": 0.003,            # 證交稅率
    "REBALANCE_FREQUENCY": "weekly", # 再平衡頻率
}

# 數據收集參數
DATA_COLLECTION_CONFIG = {
    "NEWS_SOURCES": [
        "https://tw.news.yahoo.com/finance",  # Yahoo 財經
        "https://ec.ltn.com.tw/",             # 自由時報財經
        "https://news.cnyes.com/",            # 鉅亨網
        "https://ctee.com.tw/",               # 工商時報
    ],
    "STOCK_LIST": [                     # 主要股票代碼列表（已移除無法獲取的股票）
        "2330", "2317", "2454", "6505", "2308",  # 台積電、鴻海、聯發科等
        "3535", "5443", "2363", "2344", "2481",
        "3260", "2408", "3324", "6449", "5469",
        "5284", "3704", "1560", "1316",          # 其他股票（移除5475）
        "8039", "3563", "2630",                  # 其他股票（移除6761）
        "3019", "3311", "8021", "2476",          # 其他股票（移除8027）
        "4976", "2231", "8033", "2429",          # 其他股票（移除5498）
        "4414", "6235", "1504",                  # 其他股票
        "1445",                                  # 其他股票（移除8111, 3323, 6143）
        "3059", "2614",                          # 其他股票（移除2641, 3624, 6510）
        "2449",                                  # 其他股票
    ],
    "LOOKBACK_DAYS": 500,              # 回看天數（約兩年交易日）
}

# 新聞處理參數
NEWS_PROCESSING_CONFIG = {
    "MAX_NEWS_LENGTH": 512,            # 新聞最大長度
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "SENTIMENT_MODEL": "nlptown/bert-base-multilingual-uncased-sentiment",
}

# 增強版新聞爬蟲配置
ENHANCED_NEWS_CONFIG = {
    "SCRAPING": {
        "MAX_RETRIES": 3,              # 最大重試次數
        "RETRY_DELAY": 2,              # 重試延遲（秒）
        "MIN_DELAY": 1.0,              # 最小隨機延遲（秒）
        "MAX_DELAY": 4.0,              # 最大隨機延遲（秒）
        "TIMEOUT": 30,                 # 請求超時（秒）
        "MAX_WORKERS": 5,              # 並行工作線程數
        "USE_API_FIRST": True,         # 優先使用 API
        "FETCH_CONTENT": True,         # 是否抓取詳細內容
        "DEDUPLICATION": True,         # 是否去重
    },
    "API_ENDPOINTS": {
        "YAHOO_FINANCE": "https://query1.finance.yahoo.com/v1/finance/search",
        "CNYES": "https://news.cnyes.com/api/v3/news",
    },
    "SELECTORS": {
        "CNYES": {
            "ARTICLE_LIST": ["div._1h45", "div[class*=\"item\"]", "div[class*=\"news\"]", "article", "div.post-item"],
            "TITLE": ["h3", "h2", ".title", "a[title]"],
            "CONTENT": ["div.article-content", "div.post-content", "div.content", "article .content", "div[class*=\"article\"]"],
            "TIME": [".date", ".time", "time", "[datetime]"],
        },
        "YAHOO": {
            "ARTICLE_LIST": ["div[data-module=\"Stream\"] > ul > li", "li[class*=\"stream-item\"]", "article"],
            "TITLE": ["h3 a", "h2 a", "a[data-test-locator=\"headline\"]"],
            "CONTENT": ["div.caas-body", "div.article-content", "div.content", "article .content"],
            "TIME": ["time", ".date", "[datetime]"],
        },
        "CTEE": {
            "ARTICLE_LIST": [".post-item", ".news-item", "article"],
            "TITLE": ["h3 a", "h2 a", ".title a"],
            "CONTENT": ["div.entry-content", "div.post-content", "div.content", "article .content"],
            "TIME": [".date", "time", "[datetime]"],
        },
        "LTN": {
            "ARTICLE_LIST": [".listphoto", ".list", ".news-item"],
            "TITLE": ["h3 a", "h2 a", ".title a"],
            "CONTENT": ["div.text", "div.article-content", "div.content", "article .content"],
            "TIME": [".time", ".date", "time"],
        },
    },
    "NLP": {
        "ENABLE_SEGMENTATION": True,   # 是否啟用斷詞
        "REMOVE_STOPWORDS": True,      # 是否移除停用詞
        "MIN_WORD_LENGTH": 2,          # 最小詞長
        "CUSTOM_STOPWORDS": [          # 自定義停用詞
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一個",
            "上", "也", "很", "到", "說", "要", "去", "你", "會", "著", "沒有", "看", "好",
            "自己", "這", "那", "他", "她", "它", "們", "我們", "你們", "他們", "這個",
            "那個", "什麼", "怎麼", "為什麼", "因為", "所以", "但是", "如果", "雖然",
            "然後", "而且", "或者", "不過", "可是", "只是", "就是", "就是說"
        ],
    },
}

# 特徵工程參數
FEATURE_ENGINEERING_CONFIG = {
    "LOOKBACK_PERIODS": [5, 10, 20, 60],  # 回看期間
    "VOLUME_MA_PERIODS": [5, 20],         # 成交量移動平均期間
    "PRICE_CHANGE_PERIODS": [1, 5, 10],   # 價格變化期間
}

# 日誌配置
LOGGING_CONFIG = {
    "LEVEL": "INFO",
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "FILE": PROJECT_ROOT / "logs" / "stock_selector.log"
}

# 確保日誌目錄存在
LOGGING_CONFIG["FILE"].parent.mkdir(parents=True, exist_ok=True)

# 環境變數
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# API 配置（如果需要）
API_CONFIG = {
    "TWSTOCK_API": {
        "BASE_URL": "https://mis.twse.com.tw",
        "TIMEOUT": 30
    },
    "NEWS_API": {
        "TIMEOUT": 30,
        "RETRY_COUNT": 3
    }
}

# 快取配置
CACHE_CONFIG = {
    "ENABLE_CACHE": True,
    "CACHE_DIR": PROJECT_ROOT / "cache",
    "CACHE_EXPIRY": 3600,  # 1小時
}

# 確保快取目錄存在
if CACHE_CONFIG["ENABLE_CACHE"]:
    CACHE_CONFIG["CACHE_DIR"].mkdir(parents=True, exist_ok=True)




