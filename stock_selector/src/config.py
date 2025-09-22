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
        "https://news.cnyes.com/",      # 鉅亨網
        "https://ctee.com.tw/",         # 工商時報
    ],
    "STOCK_LIST": [                     # 主要股票代碼列表（已移除無法獲取的股票）
        "2330", "2317", "2454", "6505", "2308",  # 台積電、鴻海、聯發科等
        "2881", "2882", "2886", "2891", "2892",  # 富邦金、國泰金等
        "5284", "3704", "1560", "1316",          # 其他股票（移除5475）
        "2481", "8039", "3563", "2630",          # 其他股票（移除6761）
        "3019", "3311", "8021", "2476",          # 其他股票（移除8027）
        "4976", "2231", "8033", "2429",          # 其他股票（移除5498）
        "4414", "6235", "1504", "2408",          # 其他股票
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




