# Stock Selector 選股系統

基於機器學習的台股選股系統，整合多種數據源進行智能選股。

## 🎯 功能特色

- **多數據源整合**: 股價數據 + 財經新聞
- **技術指標分析**: MA、RSI、MACD、波動率等
- **機器學習模型**: Logistic Regression + XGBoost
- **智能選股**: 基於模型預測的股票排序
- **風險控制**: 流動性篩選 + 產業分散

## 📁 專案結構

```
stock_selector/
├── data/                      # 數據目錄
│   ├── raw/                   # 原始數據
│   └── processed/             # 處理後數據
├── src/                       # 主要程式碼
│   ├── config.py              # 全域配置
│   ├── data_collection/       # 數據收集
│   ├── preprocessing/         # 前處理
│   ├── models/                # 模型訓練
│   ├── selection/             # 選股邏輯
│   ├── backtest/              # 回測
│   └── visualization/         # 視覺化
├── outputs/                   # 輸出結果
├── notebooks/                 # Jupyter 筆記本
├── main.py                    # 主程式入口
└── requirements.txt           # 依賴套件
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 運行系統

```bash
python main.py
```

### 3. 查看結果

- 股價數據: `data/raw/prices.csv`
- 新聞數據: `data/raw/news.csv`
- 特徵矩陣: `data/processed/features.npy`
- 選股結果: `outputs/top20.csv`

## 📊 技術指標

- **移動平均線**: MA(5), MA(20), MA差
- **相對強弱指標**: RSI(14)
- **MACD**: DIF, DEM, OSC
- **成交量**: Volume Change, Volume MA
- **波動率**: Volatility(10)

## 🤖 機器學習模型

1. **Logistic Regression** (Baseline)
2. **XGBoost Classifier** (主力模型)
3. **XGBoost Regressor** (回歸版本)

## 📈 選股流程

1. **數據收集**: 股價 + 新聞爬蟲
2. **特徵工程**: 技術指標 + 新聞情緒
3. **模型訓練**: 多模型集成
4. **股票排序**: 基於預測分數
5. **風險控制**: 流動性 + 產業分散

## ⚙️ 配置說明

主要配置檔案: `src/config.py`

- `TECHNICAL_INDICATORS`: 技術指標參數
- `MODEL_CONFIG`: 模型參數
- `SELECTION_CONFIG`: 選股參數
- `DATA_COLLECTION_CONFIG`: 數據收集參數

## 📝 使用範例

```python
from src.data_collection import PriceFetcher, NewsScraper
from src.preprocessing import FeatureEngineer

# 數據收集
price_fetcher = PriceFetcher()
price_df = price_fetcher.fetch_all_stocks()

news_scraper = NewsScraper()
news_df = news_scraper.scrape_all_news()

# 特徵工程
engineer = FeatureEngineer()
features = engineer.create_features(price_df, news_df)
```

## 🔧 開發狀態

- ✅ 專案結構建立
- ✅ 數據收集模組
- ✅ 前處理模組
- 🚧 模型訓練模組 (進行中)
- ⏳ 選股邏輯模組
- ⏳ 回測模組
- ⏳ 視覺化模組

## 📞 注意事項

- 本系統僅供學習和研究使用
- 投資有風險，請謹慎決策
- 建議在實際投資前進行充分回測

## 📄 授權

MIT License


