# Stock Selector 選股系統

基於機器學習的台股選股系統，整合多種數據源進行智能選股。

## 功能特色

- **多數據源整合**: 股價數據 + 財經新聞
- **技術指標分析**: MA、RSI、MACD、波動率等
- **機器學習模型**: Logistic Regression + XGBoost
- **智能選股**: 基於模型預測的股票排序
- **回測系統**: 完整的策略回測與績效分析
- **增量訓練**: 支持模型持續學習與更新
- **風險控制**: 流動性篩選 + 產業分散

## 專案結構

```
stock_selector/
├── data/                      # 數據目錄
│   ├── raw/                   # 原始數據 (prices.csv)
│   └── processed/             # 處理後數據
├── src/                       # 主要程式碼
│   ├── config.py              # 全域配置
│   ├── data_collection/       # 數據收集 (PriceFetcher)
│   ├── preprocessing/         # 前處理 (FeatureEngineer)
│   ├── models/                # 模型訓練 (ModelTrainer)
│   ├── selection/             # 選股邏輯 (StockSelector)
│   ├── backtest/              # 回測 (Backtester)
│   └── visualization/         # 視覺化
├── outputs/                   # 輸出結果
│   ├── models/                # 訓練好的模型
│   ├── checkpoints/           # 模型檢查點
│   ├── top20.csv             # 選股結果
│   └── backtest/             # 回測結果
├── fetch_data.py             # 數據抓取腳本
├── train_model.py            # 模型訓練腳本
├── quick_predict.py          # 快速預測腳本
├── continue_train.py         # 增量訓練腳本
├── run_backtest.py           # 回測腳本
└── requirements.txt          # 依賴套件
```

## 快速開始

### 1. 安裝依賴

```bash
cd stock_selector
pip install -r requirements.txt
```

### 2. 數據抓取

```bash
# 檢查現有數據
python stock_selector/fetch_data.py --check

# 獲取新數據
python stock_selector/fetch_data.py

# 強制重新獲取數據
python stock_selector/fetch_data.py --force
```

### 3. 模型訓練

```bash
# 完整訓練模型（從頭開始）
python stock_selector/train_model.py

# 繼續訓練（基於現有模型微調）
python stock_selector/continue_train.py

# 列出所有可用的 checkpoint
python stock_selector/continue_train.py --list
```

### 4. 股票預測

```bash
# 快速預測（使用現有數據）
python stock_selector/quick_predict.py

# 完整預測（包含數據獲取）
python stock_selector/predict_stocks.py
```

### 5. 回測分析

```bash
# 運行回測
python stock_selector/run_backtest.py --start-date 2025-06-01 --end-date 2025-08-31 --capital 1000000 --top-n 10
```

### 6. 查看結果

- 股價數據: `stock_selector/data/raw/prices.csv`
- 選股結果: `stock_selector/outputs/top20.csv`
- 回測報告: `stock_selector/outputs/backtest/backtest_report.txt`
- 模型檢查點: `stock_selector/outputs/checkpoints/`

## 技術指標

- **移動平均線**: MA(5), MA(20), MA差
- **相對強弱指標**: RSI(14)
- **MACD**: DIF, DEM, OSC
- **成交量**: Volume Change, Volume MA
- **波動率**: Volatility(10)

## 機器學習模型

1. **Logistic Regression** (Baseline)
2. **XGBoost Classifier** (主力模型)
3. **XGBoost Regressor** (回歸版本)

## 選股流程

1. **數據收集**: 股價數據抓取
2. **特徵工程**: 技術指標計算
3. **模型訓練**: 多模型集成訓練
4. **股票排序**: 基於預測分數排序
5. **風險控制**: 流動性 + 產業分散

## 配置說明

主要配置檔案: `stock_selector/src/config.py`

- `DATA_COLLECTION_CONFIG`: 數據收集參數
- `TECHNICAL_INDICATORS`: 技術指標參數
- `MODEL_CONFIG`: 模型參數
- `SELECTION_CONFIG`: 選股參數
- `BACKTEST_CONFIG`: 回測參數

## 使用範例

```python
from src.data_collection import PriceFetcher
from src.preprocessing import FeatureEngineer
from src.models import ModelTrainer
from src.selection import StockSelector

# 數據收集
price_fetcher = PriceFetcher()
price_df = price_fetcher.fetch_all_stocks()

# 特徵工程
engineer = FeatureEngineer()
features = engineer.create_features(price_df)

# 模型訓練
trainer = ModelTrainer()
trainer.train_models(features, labels)

# 股票選擇
selector = StockSelector()
selected_stocks = selector.select_top_stocks(features)
```

## 系統特色

### 增量更新
- 智能數據抓取：只獲取缺失的數據
- 增量模型訓練：基於現有模型微調
- 自動檢查點管理：支持模型版本控制

### 風險控制
- 流動性篩選：確保股票可交易性
- 產業分散：避免過度集中投資
- 台股交易規則：支持張為單位的交易

### 回測系統
- 完整的策略回測
- 詳細的績效分析
- 交易歷史追蹤

## 開發狀態

- 專案結構建立 (完成)
- 數據收集模組 (PriceFetcher) (完成)
- 前處理模組 (FeatureEngineer) (完成)
- 模型訓練模組 (ModelTrainer) (完成)
- 選股邏輯模組 (StockSelector) (完成)
- 回測模組 (Backtester) (完成)
- 增量訓練系統 (完成)
- 路徑處理優化 (完成)
- 視覺化模組 (進行中)

## 文檔

- [USAGE.md](stock_selector/USAGE.md) - 詳細使用說明
- [DESIGN.md](DESIGN.md) - 系統設計文檔