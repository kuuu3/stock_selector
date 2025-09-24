# 股票選股系統使用說明

## 系統架構

本系統採用模塊化設計，將數據準備、模型訓練和股票預測分離，提供更靈活的使用方式：

```
數據準備 → 模型訓練 → 股票預測 → 回測分析
   ↓           ↓         ↓         ↓
prepare_data  train.py   predict.py  run_backtest.py
```

## 新聞爬蟲系統

系統已整合統一新聞爬蟲，具備以下特性：

### 統一新聞爬蟲
- **API優先策略** - 優先使用鉅亨網API和Yahoo RSS
- **快速穩定** - 專注於最穩定的數據源，避免HTML解析問題
- **完整內容** - 鉅亨網API提供完整新聞內容，Yahoo RSS提供部分內容
- **智能去重** - 基於標題自動去重
- **多源整合** - 同時爬取鉅亨網、Yahoo財經、中央社三個新聞源
- **高效爬取** - 快速獲取新聞，避免長時間等待

## 快速開始

### 運行方式
所有腳本都支持從不同目錄運行：

```bash
# 方式1：在 stock_selector 目錄內運行
cd stock_selector
python prepare_data.py --check
python train.py --mode full
python predict.py --mode quick

# 方式2：從項目根目錄運行
python stock_selector/prepare_data.py --check
python stock_selector/train.py --mode full
python stock_selector/predict.py --mode quick
```

系統會自動處理路徑問題，無論從哪個目錄運行都能正確找到數據文件。

### 1. 數據準備
```bash
# 檢查系統狀態和數據完整性
python prepare_data.py --check

# 獲取所有數據（股價 + 新聞）
python prepare_data.py

# 只獲取股價數據
python prepare_data.py --prices-only

# 只獲取新聞數據（使用統一新聞爬蟲）
python prepare_data.py --news-only

# 跳過新聞數據處理
python prepare_data.py --no-news

# 強制重新獲取所有數據
python prepare_data.py --force

# 使用API獲取股價數據
python prepare_data.py --use-api
```

### 新聞預處理
```bash
# 處理新聞情感分析（使用預設文件）
python process_news.py

# 指定輸入文件
python process_news.py --input data/raw/unified_news.csv

# 指定輸出文件
python process_news.py --output data/processed/custom_news.csv

# 強制重新處理，覆蓋現有文件
python process_news.py --force
```

# 快速新聞爬取（僅中央社）
python quick_news_scraper.py

# 指定新聞抓取頁數
python prepare_data.py --pages 5

# 系統狀態檢查
python system_status.py
```

### 2. 模型訓練
```bash
# 完整訓練模型（從頭開始）
python train.py --mode full

# 繼續訓練（基於現有模型微調）
python train.py --mode continue

# 使用指定的 checkpoint 進行訓練
python train.py --mode continue --checkpoint 2024-05-31_14-30-25

# 列出所有可用的 checkpoint
python train.py --list
```

### 3. 股票預測
```bash
# 快速預測（使用現有數據）
python predict.py --mode quick

# 完整預測（包含數據獲取）
python predict.py --mode full

# 指定選股數量
python predict.py --top-n 10
```

### 4. 回測系統
```bash
# 基本回測（使用預設參數）
python run_backtest.py

# 自定義回測參數
python run_backtest.py --start-date 2024-01-01 --end-date 2024-12-31 --capital 1000000 --top-n 10 --rebalance-days 5

# 快速回測（較短期間）
python run_backtest.py --start-date 2024-06-01 --end-date 2024-08-31 --top-n 5
```

## 文件說明

### 數據準備相關
- `prepare_data.py` - 統一數據準備腳本（股價 + 新聞 + 情感分析）
- `data/raw/prices.csv` - 原始股價數據
- `data/processed/news_with_sentiment.csv` - 新聞情感分析數據

### 模型訓練相關
- `train.py` - 統一訓練腳本（支持完整訓練和增量訓練）
- `outputs/models/` - 當前使用的模型文件
- `outputs/checkpoints/` - 所有模型備份文件（自動創建）
- `outputs/training_logs/` - 訓練記錄

### 預測相關
- `predict.py` - 統一預測腳本（支持快速和完整預測模式）
- `outputs/top20.csv` - 選股結果
- `outputs/top20_predictions.csv` - 詳細預測結果
- `outputs/quick_predictions.csv` - 快速預測結果

### 回測相關
- `run_backtest.py` - 回測腳本
- `src/backtest/backtest.py` - 回測引擎
- `outputs/backtest/` - 回測結果目錄
  - `portfolio_performance.csv` - 投資組合表現
  - `trade_history.csv` - 交易歷史
  - `backtest_report.txt` - 回測報告

### 核心模塊
- `src/models/train.py` - 模型訓練引擎
- `src/models/predict.py` - 模型預測引擎
- `src/preprocessing/feature_engineer.py` - 特徵工程
- `src/selection/stock_selector.py` - 股票選擇器

## 工作流程

### 日常使用流程
1. **更新數據**（每日一次）
   ```bash
   python prepare_data.py
   ```

2. **快速預測**（需要時）
   ```bash
   python predict.py --mode quick
   ```

### 模型重新訓練流程
1. **獲取最新數據**
   ```bash
   python prepare_data.py --force
   ```

2. **選擇訓練方式**
   ```bash
   # 完整重新訓練（從頭開始）
   python train.py --mode full

   # 增量訓練（基於現有模型微調）
   python train.py --mode continue

   # 查看可用的 checkpoint
   python train.py --list

   # 使用特定 checkpoint 繼續訓練
   python train.py --mode continue --checkpoint 2025-09-22_21-59-32
   ```

3. **進行預測**
   ```bash
   python predict.py --mode quick
   ```

4. **回測驗證**（可選）
   ```bash
   python run_backtest.py
   ```

### 繼續訓練 vs 完整訓練

#### 增量訓練 (train.py --mode continue)
- **適用場景**：定期模型更新、小幅數據變化
- **優勢**：保留歷史知識、訓練速度快、節省時間
- **特點**：
  - 自動備份現有模型
  - 使用較小學習率微調
  - 記錄訓練歷史
  - 支持模型版本管理
  - **支持 checkpoint 選擇**：
    - 預設使用最新模型
    - 可指定特定 checkpoint 進行訓練
    - 可列出所有可用 checkpoint

#### 完整訓練 (train.py --mode full)
- **適用場景**：重大數據變化、模型架構調整
- **優勢**：從零開始、完全重新學習
- **特點**：
  - 隨機初始化參數
  - 完整訓練流程
  - 適合大規模數據變化
  - **正確的時間計算**：記錄真正的端到端訓練時間

## 性能優化

### 數據準備優化
- 自動檢查數據新舊程度
- 避免重複抓取相同數據
- 支持強制更新選項
- 增量更新機制

### 訓練優化
- GPU 自動檢測和加速
- 使用現有數據避免重複抓取
- 模型參數優化（減少訓練時間）
- **正確的標籤配置**：分類模型使用1週標籤，回歸模型使用1個月標籤

### 預測優化
- 快速預測模式（跳過數據抓取）
- 自動載入最新模型
- 結果自動保存
- 特徵工程優化

### 回測優化
- 並行處理多個再平衡點
- 高效的投資組合計算
- 詳細的交易記錄

## 輸出結果

### 預測結果格式
```csv
date,stock_code,logistic_regression_prob,xgboost_classifier_prob,xgboost_regressor_return,final_score,risk_adjusted_score
2025-09-22,4414,0.85,0.92,0.015,1.6363,1.6363
2025-09-22,2429,0.78,0.89,0.012,0.7724,0.7724
```

### 評分說明
- `final_score`: 最終評分（基於模型預測）
- `risk_adjusted_score`: 風險調整評分（考慮波動性）
- 評分 > 1.0: 強力推薦
- 評分 0.5-1.0: 推薦
- 評分 0.0-0.5: 觀察名單
- 評分 < 0.0: 不推薦

## 故障排除

### 常見問題

1. **找不到數據文件**
   - 問題：`FileNotFoundError: 找不到股價數據文件`
   - 解決：先運行數據準備
   ```bash
   python prepare_data.py
   ```

2. **路徑問題**
   - 問題：從不同目錄運行時找不到數據文件
   - 解決：系統已自動處理路徑問題，支持從任意目錄運行
   ```bash
   # 從項目根目錄運行
   python stock_selector/train.py --mode full
   
   # 從 stock_selector 目錄運行
   cd stock_selector
   python train.py --mode full
   ```

3. **模型載入失敗**
   ```bash
   python train.py --mode full
   ```

4. **預測結果異常**
   ```bash
   python predict.py --mode quick
   ```

5. **特徵數量不匹配**
   - 問題：`ValueError: X has X features, but StandardScaler is expecting Y features`
   - 解決：重新訓練模型以匹配當前特徵數量
   ```bash
   python train.py --mode full
   ```

### 數據檢查
```bash
# 檢查數據狀態
python prepare_data.py --check

# 檢查模型狀態
python predict.py --mode quick
```

## 系統監控

### 數據監控
- 數據年齡檢查
- 股票數量驗證
- 價格異常檢測

### 模型監控
- 準確率追蹤
- 預測一致性檢查
- 性能指標監控

## 最新系統改進

### 已修正的問題
1. **數據洩漏問題**
   - 特徵工程現在正確分離特徵和標籤
   - 避免未來信息洩漏到訓練數據中

2. **路徑處理問題**
   - 修正了輸出路徑多疊層問題
   - 支持相對和絕對路徑

3. **增量訓練問題**
   - 修正了標籤格式問題
   - 分類和回歸模型使用正確的標籤

4. **時間計算問題**
   - 完整訓練現在顯示正確的端到端時間

5. **標籤配置問題**
   - 回歸模型現在使用1個月前向報酬
   - 保留1週報酬作為元數據

6. **增強版新聞爬蟲**
   - 整合了智能重試機制
   - 支持多新聞源並行爬取
   - 實現了文本預處理功能
   - 添加了配置化管理系統

## 新聞爬蟲故障排除

### 常見問題

1. **403 Forbidden 錯誤**
   ```
   問題：網站拒絕訪問
   解決：使用快速新聞爬取腳本
   python quick_news_scraper.py
   ```

2. **新聞爬取速度慢**
   ```
   問題：網站反爬蟲機制導致延遲
   解決：減少爬取頁數或使用單一新聞源
   python prepare_data.py --pages 1
   ```

3. **選擇器失效**
   ```
   問題：網站結構改變導致選擇器失效
   解決：增強版爬蟲會自動嘗試多個選擇器
   ```

4. **依賴問題**
   ```
   問題：缺少 jieba 或 aiohttp
   解決：安裝依賴或使用降級模式
   pip install jieba aiohttp
   ```

### 新聞源狀態

- ✅ **鉅亨網API** - 最穩定，提供完整內容，推薦使用
- ✅ **Yahoo RSS** - 穩定，提供部分內容，推薦使用
- ✅ **中央社** - 可用，快速模式僅獲取標題
- ⚠️ **工商時報** - 被Cloudflare阻擋，暫時禁用
- ❌ **Yahoo財經HTML** - 網站問題（502錯誤），已改用RSS

### 替代方案

1. **統一新聞爬取（推薦）**
   ```bash
   python prepare_data.py --news-only
   ```

2. **手動新聞數據**
   - 可以手動下載新聞CSV文件
   - 放置在 `data/raw/` 目錄下

3. **RSS Feeds**
   - 考慮使用RSS作為替代數據源
   - 更穩定且不易被封鎖

### 系統架構改進
- **統一腳本**: 整合了多個分散的腳本
- **模塊化設計**: 清晰的職責分離
- **錯誤處理**: 完善的異常處理機制
- **日誌記錄**: 詳細的運行日誌

## 自定義配置

### 修改股票清單
編輯 `src/config.py` 中的 `STOCK_LIST`

### 調整模型參數
編輯 `src/config.py` 中的 `MODEL_CONFIG`

### 修改選股數量
編輯 `src/config.py` 中的 `SELECTION_CONFIG`

### 調整特徵工程參數
編輯 `src/preprocessing/feature_engineer.py` 中的相關參數
