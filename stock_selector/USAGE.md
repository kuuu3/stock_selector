# 股票選股系統使用說明

## 系統架構

本系統將數據抓取和預測分離，提供更靈活的使用方式：

```
數據抓取 → 模型訓練 → 股票預測
   ↓           ↓         ↓
fetch_data  train_model  quick_predict
              ↓
         continue_train
```

## 快速開始

### 1. 數據抓取
```bash
# 檢查現有數據
python fetch_data.py --check

# 獲取新數據
python fetch_data.py

# 強制重新獲取數據
python fetch_data.py --force
```

### 2. 模型訓練
```bash
# 完整訓練模型（從頭開始）
python train_model.py

# 繼續訓練（基於現有模型微調）
python continue_train.py

# 列出所有可用的 checkpoint
python continue_train.py --list

# 使用指定的 checkpoint 進行訓練
python continue_train.py --checkpoint 2024-05-31_14-30-25
```

### 3. 股票預測
```bash
# 快速預測（使用現有數據）
python quick_predict.py

# 完整預測（包含數據獲取）
python predict_stocks.py
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

### 數據抓取相關
- `fetch_data.py` - 獨立數據抓取腳本
- `data/raw/prices.csv` - 原始股價數據

### 模型訓練相關
- `train_model.py` - 完整模型訓練腳本（從頭開始，自動備份現有模型）
- `continue_train.py` - 繼續訓練腳本（基於現有模型微調）
- `outputs/models/` - 當前使用的模型文件
- `outputs/checkpoints/` - 所有模型備份文件（自動創建）
- `outputs/training_logs/` - 訓練記錄

### 預測相關
- `predict_stocks.py` - 完整預測腳本
- `quick_predict.py` - 快速預測腳本
- `outputs/top20_predictions.csv` - 預測結果

### 回測相關
- `run_backtest.py` - 回測腳本
- `src/backtest/backtest.py` - 回測引擎
- `outputs/backtest/` - 回測結果目錄
  - `portfolio_performance.csv` - 投資組合表現
  - `trade_history.csv` - 交易歷史
  - `backtest_report.txt` - 回測報告

### 工具腳本
- `test_predict.py` - 測試預測功能

## 工作流程

### 日常使用流程
1. **更新數據**（每日一次）
   ```bash
   python fetch_data.py
   ```

2. **快速預測**（需要時）
   ```bash
   python quick_predict.py
   ```

### 模型重新訓練流程
1. **獲取最新數據**
   ```bash
   python fetch_data.py --force
   ```

       2. **選擇訓練方式**
          ```bash
          # 完整重新訓練（從頭開始）
          python train_model.py

          # 繼續訓練（基於現有模型微調）
          python continue_train.py

          # 查看可用的 checkpoint
          python continue_train.py --list

          # 使用特定 checkpoint 繼續訓練
          python continue_train.py --checkpoint 2024-05-31_14-30-25
          ```

3. **進行預測**
   ```bash
   python quick_predict.py
   ```

### 繼續訓練 vs 完整訓練

#### 繼續訓練 (continue_train.py)
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

#### 完整訓練 (train_model.py)
- **適用場景**：重大數據變化、模型架構調整
- **優勢**：從零開始、完全重新學習
- **特點**：
  - 隨機初始化參數
  - 完整訓練流程
  - 適合大規模數據變化

## 性能優化

### 數據抓取優化
- 自動檢查數據新舊程度
- 避免重複抓取相同數據
- 支持強制更新選項

### 訓練優化
- GPU 自動檢測和加速
- 使用現有數據避免重複抓取
- 模型參數優化（減少訓練時間）

### 預測優化
- 快速預測模式（跳過數據抓取）
- 自動載入最新模型
- 結果自動保存

## 輸出結果

### 預測結果格式
```csv
date,stock_code,logistic_regression_prob,xgboost_classifier_prob,xgboost_regressor_return,composite_score
2024-05-31,2454,0.85,0.92,0.015,1.3968
2024-05-31,2882,0.78,0.89,0.012,1.0492
```

### 評分說明
- `composite_score`: 綜合評分（越高越好）
- 評分 > 0.5: 強力推薦
- 評分 0.0-0.5: 觀察名單
- 評分 < 0.0: 不推薦

## 故障排除

### 常見問題

1. **找不到數據文件**
   ```bash
   python fetch_data.py
   ```

2. **模型載入失敗**
   ```bash
   python train_model.py
   ```

3. **預測結果異常**
   ```bash
   python test_predict.py
   ```

### 數據檢查
```bash
# 檢查數據狀態
python fetch_data.py --check

# 檢查模型狀態
python test_predict.py
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

## 自定義配置

### 修改股票清單
編輯 `src/config.py` 中的 `STOCK_LIST`

### 調整模型參數
編輯 `src/config.py` 中的 `MODEL_CONFIG`

### 修改選股數量
編輯 `src/config.py` 中的 `SELECTION_CONFIG`
