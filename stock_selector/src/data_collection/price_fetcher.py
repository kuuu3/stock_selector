"""
股價數據收集模組
使用 twstock 套件收集台股數據
"""

import pandas as pd
import twstock
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path

from ..config import (
    RAW_PRICES_FILE, 
    DATA_COLLECTION_CONFIG,
    TECHNICAL_INDICATORS,
    API_CONFIG
)

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceFetcher:
    """股價數據收集器"""
    
    def __init__(self):
        self.stock_list = DATA_COLLECTION_CONFIG["STOCK_LIST"]
        self.lookback_days = DATA_COLLECTION_CONFIG["LOOKBACK_DAYS"]
        
    def fetch_stock_data(self, stock_code: str, days: int = None) -> pd.DataFrame:
        """
        獲取單一股票數據
        
        Args:
            stock_code: 股票代碼
            days: 回看天數
            
        Returns:
            包含股價數據的 DataFrame
        """
        if days is None:
            days = self.lookback_days
            
        try:
            # 使用 twstock 獲取數據
            stock = twstock.Stock(stock_code)
            
            # 獲取歷史數據 - 修復 twstock API 調用
            import datetime
            end_date = datetime.datetime.now()
            # 使用傳入的 days 參數，不強制限制
            start_date = end_date - datetime.timedelta(days=days)
            
            # 收集多個月的數據
            all_data = []
            current_date = start_date
            
            while current_date <= end_date:
                try:
                    data_list = stock.fetch(current_date.year, current_date.month)
                    if data_list:
                        all_data.extend(data_list)
                        logger.info(f"  獲取 {current_date.year}-{current_date.month:02d}: {len(data_list)} 筆")
                    time.sleep(0.2)  # 稍微增加延遲避免被限制
                except Exception as e:
                    logger.warning(f"獲取 {current_date.year}-{current_date.month} 數據失敗: {e}")
                
                # 移動到下一個月 - 修復日期溢出問題
                # 使用更安全的方法：先計算年月，再創建新日期
                if current_date.month == 12:
                    next_year = current_date.year + 1
                    next_month = 1
                else:
                    next_year = current_date.year
                    next_month = current_date.month + 1
                
                # 創建下個月第一天，避免日期溢出
                current_date = datetime.datetime(next_year, next_month, 1)
            
            data_list = all_data
            
            if data_list is None or len(data_list) == 0:
                logger.warning(f"無法獲取股票 {stock_code} 的數據")
                return pd.DataFrame()
            
            # 轉換為 DataFrame
            data = []
            for item in data_list:
                data.append({
                    'date': item.date,
                    'open': item.open,
                    'high': item.high,
                    'low': item.low,
                    'close': item.close,
                    'volume': item.capacity,  # twstock 中成交量字段名
                    'stock_code': stock_code
                })
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"成功獲取股票 {stock_code} 的 {len(df)} 筆數據")
            return df
            
        except Exception as e:
            logger.error(f"獲取股票 {stock_code} 數據時發生錯誤: {e}")
            return pd.DataFrame()
    
    def fetch_incremental_data(self, existing_df: pd.DataFrame = None, days_to_fetch: int = 7) -> pd.DataFrame:
        """
        增量獲取數據 - 只獲取缺失的數據
        
        Args:
            existing_df: 現有數據
            days_to_fetch: 要獲取的天數
            
        Returns:
            新增的數據 DataFrame
        """
        if existing_df is None or existing_df.empty:
            logger.info("沒有現有數據，將獲取完整數據")
            return self.fetch_all_stocks(save_to_file=False)
        
        # 找到最新日期
        latest_date = existing_df['date'].max()
        logger.info(f"現有數據最新日期: {latest_date.strftime('%Y-%m-%d')}")
        
        # 計算需要獲取的天數
        import datetime
        days_old = (datetime.datetime.now() - latest_date).days
        if days_old <= 0:
            logger.info("數據已是最新，無需更新")
            return pd.DataFrame()
        
        # 限制獲取天數
        fetch_days = min(days_old + 5, days_to_fetch)  # 多獲取5天作為緩衝
        logger.info(f"需要獲取最近 {fetch_days} 天的數據")
        
        all_new_data = []
        
        for i, stock_code in enumerate(self.stock_list):
            logger.info(f"增量獲取股票 {stock_code} ({i+1}/{len(self.stock_list)}) - 進度: {i/len(self.stock_list)*100:.1f}%")
            
            try:
                # 只獲取最近幾天的數據
                df = self.fetch_stock_data(stock_code, days=fetch_days)
                if not df.empty:
                    # 過濾掉已存在的數據
                    existing_stock_data = existing_df[existing_df['stock_code'] == stock_code]
                    if not existing_stock_data.empty:
                        existing_dates = set(existing_stock_data['date'].dt.date)
                        df = df[~df['date'].dt.date.isin(existing_dates)]
                    
                    if not df.empty:
                        all_new_data.append(df)
                        logger.info(f"  ✓ 新增 {len(df)} 筆數據")
                    else:
                        logger.info(f"  - 無新數據")
                else:
                    logger.warning(f"  ✗ 未獲取到數據")
            except Exception as e:
                logger.error(f"  ✗ 獲取股票 {stock_code} 時發生錯誤: {e}")
            
            time.sleep(0.5)
        
        if not all_new_data:
            logger.info("沒有新數據需要更新")
            return pd.DataFrame()
        
        # 合併新數據
        new_df = pd.concat(all_new_data, ignore_index=True)
        logger.info(f"總共獲取 {len(new_df)} 筆新數據")
        
        return new_df

    def fetch_all_stocks(self, save_to_file: bool = True) -> pd.DataFrame:
        """
        獲取所有股票的數據
        
        Args:
            save_to_file: 是否保存到檔案
            
        Returns:
            合併後的股價數據 DataFrame
        """
        all_data = []
        
        logger.info(f"開始獲取 {len(self.stock_list)} 支股票的數據...")
        
        for i, stock_code in enumerate(self.stock_list):
            logger.info(f"正在獲取股票 {stock_code} ({i+1}/{len(self.stock_list)}) - 進度: {i/len(self.stock_list)*100:.1f}%")
            
            try:
                df = self.fetch_stock_data(stock_code, self.lookback_days)
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"  ✓ 成功獲取 {len(df)} 筆數據")
                else:
                    logger.warning(f"  ✗ 未獲取到數據")
            except Exception as e:
                logger.error(f"  ✗ 獲取股票 {stock_code} 時發生錯誤: {e}")
            
            # 避免請求過於頻繁
            time.sleep(0.5)  # 減少延遲時間
        
        if not all_data:
            logger.error("沒有成功獲取任何股票數據")
            return pd.DataFrame()
        
        # 合併所有數據
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['stock_code', 'date']).reset_index(drop=True)
        
        logger.info(f"成功獲取 {len(combined_df)} 筆股價數據")
        
        # 保存到檔案
        if save_to_file:
            self.save_data(combined_df, RAW_PRICES_FILE)
            logger.info(f"股價數據已保存到 {RAW_PRICES_FILE}")
        
        return combined_df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算技術指標
        
        Args:
            df: 股價數據 DataFrame
            
        Returns:
            包含技術指標的 DataFrame
        """
        if df.empty:
            return df
        
        # 按股票代碼分組計算技術指標
        result_data = []
        
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].copy()
            stock_df = stock_df.sort_values('date').reset_index(drop=True)
            
            # 計算移動平均線
            ma_short = TECHNICAL_INDICATORS["MA_SHORT"]
            ma_long = TECHNICAL_INDICATORS["MA_LONG"]
            
            stock_df[f'MA_{ma_short}'] = stock_df['close'].rolling(window=ma_short).mean()
            stock_df[f'MA_{ma_long}'] = stock_df['close'].rolling(window=ma_long).mean()
            stock_df['MA_diff'] = stock_df[f'MA_{ma_short}'] - stock_df[f'MA_{ma_long}']
            
            # 計算 RSI
            rsi_period = TECHNICAL_INDICATORS["RSI_PERIOD"]
            stock_df['RSI'] = self._calculate_rsi(stock_df['close'], rsi_period)
            
            # 計算 MACD
            macd_data = self._calculate_macd(
                stock_df['close'], 
                TECHNICAL_INDICATORS["MACD_FAST"],
                TECHNICAL_INDICATORS["MACD_SLOW"],
                TECHNICAL_INDICATORS["MACD_SIGNAL"]
            )
            stock_df['MACD_DIF'] = macd_data['dif']
            stock_df['MACD_DEM'] = macd_data['dem']
            stock_df['MACD_OSC'] = macd_data['osc']
            
            # 計算成交量變化
            stock_df['volume_change'] = stock_df['volume'].pct_change()
            
            # 計算波動率
            volatility_period = TECHNICAL_INDICATORS["VOLATILITY_PERIOD"]
            stock_df['volatility'] = stock_df['close'].rolling(window=volatility_period).std()
            
            # 計算價格變化
            stock_df['price_change'] = stock_df['close'].pct_change()
            stock_df['price_change_5d'] = stock_df['close'].pct_change(periods=5)
            
            result_data.append(stock_df)
        
        result_df = pd.concat(result_data, ignore_index=True)
        return result_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """計算 RSI 指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """計算 MACD 指標"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        dif = ema_fast - ema_slow
        dem = dif.ewm(span=signal).mean()
        osc = dif - dem
        
        return {
            'dif': dif,
            'dem': dem,
            'osc': osc
        }
    
    def save_data(self, df: pd.DataFrame, file_path: Path):
        """保存數據到 CSV 檔案"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """從 CSV 檔案載入數據"""
        if file_path.exists():
            return pd.read_csv(file_path, encoding='utf-8-sig')
        else:
            logger.warning(f"檔案 {file_path} 不存在")
            return pd.DataFrame()
    
    def get_latest_data(self) -> pd.DataFrame:
        """獲取最新的股價數據"""
        # 先嘗試從檔案載入
        if RAW_PRICES_FILE.exists():
            df = self.load_data(RAW_PRICES_FILE)
            # 檢查數據是否為最新（例如：今天有數據）
            latest_date = pd.to_datetime(df['date']).max()
            today = datetime.now().date()
            
            if latest_date.date() == today:
                logger.info("使用快取的股價數據")
                return df
        
        # 如果沒有最新數據，則重新獲取
        logger.info("獲取最新股價數據...")
        return self.fetch_all_stocks()


def main():
    """主函數 - 用於測試數據收集功能"""
    fetcher = PriceFetcher()
    
    # 測試獲取單一股票數據
    test_stock = "2330"  # 台積電
    logger.info(f"測試獲取股票 {test_stock} 數據...")
    
    df = fetcher.fetch_stock_data(test_stock, days=30)
    if not df.empty:
        logger.info(f"成功獲取 {len(df)} 筆數據")
        logger.info(f"數據欄位: {df.columns.tolist()}")
        logger.info(f"最新收盤價: {df['close'].iloc[-1]}")
        
        # 計算技術指標
        df_with_indicators = fetcher.calculate_technical_indicators(df)
        logger.info(f"技術指標計算完成，新增欄位: {[col for col in df_with_indicators.columns if col not in df.columns]}")


if __name__ == "__main__":
    main()




