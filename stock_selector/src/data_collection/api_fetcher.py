"""
API數據收集模組
使用台灣證券交易所API獲取股價數據
"""

import pandas as pd
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import json

from ..config import (
    RAW_PRICES_FILE, 
    DATA_COLLECTION_CONFIG,
    TECHNICAL_INDICATORS,
    API_CONFIG
)

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIFetcher:
    """API數據收集器"""
    
    def __init__(self):
        self.stock_list = DATA_COLLECTION_CONFIG["STOCK_LIST"]
        self.lookback_days = DATA_COLLECTION_CONFIG["LOOKBACK_DAYS"]
        
        # 台灣證券交易所API配置
        self.base_url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
        self.historical_url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        
        # API限制：每5秒最多3次請求
        self.request_delay = 2.0  # 2秒延遲確保不超過限制
        
        # TPEX 全表數據緩存
        self._tpex_cache = None
        self._tpex_cache_time = None
        self._tpex_df = None  # 緩存轉換後的 DataFrame
        
    def fetch_stock_realtime(self, stock_code: str) -> Optional[Dict]:
        """
        獲取單一股票即時數據
        注意：此API可能不可靠，建議使用歷史數據
        
        Args:
            stock_code: 股票代碼
            
        Returns:
            包含即時數據的字典
        """
        try:
            # 先嘗試上市
            ex_ch = f"tse_{stock_code}.tw"
            url = f"{self.base_url}?ex_ch={ex_ch}"
            
            logger.info(f"獲取股票 {stock_code} 即時數據...")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('rtcode') == '0000' and data.get('msgArray'):
                    stock_info = data['msgArray'][0]
                    logger.info(f"成功獲取股票 {stock_code} 即時數據")
                    return stock_info
                else:
                    logger.debug(f"上市即時數據獲取失敗: {data.get('rtmessage', '未知錯誤')}")
            
            # 如果上市失敗，嘗試上櫃
            ex_ch = f"otc_{stock_code}.tw"
            url = f"{self.base_url}?ex_ch={ex_ch}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('rtcode') == '0000' and data.get('msgArray'):
                    stock_info = data['msgArray'][0]
                    logger.info(f"成功獲取股票 {stock_code} 上櫃即時數據")
                    return stock_info
                else:
                    logger.debug(f"上櫃即時數據獲取失敗: {data.get('rtmessage', '未知錯誤')}")
            
            logger.warning(f"股票 {stock_code} 即時數據獲取失敗，API可能被限制")
            return None
                
        except Exception as e:
            logger.error(f"獲取股票 {stock_code} 即時數據時發生錯誤: {e}")
            return None
        
        finally:
            time.sleep(self.request_delay)
    
    def fetch_stock_historical(self, stock_code: str, days: int = None) -> pd.DataFrame:
        """
        獲取單一股票歷史數據
        
        Args:
            stock_code: 股票代碼
            days: 回看天數
            
        Returns:
            包含歷史數據的 DataFrame
        """
        if days is None:
            days = self.lookback_days
            
        try:
            logger.info(f"開始獲取股票 {stock_code} 的歷史數據...")
            
            # 計算日期範圍
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 先嘗試 TWSE API（上市股票）
            logger.info(f"嘗試 TWSE API 獲取股票 {stock_code}...")
            tse_data = self._fetch_tse_historical(stock_code, start_date, end_date)
            
            if not tse_data.empty:
                logger.info(f"TWSE API 成功獲取股票 {stock_code} 數據: {len(tse_data)} 筆")
                return tse_data
            else:
                logger.info(f"TWSE API 無法獲取股票 {stock_code}，嘗試 TPEX API...")
                # 如果 TWSE 失敗，嘗試 TPEX API（上櫃股票）
                tpex_data = self._fetch_tpex_historical(stock_code, start_date, end_date)
                
                if not tpex_data.empty:
                    logger.info(f"TPEX API 成功獲取股票 {stock_code} 數據: {len(tpex_data)} 筆")
                    return tpex_data
                else:
                    logger.warning(f"TWSE 和 TPEX API 都無法獲取股票 {stock_code} 的數據")
                    return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"獲取股票 {stock_code} 歷史數據時發生錯誤: {e}")
            return pd.DataFrame()
    
    def _fetch_tse_historical(self, stock_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        使用 TWSE API 獲取歷史數據
        
        Args:
            stock_code: 股票代碼
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            包含歷史數據的 DataFrame
        """
        try:
            url = self.historical_url
            all_data = []
            current_date = start_date
            
            while current_date <= end_date:
                try:
                    year = current_date.year
                    month = current_date.month
                    
                    params = {
                        'response': 'json',
                        'date': f'{year}{month:02d}01',
                        'stockNo': stock_code
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('stat') == 'OK' and data.get('data'):
                            monthly_data = data['data']
                            logger.info(f"  TWSE 獲取 {year}-{month:02d}: {len(monthly_data)} 筆")
                            all_data.extend(monthly_data)
                        else:
                            logger.debug(f"  TWSE API 返回錯誤: {data.get('stat')} - {data.get('message', '')}")
                    else:
                        logger.warning(f"  TWSE API 請求失敗，狀態碼: {response.status_code}")
                    
                    time.sleep(self.request_delay)
                    
                except Exception as e:
                    logger.warning(f"TWSE 獲取 {year}-{month:02d} 數據失敗: {e}")
                
                # 移動到下一個月
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            
            if all_data:
                df = self._convert_to_dataframe(all_data, stock_code)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"TWSE API 獲取股票 {stock_code} 時發生錯誤: {e}")
            return pd.DataFrame()
    
    def _download_tpex_full_table(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        下載 TPEX 全表數據（帶緩存和日期過濾）
        
        Args:
            start_date: 開始日期（可選）
            end_date: 結束日期（可選）
        
        Returns:
            TPEX 全表數據的 DataFrame
        """
        # 檢查緩存（1小時內有效）
        if (self._tpex_df is not None and 
            self._tpex_cache_time is not None and 
            (datetime.now() - self._tpex_cache_time).total_seconds() < 3600):
            logger.info("  使用 TPEX 緩存 DataFrame")
            # 如果有日期過濾需求，進行過濾
            if start_date and end_date:
                return self._filter_tpex_by_date(self._tpex_df, start_date, end_date)
            return self._tpex_df
        
        try:
            url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
            logger.info("  下載 TPEX 全表數據...")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"  TPEX 全表下載完成: {len(data)} 筆記錄")
                    
                    # 轉換為 DataFrame
                    df = pd.DataFrame(data)
                    
                    # 轉換日期格式 - TPEX 使用民國年格式
                    def convert_tpex_date(date_str):
                        """轉換 TPEX 日期格式 (民國年YYYYMMDD -> 西元年YYYYMMDD)"""
                        if len(date_str) == 7:  # 民國年格式 1140924
                            roc_year = int(date_str[:3])
                            gregorian_year = roc_year + 1911
                            return f"{gregorian_year}{date_str[3:]}"
                        return date_str
                    
                    df['Date'] = df['Date'].apply(convert_tpex_date)
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
                    
                    # 轉換數值欄位 - 使用正確的 TPEX API 欄位名稱
                    numeric_columns = ['Close', 'Open', 'High', 'Low', 'TradingShares', 'TransactionAmount']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    # 更新緩存
                    self._tpex_df = df
                    self._tpex_cache_time = datetime.now()
                    
                    # 如果有日期過濾需求，進行過濾
                    if start_date and end_date:
                        return self._filter_tpex_by_date(df, start_date, end_date)
                    
                    logger.info(f"  TPEX DataFrame 準備完成: {len(df)} 筆記錄")
                    return df
                else:
                    logger.warning("  TPEX API 返回空數據或格式錯誤")
                    return pd.DataFrame()
            else:
                logger.warning(f"  TPEX API 請求失敗，狀態碼: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"TPEX API 下載全表時發生錯誤: {e}")
            return pd.DataFrame()
    
    def _filter_tpex_by_date(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        按日期過濾 TPEX DataFrame
        
        Args:
            df: TPEX DataFrame
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            過濾後的 DataFrame
        """
        if df.empty:
            return df
        
        mask = (df['Date'].dt.date >= start_date.date()) & (df['Date'].dt.date <= end_date.date())
        filtered_df = df[mask].copy()
        logger.info(f"  日期過濾: {len(df)} -> {len(filtered_df)} 筆記錄")
        return filtered_df
    
    def _fetch_tpex_historical(self, stock_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        使用 TPEX API 獲取歷史數據
        
        Args:
            stock_code: 股票代碼
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            包含歷史數據的 DataFrame
        """
        try:
            # 使用緩存的全表數據（帶日期過濾）
            tpex_df = self._download_tpex_full_table(start_date, end_date)
            
            if not tpex_df.empty:
                # 使用 DataFrame 查詢過濾目標股票
                stock_data = tpex_df[tpex_df["SecuritiesCompanyCode"] == stock_code].copy()
                
                if not stock_data.empty:
                    logger.info(f"  TPEX 找到股票 {stock_code} 數據: {len(stock_data)} 筆")
                    
                    # 轉換為標準格式
                    result_df = self._convert_tpex_dataframe_to_standard(stock_data, stock_code)
                    return result_df
                else:
                    logger.debug(f"  TPEX 全表中未找到股票 {stock_code}")
                    return pd.DataFrame()
            else:
                logger.warning("  TPEX 全表數據下載失敗")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"TPEX API 獲取股票 {stock_code} 時發生錯誤: {e}")
            return pd.DataFrame()
    
    
    def _convert_to_dataframe(self, data: List, stock_code: str) -> pd.DataFrame:
        """
        將API返回的數據轉換為DataFrame
        
        Args:
            data: API返回的原始數據
            stock_code: 股票代碼
            
        Returns:
            轉換後的DataFrame
        """
        df_data = []
        
        for item in data:
            try:
                date_str = item[0]  # 日期
                volume = int(item[1].replace(',', ''))  # 成交量
                amount = int(item[2].replace(',', ''))  # 成交金額
                open_price = float(item[3].replace(',', ''))  # 開盤價
                high_price = float(item[4].replace(',', ''))  # 最高價
                low_price = float(item[5].replace(',', ''))   # 最低價
                close_price = float(item[6].replace(',', '')) # 收盤價
                
                # 轉換日期格式 - TWSE 也使用民國年格式
                # 民國年格式: 114/08/01 -> 2025/08/01
                if '/' in date_str and date_str.split('/')[0].isdigit():
                    parts = date_str.split('/')
                    if len(parts[0]) == 3:  # 民國年
                        roc_year = int(parts[0])
                        gregorian_year = roc_year + 1911
                        date_str = f"{gregorian_year}/{parts[1]}/{parts[2]}"
                
                date = datetime.strptime(date_str, '%Y/%m/%d').date()
                
                df_data.append({
                    'date': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'stock_code': stock_code
                })
                
            except (ValueError, IndexError) as e:
                logger.warning(f"解析數據項時發生錯誤: {e}, 數據: {item}")
                continue
        
        if df_data:
            df = pd.DataFrame(df_data)
            df = df.sort_values('date').reset_index(drop=True)
            return df
        else:
            return pd.DataFrame()
    
    def _convert_tpex_dataframe_to_standard(self, stock_data: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        將 TPEX DataFrame 轉換為標準格式
        
        Args:
            stock_data: TPEX 股票數據 DataFrame
            stock_code: 股票代碼
            
        Returns:
            標準格式的 DataFrame
        """
        try:
            # 創建標準格式的 DataFrame - 使用正確的 TPEX API 欄位名稱
            result_df = pd.DataFrame({
                'date': stock_data['Date'].dt.date,
                'open': stock_data['Open'],
                'high': stock_data['High'],
                'low': stock_data['Low'],
                'close': stock_data['Close'],
                'volume': stock_data['TradingShares'],  # TPEX 使用 TradingShares 而不是 TradeVolume
                'stock_code': stock_code
            })
            
            # 排序並重置索引
            result_df = result_df.sort_values('date').reset_index(drop=True)
            
            return result_df
            
        except Exception as e:
            logger.error(f"轉換 TPEX DataFrame 時發生錯誤: {e}")
            return pd.DataFrame()
    
    def fetch_missing_stocks(self, existing_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        獲取缺失股票的數據（增量更新）
        
        Args:
            existing_df: 現有的股價數據
            
        Returns:
            包含缺失股票數據的 DataFrame
        """
        if existing_df is None or existing_df.empty:
            logger.info("沒有現有數據，獲取所有股票數據")
            return self.fetch_all_stocks(force_refresh=False)
        
        # 檢查哪些股票缺失數據
        existing_stocks = set(existing_df['stock_code'].astype(str).unique())
        all_stocks = set(self.stock_list)
        missing_stocks = all_stocks - existing_stocks
        
        if not missing_stocks:
            logger.info("所有配置的股票都有數據，無需更新")
            return pd.DataFrame()
        
        logger.info(f"發現 {len(missing_stocks)} 支股票缺失數據: {sorted(missing_stocks)}")
        
        # 計算日期範圍
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # 先下載 TPEX 全表數據（一次性下載，提高效率）
        logger.info("預先下載 TPEX 全表數據以提高效率...")
        tpex_df = self._download_tpex_full_table(start_date, end_date)
        if not tpex_df.empty:
            logger.info(f"TPEX 全表數據準備完成: {len(tpex_df)} 筆記錄")
        
        all_data = []
        failed_stocks = []
        tse_success = 0
        tpex_success = 0
        
        for i, stock_code in enumerate(missing_stocks):
            try:
                logger.info(f"正在獲取缺失股票 {stock_code} ({i+1}/{len(missing_stocks)}) - 進度: {i/len(missing_stocks)*100:.1f}%")
                
                # 先嘗試 TWSE API（上市股票）
                logger.info(f"  嘗試 TWSE API 獲取股票 {stock_code}...")
                tse_data = self._fetch_tse_historical(stock_code, start_date, end_date)
                
                if not tse_data.empty:
                    all_data.append(tse_data)
                    tse_success += 1
                    logger.info(f"  股票 {stock_code} TWSE 獲取成功: {len(tse_data)} 筆數據")
                else:
                    logger.info(f"  TWSE 無法獲取股票 {stock_code}，嘗試 TPEX...")
                    # 如果 TWSE 失敗，嘗試 TPEX API（上櫃股票）
                    tpex_data = self._fetch_tpex_historical(stock_code, start_date, end_date)
                    
                    if not tpex_data.empty:
                        all_data.append(tpex_data)
                        tpex_success += 1
                        logger.info(f"  股票 {stock_code} TPEX 獲取成功: {len(tpex_data)} 筆數據")
                    else:
                        failed_stocks.append(stock_code)
                        logger.warning(f"  股票 {stock_code} TWSE 和 TPEX 都獲取失敗")
                
            except Exception as e:
                failed_stocks.append(stock_code)
                logger.error(f"獲取股票 {stock_code} 時發生錯誤: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['stock_code', 'date']).reset_index(drop=True)
            
            logger.info(f"缺失股票數據獲取完成: 成功 {len(all_data)} 支股票，共 {len(combined_df)} 筆數據")
            logger.info(f"  - TWSE (上市): {tse_success} 支")
            logger.info(f"  - TPEX (上櫃): {tpex_success} 支")
            
            if failed_stocks:
                logger.warning(f"以下股票獲取失敗: {failed_stocks}")
            
            return combined_df
        else:
            logger.error("所有缺失股票數據獲取都失敗了")
            return pd.DataFrame()
    
    def fetch_all_stocks(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        獲取所有股票的數據
        
        Args:
            force_refresh: 是否強制刷新
            
        Returns:
            包含所有股票數據的DataFrame
        """
        logger.info(f"開始使用API獲取 {len(self.stock_list)} 支股票的數據...")
        
        # 計算日期範圍
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # 先下載 TPEX 全表數據（一次性下載，提高效率）
        logger.info("預先下載 TPEX 全表數據以提高效率...")
        tpex_df = self._download_tpex_full_table(start_date, end_date)
        if not tpex_df.empty:
            logger.info(f"TPEX 全表數據準備完成: {len(tpex_df)} 筆記錄")
        
        all_data = []
        failed_stocks = []
        tse_success = 0
        tpex_success = 0
        
        for i, stock_code in enumerate(self.stock_list):
            try:
                logger.info(f"正在獲取股票 {stock_code} ({i+1}/{len(self.stock_list)}) - 進度: {i/len(self.stock_list)*100:.1f}%")
                
                
                # 先嘗試 TWSE API（上市股票）
                logger.info(f"  嘗試 TWSE API 獲取股票 {stock_code}...")
                tse_data = self._fetch_tse_historical(stock_code, start_date, end_date)
                
                if not tse_data.empty:
                    all_data.append(tse_data)
                    tse_success += 1
                    logger.info(f"  股票 {stock_code} TWSE 獲取成功: {len(tse_data)} 筆數據")
                else:
                    logger.info(f"  TWSE 無法獲取股票 {stock_code}，嘗試 TPEX...")
                    # 如果 TWSE 失敗，嘗試 TPEX API（上櫃股票）
                    tpex_data = self._fetch_tpex_historical(stock_code, start_date, end_date)
                    
                    if not tpex_data.empty:
                        all_data.append(tpex_data)
                        tpex_success += 1
                        logger.info(f"  股票 {stock_code} TPEX 獲取成功: {len(tpex_data)} 筆數據")
                    else:
                        failed_stocks.append(stock_code)
                        logger.warning(f"  股票 {stock_code} TWSE 和 TPEX 都獲取失敗")
                
            except Exception as e:
                failed_stocks.append(stock_code)
                logger.error(f"獲取股票 {stock_code} 時發生錯誤: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['stock_code', 'date']).reset_index(drop=True)
            
            logger.info(f"API數據獲取完成: 成功 {len(all_data)} 支股票，共 {len(combined_df)} 筆數據")
            logger.info(f"  - TWSE (上市): {tse_success} 支")
            logger.info(f"  - TPEX (上櫃): {tpex_success} 支")
            
            if failed_stocks:
                logger.warning(f"以下股票獲取失敗: {failed_stocks}")
            
            return combined_df
        else:
            logger.error("所有股票數據獲取都失敗了")
            return pd.DataFrame()
    
    def test_api_connection(self) -> bool:
        """
        測試API連接
        
        Returns:
            連接是否成功
        """
        try:
            # 測試獲取台積電的歷史數據（TWSE）
            logger.info("測試 TWSE API 連接...")
            test_data = self._fetch_tse_historical("2330", datetime.now() - timedelta(days=30), datetime.now())
            if not test_data.empty:
                logger.info("TWSE API 連接測試成功")
                return True
            
            # 測試 TPEX API 連接
            logger.info("測試 TPEX API 連接...")
            tpex_df = self._download_tpex_full_table()
            if not tpex_df.empty:
                logger.info("TPEX API 連接測試成功")
                return True
            else:
                logger.error("TWSE 和 TPEX API 連接測試都失敗")
                return False
                
        except Exception as e:
            logger.error(f"API連接測試時發生錯誤: {e}")
            return False
