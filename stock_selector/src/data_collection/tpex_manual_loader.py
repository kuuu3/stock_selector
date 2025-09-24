"""
櫃買中心手動下載數據處理模組

處理從 https://www.tpex.org.tw/zh-tw/mainboard/trading/info/stock-pricing.html 
手動下載的 CSV 檔案

使用方式：
1. 前往櫃買中心個股日成交資訊頁面
2. 輸入股票代碼（如：3260）
3. 選擇資料年月範圍
4. 點擊「下載CSV檔(UTF-8)」
5. 將下載的檔案放入 data/manual_tpex/ 資料夾
6. 運行此模組處理數據
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class TPEXManualLoader:
    """處理櫃買中心手動下載的歷史數據"""
    
    def __init__(self, manual_data_dir: str = "data/manual_tpex"):
        """
        初始化
        
        Args:
            manual_data_dir: 手動下載數據存放目錄
        """
        self.manual_data_dir = Path(manual_data_dir)
        self.manual_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_manual_csv(self, csv_file: str) -> pd.DataFrame:
        """
        載入手動下載的 CSV 檔案
        
        Args:
            csv_file: CSV 檔案路徑或檔案名
            
        Returns:
            標準化的 DataFrame
        """
        csv_path = Path(csv_file)
        if not csv_path.is_absolute():
            if csv_path.parent.name == 'manual_tpex':
                # 如果已經是相對路徑，直接使用
                csv_path = self.manual_data_dir / csv_path.name
            else:
                # 如果是檔案名，加上目錄
                csv_path = self.manual_data_dir / csv_path
            
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到檔案: {csv_path}")
            
        logger.info(f"載入手動下載的 CSV 檔案: {csv_path}")
        
        try:
            # 櫃買中心 CSV 格式需要跳過前 4 行標題
            # 嘗試不同的編碼方式
            for encoding in ['utf-8', 'big5', 'cp950']:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, skiprows=4)
                    logger.info(f"使用編碼 {encoding} 成功載入，跳過前 4 行標題")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"編碼 {encoding} 失敗: {e}")
                    continue
            else:
                raise ValueError("無法解析 CSV 檔案編碼")
                
            # 標準化數據格式
            df = self._standardize_data(df, csv_path)
            logger.info(f"成功載入 {len(df)} 筆數據")
            
            return df
            
        except Exception as e:
            logger.error(f"載入 CSV 檔案失敗: {e}")
            raise
            
    def _standardize_data(self, df: pd.DataFrame, csv_path: str = None) -> pd.DataFrame:
        """
        標準化數據格式
        
        Args:
            df: 原始 DataFrame
            
        Returns:
            標準化的 DataFrame
        """
        logger.info("開始標準化數據格式...")
        
        # 顯示原始欄位名稱
        logger.info(f"原始欄位: {list(df.columns)}")
        
        # 根據櫃買中心實際 CSV 格式進行欄位映射
        column_mapping = {
            '日 期': 'date',
            '開盤': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盤': 'close',
            '成交仟股': 'volume',
            '成交張數': 'volume',  # 2025年後改為成交張數
            '成交仟元': 'transaction_amount',
            '筆數': 'transaction_count',
            '漲跌': 'price_change'
        }
        
        # 重命名欄位
        df = df.rename(columns=column_mapping)
        
        # 提取股票代碼（從檔案名提取）
        if csv_path:
            stock_code = self._extract_stock_code_from_filename(str(csv_path))
            if stock_code:
                df['stock_code'] = int(stock_code)
                logger.info(f"從檔案名提取股票代碼: {stock_code}")
            else:
                logger.warning("無法從檔案名提取股票代碼")
                # 嘗試從檔案名中的數字部分提取
                import re
                match = re.search(r'(\d{4})', str(csv_path))
                if match:
                    df['stock_code'] = int(match.group(1))
                    logger.info(f"使用正則表達式提取股票代碼: {match.group(1)}")
                else:
                    raise ValueError("無法從檔案名提取股票代碼")
        else:
            raise ValueError("沒有提供檔案路徑，無法提取股票代碼")
        
        # 轉換日期格式（櫃買中心使用民國年格式：113/05/08）
        if 'date' in df.columns:
            def convert_roc_date(date_str):
                """轉換民國年日期格式"""
                if pd.isna(date_str) or date_str == '':
                    return None
                try:
                    # 民國年格式：113/05/08 -> 2024/05/08
                    parts = str(date_str).split('/')
                    if len(parts) == 3:
                        roc_year = int(parts[0])
                        gregorian_year = roc_year + 1911
                        return f"{gregorian_year}/{parts[1]}/{parts[2]}"
                    return date_str
                except:
                    return date_str
            
            df['date'] = df['date'].apply(convert_roc_date)
            df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
            df['date'] = df['date'].dt.date
            
        # 轉換數值欄位
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'transaction_amount', 'transaction_count']
        for col in numeric_columns:
            if col in df.columns:
                # 移除逗號並轉換為數值
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('--', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # 成交量轉換
                if col == 'volume':
                    # 如果是成交張數（2025年後），需要乘以1000轉換為股
                    # 如果是成交仟股（2024年），已經是仟股單位，需要乘以1000轉換為股
                    df[col] = df[col] * 1000
                
        # 確保股票代碼為字符串格式
        if 'stock_code' in df.columns:
            df['stock_code'] = df['stock_code'].astype(str).str.zfill(4)
            
        # 選擇需要的欄位
        required_columns = ['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in required_columns if col in df.columns]
        df = df[available_columns]
        
        # 排序並重置索引
        df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
        
        logger.info(f"標準化完成，最終欄位: {list(df.columns)}")
        return df
        
    def process_all_manual_files(self) -> pd.DataFrame:
        """
        處理所有手動下載的 CSV 檔案
        
        Returns:
            合併後的 DataFrame
        """
        logger.info("開始處理所有手動下載的檔案...")
        
        csv_files = list(self.manual_data_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"在 {self.manual_data_dir} 中沒有找到 CSV 檔案")
            return pd.DataFrame()
            
        all_data = []
        for csv_file in csv_files:
            try:
                logger.info(f"處理檔案: {csv_file.name}")
                df = self.load_manual_csv(csv_file)
                
                # 從檔案名提取股票代碼（如果數據中沒有）
                if 'stock_code' not in df.columns:
                    stock_code = self._extract_stock_code_from_filename(csv_file.name)
                    if stock_code:
                        df['stock_code'] = stock_code
                    else:
                        logger.warning(f"無法從檔案名 {csv_file.name} 提取股票代碼")
                        continue
                        
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"處理檔案 {csv_file.name} 失敗: {e}")
                continue
                
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['stock_code', 'date']).sort_values(['stock_code', 'date'])
            logger.info(f"成功處理 {len(all_data)} 個檔案，合計 {len(combined_df)} 筆數據")
            return combined_df
        else:
            logger.error("沒有成功處理任何檔案")
            return pd.DataFrame()
            
    def _extract_stock_code_from_filename(self, filename: str) -> Optional[str]:
        """
        從檔案名提取股票代碼
        
        Args:
            filename: 檔案名
            
        Returns:
            股票代碼或 None
        """
        import re
        
        # 嘗試提取 4 位數字
        match = re.search(r'(\d{4})', filename)
        if match:
            return match.group(1)
            
        return None
        
    def save_processed_data(self, df: pd.DataFrame, output_file: str = "data/processed/tpex_historical.csv"):
        """
        保存處理後的數據
        
        Args:
            df: 處理後的 DataFrame
            output_file: 輸出檔案路徑
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"已保存處理後的數據到: {output_path}")
        
    def get_available_files(self) -> List[str]:
        """
        獲取可用的手動下載檔案列表
        
        Returns:
            檔案名列表
        """
        csv_files = list(self.manual_data_dir.glob("*.csv"))
        return [f.name for f in csv_files]


def main():
    """主函數 - 處理手動下載的數據"""
    logging.basicConfig(level=logging.INFO)
    
    loader = TPEXManualLoader()
    
    # 檢查可用檔案
    available_files = loader.get_available_files()
    if not available_files:
        print("❌ 在 data/manual_tpex/ 中沒有找到 CSV 檔案")
        print("\n📋 使用說明：")
        print("1. 前往櫃買中心個股日成交資訊頁面")
        print("   https://www.tpex.org.tw/zh-tw/mainboard/trading/info/stock-pricing.html")
        print("2. 輸入股票代碼（如：3260）")
        print("3. 選擇資料年月範圍")
        print("4. 點擊「下載CSV檔(UTF-8)」")
        print("5. 將下載的檔案放入 data/manual_tpex/ 資料夾")
        print("6. 重新運行此腳本")
        return
        
    print(f"✅ 找到 {len(available_files)} 個 CSV 檔案:")
    for file in available_files:
        print(f"  - {file}")
        
    # 處理所有檔案
    print("\n🔄 開始處理檔案...")
    combined_df = loader.process_all_manual_files()
    
    if not combined_df.empty:
        print(f"✅ 成功處理 {len(combined_df)} 筆數據")
        print(f"📊 包含股票: {sorted(combined_df['stock_code'].unique())}")
        
        # 保存處理後的數據
        loader.save_processed_data(combined_df)
        
        # 顯示統計信息
        print("\n📈 數據統計:")
        for stock_code in sorted(combined_df['stock_code'].unique()):
            stock_data = combined_df[combined_df['stock_code'] == stock_code]
            print(f"  股票 {stock_code}: {len(stock_data)} 筆數據")
            if len(stock_data) > 0:
                date_range = f"{stock_data['date'].min()} 到 {stock_data['date'].max()}"
                print(f"    日期範圍: {date_range}")
    else:
        print("❌ 沒有成功處理任何數據")


if __name__ == "__main__":
    main()
