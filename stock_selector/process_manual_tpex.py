#!/usr/bin/env python3
"""
櫃買中心手動下載數據處理腳本

使用方式：
1. 前往櫃買中心個股日成交資訊頁面：
   https://www.tpex.org.tw/zh-tw/mainboard/trading/info/stock-pricing.html

2. 下載所需股票的歷史數據：
   - 輸入股票代碼（如：3260, 3324, 5443）
   - 選擇資料年月範圍（建議選擇過去 1-3 個月）
   - 點擊「下載CSV檔(UTF-8)」

3. 將下載的 CSV 檔案放入 data/manual_tpex/ 資料夾

4. 運行此腳本處理數據：
   python process_manual_tpex.py

5. 處理後的數據將整合到主數據庫中
"""

import sys
import logging
from pathlib import Path

# 添加 src 到 Python 路徑
sys.path.insert(0, 'src')

from src.data_collection.tpex_manual_loader import TPEXManualLoader
from src.config import get_data_file_path
import pandas as pd

def main():
    """主函數"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("櫃買中心手動下載數據處理工具")
    print("=" * 60)
    
    # 初始化處理器
    loader = TPEXManualLoader()
    
    # 檢查可用檔案
    available_files = loader.get_available_files()
    
    if not available_files:
        print("\n❌ 沒有找到手動下載的 CSV 檔案")
        print("\n📋 使用說明：")
        print("1. 前往櫃買中心個股日成交資訊頁面：")
        print("   https://www.tpex.org.tw/zh-tw/mainboard/trading/info/stock-pricing.html")
        print("\n2. 下載所需股票的歷史數據：")
        print("   - 輸入股票代碼（如：3260, 3324, 5443）")
        print("   - 選擇資料年月範圍（建議選擇過去 1-3 個月）")
        print("   - 點擊「下載CSV檔(UTF-8)」")
        print("\n3. 將下載的 CSV 檔案放入 data/manual_tpex/ 資料夾")
        print("\n4. 重新運行此腳本：")
        print("   python process_manual_tpex.py")
        return
    
    print(f"\n✅ 找到 {len(available_files)} 個 CSV 檔案：")
    for i, file in enumerate(available_files, 1):
        print(f"   {i}. {file}")
    
    # 處理所有檔案
    print(f"\n🔄 開始處理 {len(available_files)} 個檔案...")
    combined_df = loader.process_all_manual_files()
    
    if combined_df.empty:
        print("❌ 沒有成功處理任何數據")
        return
    
    print(f"\n✅ 成功處理 {len(combined_df)} 筆數據")
    
    # 顯示處理結果
    print(f"\n📊 數據統計：")
    print(f"   總筆數：{len(combined_df)}")
    print(f"   股票數量：{len(combined_df['stock_code'].unique())}")
    # 過濾掉 NaN 日期
    valid_dates = combined_df['date'].dropna()
    if not valid_dates.empty:
        print(f"   日期範圍：{valid_dates.min()} 到 {valid_dates.max()}")
    else:
        print("   日期範圍：無有效日期")
    
    print(f"\n📈 各股票數據：")
    for stock_code in sorted(combined_df['stock_code'].unique()):
        stock_data = combined_df[combined_df['stock_code'] == stock_code]
        print(f"   股票 {stock_code}：{len(stock_data)} 筆數據")
        if len(stock_data) > 0:
            date_range = f"{stock_data['date'].min()} 到 {stock_data['date'].max()}"
            print(f"     日期範圍：{date_range}")
    
    # 保存處理後的數據
    output_file = "data/processed/tpex_historical.csv"
    loader.save_processed_data(combined_df, output_file)
    
    # 整合到主數據庫
    print(f"\n🔄 整合到主數據庫...")
    integrate_with_main_database(combined_df)
    
    print(f"\n🎉 處理完成！")
    print(f"   處理後的數據已保存到：{output_file}")
    print(f"   已整合到主數據庫：data/raw/prices.csv")

def integrate_with_main_database(tpex_df: pd.DataFrame):
    """
    將 TPEX 歷史數據整合到主數據庫
    
    Args:
        tpex_df: TPEX 歷史數據 DataFrame
    """
    try:
        # 載入現有數據
        price_path = get_data_file_path('raw/prices.csv')
        
        if price_path.exists():
            existing_df = pd.read_csv(price_path)
            print(f"   現有數據：{len(existing_df)} 筆")
            
            # 合併數據
            combined_df = pd.concat([existing_df, tpex_df], ignore_index=True)
            
            # 去重（保留最新的數據）
            combined_df = combined_df.drop_duplicates(
                subset=['stock_code', 'date'], 
                keep='last'
            ).sort_values(['stock_code', 'date'])
            
            # 保存合併後的數據
            combined_df.to_csv(price_path, index=False)
            
            print(f"   整合後數據：{len(combined_df)} 筆")
            print(f"   新增數據：{len(combined_df) - len(existing_df)} 筆")
            
        else:
            # 如果沒有現有數據，直接保存 TPEX 數據
            tpex_df.to_csv(price_path, index=False)
            print(f"   創建新的主數據庫：{len(tpex_df)} 筆數據")
            
    except Exception as e:
        print(f"❌ 整合到主數據庫失敗：{e}")
        raise

if __name__ == "__main__":
    main()
