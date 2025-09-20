"""
檢查數據可用性和日期範圍
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_collection import PriceFetcher

def check_data_availability():
    """檢查數據可用性"""
    print("=== 數據可用性檢查 ===")
    
    # 檢查現有數據
    price_file = Path("data/raw/prices.csv")
    if price_file.exists():
        df = pd.read_csv(price_file)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"📁 現有數據文件: {price_file}")
        print(f"📊 數據筆數: {len(df)}")
        print(f"📅 數據日期範圍:")
        print(f"   最早: {df['date'].min()}")
        print(f"   最晚: {df['date'].max()}")
        print(f"🏢 股票數量: {df['stock_code'].nunique()}")
        print(f"📈 股票清單: {sorted(df['stock_code'].unique())}")
        
        # 檢查最新日期的數據
        latest_date = df['date'].max()
        latest_data = df[df['date'] == latest_date]
        print(f"\n📅 最新交易日 ({latest_date.strftime('%Y-%m-%d')}) 數據:")
        for _, row in latest_data.iterrows():
            print(f"   {row['stock_code']}: 收盤價 {row['close']:.2f}")
        
        # 計算數據年齡
        days_old = (datetime.now() - latest_date).days
        print(f"\n⏰ 數據年齡: {days_old} 天前")
        
        if days_old > 7:
            print("⚠️  警告: 數據較舊，建議更新")
        else:
            print("✅ 數據相對較新")
            
    else:
        print("❌ 沒有找到現有數據文件")
    
    print("\n=== 嘗試獲取最新數據 ===")
    
    # 嘗試獲取最新數據
    try:
        price_fetcher = PriceFetcher()
        new_df = price_fetcher.fetch_all_stocks(save_to_file=False)
        
        if not new_df.empty:
            new_df['date'] = pd.to_datetime(new_df['date'])
            latest_new = new_df['date'].max()
            print(f"✅ 成功獲取新數據")
            print(f"📅 最新數據日期: {latest_new.strftime('%Y-%m-%d')}")
            print(f"📊 新數據筆數: {len(new_df)}")
            
            # 比較數據新舊
            if price_file.exists():
                old_latest = pd.to_datetime(df['date']).max()
                if latest_new > old_latest:
                    print("🆕 發現更新的數據！")
                else:
                    print("📅 數據日期相同，無需更新")
        else:
            print("❌ 無法獲取新數據")
            
    except Exception as e:
        print(f"❌ 獲取數據時發生錯誤: {e}")

def suggest_alternatives():
    """建議替代方案"""
    print("\n=== 建議替代方案 ===")
    print("1. 📈 使用現有數據進行預測 (2024-05-31)")
    print("2. 🔄 等待下一個交易日更新數據")
    print("3. 💰 考慮付費API獲取實時數據")
    print("4. 🧪 使用模擬數據測試模型")
    print("5. 📊 手動輸入最新股價數據")

if __name__ == "__main__":
    check_data_availability()
    suggest_alternatives()
