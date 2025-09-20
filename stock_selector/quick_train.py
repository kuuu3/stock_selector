"""
快速訓練腳本 - 使用現有數據進行訓練
"""

import sys
from pathlib import Path

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from train_model import main

if __name__ == "__main__":
    print("🚀 快速訓練模式 - 使用現有數據")
    print("💡 提示: 如果要重新獲取數據，請使用 'python train_model.py --refresh-data'")
    print()
    
    # 使用現有數據進行訓練（不重新獲取數據）
    main(force_refresh_data=False)
