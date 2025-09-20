"""
測試預測功能
"""

import sys
from pathlib import Path
import logging

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """測試導入"""
    try:
        logger.info("測試導入模組...")
        
        from src.data_collection import PriceFetcher
        logger.info("✓ PriceFetcher 導入成功")
        
        from src.preprocessing import FeatureEngineer
        logger.info("✓ FeatureEngineer 導入成功")
        
        from src.models import StockPredictor
        logger.info("✓ StockPredictor 導入成功")
        
        return True
    except Exception as e:
        logger.error(f"導入失敗: {e}")
        return False

def test_models():
    """測試模型載入"""
    try:
        logger.info("測試模型載入...")
        
        from src.models import StockPredictor
        predictor = StockPredictor()
        
        models_dir = Path("outputs/models")
        if not models_dir.exists():
            logger.error("模型目錄不存在")
            return False
        
        logger.info(f"模型目錄存在: {models_dir}")
        
        # 列出所有模型文件
        model_files = list(models_dir.glob("*.joblib"))
        logger.info(f"找到 {len(model_files)} 個模型文件:")
        for f in model_files:
            logger.info(f"  - {f.name}")
        
        return True
    except Exception as e:
        logger.error(f"模型測試失敗: {e}")
        return False

def main():
    """主測試流程"""
    logger.info("=== 開始測試預測功能 ===")
    
    # 測試1: 導入
    if not test_imports():
        logger.error("導入測試失敗")
        return
    
    # 測試2: 模型
    if not test_models():
        logger.error("模型測試失敗")
        return
    
    logger.info("=== 所有測試通過 ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
