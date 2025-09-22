"""
回測腳本 - 運行股票選股策略回測
"""

import sys
from pathlib import Path
import logging

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.backtest.backtest import Backtester

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主回測流程"""
    import argparse
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='股票選股策略回測')
    parser.add_argument('--start-date', type=str, help='回測開始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='回測結束日期 (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=1000000, help='初始資金 (台幣)')
    parser.add_argument('--top-n', type=int, default=10, help='選股數量')
    parser.add_argument('--rebalance-days', type=int, default=10, help='再平衡間隔 (天)')
    args = parser.parse_args()
    
    logger.info("=== 開始股票選股策略回測 ===")
    
    try:
        # 初始化回測器
        backtester = Backtester(initial_capital=args.capital)
        
        # 運行回測
        performance = backtester.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            top_n=args.top_n,
            rebalance_days=args.rebalance_days
        )
        
        # 生成並顯示報告
        report = backtester.generate_report(performance)
        print(report)
        
        # 保存結果
        output_dir = Path("outputs/backtest")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存績效數據
        import pandas as pd
        performance_df = pd.DataFrame(performance['portfolio_values'])
        performance_df.to_csv(output_dir / "portfolio_performance.csv", index=False)
        
        # 保存交易歷史
        if performance['trade_history']:
            trades_df = pd.DataFrame(performance['trade_history'])
            trades_df.to_csv(output_dir / "trade_history.csv", index=False)
        
        # 保存報告
        with open(output_dir / "backtest_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"回測結果已保存到: {output_dir}")
        
        # 顯示關鍵指標
        logger.info("=== 關鍵績效指標 ===")
        logger.info(f"總報酬率: {performance['total_return']:.2%}")
        logger.info(f"年化報酬率: {performance['annualized_return']:.2%}")
        logger.info(f"超額報酬: {performance['excess_return']:.2%}")
        logger.info(f"夏普比率: {performance['sharpe_ratio']:.3f}")
        logger.info(f"最大回撤: {performance['max_drawdown']:.2%}")
        logger.info(f"勝率: {performance['win_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"回測過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序執行失敗: {e}")
        import traceback
        traceback.print_exc()
