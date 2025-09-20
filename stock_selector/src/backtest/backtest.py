"""
回測模組
模擬交易並分析選股策略的歷史表現
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

from ..config import (
    BACKTEST_CONFIG,
    OUTPUTS_DIR,
    TOP20_OUTPUT_FILE
)

logger = logging.getLogger(__name__)


class Backtester:
    """回測器"""
    
    def __init__(self):
        self.initial_capital = BACKTEST_CONFIG["INITIAL_CAPITAL"]
        self.commission_rate = BACKTEST_CONFIG["COMMISSION_RATE"]
        self.tax_rate = BACKTEST_CONFIG["TAX_RATE"]
        self.rebalance_frequency = BACKTEST_CONFIG["REBALANCE_FREQUENCY"]
        
        self.portfolio = {}
        self.cash = self.initial_capital
        self.transactions = []
        self.portfolio_history = []
        
    def load_price_data(self, file_path: Path) -> pd.DataFrame:
        """
        載入股價數據
        
        Args:
            file_path: 股價數據檔案路徑
            
        Returns:
            股價數據 DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
            
            logger.info(f"載入股價數據: {len(df)} 筆")
            return df
            
        except Exception as e:
            logger.error(f"載入股價數據失敗: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_value(self, prices: pd.DataFrame, date: datetime) -> float:
        """
        計算投資組合價值
        
        Args:
            prices: 股價數據
            date: 計算日期
            
        Returns:
            投資組合總價值
        """
        total_value = self.cash
        
        for stock_code, shares in self.portfolio.items():
            if shares > 0:
                # 獲取當日股價
                stock_price = prices[
                    (prices['stock_code'] == stock_code) & 
                    (prices['date'] == date)
                ]
                
                if not stock_price.empty:
                    price = stock_price['close'].iloc[0]
                    total_value += shares * price
        
        return total_value
    
    def rebalance_portfolio(self, selected_stocks: pd.DataFrame, 
                          prices: pd.DataFrame, date: datetime):
        """
        重新平衡投資組合
        
        Args:
            selected_stocks: 選中的股票 DataFrame
            prices: 股價數據
            date: 重新平衡日期
        """
        # 計算當前投資組合價值
        current_value = self.calculate_portfolio_value(prices, date)
        
        # 清空當前持倉
        for stock_code in list(self.portfolio.keys()):
            if self.portfolio[stock_code] > 0:
                self._sell_stock(stock_code, self.portfolio[stock_code], prices, date)
        
        # 等權重分配
        num_stocks = len(selected_stocks)
        if num_stocks == 0:
            return
        
        weight_per_stock = 1.0 / num_stocks
        target_value_per_stock = current_value * weight_per_stock
        
        # 買入新選中的股票
        for _, stock in selected_stocks.iterrows():
            stock_code = stock['stock_code']
            
            # 獲取當日股價
            stock_price = prices[
                (prices['stock_code'] == stock_code) & 
                (prices['date'] == date)
            ]
            
            if not stock_price.empty:
                price = stock_price['close'].iloc[0]
                shares = int(target_value_per_stock / price)
                
                if shares > 0:
                    self._buy_stock(stock_code, shares, price, date)
    
    def _buy_stock(self, stock_code: str, shares: int, price: float, date: datetime):
        """
        買入股票
        
        Args:
            stock_code: 股票代碼
            shares: 股數
            price: 股價
            date: 交易日期
        """
        cost = shares * price
        commission = cost * self.commission_rate
        
        if self.cash >= cost + commission:
            self.cash -= (cost + commission)
            
            if stock_code not in self.portfolio:
                self.portfolio[stock_code] = 0
            self.portfolio[stock_code] += shares
            
            # 記錄交易
            self.transactions.append({
                'date': date,
                'stock_code': stock_code,
                'action': 'buy',
                'shares': shares,
                'price': price,
                'cost': cost,
                'commission': commission
            })
            
            logger.debug(f"買入 {stock_code}: {shares} 股 @ {price}")
    
    def _sell_stock(self, stock_code: str, shares: int, prices: pd.DataFrame, date: datetime):
        """
        賣出股票
        
        Args:
            stock_code: 股票代碼
            shares: 股數
            prices: 股價數據
            date: 交易日期
        """
        if stock_code not in self.portfolio or self.portfolio[stock_code] <= 0:
            return
        
        # 獲取當日股價
        stock_price = prices[
            (prices['stock_code'] == stock_code) & 
            (prices['date'] == date)
        ]
        
        if not stock_price.empty:
            price = stock_price['close'].iloc[0]
            proceeds = shares * price
            commission = proceeds * self.commission_rate
            tax = proceeds * self.tax_rate
            
            self.cash += (proceeds - commission - tax)
            self.portfolio[stock_code] -= shares
            
            # 記錄交易
            self.transactions.append({
                'date': date,
                'stock_code': stock_code,
                'action': 'sell',
                'shares': shares,
                'price': price,
                'proceeds': proceeds,
                'commission': commission,
                'tax': tax
            })
            
            logger.debug(f"賣出 {stock_code}: {shares} 股 @ {price}")
    
    def run_backtest(self, prices: pd.DataFrame, 
                    selection_strategy: callable = None) -> Dict[str, Any]:
        """
        運行回測
        
        Args:
            prices: 股價數據
            selection_strategy: 選股策略函數
            
        Returns:
            回測結果字典
        """
        logger.info("開始運行回測...")
        
        # 初始化
        self.portfolio = {}
        self.cash = self.initial_capital
        self.transactions = []
        self.portfolio_history = []
        
        # 獲取所有交易日
        trading_days = sorted(prices['date'].unique())
        
        # 模擬每日交易
        for i, date in enumerate(trading_days):
            # 計算投資組合價值
            portfolio_value = self.calculate_portfolio_value(prices, date)
            
            # 記錄投資組合歷史
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'num_positions': len([s for s in self.portfolio.values() if s > 0])
            })
            
            # 重新平衡（每週或每月）
            if self._should_rebalance(date, i):
                if selection_strategy:
                    # 使用選股策略
                    selected_stocks = selection_strategy(prices, date)
                    self.rebalance_portfolio(selected_stocks, prices, date)
                else:
                    # 簡單策略：持有所有股票
                    all_stocks = prices[prices['date'] == date][['stock_code']].drop_duplicates()
                    self.rebalance_portfolio(all_stocks, prices, date)
        
        # 計算回測結果
        results = self._calculate_results(prices)
        
        logger.info("回測完成")
        return results
    
    def _should_rebalance(self, date: datetime, day_index: int) -> bool:
        """
        判斷是否應該重新平衡
        
        Args:
            date: 當前日期
            day_index: 日期索引
            
        Returns:
            是否應該重新平衡
        """
        if self.rebalance_frequency == "daily":
            return True
        elif self.rebalance_frequency == "weekly":
            return day_index % 5 == 0  # 每5個交易日
        elif self.rebalance_frequency == "monthly":
            return day_index % 20 == 0  # 每20個交易日
        else:
            return day_index % 5 == 0  # 預設每週
    
    def _calculate_results(self, prices: pd.DataFrame) -> Dict[str, Any]:
        """
        計算回測結果
        
        Args:
            prices: 股價數據
            
        Returns:
            回測結果字典
        """
        if not self.portfolio_history:
            return {}
        
        # 轉換為 DataFrame
        history_df = pd.DataFrame(self.portfolio_history)
        history_df = history_df.sort_values('date')
        
        # 計算基本指標
        initial_value = history_df['portfolio_value'].iloc[0]
        final_value = history_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # 計算年化報酬率
        days = (history_df['date'].iloc[-1] - history_df['date'].iloc[0]).days
        years = days / 365.25
        annualized_return = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else 0
        
        # 計算最大回撤
        history_df['cumulative_max'] = history_df['portfolio_value'].cummax()
        history_df['drawdown'] = (history_df['portfolio_value'] - history_df['cumulative_max']) / history_df['cumulative_max']
        max_drawdown = history_df['drawdown'].min()
        
        # 計算夏普比率（假設無風險利率為2%）
        risk_free_rate = 0.02
        returns = history_df['portfolio_value'].pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252  # 日無風險利率
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 計算交易統計
        buy_transactions = [t for t in self.transactions if t['action'] == 'buy']
        sell_transactions = [t for t in self.transactions if t['action'] == 'sell']
        
        total_commission = sum(t.get('commission', 0) for t in self.transactions)
        total_tax = sum(t.get('tax', 0) for t in self.transactions)
        
        results = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.transactions),
            'buy_trades': len(buy_transactions),
            'sell_trades': len(sell_transactions),
            'total_commission': total_commission,
            'total_tax': total_tax,
            'portfolio_history': history_df,
            'transactions': self.transactions
        }
        
        logger.info(f"回測結果:")
        logger.info(f"  初始資金: {initial_value:,.0f}")
        logger.info(f"  最終價值: {final_value:,.0f}")
        logger.info(f"  總報酬率: {total_return:.2%}")
        logger.info(f"  年化報酬率: {annualized_return:.2%}")
        logger.info(f"  最大回撤: {max_drawdown:.2%}")
        logger.info(f"  夏普比率: {sharpe_ratio:.2f}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = "backtest_report.csv"):
        """
        保存回測結果
        
        Args:
            results: 回測結果
            filename: 檔案名稱
        """
        if not results:
            logger.warning("沒有回測結果可保存")
            return
        
        # 保存投資組合歷史
        if 'portfolio_history' in results:
            output_file = OUTPUTS_DIR / filename
            results['portfolio_history'].to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"回測結果已保存到: {output_file}")
        
        # 保存交易記錄
        if results.get('transactions'):
            transactions_file = OUTPUTS_DIR / "transactions.csv"
            transactions_df = pd.DataFrame(results['transactions'])
            transactions_df.to_csv(transactions_file, index=False, encoding='utf-8-sig')
            logger.info(f"交易記錄已保存到: {transactions_file}")


def simple_selection_strategy(prices: pd.DataFrame, date: datetime) -> pd.DataFrame:
    """
    簡單選股策略：選擇前20支股票
    
    Args:
        prices: 股價數據
        date: 選股日期
        
    Returns:
        選中的股票 DataFrame
    """
    # 獲取當日所有股票
    daily_prices = prices[prices['date'] == date]
    
    if daily_prices.empty:
        return pd.DataFrame()
    
    # 簡單策略：按成交量排序，選擇前20支
    top_stocks = daily_prices.nlargest(20, 'volume')
    
    return top_stocks[['stock_code']].drop_duplicates()


def main():
    """主函數 - 用於測試回測功能"""
    backtester = Backtester()
    
    # 載入股價數據
    prices_file = Path("data/raw/prices.csv")
    if not prices_file.exists():
        logger.error("找不到股價數據檔案")
        return
    
    prices = backtester.load_price_data(prices_file)
    if prices.empty:
        logger.error("無法載入股價數據")
        return
    
    # 運行回測
    results = backtester.run_backtest(prices, simple_selection_strategy)
    
    if results:
        # 保存結果
        backtester.save_results(results)
        
        logger.info("回測測試完成")


if __name__ == "__main__":
    main()
