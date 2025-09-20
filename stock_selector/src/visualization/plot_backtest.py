"""
回測績效視覺化模組
生成回測結果的圖表
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from typing import Dict, Any, List
from pathlib import Path

from ..config import OUTPUTS_DIR

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """回測績效視覺化器"""
    
    def __init__(self):
        self.colors = {
            'portfolio': '#2E8B57',      # 綠色 - 投資組合
            'benchmark': '#DC143C',      # 紅色 - 基準
            'drawdown': '#FF6347',       # 番茄紅 - 回撤
            'returns': '#4682B4'         # 鋼藍 - 報酬
        }
    
    def load_backtest_data(self, file_path: Path = None) -> pd.DataFrame:
        """
        載入回測數據
        
        Args:
            file_path: 回測數據檔案路徑
            
        Returns:
            回測數據 DataFrame
        """
        if file_path is None:
            file_path = OUTPUTS_DIR / "backtest_report.csv"
        
        if not file_path.exists():
            logger.warning(f"找不到回測數據檔案: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"載入回測數據: {len(df)} 筆記錄")
            return df
        except Exception as e:
            logger.error(f"載入回測數據失敗: {e}")
            return pd.DataFrame()
    
    def plot_portfolio_value(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
        """
        繪製投資組合價值曲線
        
        Args:
            df: 投資組合數據
            benchmark_df: 基準數據（可選）
        """
        if df.empty:
            logger.warning("沒有投資組合數據可視覺化")
            return
        
        plt.figure(figsize=(14, 8))
        
        # 繪製投資組合價值曲線
        plt.plot(df['date'], df['portfolio_value'], 
                color=self.colors['portfolio'], linewidth=2, 
                label='投資組合價值')
        
        # 繪製基準曲線（如果有）
        if benchmark_df is not None and not benchmark_df.empty:
            plt.plot(benchmark_df['date'], benchmark_df['value'], 
                    color=self.colors['benchmark'], linewidth=2, 
                    label='基準指數', linestyle='--')
        
        # 設置圖表
        plt.title('投資組合價值曲線', fontsize=16, fontweight='bold')
        plt.xlabel('日期')
        plt.ylabel('投資組合價值 (元)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 格式化Y軸
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # 旋轉X軸標籤
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = OUTPUTS_DIR / "portfolio_value.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"投資組合價值曲線已保存到: {output_file}")
    
    def plot_drawdown(self, df: pd.DataFrame):
        """
        繪製回撤曲線
        
        Args:
            df: 投資組合數據
        """
        if df.empty:
            logger.warning("沒有投資組合數據可視覺化")
            return
        
        # 計算回撤
        df = df.copy()
        df['cumulative_max'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['cumulative_max']) / df['cumulative_max'] * 100
        
        plt.figure(figsize=(14, 6))
        
        # 填充回撤區域
        plt.fill_between(df['date'], df['drawdown'], 0, 
                        color=self.colors['drawdown'], alpha=0.3, label='回撤')
        
        # 繪製回撤線
        plt.plot(df['date'], df['drawdown'], 
                color=self.colors['drawdown'], linewidth=1)
        
        # 設置圖表
        plt.title('投資組合回撤曲線', fontsize=16, fontweight='bold')
        plt.xlabel('日期')
        plt.ylabel('回撤 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加零線
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 旋轉X軸標籤
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = OUTPUTS_DIR / "drawdown_curve.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"回撤曲線已保存到: {output_file}")
    
    def plot_monthly_returns(self, df: pd.DataFrame):
        """
        繪製月度報酬率熱力圖
        
        Args:
            df: 投資組合數據
        """
        if df.empty:
            logger.warning("沒有投資組合數據可視覺化")
            return
        
        # 計算月度報酬率
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # 按月度分組計算報酬率
        monthly_data = df.groupby(['year', 'month'])['portfolio_value'].agg(['first', 'last'])
        monthly_returns = (monthly_data['last'] / monthly_data['first'] - 1) * 100
        
        # 創建月度報酬率矩陣
        returns_matrix = monthly_returns.unstack(level=1)
        
        plt.figure(figsize=(12, 8))
        
        # 繪製熱力圖
        sns.heatmap(returns_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': '月度報酬率 (%)'})
        
        plt.title('月度報酬率熱力圖', fontsize=16, fontweight='bold')
        plt.xlabel('月份')
        plt.ylabel('年份')
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = OUTPUTS_DIR / "monthly_returns_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"月度報酬率熱力圖已保存到: {output_file}")
    
    def plot_rolling_metrics(self, df: pd.DataFrame, window: int = 30):
        """
        繪製滾動指標
        
        Args:
            df: 投資組合數據
            window: 滾動窗口大小（天）
        """
        if df.empty:
            logger.warning("沒有投資組合數據可視覺化")
            return
        
        # 計算滾動指標
        df = df.copy()
        df['returns'] = df['portfolio_value'].pct_change()
        df['rolling_volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(252) * 100
        df['rolling_sharpe'] = df['returns'].rolling(window=window).mean() / df['returns'].rolling(window=window).std() * np.sqrt(252)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 滾動波動率
        ax1.plot(df['date'], df['rolling_volatility'], 
                color=self.colors['portfolio'], linewidth=2)
        ax1.set_title(f'滾動波動率 ({window}天窗口)', fontsize=14)
        ax1.set_ylabel('年化波動率 (%)')
        ax1.grid(True, alpha=0.3)
        
        # 滾動夏普比率
        ax2.plot(df['date'], df['rolling_sharpe'], 
                color=self.colors['benchmark'], linewidth=2)
        ax2.set_title(f'滾動夏普比率 ({window}天窗口)', fontsize=14)
        ax2.set_ylabel('夏普比率')
        ax2.set_xlabel('日期')
        ax2.grid(True, alpha=0.3)
        
        # 旋轉X軸標籤
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = OUTPUTS_DIR / "rolling_metrics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"滾動指標圖表已保存到: {output_file}")
    
    def plot_performance_metrics(self, results: Dict[str, Any]):
        """
        繪製績效指標儀表板
        
        Args:
            results: 回測結果字典
        """
        if not results:
            logger.warning("沒有回測結果可視覺化")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('回測績效指標儀表板', fontsize=16, fontweight='bold')
        
        # 1. 關鍵指標
        metrics = {
            '總報酬率': results.get('total_return', 0) * 100,
            '年化報酬率': results.get('annualized_return', 0) * 100,
            '最大回撤': results.get('max_drawdown', 0) * 100,
            '夏普比率': results.get('sharpe_ratio', 0)
        }
        
        bars = ax1.bar(metrics.keys(), metrics.values(), 
                      color=['#2E8B57', '#4682B4', '#DC143C', '#FF8C00'])
        ax1.set_title('關鍵績效指標')
        ax1.set_ylabel('數值')
        
        # 添加數值標籤
        for bar, value in zip(bars, metrics.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 2. 交易統計
        trade_metrics = {
            '總交易次數': results.get('total_trades', 0),
            '買入次數': results.get('buy_trades', 0),
            '賣出次數': results.get('sell_trades', 0)
        }
        
        ax2.pie(trade_metrics.values(), labels=trade_metrics.keys(), 
               autopct='%1.0f', startangle=90, colors=['#FFB6C1', '#87CEEB', '#DDA0DD'])
        ax2.set_title('交易統計')
        
        # 3. 成本分析
        cost_data = {
            '手續費': results.get('total_commission', 0),
            '證交稅': results.get('total_tax', 0)
        }
        
        ax3.bar(cost_data.keys(), cost_data.values(), 
               color=['#FF6347', '#4682B4'])
        ax3.set_title('交易成本')
        ax3.set_ylabel('金額 (元)')
        
        # 4. 持倉分布
        if 'portfolio_history' in results and not results['portfolio_history'].empty:
            portfolio_df = results['portfolio_history']
            ax4.plot(portfolio_df['date'], portfolio_df['num_positions'], 
                    color='#2E8B57', linewidth=2)
            ax4.set_title('持倉數量變化')
            ax4.set_ylabel('持倉數量')
            ax4.set_xlabel('日期')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = OUTPUTS_DIR / "performance_dashboard.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"績效指標儀表板已保存到: {output_file}")
    
    def generate_all_plots(self, backtest_file: Path = None, results: Dict[str, Any] = None):
        """
        生成所有回測圖表
        
        Args:
            backtest_file: 回測數據檔案路徑
            results: 回測結果字典
        """
        logger.info("開始生成所有回測視覺化圖表...")
        
        # 載入回測數據
        df = self.load_backtest_data(backtest_file)
        
        if not df.empty:
            # 生成各種圖表
            self.plot_portfolio_value(df)
            self.plot_drawdown(df)
            self.plot_monthly_returns(df)
            self.plot_rolling_metrics(df)
        
        # 生成績效指標儀表板
        if results:
            self.plot_performance_metrics(results)
        
        logger.info("所有回測視覺化圖表生成完成")


def main():
    """主函數 - 用於測試回測視覺化功能"""
    visualizer = BacktestVisualizer()
    
    # 創建測試數據
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = []
    
    initial_value = 1000000
    for i, date in enumerate(dates):
        # 模擬投資組合價值變化
        value = initial_value * (1 + np.random.normal(0.001, 0.02)) ** i
        test_data.append({
            'date': date,
            'portfolio_value': value,
            'cash': value * 0.1,
            'num_positions': np.random.randint(15, 25)
        })
    
    test_df = pd.DataFrame(test_data)
    
    # 保存測試數據
    test_file = OUTPUTS_DIR / "test_backtest_report.csv"
    test_df.to_csv(test_file, index=False)
    
    # 生成所有圖表
    visualizer.generate_all_plots(test_file)
    
    logger.info("回測視覺化測試完成")


if __name__ == "__main__":
    main()
