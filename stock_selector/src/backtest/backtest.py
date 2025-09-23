"""
回測系統 - 測試股票選股策略的歷史表現
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.preprocessing import FeatureEngineer
from src.models.predict import StockPredictor
from src.config import BACKTEST_CONFIG, get_data_file_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Backtester:
    """回測器 - 測試股票選股策略的歷史表現"""
    
    def __init__(self, initial_capital: float = None):
        """
        初始化回測器
        
        Args:
            initial_capital: 初始資金（台幣）
        """
        self.initial_capital = initial_capital or BACKTEST_CONFIG["INITIAL_CAPITAL"]
        self.current_capital = self.initial_capital
        self.positions = {}  # 當前持倉
        self.trade_history = []  # 交易歷史
        self.portfolio_values = []  # 投資組合價值歷史
        self.benchmark_values = []  # 基準（買入持有）價值歷史
        
        # 交易成本
        self.commission_rate = BACKTEST_CONFIG["COMMISSION_RATE"]
        self.tax_rate = BACKTEST_CONFIG["TAX_RATE"]
        
        # 再平衡頻率
        self.rebalance_frequency = BACKTEST_CONFIG["REBALANCE_FREQUENCY"]
        
        logger.info(f"初始化回測器 - 初始資金: {self.initial_capital:,.0f} 台幣")
    
    def load_data(self, price_file: str = None) -> pd.DataFrame:
        """載入股價數據"""
        if price_file is None:
            price_path = get_data_file_path("raw/prices.csv")
        else:
            price_path = Path(price_file)
        
        if not price_path.exists():
            raise FileNotFoundError(f"找不到股價數據文件: {price_path}")
        
        df = pd.read_csv(price_path)
        df['date'] = pd.to_datetime(df['date'])
        df['stock_code'] = df['stock_code'].astype(str)  # 確保 stock_code 是字符串類型
        df = df.sort_values(['date', 'stock_code']).reset_index(drop=True)
        
        logger.info(f"載入股價數據: {len(df)} 筆，日期範圍: {df['date'].min()} 到 {df['date'].max()}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """準備特徵數據"""
        logger.info("進行特徵工程...")
        
        # 嘗試載入新聞數據
        news_df = None
        try:
            news_csv_path = get_data_file_path("processed/news_with_sentiment.csv")
            if news_csv_path.exists():
                news_df = pd.read_csv(news_csv_path)
                news_df['analyzed_time'] = pd.to_datetime(news_df['analyzed_time'])
                logger.info(f"載入新聞數據: {len(news_df)} 筆")
            else:
                logger.info("找不到新聞情感分析數據，將只使用股價數據")
        except Exception as e:
            logger.warning(f"載入新聞數據時發生錯誤: {e}")
            news_df = None
        
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(df, news_df)
        
        if features_df.empty:
            raise ValueError("特徵工程失敗")
        
        # 創建標籤數據
        labels_df = feature_engineer.create_labels(df)
        
        # 合併特徵和標籤，避免重複列名
        features_df = features_df.merge(labels_df, on=['stock_code', 'date'], how='left', suffixes=('', '_label'))
        
        logger.info(f"特徵工程完成: {len(features_df)} 個樣本，{len(features_df.columns)} 個特徵")
        return features_df
    
    def get_predictions(self, features_df: pd.DataFrame, predictor: StockPredictor) -> pd.DataFrame:
        """獲取預測結果"""
        logger.info("進行模型預測...")
        
        # 載入訓練時使用的特徵列名
        try:
            import joblib
            feature_columns_path = Path("outputs/models/feature_columns.pkl")
            if feature_columns_path.exists():
                feature_columns = joblib.load(feature_columns_path)
                logger.info(f"載入訓練時特徵列名: {len(feature_columns)} 個")
            else:
                raise FileNotFoundError("找不到特徵列名文件")
        except Exception as e:
            logger.warning(f"載入特徵列名失敗: {e}，使用預設邏輯")
            feature_columns = [col for col in features_df.columns 
                              if col not in ['stock_code', 'date', 'future_return_1w', 'future_return_1m', 'label_1w', 'label_1m'] and 
                              features_df[col].dtype in ['float64', 'int64']]
        
        # 檢查特徵是否存在
        missing_features = [col for col in feature_columns if col not in features_df.columns]
        if missing_features:
            logger.warning(f"缺少特徵: {missing_features}")
            # 只使用存在的特徵
            feature_columns = [col for col in feature_columns if col in features_df.columns]
        
        prediction_data = features_df[feature_columns].values
        
        # 進行預測
        classification_results = predictor.predict_classification(prediction_data)
        regression_results = predictor.predict_regression(prediction_data)
        
        # 創建預測結果DataFrame
        results_df = features_df[['date', 'stock_code']].copy()
        
        # 添加分類預測結果
        if 'logistic_regression' in classification_results:
            lr_probs = classification_results['logistic_regression']['probabilities']
            results_df['lr_prob'] = lr_probs[:, 2]  # 取上漲概率
        
        if 'xgboost_classifier' in classification_results:
            xgb_probs = classification_results['xgboost_classifier']['probabilities']
            results_df['xgb_prob'] = xgb_probs[:, 2]  # 取上漲概率
        
        # 添加回歸預測結果
        if 'xgboost_regressor' in regression_results:
            results_df['xgb_return'] = regression_results['xgboost_regressor']['predictions']
        
        # 計算綜合評分
        score_components = []
        if 'lr_prob' in results_df.columns:
            score_components.append(results_df['lr_prob'] * 0.3)
        if 'xgb_prob' in results_df.columns:
            score_components.append(results_df['xgb_prob'] * 0.5)
        if 'xgb_return' in results_df.columns:
            score_components.append(results_df['xgb_return'] * 100 * 0.2)
        
        if score_components:
            results_df['composite_score'] = sum(score_components)
        else:
            results_df['composite_score'] = 0
        
        logger.info("預測完成")
        return results_df
    
    def select_stocks(self, predictions_df: pd.DataFrame, date: pd.Timestamp, top_n: int = 20) -> List[str]:
        """選擇股票"""
        # 獲取指定日期的預測結果
        date_predictions = predictions_df[predictions_df['date'] == date].copy()
        
        if date_predictions.empty:
            return []
        
        # 按綜合評分排序，選擇Top N
        top_stocks = date_predictions.nlargest(top_n, 'composite_score')['stock_code'].tolist()
        
        return top_stocks
    
    def calculate_position_size(self, stock_code: str, price: float, target_weight: float) -> int:
        """計算持倉數量（以張為單位，1張=1000股）"""
        target_value = self.current_capital * target_weight
        shares = int(target_value / price)
        # 台股交易以張為單位，1張=1000股
        shares = (shares // 1000) * 1000  # 向下取整到最接近的1000股
        return shares
    
    def execute_trade(self, stock_code: str, shares: int, price: float, action: str) -> Dict:
        """執行交易"""
        if action == 'buy':
            cost = shares * price
            commission = cost * self.commission_rate
            total_cost = cost + commission
            
            if total_cost <= self.current_capital:
                self.current_capital -= total_cost
                if stock_code in self.positions:
                    self.positions[stock_code] += shares
                else:
                    self.positions[stock_code] = shares
                
                trade = {
                    'date': datetime.now(),
                    'stock_code': stock_code,
                    'action': 'buy',
                    'shares': shares,
                    'price': price,
                    'cost': total_cost,
                    'commission': commission
                }
                self.trade_history.append(trade)
                return trade
        
        elif action == 'sell':
            if stock_code in self.positions and self.positions[stock_code] >= shares:
                proceeds = shares * price
                commission = proceeds * self.commission_rate
                tax = proceeds * self.tax_rate
                net_proceeds = proceeds - commission - tax
                
                self.current_capital += net_proceeds
                self.positions[stock_code] -= shares
                
                if self.positions[stock_code] == 0:
                    del self.positions[stock_code]
                
                trade = {
                    'date': datetime.now(),
                    'stock_code': stock_code,
                    'action': 'sell',
                    'shares': shares,
                    'price': price,
                    'proceeds': proceeds,
                    'commission': commission,
                    'tax': tax,
                    'net_proceeds': net_proceeds
                }
                self.trade_history.append(trade)
                return trade
        
        return None
    
    def rebalance_portfolio(self, target_stocks: List[str], prices_df: pd.DataFrame, date: pd.Timestamp):
        """再平衡投資組合"""
        logger.info(f"再平衡投資組合 - 目標股票: {target_stocks}")
        
        # 計算目標權重（等權重）
        target_weight = 1.0 / len(target_stocks) if target_stocks else 0
        
        # 賣出不在目標清單中的股票
        current_stocks = list(self.positions.keys())
        for stock_code in current_stocks:
            if stock_code not in target_stocks:
                # 獲取當前價格
                stock_data = prices_df[(prices_df['date'] == date) & 
                                     (prices_df['stock_code'] == stock_code)]
                if stock_data.empty:
                    logger.warning(f"找不到股票 {stock_code} 在 {date} 的價格數據")
                    continue
                
                stock_price = stock_data['close'].iloc[0]
                
                shares_to_sell = self.positions[stock_code]
                self.execute_trade(stock_code, shares_to_sell, stock_price, 'sell')
                logger.info(f"賣出 {stock_code}: {shares_to_sell} 股 @ {stock_price:.2f}")
        
        # 買入目標股票
        for stock_code in target_stocks:
            # 獲取當前價格
            stock_data = prices_df[(prices_df['date'] == date) & 
                                 (prices_df['stock_code'] == stock_code)]
            if stock_data.empty:
                logger.warning(f"找不到股票 {stock_code} 在 {date} 的價格數據")
                continue
            
            stock_price = stock_data['close'].iloc[0]
            
            # 計算目標持倉數量
            target_shares = self.calculate_position_size(stock_code, stock_price, target_weight)
            
            # 計算需要買入的數量
            current_shares = self.positions.get(stock_code, 0)
            shares_to_buy = max(0, target_shares - current_shares)
            
            if shares_to_buy > 0:
                self.execute_trade(stock_code, shares_to_buy, stock_price, 'buy')
                logger.info(f"買入 {stock_code}: {shares_to_buy} 股 @ {stock_price:.2f}")
    
    def calculate_portfolio_value(self, prices_df: pd.DataFrame, date: pd.Timestamp) -> float:
        """計算投資組合價值"""
        portfolio_value = self.current_capital
        
        for stock_code, shares in self.positions.items():
            stock_data = prices_df[(prices_df['date'] == date) & 
                                 (prices_df['stock_code'] == stock_code)]
            if not stock_data.empty:
                stock_price = stock_data['close'].iloc[0]
                portfolio_value += shares * stock_price
        
        return portfolio_value
    
    def calculate_benchmark_value(self, prices_df: pd.DataFrame, start_date: pd.Timestamp, 
                                end_date: pd.Timestamp, benchmark_stocks: List[str] = None) -> float:
        """計算基準價值（買入持有策略）"""
        if benchmark_stocks is None:
            # 使用所有股票的平均表現作為基準
            all_stocks = prices_df['stock_code'].unique()
            benchmark_stocks = all_stocks[:10]  # 取前10支股票
        
        # 計算基準組合的總價值
        benchmark_value = 0
        weight = 1.0 / len(benchmark_stocks)
        
        for stock_code in benchmark_stocks:
            start_data = prices_df[(prices_df['date'] == start_date) & 
                                 (prices_df['stock_code'] == stock_code)]
            end_data = prices_df[(prices_df['date'] == end_date) & 
                               (prices_df['stock_code'] == stock_code)]
            
            if start_data.empty or end_data.empty:
                continue
            
            start_price = start_data['close'].iloc[0]
            end_price = end_data['close'].iloc[0]
            
            shares = (self.initial_capital * weight) / start_price
            benchmark_value += shares * end_price
        
        return benchmark_value
    
    def run_backtest(self, start_date: str = None, end_date: str = None, 
                    top_n: int = 20, rebalance_days: int = 5) -> Dict:
        """運行回測"""
        logger.info("=== 開始回測 ===")
        
        # 載入數據
        prices_df = self.load_data()
        
        # 設置回測期間
        if start_date:
            start_date = pd.to_datetime(start_date)
            prices_df = prices_df[prices_df['date'] >= start_date]
        else:
            start_date = prices_df['date'].min()
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            prices_df = prices_df[prices_df['date'] <= end_date]
        else:
            end_date = prices_df['date'].max()
        
        logger.info(f"回測期間: {start_date} 到 {end_date}")
        
        # 準備特徵和預測
        features_df = self.prepare_features(prices_df)
        predictor = StockPredictor()
        if not predictor.load_models():
            raise RuntimeError("無法載入模型")
        
        predictions_df = self.get_predictions(features_df, predictor)
        
        # 獲取所有交易日期
        all_dates = sorted(predictions_df['date'].unique())
        rebalance_dates = all_dates[::rebalance_days]  # 每N天再平衡一次
        
        logger.info(f"回測日期: {len(rebalance_dates)} 個再平衡點")
        
        # 初始化基準組合
        benchmark_stocks = predictions_df[predictions_df['date'] == all_dates[0]].nlargest(10, 'composite_score')['stock_code'].tolist()
        
        # 運行回測
        for i, date in enumerate(rebalance_dates):
            logger.info(f"處理日期 {i+1}/{len(rebalance_dates)}: {date}")
            
            # 選擇股票
            target_stocks = self.select_stocks(predictions_df, date, top_n)
            
            if target_stocks:
                # 再平衡投資組合
                self.rebalance_portfolio(target_stocks, prices_df, date)
            
            # 計算投資組合價值
            portfolio_value = self.calculate_portfolio_value(prices_df, date)
            self.portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': self.current_capital,
                'positions': dict(self.positions)
            })
            
            # 計算基準價值
            benchmark_value = self.calculate_benchmark_value(prices_df, all_dates[0], date, benchmark_stocks)
            self.benchmark_values.append({
                'date': date,
                'value': benchmark_value
            })
        
        # 計算績效指標
        performance = self.calculate_performance()
        
        logger.info("=== 回測完成 ===")
        return performance
    
    def calculate_performance(self) -> Dict:
        """計算績效指標"""
        if not self.portfolio_values:
            return {}
        
        # 提取價值序列
        portfolio_values = [pv['value'] for pv in self.portfolio_values]
        benchmark_values = [bv['value'] for bv in self.benchmark_values]
        
        # 計算總報酬率
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        benchmark_return = (benchmark_values[-1] - self.initial_capital) / self.initial_capital
        
        # 計算年化報酬率
        days = (self.portfolio_values[-1]['date'] - self.portfolio_values[0]['date']).days
        annualized_return = (1 + total_return) ** (365 / days) - 1
        benchmark_annualized = (1 + benchmark_return) ** (365 / days) - 1
        
        # 計算夏普比率
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # 計算最大回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # 計算勝率
        winning_trades = len([t for t in self.trade_history if t['action'] == 'sell' and t['net_proceeds'] > 0])
        total_trades = len([t for t in self.trade_history if t['action'] == 'sell'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        performance = {
            'initial_capital': self.initial_capital,
            'final_value': portfolio_values[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'benchmark_return': benchmark_return,
            'benchmark_annualized': benchmark_annualized,
            'excess_return': total_return - benchmark_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trade_history),
            'winning_trades': winning_trades,
            'portfolio_values': self.portfolio_values,
            'benchmark_values': self.benchmark_values,
            'trade_history': self.trade_history
        }
        
        return performance
    
    def generate_report(self, performance: Dict) -> str:
        """生成回測報告"""
        report = f"""
=== 股票選股策略回測報告 ===

基本資訊:
- 初始資金: {performance['initial_capital']:,.0f} 台幣
- 最終價值: {performance['final_value']:,.0f} 台幣
- 回測期間: {self.portfolio_values[0]['date'].strftime('%Y-%m-%d')} 到 {self.portfolio_values[-1]['date'].strftime('%Y-%m-%d')}

績效指標:
- 總報酬率: {performance['total_return']:.2%}
- 年化報酬率: {performance['annualized_return']:.2%}
- 基準報酬率: {performance['benchmark_return']:.2%}
- 超額報酬: {performance['excess_return']:.2%}
- 夏普比率: {performance['sharpe_ratio']:.3f}
- 最大回撤: {performance['max_drawdown']:.2%}

交易統計:
- 總交易次數: {performance['total_trades']}
- 獲利交易: {performance['winning_trades']}
- 勝率: {performance['win_rate']:.2%}

風險評估:
- 策略表現: {'優秀' if performance['excess_return'] > 0.05 else '良好' if performance['excess_return'] > 0 else '需改進'}
- 風險等級: {'低' if abs(performance['max_drawdown']) < 0.1 else '中' if abs(performance['max_drawdown']) < 0.2 else '高'}
"""
        
        return report


def main():
    """主函數"""
    try:
        # 初始化回測器
        backtester = Backtester(initial_capital=1000000)  # 100萬台幣
        
        # 運行回測
        performance = backtester.run_backtest(
            start_date="2024-01-01",
            end_date="2024-12-31",
            top_n=10,
            rebalance_days=5
        )
        
        # 生成報告
        report = backtester.generate_report(performance)
        print(report)
        
        # 保存結果
        output_dir = Path("outputs/backtest")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存績效數據
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
        
    except Exception as e:
        logger.error(f"回測過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    main()