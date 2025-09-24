"""
選股邏輯模組
基於模型預測結果進行股票選擇和排序
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

from ..config import (
    SELECTION_CONFIG,
    TOP20_OUTPUT_FILE,
    OUTPUTS_DIR
)
from ..models import StockPredictor

logger = logging.getLogger(__name__)


class StockSelector:
    """股票選擇器"""
    
    def __init__(self):
        self.predictor = StockPredictor()
        self.selected_stocks = None
        
    def load_models(self) -> bool:
        """
        載入預訓練模型
        
        Returns:
            是否成功載入模型
        """
        return self.predictor.load_models()
    
    def filter_by_liquidity(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        根據流動性篩選股票
        
        Args:
            stock_data: 股票數據 DataFrame
            
        Returns:
            篩選後的股票數據
        """
        min_volume = SELECTION_CONFIG["MIN_VOLUME"]
        min_market_cap = SELECTION_CONFIG["MIN_MARKET_CAP"]
        min_price = SELECTION_CONFIG["MIN_PRICE"]
        
        # 成交量篩選
        if 'volume' in stock_data.columns:
            stock_data = stock_data[stock_data['volume'] >= min_volume]
        
        # 價格篩選
        if 'close' in stock_data.columns:
            stock_data = stock_data[stock_data['close'] >= min_price]
        
        # 市值篩選（如果有市值數據）
        if 'market_cap' in stock_data.columns:
            stock_data = stock_data[stock_data['market_cap'] >= min_market_cap]
        
        logger.info(f"流動性篩選後剩餘 {len(stock_data)} 支股票")
        
        return stock_data
    
    def filter_by_industry_diversification(self, stock_data: pd.DataFrame, 
                                         top_n: int = None) -> pd.DataFrame:
        """
        根據產業分散化篩選股票
        
        Args:
            stock_data: 股票數據 DataFrame
            top_n: 每產業最大選股數量
            
        Returns:
            產業分散化後的股票數據
        """
        if top_n is None:
            top_n = SELECTION_CONFIG["TOP_N_STOCKS"] // 5  # 假設有5個主要產業
        
        # 如果有產業分類數據
        if 'industry' in stock_data.columns:
            diversified_stocks = []
            
            for industry in stock_data['industry'].unique():
                industry_stocks = stock_data[stock_data['industry'] == industry]
                # 每個產業最多選擇 top_n 支股票
                industry_top = industry_stocks.nlargest(top_n, 'final_score')
                diversified_stocks.append(industry_top)
            
            result = pd.concat(diversified_stocks, ignore_index=True)
            logger.info(f"產業分散化後選出 {len(result)} 支股票")
            return result
        
        # 如果沒有產業數據，直接返回原數據
        return stock_data
    
    def calculate_risk_adjusted_score(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        計算風險調整後的評分
        
        Args:
            stock_data: 股票數據 DataFrame
            
        Returns:
            包含風險調整評分的 DataFrame
        """
        result_df = stock_data.copy()
        
        # 如果有波動率數據，進行風險調整
        if 'volatility' in result_df.columns:
            # 使用夏普比率概念：預期收益 / 風險
            result_df['risk_adjusted_score'] = (
                result_df['final_score'] / (result_df['volatility'] + 1e-8)
            )
        else:
            result_df['risk_adjusted_score'] = result_df['final_score']
        
        # 如果有Beta數據，進一步調整
        if 'beta' in result_df.columns:
            # Beta > 1 的股票降低評分（高風險）
            result_df['risk_adjusted_score'] = result_df['risk_adjusted_score'] / (
                1 + (result_df['beta'] - 1) * 0.5
            )
        
        return result_df
    
    def rank_stocks(self, stock_data: pd.DataFrame, 
                   ranking_method: str = "risk_adjusted") -> pd.DataFrame:
        """
        股票排序
        
        Args:
            stock_data: 股票數據 DataFrame
            ranking_method: 排序方法 ("final_score", "risk_adjusted", "expected_return")
            
        Returns:
            排序後的股票 DataFrame
        """
        result_df = stock_data.copy()
        
        # 根據選擇的方法排序
        if ranking_method == "risk_adjusted":
            if 'risk_adjusted_score' not in result_df.columns:
                result_df = self.calculate_risk_adjusted_score(result_df)
            result_df = result_df.sort_values('risk_adjusted_score', ascending=False)
        elif ranking_method == "expected_return":
            if 'expected_return' in result_df.columns:
                result_df = result_df.sort_values('expected_return', ascending=False)
            else:
                result_df = result_df.sort_values('final_score', ascending=False)
        else:  # final_score
            result_df = result_df.sort_values('final_score', ascending=False)
        
        # 重新設置排名
        result_df = result_df.reset_index(drop=True)
        result_df['rank'] = range(1, len(result_df) + 1)
        
        logger.info(f"股票排序完成，使用 {ranking_method} 方法")
        
        return result_df
    
    def select_top_stocks(self, stock_data: pd.DataFrame, 
                         top_n: int = None) -> pd.DataFrame:
        """
        選擇前N支股票
        
        Args:
            stock_data: 股票數據 DataFrame
            top_n: 選擇的股票數量
            
        Returns:
            前N支股票 DataFrame
        """
        if top_n is None:
            top_n = SELECTION_CONFIG["TOP_N_STOCKS"]
        
        # 流動性篩選
        filtered_data = self.filter_by_liquidity(stock_data)
        
        # 產業分散化篩選
        diversified_data = self.filter_by_industry_diversification(filtered_data, top_n)
        
        # 風險調整評分
        risk_adjusted_data = self.calculate_risk_adjusted_score(diversified_data)
        
        # 去重：每支股票只保留評分最高的樣本
        if 'stock_code' in risk_adjusted_data.columns:
            # 按股票代碼分組，保留每支股票評分最高的樣本
            deduplicated_data = risk_adjusted_data.loc[
                risk_adjusted_data.groupby('stock_code')['final_score'].idxmax()
            ].reset_index(drop=True)
            logger.info(f"去重後剩餘 {len(deduplicated_data)} 支唯一股票")
        else:
            deduplicated_data = risk_adjusted_data
            logger.warning("沒有 stock_code 欄位，無法去重")
        
        # 排序
        ranked_data = self.rank_stocks(deduplicated_data)
        
        # 選擇前N支
        top_stocks = ranked_data.head(top_n)
        
        logger.info(f"選出前 {len(top_stocks)} 支股票")
        
        return top_stocks
    
    def generate_selection_summary(self, selected_stocks: pd.DataFrame) -> Dict[str, Any]:
        """
        生成選股摘要
        
        Args:
            selected_stocks: 選中的股票 DataFrame
            
        Returns:
            選股摘要字典
        """
        summary = {
            'total_stocks': len(selected_stocks),
            'avg_score': selected_stocks['final_score'].mean(),
            'avg_confidence': selected_stocks.get('confidence', pd.Series([0] * len(selected_stocks))).mean(),
            'avg_expected_return': selected_stocks.get('expected_return', pd.Series([0] * len(selected_stocks))).mean()
        }
        
        # 預測方向分布
        if 'prediction_class' in selected_stocks.columns:
            direction_dist = selected_stocks['prediction_class'].value_counts()
            summary['direction_distribution'] = {
                'up': direction_dist.get(1, 0),
                'down': direction_dist.get(-1, 0),
                'neutral': direction_dist.get(0, 0)
            }
        
        # 產業分布（如果有產業數據）
        if 'industry' in selected_stocks.columns:
            industry_dist = selected_stocks['industry'].value_counts()
            summary['industry_distribution'] = industry_dist.to_dict()
        
        logger.info("選股摘要:")
        logger.info(f"  總股票數: {summary['total_stocks']}")
        logger.info(f"  平均評分: {summary['avg_score']:.4f}")
        logger.info(f"  平均信心度: {summary['avg_confidence']:.4f}")
        logger.info(f"  平均預期報酬: {summary['avg_expected_return']:.4f}")
        
        return summary
    
    def save_selection_results(self, selected_stocks: pd.DataFrame, 
                             filename: str = None) -> Path:
        """
        保存選股結果
        
        Args:
            selected_stocks: 選中的股票 DataFrame
            filename: 檔案名稱
            
        Returns:
            保存檔案的路徑
        """
        if filename is None:
            # 使用預設的完整路徑
            output_path = TOP20_OUTPUT_FILE
        else:
            # 將 filename 轉換為 Path 對象
            filename_path = Path(filename)
            
            # 檢查是否已經是完整路徑（包含 outputs 目錄）
            if str(filename_path).startswith('outputs/') or filename_path.is_absolute():
                # 如果已經是完整路徑，直接使用
                output_path = filename_path
            else:
                # 如果只是檔案名，則組合路徑
                output_path = OUTPUTS_DIR / filename_path
        
        # 確保輸出目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存為CSV
        selected_stocks.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"選股結果已保存到: {output_path}")
        
        return output_path
    
    def select_stocks(self, features: np.ndarray, stock_codes: List[str], 
                     stock_data: pd.DataFrame = None, 
                     top_n: int = None) -> pd.DataFrame:
        """
        完整的選股流程
        
        Args:
            features: 特徵矩陣
            stock_codes: 股票代碼列表
            stock_data: 股票基本數據（可選）
            top_n: 選擇的股票數量
            
        Returns:
            選中的股票 DataFrame
        """
        logger.info("=== 開始選股流程 ===")
        
        # 載入模型
        if not self.load_models():
            logger.error("無法載入模型")
            return pd.DataFrame()
        
        # 進行預測
        prediction_results = self.predictor.predict_stocks(
            features, stock_codes, top_n=None
        )
        
        if prediction_results.empty:
            logger.error("預測失敗")
            return pd.DataFrame()
        
        # 合併股票基本數據（如果有）
        if stock_data is not None:
            prediction_results = prediction_results.merge(
                stock_data, on='stock_code', how='left'
            )
        
        # 選擇前N支股票
        selected_stocks = self.select_top_stocks(prediction_results, top_n)
        
        # 生成摘要
        summary = self.generate_selection_summary(selected_stocks)
        
        # 保存結果
        self.save_selection_results(selected_stocks)
        
        logger.info("=== 選股流程完成 ===")
        
        return selected_stocks
    
    def get_stock_recommendations(self, selected_stocks: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        生成股票推薦說明
        
        Args:
            selected_stocks: 選中的股票 DataFrame
            
        Returns:
            推薦說明列表
        """
        recommendations = []
        
        for _, stock in selected_stocks.iterrows():
            recommendation = {
                'stock_code': stock['stock_code'],
                'rank': stock['rank'],
                'final_score': stock['final_score'],
                'recommendation': self._generate_stock_recommendation(stock)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_stock_recommendation(self, stock: pd.Series) -> str:
        """
        為單一股票生成推薦說明
        
        Args:
            stock: 股票數據 Series
            
        Returns:
            推薦說明文字
        """
        score = stock['final_score']
        prediction_class = stock.get('prediction_class', 0)
        confidence = stock.get('confidence', 0)
        expected_return = stock.get('expected_return', 0)
        
        # 根據評分生成推薦等級
        if score > 0.5:
            recommendation_level = "強力推薦"
        elif score > 0.2:
            recommendation_level = "推薦"
        elif score > -0.2:
            recommendation_level = "中性"
        else:
            recommendation_level = "不推薦"
        
        # 根據預測方向生成說明
        direction_text = {
            1: "上漲",
            -1: "下跌",
            0: "盤整"
        }.get(prediction_class, "未知")
        
        recommendation = (
            f"{recommendation_level}，預期{direction_text}，"
            f"信心度{confidence:.2f}，預期報酬{expected_return:.2%}"
        )
        
        return recommendation


def main():
    """主函數 - 用於測試選股功能"""
    selector = StockSelector()
    
    # 創建測試數據
    n_stocks = 50
    n_features = 20
    
    test_features = np.random.randn(n_stocks, n_features)
    test_stock_codes = [f"233{i:02d}" for i in range(n_stocks)]
    
    # 創建測試股票數據
    test_stock_data = pd.DataFrame({
        'stock_code': test_stock_codes,
        'volume': np.random.randint(1000000, 10000000, n_stocks),
        'close': np.random.uniform(10, 500, n_stocks),
        'industry': np.random.choice(['科技', '金融', '傳產', '生技', '能源'], n_stocks)
    })
    
    # 進行選股
    selected_stocks = selector.select_stocks(
        test_features, test_stock_codes, test_stock_data, top_n=10
    )
    
    if not selected_stocks.empty:
        logger.info("選股結果:")
        logger.info(selected_stocks[['stock_code', 'rank', 'final_score', 'confidence']].to_string())
        
        # 生成推薦說明
        recommendations = selector.get_stock_recommendations(selected_stocks.head(5))
        logger.info("\n推薦說明:")
        for rec in recommendations:
            logger.info(f"{rec['stock_code']}: {rec['recommendation']}")


if __name__ == "__main__":
    main()


