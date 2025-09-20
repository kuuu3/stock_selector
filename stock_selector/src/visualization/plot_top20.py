"""
Top20 選股視覺化模組
生成選股結果的圖表
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from typing import List, Dict, Any
from pathlib import Path

from ..config import OUTPUTS_DIR

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class Top20Visualizer:
    """Top20 選股視覺化器"""
    
    def __init__(self):
        self.colors = {
            'up': '#2E8B57',      # 綠色 - 上漲
            'down': '#DC143C',    # 紅色 - 下跌
            'neutral': '#4682B4'  # 藍色 - 平盤
        }
    
    def load_top20_data(self, file_path: Path = None) -> pd.DataFrame:
        """
        載入 Top20 數據
        
        Args:
            file_path: 數據檔案路徑
            
        Returns:
            Top20 數據 DataFrame
        """
        if file_path is None:
            file_path = OUTPUTS_DIR / "top20.csv"
        
        if not file_path.exists():
            logger.warning(f"找不到 Top20 數據檔案: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"載入 Top20 數據: {len(df)} 支股票")
            return df
        except Exception as e:
            logger.error(f"載入 Top20 數據失敗: {e}")
            return pd.DataFrame()
    
    def plot_stock_scores(self, df: pd.DataFrame, top_n: int = 20):
        """
        繪製股票評分條狀圖
        
        Args:
            df: 股票數據
            top_n: 顯示前N支股票
        """
        if df.empty:
            logger.warning("沒有數據可視覺化")
            return
        
        # 選擇前N支股票
        top_stocks = df.head(top_n)
        
        # 創建圖表
        plt.figure(figsize=(12, 8))
        
        # 根據預測方向設置顏色
        colors = []
        for _, stock in top_stocks.iterrows():
            prediction = stock.get('prediction_class', 0)
            if prediction == 1:
                colors.append(self.colors['up'])
            elif prediction == -1:
                colors.append(self.colors['down'])
            else:
                colors.append(self.colors['neutral'])
        
        # 繪製條狀圖
        bars = plt.barh(range(len(top_stocks)), top_stocks['final_score'], color=colors)
        
        # 設置標籤
        plt.yticks(range(len(top_stocks)), top_stocks['stock_code'])
        plt.xlabel('最終評分')
        plt.title(f'Top {top_n} 股票評分')
        
        # 添加數值標籤
        for i, (bar, score) in enumerate(zip(bars, top_stocks['final_score'])):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', ha='left')
        
        # 添加圖例
        legend_elements = [
            plt.Rectangle((0,0),1,1, color=self.colors['up'], label='預期上漲'),
            plt.Rectangle((0,0),1,1, color=self.colors['down'], label='預期下跌'),
            plt.Rectangle((0,0),1,1, color=self.colors['neutral'], label='預期平盤')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = OUTPUTS_DIR / "top20_scores.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"股票評分圖表已保存到: {output_file}")
    
    def plot_confidence_distribution(self, df: pd.DataFrame):
        """
        繪製信心度分布圖
        
        Args:
            df: 股票數據
        """
        if df.empty or 'confidence' not in df.columns:
            logger.warning("沒有信心度數據可視覺化")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 信心度直方圖
        ax1.hist(df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('信心度')
        ax1.set_ylabel('股票數量')
        ax1.set_title('信心度分布')
        ax1.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                   label=f'平均信心度: {df["confidence"].mean():.3f}')
        ax1.legend()
        
        # 信心度箱線圖
        ax2.boxplot(df['confidence'], patch_artist=True, 
                   boxprops=dict(facecolor='lightgreen', alpha=0.7))
        ax2.set_ylabel('信心度')
        ax2.set_title('信心度箱線圖')
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = OUTPUTS_DIR / "confidence_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"信心度分布圖表已保存到: {output_file}")
    
    def plot_prediction_direction(self, df: pd.DataFrame):
        """
        繪製預測方向分布圖
        
        Args:
            df: 股票數據
        """
        if df.empty or 'prediction_class' not in df.columns:
            logger.warning("沒有預測方向數據可視覺化")
            return
        
        # 計算方向分布
        direction_counts = df['prediction_class'].value_counts()
        
        # 設置標籤
        labels = {
            1: '預期上漲',
            -1: '預期下跌',
            0: '預期平盤'
        }
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 圓餅圖
        colors = [self.colors['up'], self.colors['down'], self.colors['neutral']]
        wedges, texts, autotexts = ax1.pie(direction_counts.values, 
                                          labels=[labels.get(i, f'類別{i}') for i in direction_counts.index],
                                          colors=[colors[i] for i in direction_counts.index],
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('預測方向分布')
        
        # 條狀圖
        bars = ax2.bar([labels.get(i, f'類別{i}') for i in direction_counts.index], 
                      direction_counts.values,
                      color=[colors[i] for i in direction_counts.index])
        ax2.set_ylabel('股票數量')
        ax2.set_title('預測方向數量')
        
        # 添加數值標籤
        for bar, count in zip(bars, direction_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = OUTPUTS_DIR / "prediction_direction.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"預測方向圖表已保存到: {output_file}")
    
    def plot_expected_returns(self, df: pd.DataFrame, top_n: int = 10):
        """
        繪製預期報酬率圖
        
        Args:
            df: 股票數據
            top_n: 顯示前N支股票
        """
        if df.empty or 'expected_return' not in df.columns:
            logger.warning("沒有預期報酬率數據可視覺化")
            return
        
        # 選擇前N支股票
        top_stocks = df.head(top_n)
        
        # 創建圖表
        plt.figure(figsize=(12, 8))
        
        # 根據預期報酬率設置顏色
        colors = ['green' if ret > 0 else 'red' for ret in top_stocks['expected_return']]
        
        # 繪製條狀圖
        bars = plt.barh(range(len(top_stocks)), top_stocks['expected_return'], color=colors, alpha=0.7)
        
        # 設置標籤
        plt.yticks(range(len(top_stocks)), top_stocks['stock_code'])
        plt.xlabel('預期報酬率')
        plt.title(f'Top {top_n} 股票預期報酬率')
        
        # 添加數值標籤
        for i, (bar, ret) in enumerate(zip(bars, top_stocks['expected_return'])):
            plt.text(bar.get_width() + 0.001 if ret >= 0 else bar.get_width() - 0.001, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{ret:.2%}', va='center', 
                    ha='left' if ret >= 0 else 'right')
        
        # 添加零線
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = OUTPUTS_DIR / "expected_returns.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"預期報酬率圖表已保存到: {output_file}")
    
    def create_summary_dashboard(self, df: pd.DataFrame):
        """
        創建綜合儀表板
        
        Args:
            df: 股票數據
        """
        if df.empty:
            logger.warning("沒有數據創建儀表板")
            return
        
        # 創建子圖
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Stock Selector 綜合儀表板', fontsize=16, fontweight='bold')
        
        # 1. Top 10 股票評分
        top10 = df.head(10)
        colors1 = [self.colors['up'] if p == 1 else self.colors['down'] if p == -1 else self.colors['neutral'] 
                  for p in top10.get('prediction_class', [0] * len(top10))]
        
        ax1.barh(range(len(top10)), top10['final_score'], color=colors1)
        ax1.set_yticks(range(len(top10)))
        ax1.set_yticklabels(top10['stock_code'])
        ax1.set_xlabel('最終評分')
        ax1.set_title('Top 10 股票評分')
        
        # 2. 信心度分布
        if 'confidence' in df.columns:
            ax2.hist(df['confidence'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                       label=f'平均: {df["confidence"].mean():.3f}')
            ax2.set_xlabel('信心度')
            ax2.set_ylabel('股票數量')
            ax2.set_title('信心度分布')
            ax2.legend()
        
        # 3. 預測方向分布
        if 'prediction_class' in df.columns:
            direction_counts = df['prediction_class'].value_counts()
            labels = {1: '上漲', -1: '下跌', 0: '平盤'}
            colors3 = [self.colors['up'], self.colors['down'], self.colors['neutral']]
            
            ax3.pie(direction_counts.values, 
                   labels=[labels.get(i, f'類別{i}') for i in direction_counts.index],
                   colors=[colors3[i] for i in direction_counts.index],
                   autopct='%1.1f%%', startangle=90)
            ax3.set_title('預測方向分布')
        
        # 4. 預期報酬率分布
        if 'expected_return' in df.columns:
            ax4.hist(df['expected_return'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            ax4.axvline(df['expected_return'].mean(), color='red', linestyle='--',
                       label=f'平均: {df["expected_return"].mean():.2%}')
            ax4.set_xlabel('預期報酬率')
            ax4.set_ylabel('股票數量')
            ax4.set_title('預期報酬率分布')
            ax4.legend()
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = OUTPUTS_DIR / "summary_dashboard.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"綜合儀表板已保存到: {output_file}")
    
    def generate_all_plots(self, file_path: Path = None):
        """
        生成所有圖表
        
        Args:
            file_path: Top20 數據檔案路徑
        """
        logger.info("開始生成所有視覺化圖表...")
        
        # 載入數據
        df = self.load_top20_data(file_path)
        if df.empty:
            logger.error("無法載入數據，跳過視覺化")
            return
        
        # 生成各種圖表
        self.plot_stock_scores(df)
        self.plot_confidence_distribution(df)
        self.plot_prediction_direction(df)
        self.plot_expected_returns(df)
        self.create_summary_dashboard(df)
        
        logger.info("所有視覺化圖表生成完成")


def main():
    """主函數 - 用於測試視覺化功能"""
    visualizer = Top20Visualizer()
    
    # 創建測試數據
    test_data = []
    for i in range(20):
        test_data.append({
            'stock_code': f'233{i:02d}',
            'rank': i + 1,
            'final_score': np.random.uniform(-0.5, 0.8),
            'prediction_class': np.random.choice([-1, 0, 1]),
            'confidence': np.random.uniform(0.3, 0.9),
            'expected_return': np.random.uniform(-0.1, 0.15)
        })
    
    test_df = pd.DataFrame(test_data)
    
    # 生成所有圖表
    visualizer.generate_all_plots()
    
    logger.info("視覺化測試完成")


if __name__ == "__main__":
    main()
