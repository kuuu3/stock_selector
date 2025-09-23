"""
特徵工程模組
整合股價數據和新聞數據，生成模型特徵
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging
from pathlib import Path

from ..config import (
    TECHNICAL_INDICATORS,
    FEATURE_ENGINEERING_CONFIG,
    PROCESSED_FEATURES_FILE,
    PROCESSED_LABELS_FILE
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特徵工程器"""
    
    def __init__(self):
        self.lookback_periods = FEATURE_ENGINEERING_CONFIG["LOOKBACK_PERIODS"]
        self.volume_ma_periods = FEATURE_ENGINEERING_CONFIG["VOLUME_MA_PERIODS"]
        self.price_change_periods = FEATURE_ENGINEERING_CONFIG["PRICE_CHANGE_PERIODS"]
    
    def create_features(self, price_df: pd.DataFrame, news_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        創建特徵矩陣（不包含標籤）
        
        Args:
            price_df: 股價數據
            news_df: 新聞數據（可選）
            
        Returns:
            特徵矩陣 DataFrame（不包含標籤欄位）
        """
        logger.info("開始創建特徵...")
        
        # 創建基礎技術指標特徵
        feature_df = self._create_technical_features(price_df)
        
        # 創建衍生特徵
        feature_df = self._create_derived_features(feature_df)
        
        # 如果有新聞數據，創建新聞特徵
        if news_df is not None and not news_df.empty:
            feature_df = self._create_news_features(feature_df, news_df)
        
        # 清理數據
        feature_df = self._clean_features(feature_df)
        
        logger.info(f"特徵創建完成，共 {len(feature_df)} 筆樣本，{len(feature_df.columns)} 個特徵")
        
        return feature_df
    
    def create_labels(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        創建標籤數據
        
        Args:
            price_df: 股價數據
            
        Returns:
            標籤 DataFrame
        """
        logger.info("開始創建標籤...")
        
        # 創建基礎特徵用於計算標籤
        base_df = self._create_technical_features(price_df)
        
        # 創建標籤
        labels_df = self._create_labels(base_df)
        
        # 只保留標籤相關欄位
        label_columns = ['stock_code', 'date', 'future_return_1w', 'future_return_1m', 'label_1w', 'label_1m']
        labels_df = labels_df[label_columns]
        
        logger.info(f"標籤創建完成，共 {len(labels_df)} 筆樣本")
        
        return labels_df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建技術指標特徵"""
        result_data = []
        
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].copy()
            stock_df = stock_df.sort_values('date').reset_index(drop=True)
            
            # 基礎技術指標
            features = {
                'date': stock_df['date'],
                'stock_code': stock_df['stock_code'],
                'close': stock_df['close'],
                'volume': stock_df['volume'],
                'open': stock_df['open'],
                'high': stock_df['high'],
                'low': stock_df['low']
            }
            
            # 移動平均線
            ma_short = TECHNICAL_INDICATORS["MA_SHORT"]
            ma_long = TECHNICAL_INDICATORS["MA_LONG"]
            
            features[f'MA_{ma_short}'] = stock_df['close'].rolling(ma_short).mean()
            features[f'MA_{ma_long}'] = stock_df['close'].rolling(ma_long).mean()
            features['MA_diff'] = features[f'MA_{ma_short}'] - features[f'MA_{ma_long}']
            features['MA_ratio'] = features[f'MA_{ma_short}'] / features[f'MA_{ma_long}']
            
            # RSI
            features['RSI'] = self._calculate_rsi(stock_df['close'], TECHNICAL_INDICATORS["RSI_PERIOD"])
            
            # MACD
            macd = self._calculate_macd(stock_df['close'])
            features['MACD_DIF'] = macd['dif']
            features['MACD_DEM'] = macd['dem']
            features['MACD_OSC'] = macd['osc']
            
            # 成交量變化
            features['volume_change'] = stock_df['volume'].pct_change()
            features['volume_ma_ratio'] = stock_df['volume'] / stock_df['volume'].rolling(20).mean()
            
            # 波動率
            features['volatility'] = stock_df['close'].rolling(TECHNICAL_INDICATORS["VOLATILITY_PERIOD"]).std()
            
            # 價格變化
            features['price_change'] = stock_df['close'].pct_change(fill_method=None)
            
            result_data.append(pd.DataFrame(features))
        
        return pd.concat(result_data, ignore_index=True)
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建衍生特徵"""
        result_df = df.copy()
        
        # 多期間價格變化
        for period in self.price_change_periods:
            result_df[f'price_change_{period}d'] = result_df.groupby('stock_code')['close'].pct_change(periods=period, fill_method=None)
        
        # 多期間移動平均線
        for period in self.lookback_periods:
            result_df[f'MA_{period}'] = result_df.groupby('stock_code')['close'].rolling(period).mean().reset_index(0, drop=True)
            result_df[f'MA_{period}_ratio'] = result_df['close'] / result_df[f'MA_{period}']
        
        # 成交量移動平均
        for period in self.volume_ma_periods:
            result_df[f'volume_ma_{period}'] = result_df.groupby('stock_code')['volume'].rolling(period).mean().reset_index(0, drop=True)
            result_df[f'volume_ratio_{period}'] = result_df['volume'] / result_df[f'volume_ma_{period}']
        
        # 價格位置特徵
        result_df['price_position'] = (result_df['close'] - result_df['low']) / (result_df['high'] - result_df['low'])
        
        # 成交量價格關係
        result_df['volume_price_trend'] = result_df['volume_change'] * result_df['price_change']
        
        return result_df
    
    def _create_news_features(self, feature_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """創建新聞特徵"""
        result_df = feature_df.copy()
        
        # 確保日期格式正確
        feature_df['date'] = pd.to_datetime(feature_df['date']).dt.date
        
        # 檢查新聞數據是否包含情感分析結果
        if 'sentiment' in news_df.columns:
            logger.info("使用包含情感分析的新聞數據")
            # 使用情感分析數據
            result_df = self._create_sentiment_features(result_df, news_df)
        else:
            logger.info("使用基礎新聞數據")
            # 使用基礎新聞數據
            result_df = self._create_basic_news_features(result_df, news_df)
        
        return result_df
    
    def _create_sentiment_features(self, feature_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """創建基於情感分析的新聞特徵"""
        result_df = feature_df.copy()
        
        # 處理新聞數據
        if 'analyzed_time' in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['analyzed_time']).dt.date
        elif 'scraped_time' in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['scraped_time']).dt.date
        elif 'publish_time' in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['publish_time']).dt.date
        else:
            logger.error("新聞數據中找不到日期欄位")
            return result_df
        
        # 按日期和股票代碼分組統計
        daily_news_stats = []
        
        for date in feature_df['date'].unique():
            date_news = news_df[news_df['date'] == date]
            
            if not date_news.empty:
                # 總體新聞統計
                total_news_count = len(date_news)
                positive_count = len(date_news[date_news['sentiment'] == 'positive'])
                negative_count = len(date_news[date_news['sentiment'] == 'negative'])
                neutral_count = len(date_news[date_news['sentiment'] == 'neutral'])
                
                avg_sentiment_score = date_news['sentiment_score'].mean()
                avg_confidence = date_news['confidence'].mean()
                
                daily_news_stats.append({
                    'date': date,
                    'news_count': total_news_count,
                    'positive_news_count': positive_count,
                    'negative_news_count': negative_count,
                    'neutral_news_count': neutral_count,
                    'avg_sentiment_score': avg_sentiment_score,
                    'avg_confidence': avg_confidence,
                    'sentiment_ratio': (positive_count - negative_count) / max(total_news_count, 1)
                })
        
        if daily_news_stats:
            news_stats_df = pd.DataFrame(daily_news_stats)
            
            # 合併到特徵數據
            result_df = result_df.merge(news_stats_df, on='date', how='left')
            
            # 填充缺失值
            news_columns = ['news_count', 'positive_news_count', 'negative_news_count', 
                          'neutral_news_count', 'avg_sentiment_score', 'avg_confidence', 'sentiment_ratio']
            for col in news_columns:
                result_df[col] = result_df[col].fillna(0)
            
            # 計算移動平均
            for col in news_columns:
                result_df[f'{col}_ma5'] = result_df.groupby('stock_code')[col].rolling(5).mean().reset_index(0, drop=True)
                result_df[f'{col}_ma20'] = result_df.groupby('stock_code')[col].rolling(20).mean().reset_index(0, drop=True)
        else:
            logger.warning("沒有找到匹配的新聞數據")
            # 添加空的新聞特徵
            news_columns = ['news_count', 'positive_news_count', 'negative_news_count', 
                          'neutral_news_count', 'avg_sentiment_score', 'avg_confidence', 'sentiment_ratio']
            for col in news_columns:
                result_df[col] = 0
                result_df[f'{col}_ma5'] = 0
                result_df[f'{col}_ma20'] = 0
        
        return result_df
    
    def _create_basic_news_features(self, feature_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """創建基礎新聞特徵（無情感分析）"""
        result_df = feature_df.copy()
        
        # 處理新聞數據
        if 'scraped_time' in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['scraped_time']).dt.date
        else:
            news_df['date'] = pd.to_datetime(news_df['publish_time']).dt.date
        
        # 計算每日新聞數量
        daily_news_count = news_df.groupby('date').size().reset_index(name='news_count')
        
        # 合併新聞特徵
        result_df = result_df.merge(daily_news_count, on='date', how='left')
        result_df['news_count'] = result_df['news_count'].fillna(0)
        
        # 計算新聞特徵的移動平均
        result_df['news_count_ma5'] = result_df.groupby('stock_code')['news_count'].rolling(5).mean().reset_index(0, drop=True)
        result_df['news_count_ma20'] = result_df.groupby('stock_code')['news_count'].rolling(20).mean().reset_index(0, drop=True)
        
        return result_df
    
    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建標籤"""
        result_df = df.copy()
        
        # 未來1週報酬率
        result_df['future_return_1w'] = result_df.groupby('stock_code')['close'].pct_change(periods=5, fill_method=None)
        
        # 未來1個月報酬率
        result_df['future_return_1m'] = result_df.groupby('stock_code')['close'].pct_change(periods=20, fill_method=None)
        
        # 分類標籤：漲(1)、跌(-1)、平(0)
        result_df['label_1w'] = result_df['future_return_1w'].apply(self._create_classification_label)
        result_df['label_1m'] = result_df['future_return_1m'].apply(self._create_classification_label)
        
        return result_df
    
    def _create_classification_label(self, return_rate: float) -> int:
        """創建分類標籤"""
        if pd.isna(return_rate):
            return 0
        elif return_rate > 0.02:  # 上漲超過2%
            return 1
        elif return_rate < -0.02:  # 下跌超過2%
            return -1
        else:
            return 0
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理特徵數據"""
        logger.info(f"清理前數據量: {len(df)}")
        
        # 移除無效的股票代碼
        df['stock_code'] = df['stock_code'].astype(str)
        df = df[df['stock_code'].str.len() == 4]
        logger.info(f"移除無效股票代碼後: {len(df)}")
        
        # 只移除關鍵特徵為NaN的行，其他用0填充
        key_features = ['MA_3', 'MA_5', 'MA_diff', 'RSI', 'MACD_DIF']
        df = df.dropna(subset=key_features, how='all')  # 只要不是所有關鍵特徵都是NaN就保留
        logger.info(f"移除關鍵特徵全為NaN後: {len(df)}")
        
        # 用前向填充和0填充處理其他NaN
        df = df.ffill().fillna(0)
        logger.info(f"填充NaN後: {len(df)}")
        
        # 移除極端值（更寬鬆的處理）
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['label_1w', 'label_1m', 'future_return_1w', 'future_return_1m']:
                # 只移除極端異常值，保留更多數據
                q01, q99 = df[col].quantile([0.005, 0.995])
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        logger.info(f"最終清理後數據量: {len(df)}")
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """計算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """計算MACD"""
        fast = TECHNICAL_INDICATORS["MACD_FAST"]
        slow = TECHNICAL_INDICATORS["MACD_SLOW"]
        signal = TECHNICAL_INDICATORS["MACD_SIGNAL"]
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        dif = ema_fast - ema_slow
        dem = dif.ewm(span=signal).mean()
        osc = dif - dem
        
        return {'dif': dif, 'dem': dem, 'osc': osc}
    
    def save_features(self, df: pd.DataFrame):
        """保存特徵和標籤"""
        # 分離特徵和標籤
        feature_columns = [col for col in df.columns if not col.startswith('label_') and col not in ['future_return_1w', 'future_return_1m']]
        label_columns = ['label_1w', 'label_1m', 'future_return_1w', 'future_return_1m']
        
        features = df[feature_columns]
        labels = df[label_columns]
        
        # 保存為numpy格式
        np.save(PROCESSED_FEATURES_FILE, features.values)
        np.save(PROCESSED_LABELS_FILE, labels.values)
        
        logger.info(f"特徵已保存到 {PROCESSED_FEATURES_FILE}")
        logger.info(f"標籤已保存到 {PROCESSED_LABELS_FILE}")


def main():
    """測試特徵工程功能"""
    engineer = FeatureEngineer()
    
    # 創建測試數據
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = []
    
    for stock_code in ['2330', '2317']:
        for i, date in enumerate(dates):
            test_data.append({
                'date': date,
                'stock_code': stock_code,
                'close': 100 + i * 0.1 + np.random.normal(0, 1),
                'volume': 1000000 + np.random.normal(0, 100000),
                'open': 100 + i * 0.1,
                'high': 101 + i * 0.1,
                'low': 99 + i * 0.1
            })
    
    df = pd.DataFrame(test_data)
    
    # 創建特徵
    features = engineer.create_features(df)
    
    print(f"特徵形狀: {features.shape}")
    print(f"特徵列: {features.columns.tolist()}")


if __name__ == "__main__":
    main()


