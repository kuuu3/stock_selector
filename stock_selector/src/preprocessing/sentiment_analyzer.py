"""
新聞情感分析模組
支持多種情感分析方法：詞典法、機器學習模型、深度學習模型
"""

import pandas as pd
import numpy as np
import re
import jieba
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self):
        self.positive_words = set()
        self.negative_words = set()
        self.neutral_words = set()
        self.intensity_words = {}
        self.financial_keywords = set()
        
        # 初始化情感詞典
        self._load_sentiment_dictionaries()
        
    def _load_sentiment_dictionaries(self):
        """載入情感詞典"""
        try:
            # 基礎正面詞彙
            self.positive_words = {
                '上漲', '上揚', '攀升', '成長', '增長', '獲利', '盈餘', '利多', '看好',
                '樂觀', '積極', '強勁', '突破', '創新', '領先', '優異', '傑出', '亮眼',
                '豐收', '豐厚', '豐碩', '大幅', '顯著', '明顯', '強勁', '穩健', '穩定',
                '熱門', '搶手', '暢銷', '爆紅', '火熱', '炙手可熱', '前景看好', '未來可期',
                '投資價值', '投資機會', '潛力股', '成長股', '績優股', '藍籌股'
            }
            
            # 基礎負面詞彙
            self.negative_words = {
                '下跌', '下挫', '滑落', '衰退', '萎縮', '虧損', '赤字', '利空', '看壞',
                '悲觀', '消極', '疲弱', '跌破', '創新低', '落後', '劣勢', '不佳', '黯淡',
                '慘淡', '慘重', '嚴重', '大幅', '顯著', '明顯', '疲軟', '不穩', '動盪',
                '冷門', '滯銷', '低迷', '冷淡', '前景不明', '風險高', '投資風險',
                '避險', '減持', '拋售', '賣壓', '恐慌', '擔憂', '疑慮', '不確定性'
            }
            
            # 中性詞彙
            self.neutral_words = {
                '持平', '平穩', '維持', '不變', '穩定', '正常', '一般', '普通',
                '觀察', '關注', '監控', '評估', '分析', '研究', '調查', '檢視'
            }
            
            # 強度詞彙
            self.intensity_words = {
                '極度': 3, '非常': 2.5, '很': 2, '相當': 2, '頗': 1.5, '有點': 1.2,
                '略微': 1.1, '稍微': 1.1, '大幅': 2.5, '顯著': 2, '明顯': 1.8,
                '輕微': 1.2, '稍微': 1.1, '略': 1.1, '微': 1.1, '小': 1.1
            }
            
            # 金融關鍵詞
            self.financial_keywords = {
                '股價', '股東', '股息', '股利', '財報', '營收', '獲利', '盈餘', '利潤',
                '營收', '收入', '成本', '費用', '負債', '資產', '資本', '資金', '投資',
                '交易', '買賣', '成交', '成交量', '成交額', '市值', '本益比', '股價淨值比',
                'ROE', 'ROA', 'EPS', 'PER', 'PBR', '殖利率', '股息率', '股東權益報酬率'
            }
            
            logger.info(f"情感詞典載入完成: 正面詞彙 {len(self.positive_words)} 個, "
                       f"負面詞彙 {len(self.negative_words)} 個, "
                       f"中性詞彙 {len(self.neutral_words)} 個")
            
        except Exception as e:
            logger.error(f"載入情感詞典時發生錯誤: {e}")
    
    def analyze_sentiment(self, text: str, method: str = 'dictionary') -> Dict:
        """
        分析文本情感
        
        Args:
            text: 要分析的文本
            method: 分析方法 ('dictionary', 'rule_based', 'hybrid')
            
        Returns:
            情感分析結果字典
        """
        if not text or not isinstance(text, str):
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'positive_words': [],
                'negative_words': [],
                'method': method
            }
        
        if method == 'dictionary':
            return self._analyze_with_dictionary(text)
        elif method == 'rule_based':
            return self._analyze_with_rules(text)
        elif method == 'hybrid':
            return self._analyze_hybrid(text)
        else:
            logger.warning(f"未知的分析方法: {method}, 使用預設方法")
            return self._analyze_with_dictionary(text)
    
    def _analyze_with_dictionary(self, text: str) -> Dict:
        """使用詞典法分析情感"""
        # 文本預處理
        clean_text = self._preprocess_text(text)
        words = list(jieba.cut(clean_text))
        
        positive_score = 0.0
        negative_score = 0.0
        positive_words_found = []
        negative_words_found = []
        
        for i, word in enumerate(words):
            word = word.strip()
            if not word:
                continue
            
            # 檢查是否為正面詞彙
            if word in self.positive_words:
                intensity = self._get_word_intensity(words, i)
                positive_score += intensity
                positive_words_found.append(word)
            
            # 檢查是否為負面詞彙
            elif word in self.negative_words:
                intensity = self._get_word_intensity(words, i)
                negative_score += intensity
                negative_words_found.append(word)
            
            # 檢查是否為金融關鍵詞
            elif word in self.financial_keywords:
                # 金融關鍵詞本身不帶情感，但會影響周圍詞彙的權重
                continue
        
        # 計算最終分數
        total_score = positive_score - negative_score
        
        # 確定情感類別
        if total_score > 0.5:
            sentiment = 'positive'
            confidence = min(abs(total_score) / 5.0, 1.0)  # 正規化到 0-1
        elif total_score < -0.5:
            sentiment = 'negative'
            confidence = min(abs(total_score) / 5.0, 1.0)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'score': total_score,
            'confidence': confidence,
            'positive_words': positive_words_found,
            'negative_words': negative_words_found,
            'method': 'dictionary'
        }
    
    def _analyze_with_rules(self, text: str) -> Dict:
        """使用規則法分析情感"""
        # 基礎詞典分析
        dict_result = self._analyze_with_dictionary(text)
        
        # 添加規則修正
        score = dict_result['score']
        
        # 規則1: 否定詞反轉
        negation_words = ['不', '沒', '無', '非', '未', '別', '勿', '莫']
        words = list(jieba.cut(text))
        
        for i, word in enumerate(words):
            if word in negation_words and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in self.positive_words:
                    score -= 1.0  # 正面詞彙被否定，減分
                elif next_word in self.negative_words:
                    score += 1.0  # 負面詞彙被否定，加分
        
        # 規則2: 疑問句降低信心度
        if '？' in text or '?' in text or '嗎' in text:
            dict_result['confidence'] *= 0.7
        
        # 規則3: 感嘆號增加強度
        if '！' in text or '!' in text:
            score *= 1.2
        
        # 更新結果
        dict_result['score'] = score
        if score > 0.5:
            dict_result['sentiment'] = 'positive'
        elif score < -0.5:
            dict_result['sentiment'] = 'negative'
        else:
            dict_result['sentiment'] = 'neutral'
        
        dict_result['method'] = 'rule_based'
        return dict_result
    
    def _analyze_hybrid(self, text: str) -> Dict:
        """混合方法分析情感"""
        # 詞典法結果
        dict_result = self._analyze_with_dictionary(text)
        
        # 規則法結果
        rule_result = self._analyze_with_rules(text)
        
        # 簡單加權平均
        final_score = (dict_result['score'] * 0.6 + rule_result['score'] * 0.4)
        final_confidence = (dict_result['confidence'] + rule_result['confidence']) / 2
        
        # 確定最終情感
        if final_score > 0.3:
            sentiment = 'positive'
        elif final_score < -0.3:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': final_score,
            'confidence': final_confidence,
            'positive_words': dict_result['positive_words'] + rule_result['positive_words'],
            'negative_words': dict_result['negative_words'] + rule_result['negative_words'],
            'method': 'hybrid'
        }
    
    def _preprocess_text(self, text: str) -> str:
        """文本預處理"""
        # 移除特殊字符但保留中文標點
        text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？；：""''（）【】]', ' ', text)
        
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _get_word_intensity(self, words: List[str], word_index: int) -> float:
        """獲取詞彙強度"""
        base_intensity = 1.0
        
        # 檢查前一個詞是否為強度詞彙
        if word_index > 0:
            prev_word = words[word_index - 1]
            if prev_word in self.intensity_words:
                base_intensity *= self.intensity_words[prev_word]
        
        # 檢查後一個詞是否為強度詞彙
        if word_index < len(words) - 1:
            next_word = words[word_index + 1]
            if next_word in self.intensity_words:
                base_intensity *= self.intensity_words[next_word]
        
        return base_intensity
    
    def analyze_news_batch(self, news_df: pd.DataFrame, 
                          text_column: str = 'title',
                          method: str = 'hybrid') -> pd.DataFrame:
        """
        批量分析新聞情感
        
        Args:
            news_df: 新聞數據 DataFrame
            text_column: 要分析的文本欄位名稱
            method: 分析方法
            
        Returns:
            包含情感分析結果的 DataFrame
        """
        if news_df.empty:
            logger.warning("輸入的新聞數據為空")
            return news_df
        
        logger.info(f"開始批量分析 {len(news_df)} 則新聞的情感...")
        
        results = []
        for idx, row in news_df.iterrows():
            text = row[text_column] if text_column in row else ''
            
            # 分析情感
            sentiment_result = self.analyze_sentiment(text, method)
            
            # 添加到結果
            results.append({
                'index': idx,
                'sentiment': sentiment_result['sentiment'],
                'sentiment_score': sentiment_result['score'],
                'confidence': sentiment_result['confidence'],
                'positive_words': '|'.join(sentiment_result['positive_words']),
                'negative_words': '|'.join(sentiment_result['negative_words'])
            })
            
            if (idx + 1) % 100 == 0:
                logger.info(f"已處理 {idx + 1}/{len(news_df)} 則新聞")
        
        # 轉換為 DataFrame
        sentiment_df = pd.DataFrame(results)
        
        # 合併到原始數據
        result_df = news_df.copy()
        result_df['sentiment'] = sentiment_df['sentiment'].values
        result_df['sentiment_score'] = sentiment_df['sentiment_score'].values
        result_df['sentiment_confidence'] = sentiment_df['confidence'].values
        result_df['positive_words'] = sentiment_df['positive_words'].values
        result_df['negative_words'] = sentiment_df['negative_words'].values
        
        # 統計結果
        sentiment_counts = result_df['sentiment'].value_counts()
        logger.info(f"情感分析完成:")
        logger.info(f"  正面: {sentiment_counts.get('positive', 0)} 則")
        logger.info(f"  負面: {sentiment_counts.get('negative', 0)} 則")
        logger.info(f"  中性: {sentiment_counts.get('neutral', 0)} 則")
        logger.info(f"  平均信心度: {result_df['sentiment_confidence'].mean():.3f}")
        
        return result_df
    
    def get_stock_sentiment(self, news_df: pd.DataFrame, stock_code: str) -> Dict:
        """
        獲取特定股票的情感分析結果
        
        Args:
            news_df: 包含情感分析結果的新聞數據
            stock_code: 股票代碼
            
        Returns:
            該股票的情感分析統計
        """
        # 過濾包含該股票代碼的新聞
        stock_news = news_df[news_df['title'].str.contains(stock_code, na=False)]
        
        if stock_news.empty:
            return {
                'stock_code': stock_code,
                'news_count': 0,
                'sentiment_distribution': {},
                'average_sentiment_score': 0.0,
                'average_confidence': 0.0,
                'dominant_sentiment': 'neutral'
            }
        
        # 計算統計
        sentiment_dist = stock_news['sentiment'].value_counts().to_dict()
        avg_score = stock_news['sentiment_score'].mean()
        avg_confidence = stock_news['sentiment_confidence'].mean()
        
        # 確定主要情感
        dominant_sentiment = max(sentiment_dist, key=sentiment_dist.get) if sentiment_dist else 'neutral'
        
        return {
            'stock_code': stock_code,
            'news_count': len(stock_news),
            'sentiment_distribution': sentiment_dist,
            'average_sentiment_score': avg_score,
            'average_confidence': avg_confidence,
            'dominant_sentiment': dominant_sentiment
        }


def main():
    """測試情感分析功能"""
    analyzer = SentimentAnalyzer()
    
    # 測試文本
    test_texts = [
        "台積電股價大幅上漲，投資人看好未來前景",
        "市場對該公司財報表現感到失望，股價持續下跌",
        "公司營收持平，維持穩定表現",
        "這支股票真的很不錯！",
        "我不認為這支股票值得投資",
        "股價上漲了，但是風險也很高"
    ]
    
    print("=== 情感分析測試 ===")
    for i, text in enumerate(test_texts, 1):
        print(f"\\n測試 {i}: {text}")
        
        # 使用混合方法分析
        result = analyzer.analyze_sentiment(text, method='hybrid')
        print(f"  情感: {result['sentiment']}")
        print(f"  分數: {result['score']:.3f}")
        print(f"  信心度: {result['confidence']:.3f}")
        if result['positive_words']:
            print(f"  正面詞彙: {result['positive_words']}")
        if result['negative_words']:
            print(f"  負面詞彙: {result['negative_words']}")


if __name__ == "__main__":
    main()
