"""
文本清理與預處理模組
處理新聞文本，包括斷詞、去停用詞等
"""

import pandas as pd
import jieba
import re
from typing import List, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TextCleaner:
    """文本清理器"""
    
    def __init__(self):
        # 停用詞列表
        self.stop_words = self._load_stop_words()
        
        # 財經相關停用詞
        self.finance_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一個',
            '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有', '看', '好',
            '自己', '這', '那', '什麼', '怎麼', '為什麼', '因為', '所以', '但是', '然後'
        }
        
        # 合併停用詞
        self.all_stop_words = self.stop_words.union(self.finance_stop_words)
    
    def _load_stop_words(self) -> set:
        """載入停用詞"""
        try:
            # 嘗試載入中文停用詞
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            
            # 簡體中文停用詞
            chinese_stop_words = set(stopwords.words('chinese'))
            return chinese_stop_words
        except:
            # 如果無法載入，使用基本停用詞
            return set()
    
    def clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理後的文本
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 移除HTML標籤
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除特殊字符，保留中文、英文、數字
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
        
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除數字（可選）
        # text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def segment_text(self, text: str) -> List[str]:
        """
        中文斷詞
        
        Args:
            text: 清理後的文本
            
        Returns:
            斷詞後的詞彙列表
        """
        if not text:
            return []
        
        # 使用 jieba 斷詞
        words = jieba.lcut(text)
        
        # 過濾停用詞和短詞
        filtered_words = [
            word for word in words 
            if len(word) > 1 and word not in self.all_stop_words
        ]
        
        return filtered_words
    
    def process_news_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批次處理新聞數據
        
        Args:
            df: 包含新聞標題的 DataFrame
            
        Returns:
            處理後的 DataFrame
        """
        result_df = df.copy()
        
        # 清理標題
        result_df['cleaned_title'] = result_df['title'].apply(self.clean_text)
        
        # 斷詞
        result_df['segmented_words'] = result_df['cleaned_title'].apply(self.segment_text)
        
        # 計算詞彙數量
        result_df['word_count'] = result_df['segmented_words'].apply(len)
        
        # 提取股票代碼（如果有）
        result_df['stock_mentions'] = result_df['title'].apply(self._extract_stock_codes)
        
        return result_df
    
    def _extract_stock_codes(self, text: str) -> List[str]:
        """從文本中提取股票代碼"""
        if pd.isna(text):
            return []
        
        # 股票代碼模式
        pattern = r'\b(\d{4})\b'
        matches = re.findall(pattern, text)
        
        # 過濾可能的股票代碼
        stock_codes = []
        for match in matches:
            if 1000 <= int(match) <= 9999:
                stock_codes.append(match)
        
        return list(set(stock_codes))


def main():
    """測試文本清理功能"""
    cleaner = TextCleaner()
    
    # 測試文本
    test_text = "台積電(2330)今日股價上漲3.5%，成交量放大至10萬張。"
    
    print(f"原始文本: {test_text}")
    
    # 清理文本
    cleaned = cleaner.clean_text(test_text)
    print(f"清理後: {cleaned}")
    
    # 斷詞
    words = cleaner.segment_text(cleaned)
    print(f"斷詞結果: {words}")
    
    # 提取股票代碼
    stock_codes = cleaner._extract_stock_codes(test_text)
    print(f"股票代碼: {stock_codes}")


if __name__ == "__main__":
    main()




