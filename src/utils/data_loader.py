import pandas as pd
import ast
from typing import List, Dict, Optional
import os


class ProductDataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def preprocess_data(self) -> pd.DataFrame:
        if self.df is None:
            self.load_data()

        self.df['first_image_url'] = self.df['product_images'].apply(self._extract_first_image_url)
        self.df['text_for_embedding'] = self.df['product_name'] + '. ' + self.df['details'].fillna('')
        
        return self.df
    
    def _extract_first_image_url(self, image_data_str: str) -> str:
        if not image_data_str or not isinstance(image_data_str, str):
            return ""

        if image_data_str.startswith('http'):
            return image_data_str.strip()
            
        try:
            image_data = ast.literal_eval(image_data_str)
            
            for item in image_data:
                if isinstance(item, dict):
                    return list(item.keys())[0]
            
            return ""
        except (SyntaxError, ValueError):
            return image_data_str.strip()
    
    def get_product_details(self, product_indices: List[int]) -> List[Dict]:
        if self.df is None:
            self.preprocess_data()
            
        products = []
        for idx in product_indices:
            if 0 <= idx < len(self.df):
                image_url = self.df.loc[idx, 'first_image_url']

                if not image_url or not isinstance(image_url, str):
                    image_url = self.df.loc[idx, 'product_images']
                    if not (isinstance(image_url, str) and image_url.startswith('http')):
                        image_url = None
                
                product = {
                    'name': self.df.loc[idx, 'product_name'],
                    'details': self.df.loc[idx, 'details'],
                    'image_url': image_url,
                    'link': self.df.loc[idx, 'link']
                }
                products.append(product)
                
        return products