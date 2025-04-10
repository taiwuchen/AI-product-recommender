import pandas as pd
import ast
from typing import List, Dict, Optional
import os


class ProductDataLoader:
    """
    Class to load and preprocess product data from CSV files.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def preprocess_data(self) -> pd.DataFrame:
        if self.df is None:
            self.load_data()
        
        # Extract the first image URL for each product
        self.df['first_image_url'] = self.df['product_images'].apply(self._extract_first_image_url)
        
        # Create a combined text field for text embedding
        self.df['text_for_embedding'] = self.df['product_name'] + '. ' + self.df['details'].fillna('')
        
        return self.df
    
    def _extract_first_image_url(self, image_data_str: str) -> str:
        try:
            # Convert string representation of list to actual list
            image_data = ast.literal_eval(image_data_str)
            
            # Get the first image URL (key of the first dictionary)
            for item in image_data:
                if isinstance(item, dict):
                    return list(item.keys())[0]
            
            return ""
        except (SyntaxError, ValueError):
            return ""
    
    def get_product_details(self, product_indices: List[int]) -> List[Dict]:
        if self.df is None:
            self.preprocess_data()
            
        products = []
        for idx in product_indices:
            if 0 <= idx < len(self.df):
                # Get direct image URL from product_images column (already contains a valid URL)
                image_url = self.df.loc[idx, 'product_images']
                
                product = {
                    'name': self.df.loc[idx, 'product_name'],
                    'details': self.df.loc[idx, 'details'],
                    'image_url': image_url,
                    'link': self.df.loc[idx, 'link']
                }
                products.append(product)
                
        return products