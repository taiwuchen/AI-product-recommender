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
        
        # Extract the first image URL for each product
        self.df['first_image_url'] = self.df['product_images'].apply(self._extract_first_image_url)
        
        # Create a combined text field for text embedding
        self.df['text_for_embedding'] = self.df['product_name'] + '. ' + self.df['details'].fillna('')
        
        return self.df
    
    def _extract_first_image_url(self, image_data_str: str) -> str:
        if not image_data_str or not isinstance(image_data_str, str):
            return ""
            
        # If it looks like a direct URL, return it as is
        if image_data_str.startswith('http'):
            return image_data_str.strip()
            
        # Otherwise try to parse it as a JSON structure
        try:
            # Convert string representation of list to actual list
            image_data = ast.literal_eval(image_data_str)
            
            # Get the first image URL (key of the first dictionary)
            for item in image_data:
                if isinstance(item, dict):
                    return list(item.keys())[0]
            
            return ""
        except (SyntaxError, ValueError):
            # If we can't parse it, return the raw string (might be a URL with special chars)
            return image_data_str.strip()
    
    def get_product_details(self, product_indices: List[int]) -> List[Dict]:
        if self.df is None:
            self.preprocess_data()
            
        products = []
        for idx in product_indices:
            if 0 <= idx < len(self.df):
                # Get the processed image URL from first_image_url column
                image_url = self.df.loc[idx, 'first_image_url']
                
                # Make sure we have a valid image URL
                if not image_url or not isinstance(image_url, str):
                    # Fallback to raw product_images as a direct URL
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