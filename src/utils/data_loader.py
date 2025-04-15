import pandas as pd
from typing import List, Dict, Optional


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

        self.df['image_url'] = self.df['product_images'].apply(lambda x: x.strip() if isinstance(x, str) and x.startswith('http') else "")
        self.df['text_for_embedding'] = self.df['product_name'] + '. ' + self.df['details'].fillna('')
        
        return self.df
    
    def get_product_details(self, product_indices: List[int]) -> List[Dict]:
        if self.df is None:
            self.preprocess_data()
            
        products = []
        for idx in product_indices:
            if 0 <= idx < len(self.df):
                image_url = self.df.loc[idx, 'image_url']
                # Set to None if the extracted URL was empty
                if not image_url: 
                    image_url = None
                
                product = {
                    'name': self.df.loc[idx, 'product_name'],
                    'details': self.df.loc[idx, 'details'],
                    'image_url': image_url,
                    'link': self.df.loc[idx, 'link']
                }
                products.append(product)
                
        return products
