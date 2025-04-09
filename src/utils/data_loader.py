import pandas as pd
import ast
from typing import List, Dict, Optional
import os


class ProductDataLoader:
    """
    Class to load and preprocess product data from CSV files.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the CSV file containing product data.
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the data from CSV file.
        
        Returns:
            pd.DataFrame: DataFrame containing the product data.
        """
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the data for embedding generation.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        if self.df is None:
            self.load_data()
        
        # Extract the first image URL for each product
        self.df['first_image_url'] = self.df['product_images'].apply(self._extract_first_image_url)
        
        # Create a combined text field for text embedding
        self.df['text_for_embedding'] = self.df['product_name'] + '. ' + self.df['details'].fillna('')
        
        return self.df
    
    def _extract_first_image_url(self, image_data_str: str) -> str:
        """
        Extract the first image URL from the product_images column.
        
        Args:
            image_data_str (str): String representation of image data.
            
        Returns:
            str: First image URL or empty string if extraction fails.
        """
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
        """
        Get details for a list of product indices.
        
        Args:
            product_indices (List[int]): List of product indices.
            
        Returns:
            List[Dict]: List of dictionaries with product details.
        """
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