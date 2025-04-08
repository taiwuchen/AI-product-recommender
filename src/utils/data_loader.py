import pandas as pd
import numpy as np
import ast
import os
import logging
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class to load and preprocess product data."""
    
    def __init__(self, csv_path: str):
        """
        Initialize DataLoader with path to CSV file.
        
        Args:
            csv_path: Path to the CSV file with product data
        """
        self.csv_path = csv_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame with product data
        """
        logger.info(f"Loading data from {self.csv_path}")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Data file not found at {self.csv_path}")
            
        self.data = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.data)} products")
        return self.data
        
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the data for use in the recommender system.
        
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            self.load_data()
            
        # Clean up column names
        self.data.columns = [col.strip() for col in self.data.columns]
        
        # Convert product_images from string to list of dicts
        try:
            self.data['product_images'] = self.data['product_images'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            
            # Extract first image URL for each product
            self.data['primary_image_url'] = self.data['product_images'].apply(
                lambda imgs: list(imgs[0].keys())[0] if isinstance(imgs, list) and len(imgs) > 0 else None
            )
            
            logger.info("Successfully processed image URLs")
        except (SyntaxError, ValueError) as e:
            logger.error(f"Error processing image data: {e}")
            # If there's an error, create a fallback column
            self.data['primary_image_url'] = None
            
        # Clean text fields
        if 'details' in self.data.columns:
            self.data['details'] = self.data['details'].fillna('').astype(str)
        
        if 'product_name' in self.data.columns:
            self.data['product_name'] = self.data['product_name'].fillna('').astype(str)
            
        # Create combined text field for embedding
        self.data['text_for_embedding'] = self.data['product_name'] + ". " + self.data['details']
        
        # Use index as product_id if no explicit ID column
        if 'product_id' not in self.data.columns:
            self.data['product_id'] = self.data.index.astype(str)
            
        logger.info("Data preprocessing completed")
        return self.data
    
    def get_product_data(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Get processed product data for embedding generation.
        
        Returns:
            Tuple of (product IDs, text for embedding, image URLs)
        """
        if self.data is None or 'text_for_embedding' not in self.data.columns:
            self.preprocess_data()
            
        product_ids = self.data['product_id'].tolist()
        texts = self.data['text_for_embedding'].tolist()
        image_urls = self.data['primary_image_url'].tolist()
        
        return product_ids, texts, image_urls
    
    def create_product_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Create a dictionary of product metadata.
        
        Returns:
            Dictionary with product ID as key and product details as value
        """
        if self.data is None:
            self.preprocess_data()
            
        product_data = {}
        for _, row in self.data.iterrows():
            product_id = row['product_id']
            product_data[product_id] = {
                'product_id': product_id,
                'product_name': row['product_name'],
                'details': row['details'],
                'link': row.get('link', ''),
                'primary_image_url': row.get('primary_image_url', '')
            }
            
        logger.info(f"Created metadata dictionary with {len(product_data)} products")
        return product_data