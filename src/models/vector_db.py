import numpy as np
import faiss
import pickle
import os
import logging
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Vector database using FAISS for similarity search."""
    
    def __init__(self):
        """Initialize the vector database."""
        self.text_index = None
        self.image_index = None
        self.hybrid_index = None
        self.product_ids = []
        self.product_data = {}
        
    def build_text_index(self, product_ids: List[str], text_embeddings: List[np.ndarray]) -> None:
        """
        Build the text embedding index.
        
        Args:
            product_ids: List of product IDs
            text_embeddings: List of text embedding vectors
        """
        embeddings = np.array(text_embeddings).astype('float32')
        dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.text_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.text_index.add(embeddings)
        
        # Store product IDs
        self.product_ids = product_ids
        
        logger.info(f"Built text index with {len(product_ids)} products and dimension {dimension}")
        
    def build_image_index(self, product_ids: List[str], image_embeddings: List[np.ndarray]) -> None:
        """
        Build the image embedding index.
        
        Args:
            product_ids: List of product IDs
            image_embeddings: List of image embedding vectors
        """
        embeddings = np.array(image_embeddings).astype('float32')
        dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.image_index = faiss.IndexFlatIP(dimension)
        self.image_index.add(embeddings)
        
        # Store product IDs if not already stored
        if not self.product_ids:
            self.product_ids = product_ids
            
        logger.info(f"Built image index with {len(product_ids)} products and dimension {dimension}")
        
    def build_hybrid_index(self, product_ids: List[str], hybrid_embeddings: List[np.ndarray]) -> None:
        """
        Build the hybrid embedding index.
        
        Args:
            product_ids: List of product IDs
            hybrid_embeddings: List of hybrid embedding vectors
        """
        embeddings = np.array(hybrid_embeddings).astype('float32')
        dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.hybrid_index = faiss.IndexFlatIP(dimension)
        self.hybrid_index.add(embeddings)
        
        # Store product IDs if not already stored
        if not self.product_ids:
            self.product_ids = product_ids
            
        logger.info(f"Built hybrid index with {len(product_ids)} products and dimension {dimension}")
        
    def store_product_data(self, product_data: Dict[str, Dict]) -> None:
        """
        Store product metadata for retrieval.
        
        Args:
            product_data: Dictionary with product metadata
        """
        self.product_data = product_data
        logger.info(f"Stored metadata for {len(product_data)} products")
        
    def search_by_text(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar products using text embedding.
        
        Args:
            query_embedding: Query text embedding
            k: Number of results to return
            
        Returns:
            List of similar product metadata
        """
        if self.text_index is None:
            logger.error("Text index is not built")
            return []
            
        # Convert to numpy array and ensure correct shape and type
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        
        # Search the index
        distances, indices = self.text_index.search(query_embedding, k)
        
        # Get product data for results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.product_ids) and idx >= 0:
                product_id = self.product_ids[idx]
                if product_id in self.product_data:
                    result = self.product_data[product_id].copy()
                    result['similarity_score'] = float(distances[0][i])
                    results.append(result)
                    
        return results
        
    def search_by_image(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar products using image embedding.
        
        Args:
            query_embedding: Query image embedding
            k: Number of results to return
            
        Returns:
            List of similar product metadata
        """
        if self.image_index is None:
            logger.error("Image index is not built")
            return []
            
        # Convert to numpy array and ensure correct shape and type
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        
        # Search the index
        distances, indices = self.image_index.search(query_embedding, k)
        
        # Get product data for results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.product_ids) and idx >= 0:
                product_id = self.product_ids[idx]
                if product_id in self.product_data:
                    result = self.product_data[product_id].copy()
                    result['similarity_score'] = float(distances[0][i])
                    results.append(result)
                    
        return results
        
    def search_by_hybrid(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar products using hybrid embedding.
        
        Args:
            query_embedding: Query hybrid embedding
            k: Number of results to return
            
        Returns:
            List of similar product metadata
        """
        if self.hybrid_index is None:
            logger.error("Hybrid index is not built")
            return []
            
        # Convert to numpy array and ensure correct shape and type
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        
        # Search the index
        distances, indices = self.hybrid_index.search(query_embedding, k)
        
        # Get product data for results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.product_ids) and idx >= 0:
                product_id = self.product_ids[idx]
                if product_id in self.product_data:
                    result = self.product_data[product_id].copy()
                    result['similarity_score'] = float(distances[0][i])
                    results.append(result)
                    
        return results
    
    def save_indexes(self, save_dir: str) -> None:
        """
        Save FAISS indexes and associated data to disk.
        
        Args:
            save_dir: Directory to save indexes
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save indexes
        if self.text_index is not None:
            faiss.write_index(self.text_index, os.path.join(save_dir, "text_index.faiss"))
        
        if self.image_index is not None:
            faiss.write_index(self.image_index, os.path.join(save_dir, "image_index.faiss"))
            
        if self.hybrid_index is not None:
            faiss.write_index(self.hybrid_index, os.path.join(save_dir, "hybrid_index.faiss"))
            
        # Save product IDs and metadata
        with open(os.path.join(save_dir, "product_ids.pkl"), "wb") as f:
            pickle.dump(self.product_ids, f)
            
        with open(os.path.join(save_dir, "product_data.pkl"), "wb") as f:
            pickle.dump(self.product_data, f)
            
        logger.info(f"Saved indexes and data to {save_dir}")
        
    def load_indexes(self, load_dir: str) -> None:
        """
        Load FAISS indexes and associated data from disk.
        
        Args:
            load_dir: Directory to load indexes from
        """
        # Load indexes if files exist
        text_index_path = os.path.join(load_dir, "text_index.faiss")
        if os.path.exists(text_index_path):
            self.text_index = faiss.read_index(text_index_path)
            logger.info(f"Loaded text index from {text_index_path}")
            
        image_index_path = os.path.join(load_dir, "image_index.faiss")
        if os.path.exists(image_index_path):
            self.image_index = faiss.read_index(image_index_path)
            logger.info(f"Loaded image index from {image_index_path}")
            
        hybrid_index_path = os.path.join(load_dir, "hybrid_index.faiss")
        if os.path.exists(hybrid_index_path):
            self.hybrid_index = faiss.read_index(hybrid_index_path)
            logger.info(f"Loaded hybrid index from {hybrid_index_path}")
            
        # Load product IDs and metadata
        product_ids_path = os.path.join(load_dir, "product_ids.pkl")
        if os.path.exists(product_ids_path):
            with open(product_ids_path, "rb") as f:
                self.product_ids = pickle.load(f)
            logger.info(f"Loaded product IDs from {product_ids_path}")
            
        product_data_path = os.path.join(load_dir, "product_data.pkl")
        if os.path.exists(product_data_path):
            with open(product_data_path, "rb") as f:
                self.product_data = pickle.load(f)
            logger.info(f"Loaded product data from {product_data_path}")