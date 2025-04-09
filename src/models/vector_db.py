import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Union


class VectorDatabase:
    """
    Class to manage vector databases using FAISS.
    """
    
    def __init__(self, dimension_text: int = 512, dimension_image: int = 1280):
        """
        Initialize the vector database.
        
        Args:
            dimension_text (int): Dimension of text embeddings.
            dimension_image (int): Dimension of image embeddings.
        """
        self.dimension_text = dimension_text
        self.dimension_image = dimension_image
        
        # Initialize indices
        self.index_text = faiss.IndexFlatL2(dimension_text)
        self.index_image = faiss.IndexFlatL2(dimension_image)
        
        # Keep track of product ids
        self.product_ids = []
        
    def add_text_embeddings(self, embeddings: np.ndarray, ids: List[int]) -> None:
        """
        Add text embeddings to the index.
        
        Args:
            embeddings (np.ndarray): Array of text embedding vectors.
            ids (List[int]): List of product IDs.
        """
        if len(embeddings) == 0:
            return
            
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Add to index
        self.index_text.add(embeddings)
        self.product_ids.extend(ids)
        
    def add_image_embeddings(self, embeddings: np.ndarray, ids: List[int]) -> None:
        """
        Add image embeddings to the index.
        
        Args:
            embeddings (np.ndarray): Array of image embedding vectors.
            ids (List[int]): List of product IDs.
        """
        if len(embeddings) == 0:
            return
            
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Add to index
        self.index_image.add(embeddings)
        
    def search_by_text(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar products by text.
        
        Args:
            query_embedding (np.ndarray): Text embedding of the query.
            k (int): Number of results to return.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (distances, indices).
        """
        # Ensure query is float32
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search index
        distances, indices = self.index_text.search(query_embedding, k)
        
        return distances, indices
    
    def search_by_image(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar products by image.
        
        Args:
            query_embedding (np.ndarray): Image embedding of the query.
            k (int): Number of results to return.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (distances, indices).
        """
        # Ensure query is float32
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search index
        distances, indices = self.index_image.search(query_embedding, k)
        
        return distances, indices
    
    def hybrid_search(self, text_embedding: Optional[np.ndarray] = None, 
                     image_embedding: Optional[np.ndarray] = None, 
                     k: int = 5, alpha: float = 0.5) -> List[int]:
        """
        Perform hybrid search combining text and image similarity.
        
        Args:
            text_embedding (np.ndarray, optional): Text embedding of the query.
            image_embedding (np.ndarray, optional): Image embedding of the query.
            k (int): Number of results to return.
            alpha (float): Weight for text search (1-alpha will be weight for image search).
            
        Returns:
            List[int]: List of product indices.
        """
        results = {}
        
        # If we have text embedding, search by text
        if text_embedding is not None:
            text_distances, text_indices = self.search_by_text(text_embedding, k=k*2)
            
            # Add to results dict with weight
            for i, idx in enumerate(text_indices[0]):
                if idx not in results:
                    results[idx] = 0
                results[idx] += alpha * (1 - text_distances[0][i] / max(text_distances[0]))
        
        # If we have image embedding, search by image
        if image_embedding is not None:
            image_distances, image_indices = self.search_by_image(image_embedding, k=k*2)
            
            # Add to results dict with weight
            for i, idx in enumerate(image_indices[0]):
                if idx not in results:
                    results[idx] = 0
                results[idx] += (1 - alpha) * (1 - image_distances[0][i] / max(image_distances[0]))
        
        # Sort by combined score and get top k
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Return just the indices
        return [idx for idx, _ in sorted_results]
    
    def save_indices(self, save_dir: str) -> None:
        """
        Save the FAISS indices to disk.
        
        Args:
            save_dir (str): Directory to save indices to.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save text index
        faiss.write_index(self.index_text, os.path.join(save_dir, 'text_index.faiss'))
        
        # Save image index
        faiss.write_index(self.index_image, os.path.join(save_dir, 'image_index.faiss'))
        
        # Save product IDs
        with open(os.path.join(save_dir, 'product_ids.pkl'), 'wb') as f:
            pickle.dump(self.product_ids, f)
    
    def load_indices(self, save_dir: str) -> None:
        """
        Load the FAISS indices from disk.
        
        Args:
            save_dir (str): Directory to load indices from.
        """
        # Load text index
        self.index_text = faiss.read_index(os.path.join(save_dir, 'text_index.faiss'))
        
        # Load image index
        self.index_image = faiss.read_index(os.path.join(save_dir, 'image_index.faiss'))
        
        # Load product IDs
        with open(os.path.join(save_dir, 'product_ids.pkl'), 'rb') as f:
            self.product_ids = pickle.load(f)