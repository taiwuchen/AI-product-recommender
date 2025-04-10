import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Union


class VectorDatabase:
    """
    Class to manage vector databases using FAISS.
    """
    
    def __init__(self, dimension_text: int = 768, dimension_image: int = 1280):
        """
        Initialize the vector database.
        
        Args:
            dimension_text (int): Dimension of text embeddings (default is 768 for Vertex AI).
            dimension_image (int): Dimension of image embeddings.
        """
        self.dimension_text = dimension_text
        self.dimension_image = dimension_image
        
        # Initialize indices
        self.index_text = faiss.IndexFlatL2(dimension_text)
        self.index_image = faiss.IndexFlatL2(dimension_image)
        
        # Keep track of product ids
        self.product_ids = []
        
        # Store original data for keyword filtering
        self.product_texts = []
        
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
        
        # Check if the embedding dimension matches the index
        if embeddings.shape[1] != self.dimension_text:
            print(f"Warning: Embedding dimension mismatch. Expected {self.dimension_text}, got {embeddings.shape[1]}.")
            # Create a new index with the correct dimension
            self.dimension_text = embeddings.shape[1]
            self.index_text = faiss.IndexFlatL2(self.dimension_text)
            print(f"Created new text index with dimension {self.dimension_text}")
            
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
    
    def set_product_texts(self, texts: List[str]) -> None:
        """
        Store original product text data for keyword filtering.
        
        Args:
            texts (List[str]): List of product text data.
        """
        self.product_texts = texts
        
    def search_by_text(self, query_embedding: np.ndarray, k: int = 5, 
                       query_text: Optional[str] = None, keyword_boost: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar products by text, with optional keyword boosting.
        
        Args:
            query_embedding (np.ndarray): Text embedding of the query.
            k (int): Number of results to return.
            query_text (str, optional): The original query text for keyword matching.
            keyword_boost (bool): Whether to boost results containing query keywords.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (distances, indices).
        """
        # Check if the embedding dimension matches the index
        if query_embedding.shape[0] != self.dimension_text:
            print(f"Warning: Query embedding dimension mismatch. Expected {self.dimension_text}, got {query_embedding.shape[0]}.")
            # We can't search with mismatched dimensions, so we need to return empty results
            return np.array([[0.0] * k]), np.array([[0] * k])
            
        # Ensure query is float32
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Get more results than needed for filtering
        search_k = min(k * 3, len(self.product_ids)) if keyword_boost and query_text else k
        
        # Search index
        distances, indices = self.index_text.search(query_embedding, search_k)
        
        if keyword_boost and query_text and len(self.product_texts) > 0:
            # Extract important keywords from query (simple approach)
            keywords = [kw.lower() for kw in query_text.split() if len(kw) > 2]
            
            if keywords:
                # Score based on keyword presence
                keyword_scores = {}
                for i, idx in enumerate(indices[0]):
                    if idx >= len(self.product_texts):
                        continue
                        
                    text = self.product_texts[idx].lower()
                    # Base score from vector similarity
                    score = 1.0 - distances[0][i] / max(distances[0]) if max(distances[0]) > 0 else 1.0
                    
                    # Boost for each keyword present
                    keyword_matches = sum(1 for kw in keywords if kw in text)
                    if keyword_matches > 0:
                        # Significantly boost score for keyword matches
                        score *= (1.0 + 0.5 * keyword_matches)
                        
                    keyword_scores[idx] = score
                    
                # Sort by new scores
                if keyword_scores:
                    sorted_results = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                    
                    # Convert to numpy arrays
                    new_indices = np.array([[idx for idx, _ in sorted_results]])
                    new_distances = np.array([[1.0 - score for _, score in sorted_results]])
                    
                    return new_distances, new_indices
        
        # Return original results if no keyword boosting or no keywords found
        return distances, indices[:, :k]
    
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
                     k: int = 5, alpha: float = 0.5,
                     query_text: Optional[str] = None) -> List[int]:
        """
        Perform hybrid search combining text and image similarity.
        
        Args:
            text_embedding (np.ndarray, optional): Text embedding of the query.
            image_embedding (np.ndarray, optional): Image embedding of the query.
            k (int): Number of results to return.
            alpha (float): Weight for text search (1-alpha will be weight for image search).
            query_text (str, optional): Original query text for keyword matching.
            
        Returns:
            List[int]: List of product indices.
        """
        results = {}
        
        # If we have text embedding, search by text
        if text_embedding is not None:
            text_distances, text_indices = self.search_by_text(
                text_embedding, k=k*2, query_text=query_text)
            
            # Add to results dict with weight
            for i, idx in enumerate(text_indices[0]):
                if idx not in results:
                    results[idx] = 0
                results[idx] += alpha * (1 - text_distances[0][i] / max(text_distances[0]) if max(text_distances[0]) > 0 else 1.0)
        
        # If we have image embedding, search by image
        if image_embedding is not None:
            image_distances, image_indices = self.search_by_image(image_embedding, k=k*2)
            
            # Add to results dict with weight
            for i, idx in enumerate(image_indices[0]):
                if idx not in results:
                    results[idx] = 0
                results[idx] += (1 - alpha) * (1 - image_distances[0][i] / max(image_distances[0]) if max(image_distances[0]) > 0 else 1.0)
        
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
        
        # Save product IDs and texts
        with open(os.path.join(save_dir, 'product_ids.pkl'), 'wb') as f:
            pickle.dump(self.product_ids, f)
            
        # Save product texts for keyword filtering
        with open(os.path.join(save_dir, 'product_texts.pkl'), 'wb') as f:
            pickle.dump(self.product_texts, f)
            
        # Save dimensions
        with open(os.path.join(save_dir, 'dimensions.pkl'), 'wb') as f:
            pickle.dump({'text': self.dimension_text, 'image': self.dimension_image}, f)
    
    def load_indices(self, save_dir: str) -> None:
        """
        Load the FAISS indices from disk.
        
        Args:
            save_dir (str): Directory to load indices from.
        """
        # Try to load dimensions if available
        try:
            with open(os.path.join(save_dir, 'dimensions.pkl'), 'rb') as f:
                dimensions = pickle.load(f)
                self.dimension_text = dimensions.get('text', self.dimension_text)
                self.dimension_image = dimensions.get('image', self.dimension_image)
        except (FileNotFoundError, EOFError):
            # Backward compatibility - dimensions will be inferred from loaded indices
            pass
            
        # Load text index
        self.index_text = faiss.read_index(os.path.join(save_dir, 'text_index.faiss'))
        
        # Update dimension from loaded index
        if hasattr(self.index_text, 'd'):
            self.dimension_text = self.index_text.d
        
        # Load image index
        self.index_image = faiss.read_index(os.path.join(save_dir, 'image_index.faiss'))
        
        # Update dimension from loaded index
        if hasattr(self.index_image, 'd'):
            self.dimension_image = self.index_image.d
        
        # Load product IDs
        with open(os.path.join(save_dir, 'product_ids.pkl'), 'rb') as f:
            self.product_ids = pickle.load(f)
            
        # Try to load product texts if available
        try:
            with open(os.path.join(save_dir, 'product_texts.pkl'), 'rb') as f:
                self.product_texts = pickle.load(f)
        except (FileNotFoundError, EOFError):
            # Backward compatibility
            self.product_texts = []