import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List, Optional
from google.cloud import aiplatform

from .base_embedding import BaseEmbeddingGenerator


class TextEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Class to generate text embeddings using Google Vertex AI or TensorFlow.
    """
    
    def __init__(self, google_credentials_path: Optional[str] = None, vertex_ai_region: Optional[str] = None):
        """
        Initialize the text embedding generator.
        
        Args:
            google_credentials_path (str, optional): Path to Google Cloud service account credentials.
            vertex_ai_region (str, optional): Google Cloud region for Vertex AI.
        """
        super().__init__(google_credentials_path, vertex_ai_region)
        self._load_text_model()
    
    def _load_text_model(self):
        """Load TensorFlow model for text embeddings as fallback."""
        try:
            print("Loading text embedding model...")
            
            # Disable SSL verification for TensorFlow Hub
            import os
            os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub_cache'
            
            # Load Universal Sentence Encoder for text embeddings
            self.text_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            print("Text embedding model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load text embedding model: {e}")
            print("Using random text embeddings as fallback")
            self.text_model = None
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using Google Vertex AI or TensorFlow model.
        
        Args:
            text (str): Text to generate embedding for.
            
        Returns:
            np.ndarray: Text embedding vector.
        """
        if not text.strip():
            # Return a zero vector for empty text
            return np.zeros(512)
            
        try:
            if self.initialized:
                # Use Google Vertex AI
                endpoint = aiplatform.Endpoint("projects/{project}/locations/{location}/endpoints/{id}")
                response = endpoint.predict(instances=[text])
                embedding = np.array(response.predictions[0])
                return embedding
            else:
                # Use TensorFlow model as fallback
                embedding = self.text_model([text])[0].numpy()
                return embedding
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            # Return a random embedding as a last resort
            return np.random.randn(512).astype(np.float32)
    
    def generate_batch_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for.
            
        Returns:
            np.ndarray: Array of text embedding vectors.
        """
        embeddings = []
        for text in texts:
            embedding = self.generate_text_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)