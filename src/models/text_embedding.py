import numpy as np
from typing import List, Optional
import os
import vertexai
from vertexai.language_models import TextEmbeddingModel

from .base_embedding import BaseEmbeddingGenerator


class TextEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self, google_credentials_path: Optional[str] = None, vertex_ai_region: Optional[str] = None):
        super().__init__(google_credentials_path, vertex_ai_region)
        
        if self.initialized:
            try:
                # Get project ID from environment or use the specified project ID
                project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "authentic-arch-456221-j3")
                
                # Initialize Vertex AI with project and location
                vertexai.init(project=project_id, location=self.vertex_ai_region)
                
                # Use the specified text-embedding-large model
                model_id = os.environ.get("VERTEX_EMBEDDING_MODEL", "text-embedding-large-exp-03-07")
                
                print(f"Initializing Vertex AI Text-Embeddings API with project: {project_id} and model: {model_id}")
                # Initialize the Vertex AI Embedding model directly using the provided structure
                self.text_embedding_model = TextEmbeddingModel.from_pretrained(model_id)
                print(f"Vertex AI Text-Embeddings API initialized successfully with model: {model_id}")
                self.using_vertex_ai = True
            except Exception as e:
                print(f"Warning: Failed to initialize Vertex AI Text-Embeddings API: {e}")
                print("This will result in random embeddings being used as fallback.")
                self.text_embedding_model = None
                self.using_vertex_ai = False
        else:
            self.text_embedding_model = None
            self.using_vertex_ai = False
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            # Return a zero vector for empty text
            return np.zeros(768)  # Vertex AI embedding dimension
            
        try:
            if self.using_vertex_ai and self.text_embedding_model:
                # Use Vertex AI Text Embeddings API directly
                embeddings = self.text_embedding_model.get_embeddings([text])
                if embeddings and len(embeddings) > 0 and embeddings[0].values:
                    return np.array(embeddings[0].values)
                else:
                    raise ValueError("Empty embedding response from Vertex AI")
            else:
                # No fallback to TensorFlow, use random embedding instead
                print("Warning: Using random embeddings as fallback. Vertex AI is not initialized.")
                return np.random.randn(768).astype(np.float32)  # Random with Vertex AI dimension
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            # Return a random embedding as a last resort
            return np.random.randn(768).astype(np.float32)
    
    def generate_batch_text_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
            
        try:
            if self.using_vertex_ai and self.text_embedding_model:
                # Use Vertex AI Text Embeddings API for batch processing
                # Process in smaller batches to avoid API limitations
                batch_size = 100  # Adjust based on Vertex AI limits
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    # Filter out empty strings to avoid API errors
                    valid_texts = [t for t in batch_texts if t.strip()]
                    valid_indices = [i for i, t in enumerate(batch_texts) if t.strip()]
                    
                    if valid_texts:
                        # Get embeddings for valid texts
                        batch_embeddings = self.text_embedding_model.get_embeddings(valid_texts)
                        
                        # Realign with original indices (including empty strings)
                        batch_result = [None] * len(batch_texts)
                        for idx, emb in zip(valid_indices, batch_embeddings):
                            batch_result[idx] = np.array(emb.values if hasattr(emb, 'values') else emb)
                            
                        # Replace None values with zero vectors
                        for j in range(len(batch_result)):
                            if batch_result[j] is None:
                                batch_result[j] = np.zeros(768)  # Vertex AI embedding dimension
                                
                        all_embeddings.extend(batch_result)
                
                return np.array(all_embeddings)
            else:
                # No fallback to TensorFlow, use random embeddings instead
                print("Warning: Using random embeddings as fallback for batch. Vertex AI is not initialized.")
                return np.array([np.random.randn(768).astype(np.float32) for _ in texts])
        except Exception as e:
            print(f"Error generating batch text embeddings: {e}")
            # Return random embeddings as a last resort
            return np.array([np.random.randn(768).astype(np.float32) for _ in texts])