import numpy as np
from typing import List, Optional
import os
import vertexai
from vertexai.language_models import TextEmbeddingModel
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.oauth2 import service_account

# Base class moved from base_embedding.py
class BaseEmbeddingGenerator:
    
    def __init__(self, google_credentials_path: Optional[str] = None, vertex_ai_region: Optional[str] = None):
        load_dotenv()
        # Initialize Google Cloud credentials
        self.credentials = None
        self.initialized = False
        self.vertex_ai_region = vertex_ai_region or os.environ.get("VERTEX_AI_REGION", "us-central1")
        
        try:
            if google_credentials_path and os.path.exists(google_credentials_path):
                self.credentials = service_account.Credentials.from_service_account_file(google_credentials_path)
                aiplatform.init(credentials=self.credentials, project=os.environ.get("GOOGLE_CLOUD_PROJECT"), 
                               location=self.vertex_ai_region)
                self.initialized = True
            else:
                # Use default credentials if path not provided or file doesn't exist
                aiplatform.init(location=self.vertex_ai_region)
                self.initialized = True
                
            print(f"Google Cloud Vertex AI initialized successfully with region: {self.vertex_ai_region}")
        except Exception as e:
            print(f"Warning: Google Cloud initialization failed: {e}")


class TextEmbeddingGenerator(BaseEmbeddingGenerator):
    def __init__(self, google_credentials_path: Optional[str] = None, vertex_ai_region: Optional[str] = None):
        super().__init__(google_credentials_path, vertex_ai_region)
        
        print(f"TextEmbeddingGenerator initialization: credentials path={google_credentials_path}, region={self.vertex_ai_region}")
        
        if self.initialized:
            try:
                # Get project ID from environment or use the specified project ID
                project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "authentic-arch-456221-j3")
                
                print(f"Attempting to initialize Vertex AI with project: {project_id}, region: {self.vertex_ai_region}")
                # Initialize Vertex AI with project and location
                vertexai.init(project=project_id, location=self.vertex_ai_region)
                
                # Use the recommended text-embedding-005 model (gecko is being discontinued)
                model_id = os.environ.get("VERTEX_EMBEDDING_MODEL", "text-embedding-005")
                
                print(f"Initializing Vertex AI Text-Embeddings API with project: {project_id} and model: {model_id}")
                # Initialize the Vertex AI Embedding model directly using the provided structure
                self.text_embedding_model = TextEmbeddingModel.from_pretrained(model_id)
                print(f"✅ Vertex AI Text-Embeddings API initialized successfully with model: {model_id}")
                self.using_vertex_ai = True
            except Exception as e:
                print(f"❌ ERROR: Failed to initialize Vertex AI Text-Embeddings API: {e}")
                print("The application requires Google Vertex AI Text Embeddings. Please check your credentials.")
                self.text_embedding_model = None
                self.using_vertex_ai = False
                raise RuntimeError(f"Vertex AI initialization failed: {e}")
        else:
            print("❌ ERROR: Base authentication failed")
            self.text_embedding_model = None
            self.using_vertex_ai = False
            raise RuntimeError("Failed to authenticate with Google Cloud")
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            # Return a zero vector for empty text
            print("Empty text provided, returning zero vector")
            return np.zeros(768, dtype=np.float32)  # Vertex AI embedding dimension
            
        try:
            embeddings = self.text_embedding_model.get_embeddings([text])
            if embeddings and len(embeddings) > 0 and embeddings[0].values:
                emb = np.array(embeddings[0].values)
                return emb
            else:
                raise ValueError("Empty embedding response from Vertex AI")
        except Exception as e:
            print(f"❌ ERROR generating text embedding: {e}")
            raise RuntimeError(f"Failed to generate text embedding: {e}")
    
    def generate_batch_text_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
            
        try:
            # Process texts in batches - gecko allows larger batches
            batch_size = 5  # Gecko model allows larger batches than text-embedding-large
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                # Filter out empty strings to avoid API errors
                valid_texts = [t for t in batch_texts if t.strip()]
                valid_indices = [j for j, t in enumerate(batch_texts) if t.strip()]
                
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
                            batch_result[j] = np.zeros(768, dtype=np.float32)  # Embedding dimension
                            
                    all_embeddings.extend(batch_result)
            
            stacked = np.array(all_embeddings)
            print(f"✅ Generated {len(all_embeddings)} embeddings with shape: {stacked.shape}")
            return stacked
        except Exception as e:
            print(f"❌ ERROR generating batch text embeddings: {e}")
            raise RuntimeError(f"Failed to generate batch text embeddings: {e}")
