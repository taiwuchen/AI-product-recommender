import os
import numpy as np
import logging
import base64
from io import BytesIO
import requests
from PIL import Image
from typing import List, Dict, Any, Optional, Union
from google.cloud import aiplatform
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Class to generate embeddings using Google Vertex AI."""
    
    def __init__(self, project_id: str = None, location: str = "us-central1"):
        """
        Initialize EmbeddingGenerator with GCP project details.
        
        Args:
            project_id: Google Cloud project ID (if None, will try to get from environment)
            location: Google Cloud region
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable.")
            
        self.location = location
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)
        logger.info(f"Initialized Vertex AI with project: {self.project_id}, location: {self.location}")
        
        # Text embedding model
        self.text_model_name = "textembedding-gecko@latest"
        # Multimodal embedding model for images
        self.multimodal_model_name = "multimodalembedding@latest"
        
    def generate_text_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text strings.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating text embeddings for {len(texts)} products")
        
        # Create Vertex AI endpoint
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.text_model_name}"
        )
        
        embeddings = []
        batch_size = 5  # Process in small batches to avoid API limits
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                instances = [{"content": text} for text in batch_texts]
                response = endpoint.predict(instances=instances)
                
                # Extract embeddings from response
                batch_embeddings = [np.array(prediction["embeddings"]["values"]) 
                                   for prediction in response.predictions]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1} of {(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error generating text embeddings for batch {i//batch_size + 1}: {e}")
                # Add empty embeddings for failed items
                embeddings.extend([np.zeros(768) for _ in range(len(batch_texts))])
                
        return embeddings
    
    def _download_image(self, image_url: str) -> Optional[Image.Image]:
        """
        Download an image from a URL.
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            PIL Image object or None if download failed
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.error(f"Error downloading image from {image_url}: {e}")
            # Return a blank image
            return Image.new('RGB', (224, 224), color='white')
    
    def generate_image_embeddings(self, image_urls: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of image URLs.
        
        Args:
            image_urls: List of image URLs to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating image embeddings for {len(image_urls)} products")
        
        # Create Vertex AI endpoint
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.multimodal_model_name}"
        )
        
        embeddings = []
        batch_size = 5  # Process in small batches
        
        for i in range(0, len(image_urls), batch_size):
            batch_urls = image_urls[i:i+batch_size]
            batch_embeddings = []
            
            for url in batch_urls:
                try:
                    if not url:
                        raise ValueError("Empty URL")
                        
                    # Download the image
                    image = self._download_image(url)
                    
                    # Convert image to base64
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    # Create prediction instance
                    instance = {
                        "image": {
                            "bytesBase64Encoded": img_str
                        }
                    }
                    
                    # Get embedding
                    response = endpoint.predict(instances=[instance])
                    embedding = np.array(response.predictions[0]["imageEmbedding"])
                    batch_embeddings.append(embedding)
                    
                except Exception as e:
                    logger.error(f"Error generating image embedding: {e}")
                    # Add empty embedding for failed item
                    batch_embeddings.append(np.zeros(1408))  # Default size for multimodal embeddings
            
            embeddings.extend(batch_embeddings)
            logger.info(f"Generated image embeddings for batch {i//batch_size + 1} of {(len(image_urls)-1)//batch_size + 1}")
                
        return embeddings
    
    def generate_hybrid_embeddings(
        self, 
        texts: List[str], 
        image_urls: List[str],
        text_weight: float = 0.5
    ) -> List[np.ndarray]:
        """
        Generate hybrid embeddings by combining text and image embeddings.
        
        Args:
            texts: List of text strings
            image_urls: List of image URLs
            text_weight: Weight of text embeddings in the hybrid (0.0-1.0)
            
        Returns:
            List of hybrid embedding vectors
        """
        assert 0.0 <= text_weight <= 1.0, "Weight must be between 0.0 and 1.0"
        assert len(texts) == len(image_urls), "Number of texts and images must match"
        
        # Get embeddings
        text_embeddings = self.generate_text_embeddings(texts)
        image_embeddings = self.generate_image_embeddings(image_urls)
        
        # Normalize embeddings
        normalized_text_embeddings = [
            embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding
            for embedding in text_embeddings
        ]
        
        normalized_image_embeddings = [
            embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding
            for embedding in image_embeddings
        ]
        
        # Combine embeddings
        image_weight = 1.0 - text_weight
        
        hybrid_embeddings = []
        for text_emb, img_emb in zip(normalized_text_embeddings, normalized_image_embeddings):
            # Since dimensions don't match, we concatenate and then normalize
            combined = np.concatenate([
                text_emb * text_weight,
                img_emb * image_weight
            ])
            # Normalize the combined embedding
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
            hybrid_embeddings.append(combined)
            
        return hybrid_embeddings
