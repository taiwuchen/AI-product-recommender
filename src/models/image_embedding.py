import numpy as np
import tensorflow as tf
import urllib.parse
import requests
from PIL import Image
from io import BytesIO
from typing import List, Optional
from google.cloud import aiplatform

from .base_embedding import BaseEmbeddingGenerator


class ImageEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Class to generate image embeddings using Google Vertex AI or TensorFlow.
    """
    
    def __init__(self, google_credentials_path: Optional[str] = None, vertex_ai_region: Optional[str] = None):
        """
        Initialize the image embedding generator.
        
        Args:
            google_credentials_path (str, optional): Path to Google Cloud service account credentials.
            vertex_ai_region (str, optional): Google Cloud region for Vertex AI.
        """
        super().__init__(google_credentials_path, vertex_ai_region)
        self._load_image_model()
    
    def _load_image_model(self):
        """Load TensorFlow model for image embeddings as fallback."""
        try:
            print("Loading image embedding model...")
            
            # Use MobileNetV2 for image embeddings
            base_model = tf.keras.applications.MobileNetV2(
                include_top=False, weights='imagenet', input_shape=(224, 224, 3)
            )
            self.image_model = tf.keras.Model(
                inputs=base_model.input,
                outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            )
            print("Image embedding model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load image embedding model: {e}")
            print("Using random image embeddings as fallback")
            self.image_model = None
    
    def download_image(self, image_url: str) -> Image.Image:
        """
        Download an image from a URL.
        
        Args:
            image_url (str): URL of the image.
            
        Returns:
            Image.Image: PIL Image object.
        """
        try:
            # Check if URL is valid before making a request
            if not image_url or not isinstance(image_url, str):
                print(f"Invalid URL: {image_url}")
                return Image.new('RGB', (224, 224), color='white')
                
            # Parse and validate URL
            parsed = urllib.parse.urlparse(image_url)
            if not parsed.scheme or not parsed.netloc:
                # Add https:// if missing
                if parsed.netloc == "" and parsed.path != "":
                    image_url = f"https://{image_url}"
                else:
                    print(f"Invalid URL '{image_url}': No scheme or host provided")
                    return Image.new('RGB', (224, 224), color='white')
                    
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.zara.com/'
            }
            
            # Download the image
            response = requests.get(image_url, stream=True, headers=headers)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error downloading image from {image_url}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='white')
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for the model.
        
        Args:
            image (Image.Image): PIL Image object.
            
        Returns:
            np.ndarray: Preprocessed image array.
        """
        # Resize the image to the required dimensions
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        img_array = img_array.astype(np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def generate_image_embedding(self, image_url: str) -> np.ndarray:
        """
        Generate embedding for an image using Google Vertex AI or TensorFlow model.
        
        Args:
            image_url (str): URL of the image.
            
        Returns:
            np.ndarray: Image embedding vector.
        """
        try:
            # Download and preprocess the image
            image = self.download_image(image_url)
            img_array = self.preprocess_image(image)
            
            if self.initialized:
                # Use Google Vertex AI
                # Convert to bytes for API
                image_bytes = BytesIO()
                image.save(image_bytes, format='JPEG')
                img_bytes = image_bytes.getvalue()
                
                endpoint = aiplatform.Endpoint("projects/{project}/locations/{location}/endpoints/{id}")
                response = endpoint.predict(instances=[{"bytes_inputs": {"b64": img_bytes}}])
                embedding = np.array(response.predictions[0])
                return embedding
            else:
                # Use TensorFlow model as fallback
                embedding = self.image_model.predict(img_array)[0]
                return embedding
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            # Return a random embedding as a last resort
            return np.random.randn(1280).astype(np.float32)
    
    def generate_batch_image_embeddings(self, image_urls: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Args:
            image_urls (List[str]): List of image URLs.
            
        Returns:
            np.ndarray: Array of image embedding vectors.
        """
        embeddings = []
        for url in image_urls:
            embedding = self.generate_image_embedding(url)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def generate_embedding_from_pil_image(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding directly from a PIL image object.
        
        Args:
            image (Image.Image): PIL Image object.
            
        Returns:
            np.ndarray: Image embedding vector.
        """
        try:
            img_array = self.preprocess_image(image)
            
            if self.initialized:
                # Use Google Vertex AI
                # Convert to bytes for API
                image_bytes = BytesIO()
                image.save(image_bytes, format='JPEG')
                img_bytes = image_bytes.getvalue()
                
                endpoint = aiplatform.Endpoint("projects/{project}/locations/{location}/endpoints/{id}")
                response = endpoint.predict(instances=[{"bytes_inputs": {"b64": img_bytes}}])
                embedding = np.array(response.predictions[0])
                return embedding
            else:
                # Use TensorFlow model as fallback
                embedding = self.image_model.predict(img_array)[0]
                return embedding
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            # Return a random embedding as a last resort
            return np.random.randn(1280).astype(np.float32)