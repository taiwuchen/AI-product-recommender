import os
import ssl
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Dict, Union, Optional
from dotenv import load_dotenv
import tensorflow as tf
import tensorflow_hub as hub
from google.cloud import aiplatform
from google.oauth2 import service_account
import urllib.parse


# Fix SSL certificate verification on macOS
def fix_certificate_verification():
    """Fix SSL certificate verification issues on macOS"""
    try:
        # Use certifi if available
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        
        # For macOS, run the certificate install command for Python
        if os.path.exists('/Applications/Python 3.10/Install Certificates.command'):
            import subprocess
            subprocess.run(['/Applications/Python 3.10/Install Certificates.command'], check=False, shell=True)
        
        # Create unverified HTTPS context if needed
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception as e:
        print(f"Warning: Could not fix SSL certificate verification: {e}")


class EmbeddingGenerator:
    """
    Class to generate text and image embeddings using Google Vertex AI.
    """
    
    def __init__(self, google_credentials_path: str = None, vertex_ai_region: str = "us-west2"):
        """
        Initialize the embedding generator.
        
        Args:
            google_credentials_path (str, optional): Path to Google Cloud service account credentials.
            vertex_ai_region (str, optional): Google Cloud region for Vertex AI. Default is "us-west2".
        """
        load_dotenv()
        
        # Fix SSL certificate issues before making any requests
        fix_certificate_verification()
        
        # Initialize Google Cloud credentials
        self.credentials = None
        self.initialized = False
        
        # Valid Vertex AI regions - updated list
        self.supported_regions = {
            'asia-east1', 'asia-east2', 'asia-northeast1', 'asia-northeast2', 'asia-northeast3', 
            'asia-south1', 'asia-southeast1', 'asia-southeast2', 'australia-southeast1', 
            'australia-southeast2', 'europe-central2', 'europe-north1', 'europe-southwest1', 
            'europe-west1', 'europe-west2', 'europe-west3', 'europe-west4', 'europe-west6', 
            'europe-west8', 'europe-west9', 'europe-west12', 'global', 'me-central1', 
            'me-central2', 'me-west1', 'northamerica-northeast1', 'northamerica-northeast2', 
            'southamerica-east1', 'southamerica-west1', 'us-central1', 'us-east1', 'us-east4', 
            'us-east5', 'us-south1', 'us-west1', 'us-west2', 'us-west3', 'us-west4', 'africa-south1'
        }
        
        # Use provided region if valid, otherwise default to us-central1
        self.vertex_ai_region = vertex_ai_region if vertex_ai_region in self.supported_regions else "us-central1"
        
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
            print("Will use backup TensorFlow models for embeddings")
            
        # Load TensorFlow models as backup
        self._load_tensorflow_models()
    
    def _load_tensorflow_models(self):
        """Load TensorFlow models for text and image embeddings as fallback."""
        try:
            print("Loading TensorFlow models as fallback...")
            
            # Disable SSL verification for TensorFlow Hub
            os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub_cache'
            
            # Load Universal Sentence Encoder for text embeddings without context manager
            self.text_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            
            # Use MobileNetV2 for image embeddings
            base_model = tf.keras.applications.MobileNetV2(
                include_top=False, weights='imagenet', input_shape=(224, 224, 3)
            )
            self.image_model = tf.keras.Model(
                inputs=base_model.input,
                outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            )
            print("TensorFlow models loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load TensorFlow models: {e}")
            print("Using random embeddings as fallback")
            self.text_model = None
            self.image_model = None
        
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
    
    def generate_multimodal_description(self, text_query: str, image: Optional[Image.Image] = None) -> str:
        """
        Generate a description using Gemini's multimodal capabilities via OpenRouter API.
        
        Args:
            text_query (str): Text query
            image (Image.Image, optional): Image to analyze alongside text
            
        Returns:
            str: Generated description
        """
        openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
        if not openrouter_api_key:
            return "OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable."
            
        try:
            import requests
            import base64
            import io
            
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Add text content
            messages[0]["content"].append({
                "type": "text",
                "text": text_query
            })
            
            # Add image if provided
            if image:
                # Convert PIL image to base64
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_str}"
                    }
                })
            
            # Call OpenRouter API with Gemini
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://ai-product-recommender.app",  # Replace with your actual site URL
                    "X-Title": "AI Product Recommender",  # Replace with your actual site name
                },
                json={
                    "model": "google/gemini-2.5-pro-exp-03-25:free",  # Using Gemini multimodal model
                    "messages": messages,
                    "max_tokens": 300,
                    "temperature": 0.7
                }
            )
            
            response_data = response.json()
            
            # Extract and format the response
            if response.status_code == 200 and "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response_data.get('error', {}).get('message', 'Unknown error')}"
                
        except Exception as e:
            return f"Error generating multimodal description: {str(e)}"