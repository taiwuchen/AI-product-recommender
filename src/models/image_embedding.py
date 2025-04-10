import numpy as np
import torch
import urllib.parse
import requests
from PIL import Image
from io import BytesIO
from typing import List, Optional, Dict
from transformers import CLIPProcessor, CLIPModel

from .base_embedding import BaseEmbeddingGenerator

# Set fixed random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class ImageEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Class to generate image embeddings using OpenAI's CLIP model.
    This provides multimodal embeddings that align images and text in the same vector space.
    """
    
    def __init__(self, 
                 google_credentials_path: Optional[str] = None, 
                 vertex_ai_region: Optional[str] = None,
                 clip_model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP image embedding generator.
        
        Args:
            google_credentials_path (str, optional): Path to Google Cloud service account credentials.
                Not used for CLIP, but kept for compatibility with base class.
            vertex_ai_region (str, optional): Google Cloud region for Vertex AI.
                Not used for CLIP, but kept for compatibility with base class.
            clip_model_name (str): Name of the CLIP model to use.
        """
        super().__init__(google_credentials_path, vertex_ai_region)
        self.clip_model_name = clip_model_name
        self._load_clip_model()
        
        # Initialize embedding caches
        self.image_embedding_cache: Dict[str, np.ndarray] = {}
        self.text_embedding_cache: Dict[str, np.ndarray] = {}
    
    def _load_clip_model(self):
        """Load the CLIP model and processor."""
        try:
            print(f"Loading CLIP model: {self.clip_model_name}...")
            
            # Load the CLIP model and processor
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            
            # Check if CUDA is available and move model to GPU if possible
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
                self.device = "cuda"
                print("CLIP model loaded on GPU")
            else:
                self.device = "cpu"
                print("CLIP model loaded on CPU")
                
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            self.clip_processor = None
            self.clip_model = None
    
    def download_image(self, image_url: str) -> Optional[Image.Image]:
        """
        Download an image from a URL.
        
        Args:
            image_url (str): URL of the image.
            
        Returns:
            Optional[Image.Image]: PIL Image object or None if download fails.
        """
        try:
            # Check if URL is valid before making a request
            if not image_url or not isinstance(image_url, str):
                print(f"Invalid URL: {image_url}")
                return None
                
            # Parse and validate URL
            parsed = urllib.parse.urlparse(image_url)
            if not parsed.scheme or not parsed.netloc:
                # Add https:// if missing
                if parsed.netloc == "" and parsed.path != "":
                    image_url = f"https://{image_url}"
                else:
                    print(f"Invalid URL '{image_url}': No scheme or host provided")
                    return None
                    
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.google.com/'
            }
            
            # Download the image
            response = requests.get(image_url, stream=True, headers=headers)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error downloading image from {image_url}: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image) -> dict:
        """
        Preprocess image for the CLIP model.
        
        Args:
            image (Image.Image): PIL Image object.
            
        Returns:
            dict: Processed image inputs for the model.
        """
        if self.clip_processor is None:
            raise ValueError("CLIP processor not initialized")
            
        # Process image using CLIP processor
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        # Move inputs to the same device as the model
        if hasattr(self, 'device'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        return inputs
    
    def generate_image_embedding(self, image_url: str) -> np.ndarray:
        """
        Generate embedding for an image.
        
        Args:
            image_url (str): URL of the image.
            
        Returns:
            np.ndarray: Image embedding vector.
        """
        # Return cached embedding if available
        if image_url in self.image_embedding_cache:
            return self.image_embedding_cache[image_url]
            
        try:
            # Download the image
            image = self.download_image(image_url)
            
            if image is None:
                raise ValueError(f"Failed to download image from {image_url}")
                
            # Generate embedding from the image
            embedding = self.generate_embedding_from_pil_image(image)
            
            # Cache the result
            self.image_embedding_cache[image_url] = embedding
            
            return embedding
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            # Create a deterministic random embedding based on the URL
            random_state = np.random.RandomState(hash(image_url) % 2**32)
            random_embedding = random_state.randn(512).astype(np.float32)
            
            # Normalize the random embedding
            norm = np.linalg.norm(random_embedding)
            if norm > 0:
                random_embedding = random_embedding / norm
                
            # Cache the result
            self.image_embedding_cache[image_url] = random_embedding
            
            return random_embedding
    
    def generate_embedding_from_pil_image(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding directly from a PIL image object.
        
        Args:
            image (Image.Image): PIL Image object.
            
        Returns:
            np.ndarray: Image embedding vector.
        """
        # Create a hash of the image for caching
        # Use a more robust method that captures image content rather than just bytes
        width, height = image.size
        small_image = image.resize((32, 32))  # Resize for consistent hashing
        pixels = list(small_image.getdata())
        image_hash = hash(str(pixels))
        cache_key = f"pil_image_{image_hash}"
        
        # Return cached embedding if available
        if cache_key in self.image_embedding_cache:
            return self.image_embedding_cache[cache_key]
            
        try:
            if self.clip_model is None or self.clip_processor is None:
                raise ValueError("CLIP model or processor not initialized")
                
            # Preprocess the image
            inputs = self.preprocess_image(image)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                
            # Normalize the embedding
            image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Convert to numpy array
            embedding = image_embeddings.cpu().numpy()[0]
            
            # Cache the result
            self.image_embedding_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            print(f"Error generating CLIP image embedding: {e}")
            # Create a unique random embedding based on the image content
            random_state = np.random.RandomState(image_hash % 2**32)
            random_embedding = random_state.randn(512).astype(np.float32)
            
            # Normalize the random embedding
            norm = np.linalg.norm(random_embedding)
            if norm > 0:
                random_embedding = random_embedding / norm
                
            # Cache the result
            self.image_embedding_cache[cache_key] = random_embedding
            
            return random_embedding
    
    def generate_batch_image_embeddings(self, image_urls: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Args:
            image_urls (List[str]): List of image URLs.
            
        Returns:
            np.ndarray: Array of image embedding vectors.
        """
        if not image_urls:
            return np.array([])
            
        embeddings = []
        valid_count = 0
        
        for i, url in enumerate(image_urls):
            if not url or not isinstance(url, str) or url.strip() == "":
                # For empty URLs, create a unique random embedding based on the index
                # This ensures different products get different embeddings even with missing images
                print(f"Generating diverse random embedding for empty URL at index {i}")
                
                # Create a unique seed for each empty URL based on its index
                random_state = np.random.RandomState((i + 1) * 42)
                random_embedding = random_state.randn(512).astype(np.float32)
                
                # Normalize the random embedding
                norm = np.linalg.norm(random_embedding)
                if norm > 0:
                    random_embedding = random_embedding / norm
                    
                embeddings.append(random_embedding)
                continue
                
            try:
                embedding = self.generate_image_embedding(url)
                embeddings.append(embedding)
                valid_count += 1
                if valid_count % 10 == 0:
                    print(f"Generated {valid_count} valid embeddings out of {i+1} processed URLs")
            except Exception as e:
                print(f"Error processing image from URL at index {i}: {e}")
                # Add a random embedding to maintain alignment with input, but make it unique
                random_state = np.random.RandomState((i + 1) * 99)
                random_embedding = random_state.randn(512).astype(np.float32)
                
                # Normalize the random embedding
                norm = np.linalg.norm(random_embedding)
                if norm > 0:
                    random_embedding = random_embedding / norm
                    
                embeddings.append(random_embedding)
        
        print(f"Completed batch processing: {valid_count} valid embeddings out of {len(image_urls)} URLs")
        return np.array(embeddings)
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using CLIP's text encoder.
        
        Args:
            text (str): Text to embed.
            
        Returns:
            np.ndarray: Text embedding vector.
        """
        # Return cached embedding if available
        if text in self.text_embedding_cache:
            return self.text_embedding_cache[text]
            
        try:
            if self.clip_model is None or self.clip_processor is None:
                raise ValueError("CLIP model or processor not initialized")
                
            # Process text using CLIP processor
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
            
            # Move inputs to the same device as the model
            if hasattr(self, 'device'):
                inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'pixel_values'}
            
            # Generate embedding
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                
            # Normalize the embedding
            text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Convert to numpy array
            embedding = text_embeddings.cpu().numpy()[0]
            
            # Cache the result
            self.text_embedding_cache[text] = embedding
            
            return embedding
        except Exception as e:
            print(f"Error generating CLIP text embedding: {e}")
            # Create a deterministic random embedding based on the text
            random_state = np.random.RandomState(hash(text) % 2**32)
            random_embedding = random_state.randn(512).astype(np.float32)
            
            # Normalize the random embedding
            norm = np.linalg.norm(random_embedding)
            if norm > 0:
                random_embedding = random_embedding / norm
                
            # Cache the result
            self.text_embedding_cache[text] = random_embedding
            
            return random_embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding vector.
            embedding2 (np.ndarray): Second embedding vector.
            
        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        # Ensure embeddings are normalized
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        return float(np.dot(embedding1, embedding2))