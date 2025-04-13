import numpy as np
import torch
import urllib.parse
import requests
from PIL import Image
from io import BytesIO
from typing import List, Optional, Dict
from transformers import CLIPProcessor, CLIPModel

class ImageEmbeddingGenerator:
    
    def __init__(self, 
                 google_credentials_path: Optional[str] = None, 
                 vertex_ai_region: Optional[str] = None,
                 clip_model_name: str = "openai/clip-vit-base-patch32"):

        self.clip_model_name = clip_model_name
        self._load_clip_model()
        
        # Initialize embedding caches
        self.image_embedding_cache: Dict[str, np.ndarray] = {}
        self.text_embedding_cache: Dict[str, np.ndarray] = {}
    
    def _load_clip_model(self):
        try:
            print(f"Loading CLIP model: {self.clip_model_name}...")
            
            # Load the CLIP model and processor
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            
            # Check if CUDA is available and move model to GPU if possible
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
                self.device = "cuda"
            else:
                self.device = "cpu"
                
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            self.clip_processor = None
            self.clip_model = None
    
    def download_image(self, image_url: str, convert_to_rgb: bool = True, referer: str = 'https://www.google.com/') -> Optional[Image.Image]:
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
                    
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': referer
            }
            
            # Download the image
            response = requests.get(image_url, stream=True, headers=headers)
            response.raise_for_status()
            
            # Open image and optionally convert to RGB (needed for CLIP model)
            image = Image.open(BytesIO(response.content))
            if convert_to_rgb:
                image = image.convert('RGB')
                
            return image
        except Exception as e:
            print(f"Error downloading image from {image_url}: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image) -> dict:
        if self.clip_processor is None:
            raise ValueError("CLIP processor not initialized")
            
        # Process image using CLIP processor
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        # Move inputs to the same device as the model
        if hasattr(self, 'device'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        return inputs 
    
    def generate_embedding_from_pil_image(self, image: Image.Image) -> np.ndarray:
        width, height = image.size
        small_image = image.resize((32, 32))  # Resize for consistent hashing
        pixels = list(small_image.getdata())
        image_hash = hash(str(pixels))
        cache_key = f"pil_image_{image_hash}"
        
        # Return cached embedding if available
        if cache_key in self.image_embedding_cache:
            return self.image_embedding_cache[cache_key]
            
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
    
    def generate_image_embedding(self, image_url: str) -> np.ndarray:
        # Return cached embedding if available
        if image_url in self.image_embedding_cache:
            return self.image_embedding_cache[image_url]
            
        # Download the image
        image = self.download_image(image_url, convert_to_rgb=True)
        
        if image is None:
            raise ValueError(f"Failed to download image from {image_url}")
            
        # Generate embedding from the image
        embedding = self.generate_embedding_from_pil_image(image)
        
        # Cache the result
        self.image_embedding_cache[image_url] = embedding
        
        return embedding

    def generate_batch_image_embeddings(self, image_urls: List[str]) -> np.ndarray:
        if not image_urls:
            return np.array([])
            
        embeddings = []
        valid_indices = []
        
        # First pass: process all valid URLs and keep track of their indices
        for i, url in enumerate(image_urls):
            if url and isinstance(url, str) and url.strip():
                try:
                    embedding = self.generate_image_embedding(url)
                    embeddings.append(embedding)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Skipping URL {url} due to error: {e}")
        
        # If we couldn't process any URLs, return empty array
        if not embeddings:
            return np.array([])
            
        return np.array(embeddings)

