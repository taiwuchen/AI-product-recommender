# Export components from respective modules
from .base_embedding import fix_certificate_verification, BaseEmbeddingGenerator
from .text_embedding import TextEmbeddingGenerator
from .image_embedding import ImageEmbeddingGenerator
import os
import base64
import io
import json
import requests
from typing import Optional
from PIL import Image

class EmbeddingGenerator:
    """
    Combined class for backward compatibility that uses both text and image embedding generators.
    """
    
    def __init__(self, google_credentials_path: Optional[str] = None, vertex_ai_region: Optional[str] = None):
        """
        Initialize both text and image embedding generators.
        
        Args:
            google_credentials_path (str, optional): Path to Google Cloud service account credentials.
            vertex_ai_region (str, optional): Google Cloud region for Vertex AI.
        """
        self.text_generator = TextEmbeddingGenerator(google_credentials_path, vertex_ai_region)
        self.image_generator = ImageEmbeddingGenerator(google_credentials_path, vertex_ai_region)
        
        # For backward compatibility - handle the case where text_model no longer exists
        self.text_model = getattr(self.text_generator, 'text_embedding_model', None)
        self.image_model = self.image_generator.image_model
        self.initialized = self.text_generator.initialized
    
    # Forward text methods to text generator
    def generate_text_embedding(self, text):
        return self.text_generator.generate_text_embedding(text)
    
    def generate_batch_text_embeddings(self, texts):
        return self.text_generator.generate_batch_text_embeddings(texts)
    
    # Forward image methods to image generator
    def download_image(self, image_url):
        return self.image_generator.download_image(image_url)
    
    def preprocess_image(self, image):
        return self.image_generator.preprocess_image(image)
    
    def generate_image_embedding(self, image_url):
        return self.image_generator.generate_image_embedding(image_url)
    
    def generate_batch_image_embeddings(self, image_urls):
        return self.image_generator.generate_batch_image_embeddings(image_urls)
        
    def generate_multimodal_description(self, text_query: str, image: Optional[Image.Image] = None) -> str:
        """
        Generate a description using Gemini's multimodal capabilities via OpenRouter API.
        
        Args:
            text_query (str): Text query
            image (Image.Image, optional): Image to analyze alongside text
            
        Returns:
            str: Generated description
        """
        # Use environment variable with fallback to hardcoded key
        openrouter_api_key = os.environ.get('OPENROUTER_API_KEY', os.environ.get('OPENROUTER_API_KEY'))
            
        try:
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
                    "HTTP-Referer": "https://ai-product-recommender.app",
                    "X-Title": "AI Product Recommender",
                },
                data=json.dumps({
                    "model": "google/gemini-2.5-pro-exp-03-25:free",  # Using Gemini multimodal model
                    "messages": messages,
                    "max_tokens": 300,
                    "temperature": 0.7
                })
            )
            
            response_data = response.json()
            
            # Extract and format the response
            if response.status_code == 200 and "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response_data.get('error', {}).get('message', 'Unknown error')}"
                
        except Exception as e:
            return f"Error generating multimodal description: {str(e)}"