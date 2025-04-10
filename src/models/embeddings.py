# Export components from respective modules
from .base_embedding import fix_certificate_verification, BaseEmbeddingGenerator
from .text_embedding import TextEmbeddingGenerator
from .image_embedding import ImageEmbeddingGenerator
from typing import Optional

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
        # text_model is now text_embedding_model in the new implementation
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