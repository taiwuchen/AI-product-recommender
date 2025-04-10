# Export components from respective modules
from .text_embedding import fix_certificate_verification, BaseEmbeddingGenerator, TextEmbeddingGenerator
from .image_embedding import ImageEmbeddingGenerator
from typing import Optional

class EmbeddingGenerator:
    
    def __init__(self, google_credentials_path: Optional[str] = None, vertex_ai_region: Optional[str] = None):
        self.text_generator = TextEmbeddingGenerator(google_credentials_path, vertex_ai_region)
        self.image_generator = ImageEmbeddingGenerator(clip_model_name="openai/clip-vit-base-patch32")
        self.text_model = getattr(self.text_generator, 'text_embedding_model', None)
        self.image_model = self.image_generator.image_model
        self.initialized = self.text_generator.initialized
    
    # Text methods to text generator
    def generate_text_embedding(self, text):
        return self.text_generator.generate_text_embedding(text)
    
    def generate_batch_text_embeddings(self, texts):
        return self.text_generator.generate_batch_text_embeddings(texts)
    
    
    
    # Image methods to image generator
    def download_image(self, image_url):
        return self.image_generator.download_image(image_url)
    
    def preprocess_image(self, image):
        return self.image_generator.preprocess_image(image)
    
    def generate_image_embedding(self, image_url):
        return self.image_generator.generate_image_embedding(image_url)
    
    def generate_batch_image_embeddings(self, image_urls):
        return self.image_generator.generate_batch_image_embeddings(image_urls)