import logging

import requests
import numpy as np
import torch
from PIL import ImageOps
from transformers import CLIPModel, CLIPProcessor

from product_search.utils.images import load_image

MODEL_ID = "openai/clip-vit-base-patch32"


class ImageEmbeddingGenerator:
    def __init__(self):
        # Bound CPU inference threads for cached models shared by Streamlit workers.
        torch.set_num_threads(1)
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False)
        self.model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def generate_embedding_from_pil_image(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=1, keepdim=True)
        return features.cpu().numpy()[0]

    def generate_batch_image_embeddings(self, image_urls, cache_dir):
        embeddings, product_ids, failures = [], [], []
        for product_id, url in enumerate(image_urls):
            try:
                image = load_image(url, cache_dir)
            except (OSError, ValueError, requests.RequestException) as exc:
                failures.append(product_id)
                logging.warning("Image unavailable for product %s: %s", product_id, type(exc).__name__)
                continue
            embeddings.append(self.generate_embedding_from_pil_image(image))
            product_ids.append(product_id)
        return np.asarray(embeddings, dtype=np.float32), product_ids, failures
