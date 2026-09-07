import os

import numpy as np
import requests

MODEL_ID = "openai/text-embedding-3-small"
DIMENSIONS = 1536
ENDPOINT = "https://openrouter.ai/api/v1/embeddings"


class TextEmbeddingGenerator:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY in .env to use semantic search.")

    def generate_text_embedding(self, text):
        single = isinstance(text, str)
        if not single and not isinstance(text, list):
            raise ValueError("Embedding input must be text or a list of texts.")
        texts = [text] if single else text
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError("Embedding input must contain nonempty text.")
        batches = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            response = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": MODEL_ID, "input": batch, "dimensions": DIMENSIONS, "encoding_format": "float"},
                timeout=(5, 30),
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise ValueError("OpenRouter returned an incomplete embedding batch.")
            if any(not isinstance(row, dict) or type(row.get("index")) is not int for row in rows):
                raise ValueError("OpenRouter returned invalid embedding indices.")
            if sorted(row["index"] for row in rows) != list(range(len(batch))):
                raise ValueError("OpenRouter returned missing or duplicate embedding indices.")
            rows = sorted(rows, key=lambda row: row["index"])
            vectors = np.asarray([row.get("embedding") for row in rows], dtype=np.float32)
            if vectors.shape != (len(batch), DIMENSIONS) or not np.isfinite(vectors).all():
                raise ValueError("OpenRouter returned invalid embedding vectors.")
            if np.any(np.linalg.norm(vectors, axis=1) == 0):
                raise ValueError("OpenRouter returned zero embedding vectors.")
            batches.append(vectors)
        embeddings = np.concatenate(batches)
        return embeddings[0] if single else embeddings
