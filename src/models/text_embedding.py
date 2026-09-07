import os

import numpy as np
from google import genai
from google.genai import types

MODEL_ID = "text-embedding-005"


class TextEmbeddingGenerator:
    def __init__(self):
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError("Set GOOGLE_CLOUD_PROJECT in .env to use semantic search.")
        self.client = genai.Client(vertexai=True, project=project,
                                   location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                                   http_options=types.HttpOptions(timeout=30000))

    def generate_text_embedding(self, text, task_type="RETRIEVAL_QUERY"):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        if not texts or any(not value.strip() for value in texts):
            raise ValueError("Embedding input must contain nonempty text.")
        values = []
        for start in range(0, len(texts), 5):
            batch = texts[start:start + 5]
            response = self.client.models.embed_content(model=MODEL_ID, contents=batch,
                        config=types.EmbedContentConfig(task_type=task_type, auto_truncate=False))
            if not response.embeddings or len(response.embeddings) != len(batch):
                raise ValueError("Embedding service returned an incomplete batch.")
            values.extend(result.values for result in response.embeddings)
        embeddings = np.asarray(values, dtype=np.float32)
        return embeddings[0] if single else embeddings
