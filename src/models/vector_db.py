import json
from pathlib import Path

import faiss
import numpy as np


class VectorDatabase:
    def __init__(self):
        self.index_text = None
        self.index_image = None
        self.text_ids = []
        self.image_ids = []
        self.product_texts = []

    @staticmethod
    def _index(embeddings, ids):
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("Embeddings must have one row per unique product ID.")
        if not len(vectors) or not np.isfinite(vectors).all():
            raise ValueError("Embeddings must be nonempty and finite.")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Zero embeddings cannot be searched.")
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors))
        return index

    def add_text_embeddings(self, embeddings, ids):
        self.index_text = self._index(embeddings, ids)
        self.text_ids = list(ids)

    def add_image_embeddings(self, embeddings, ids):
        self.index_image = self._index(embeddings, ids)
        self.image_ids = list(ids)

    def set_product_texts(self, texts):
        if len(texts) != len(self.text_ids):
            raise ValueError("Text metadata must match the text index.")
        self.product_texts = list(texts)

    @staticmethod
    def _search(index, embedding, k):
        if k < 1:
            raise ValueError("Result count must be positive.")
        if index is None or index.ntotal == 0:
            return np.empty((1, 0)), np.empty((1, 0), dtype=int)
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != index.d or not np.isfinite(vector).all() or not np.linalg.norm(vector):
            raise ValueError("Query embedding is invalid or has the wrong dimension.")
        return index.search(vector, min(k, index.ntotal))

    def search_by_text(self, query_embedding, k=5, query_text=None, keyword_boost=False):
        count = k * 3 if keyword_boost and query_text else k
        distances, positions = self._search(self.index_text, query_embedding, count)
        if keyword_boost and query_text and positions.size:
            keywords = [word.lower() for word in query_text.split() if len(word) > 2]
            largest = float(distances.max())
            scores = 1 - distances[0] / largest if largest > 0 else np.ones(positions.shape[1])
            for i, position in enumerate(positions[0]):
                matches = sum(word in self.product_texts[position].lower() for word in keywords)
                scores[i] *= 1 + 0.5 * matches
            order = np.argsort(-scores, kind="stable")[:k]
            positions = positions[:, order]
            distances = distances[:, order]
        else:
            positions, distances = positions[:, :k], distances[:, :k]
        ids = np.array([[self.text_ids[p] for p in positions[0]]], dtype=int)
        return distances, ids

    def search_by_image(self, query_embedding, k=5):
        distances, positions = self._search(self.index_image, query_embedding, k)
        ids = np.array([[self.image_ids[p] for p in positions[0]]], dtype=int)
        return distances, ids

    def save_indices(self, save_dir):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        for name, index in [("text", self.index_text), ("image", self.index_image)]:
            if index is not None:
                faiss.write_index(index, str(path / f"{name}.faiss"))
        (path / "metadata.json").write_text(json.dumps({
            "text_ids": self.text_ids, "image_ids": self.image_ids,
            "product_texts": self.product_texts,
        }))

    def load_indices(self, save_dir):
        path = Path(save_dir)
        metadata = json.loads((path / "metadata.json").read_text())
        self.text_ids = metadata["text_ids"]
        self.image_ids = metadata["image_ids"]
        self.product_texts = metadata["product_texts"]
        for name, ids in [("text", self.text_ids), ("image", self.image_ids)]:
            index = faiss.read_index(str(path / f"{name}.faiss")) if ids else None
            if index is not None and (index.ntotal != len(ids) or len(set(ids)) != len(ids)):
                raise ValueError("Saved index does not match its product IDs.")
            setattr(self, f"index_{name}", index)
        if len(self.product_texts) != len(self.text_ids):
            raise ValueError("Saved text metadata does not match the index.")
