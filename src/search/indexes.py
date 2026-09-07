import hashlib
import json
import os
import tempfile
from pathlib import Path

from src.models.vector_db import VectorDatabase


def catalog_fingerprint(data_path):
    return hashlib.sha256(Path(data_path).read_bytes()).hexdigest()


def index_key(data_path, kind, model_id):
    signature = {"catalog": catalog_fingerprint(data_path), "kind": kind,
                 "model": model_id, "preprocessing": 1, "task": "RETRIEVAL_DOCUMENT" if kind == "text" else "image"}
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def load_or_build(data_path, df, indexes_dir, kind, model_id, generator_factory):
    key = index_key(data_path, kind, model_id)
    root = Path(indexes_dir)
    destination = root / key
    db = VectorDatabase()
    if destination.exists():
        db.load_indices(destination)
        return db, json.loads((destination / "build.json").read_text())
    generator = generator_factory()
    if kind == "text":
        vectors = generator.generate_text_embedding(df["text_for_embedding"].tolist(), task_type="RETRIEVAL_DOCUMENT")
        db.add_text_embeddings(vectors, list(range(len(df))))
        db.set_product_texts(df["text_for_embedding"].tolist())
        failed = []
        count = len(df)
    elif kind == "image":
        vectors, ids, failed = generator.generate_batch_image_embeddings(df["image_url"].tolist(), root / "images")
        if not ids:
            raise ValueError("No catalog images could be indexed. Check the image URLs and network connection.")
        db.add_image_embeddings(vectors, ids)
        count = len(ids)
    else:
        raise ValueError("Index kind must be text or image.")
    report = {"kind": kind, "model": model_id, "catalog_sha256": catalog_fingerprint(data_path),
              "indexed": count, "total": len(df), "failed_image_ids": failed}
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=root) as staging:
        db.save_indices(staging)
        (Path(staging) / "build.json").write_text(json.dumps(report, indent=2))
        try:
            os.rename(staging, destination)
        except OSError:
            if not destination.exists():
                raise
    return db, report
