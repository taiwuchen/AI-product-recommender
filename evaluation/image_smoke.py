import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from product_search.config import DEFAULT_CATALOG, INDEXES_DIR, ROOT
from product_search.models.image_embedding import MODEL_ID, ImageEmbeddingGenerator
from product_search.search.indexes import load_or_build
from product_search.utils.data_loader import ProductDataLoader
from product_search.utils.images import load_image


def run():
    data_path = DEFAULT_CATALOG
    df = ProductDataLoader(data_path).preprocess_data()
    with ThreadPoolExecutor(max_workers=1) as pool:
        model = pool.submit(ImageEmbeddingGenerator).result()
    db, report = load_or_build(data_path, df, INDEXES_DIR, "image", MODEL_ID, lambda: model)
    reference_id = 27
    if reference_id not in db.image_ids:
        raise ValueError("Reference image is unavailable; choose and document another reference.")
    reference = load_image(df.iloc[reference_id]["image_url"], INDEXES_DIR / "images")
    runs = []
    for _ in range(3):
        with ThreadPoolExecutor(max_workers=1) as pool:
            start = time.perf_counter()
            embedding = pool.submit(model.generate_embedding_from_pil_image, reference).result()
            _, ids = db.search_by_image(embedding)
            runs.append({"retrieved_product_ids": ids[0].tolist(), "search_ms": (time.perf_counter() - start) * 1000})
            if int(ids[0][0]) != reference_id:
                raise AssertionError("Reference photo did not retrieve the correct product first.")
    report.update(generated_at=datetime.now(timezone.utc).isoformat(), reference_product_id=reference_id,
                  worker_thread_runs=runs, passed=True,
                  scope="Exact catalog-image retrieval across successive worker threads; not an independent image relevance benchmark.")
    (ROOT / "evaluation/image-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
