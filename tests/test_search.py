import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image

from evaluation.run import metrics
from product_search.models.vector_db import VectorDatabase
from product_search.search.explanations import product_evidence
from product_search.search.indexes import index_key, load_or_build
from product_search.search.keyword import KeywordSearch
from product_search.utils.data_loader import ProductDataLoader


class SearchTests(unittest.TestCase):
    def test_failed_middle_image_keeps_correct_product_after_reload(self):
        from product_search.models.image_embedding import ImageEmbeddingGenerator
        generator = ImageEmbeddingGenerator.__new__(ImageEmbeddingGenerator)
        generator.generate_embedding_from_pil_image = Mock(side_effect=[[1, 0], [0, 1]])
        image = Image.new("RGB", (4, 4))
        with patch("product_search.models.image_embedding.load_image", side_effect=[image, ValueError("missing"), image]):
            vectors, ids, failures = generator.generate_batch_image_embeddings(["a", "b", "c"], ".")
        self.assertEqual(failures, [1])
        db = VectorDatabase()
        db.add_image_embeddings(vectors, ids)
        with tempfile.TemporaryDirectory() as directory:
            db.save_indices(directory)
            loaded = VectorDatabase()
            loaded.load_indices(directory)
        distances, found = loaded.search_by_image([0, 1], k=5)
        self.assertEqual(found.tolist(), [[2, 0]])
        self.assertEqual(distances.shape, found.shape)

    def test_empty_search_does_not_invent_product_zero(self):
        db = VectorDatabase()
        self.assertEqual(db.search_by_text([1, 0])[1].size, 0)
        self.assertEqual(db.search_by_image([1, 0])[1].size, 0)

    def test_small_text_catalog_and_nonsequential_ids(self):
        db = VectorDatabase()
        db.add_text_embeddings([[1, 0], [0, 1]], [7, 42])
        db.set_product_texts(["linen", "denim"])
        for boosted in [True, False]:
            distances, ids = db.search_by_text([0, 1], query_text="denim", keyword_boost=boosted)
            self.assertEqual(ids.tolist(), [[42, 7]])
            self.assertEqual(distances.shape, ids.shape)
        with self.assertRaises(ValueError):
            db.search_by_text([1, 2, 3])

    def test_repeated_load_never_calls_model_or_changes_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "catalog.csv"
            source.write_text("catalog v1")
            df = pd.DataFrame({"text_for_embedding": ["linen", "denim"]})
            factory = Mock()
            factory.return_value.generate_text_embedding.return_value = np.eye(2)
            root = Path(directory) / "indexes"
            first, _ = load_or_build(source, df, root, "text", "model", factory)
            before = {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
            second, _ = load_or_build(source, df, root, "text", "model", Mock(side_effect=AssertionError("rebuilt")))
            after = {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(first.text_ids, second.text_ids)
            old_key = index_key(source, "text", "model")
            source.write_text("catalog v2")
            self.assertNotEqual(old_key, index_key(source, "text", "model"))
            self.assertNotEqual(index_key(source, "text", "model"), index_key(source, "text", "different"))

    def test_corrupt_metadata_fails_without_silent_rebuild(self):
        db = VectorDatabase()
        db.add_image_embeddings([[1, 0]], [2])
        with tempfile.TemporaryDirectory() as directory:
            db.save_indices(directory)
            path = Path(directory) / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["image_ids"] = [2, 3]
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                VectorDatabase().load_indices(directory)

    def test_invalid_vectors_fail_before_saving(self):
        for vectors, ids in [([[0, 0]], [1]), ([[float("nan"), 1]], [1]), ([[1, 0]], [1, 2])]:
            with self.assertRaises(ValueError):
                VectorDatabase().add_image_embeddings(vectors, ids)

    def test_keyword_unknown_words_return_no_results(self):
        search = KeywordSearch(["linen jacket", "denim jacket"])
        self.assertEqual(search.search("astronaut"), [])
        self.assertEqual(search.search("denim"), [1])

    def test_explanation_cannot_borrow_attributes_or_invent_synonyms(self):
        product = {"name": "FAUX LEATHER JACKET", "details": "Lapel collar. Hip pockets with zip fastening. Button-up front."}
        label, excerpts = product_evidence(product, "a zip closure leather jacket")
        self.assertEqual(label, "Matching words in the listing")
        for excerpt in excerpts:
            self.assertTrue(excerpt == product["name"] or excerpt in product["details"])
        self.assertNotIn("zip closure", " ".join(excerpts))
        label, _ = product_evidence(product, "waterproof")
        self.assertEqual(label, "Listing details")

    def test_empty_product_description_is_not_nan(self):
        loader = ProductDataLoader("ZARA_jackets_men.csv")
        product = loader.get_product_details([53])[0]
        self.assertEqual(product["details"], "")
        with self.assertRaises(ValueError):
            loader.get_product_details([-1])

    def test_metrics_use_fixed_cutoff_and_relevant_set(self):
        score = metrics(["a", "x", "b"], ["a", "b"])
        self.assertEqual(score["precision_at_5"], .4)
        self.assertEqual(score["recall_at_5"], 1)
        self.assertEqual(score["reciprocal_rank_at_5"], 1)
        self.assertAlmostEqual(score["ndcg_at_5"], .919720789, places=6)
        self.assertEqual(metrics([], ["a"])["ndcg_at_5"], 0)


if __name__ == "__main__":
    unittest.main()
