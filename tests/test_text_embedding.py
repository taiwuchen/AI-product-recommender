import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from product_search.models.text_embedding import DIMENSIONS, ENDPOINT, MODEL_ID, TextEmbeddingGenerator


def response_for(values):
    return Mock(json=Mock(return_value={"data": [
        {"index": i, "embedding": [value] * DIMENSIONS} for i, value in enumerate(values)
    ]}))


class TextEmbeddingTests(unittest.TestCase):
    def setUp(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            self.model = TextEmbeddingGenerator()

    @patch("product_search.models.text_embedding.requests.post")
    def test_batches_restore_input_order_from_response_indices(self, post):
        first = response_for(range(1, 65))
        first.json.return_value["data"].reverse()
        post.side_effect = [first, response_for([65])]
        texts = [f"product {i}" for i in range(65)]
        vectors = self.model.generate_text_embedding(texts)
        np.testing.assert_array_equal(vectors[:, 0], np.arange(1, 66))
        self.assertEqual(vectors.dtype, np.float32)
        calls = post.call_args_list
        self.assertEqual(calls[0].args, (ENDPOINT,))
        self.assertEqual(calls[0].kwargs["headers"], {"Authorization": "Bearer test-key"})
        self.assertEqual(calls[0].kwargs["json"], {"model": MODEL_ID, "input": texts[:64],
                         "dimensions": DIMENSIONS, "encoding_format": "float"})
        self.assertEqual(calls[1].kwargs["json"]["input"], texts[64:])
        self.assertEqual(calls[0].kwargs["timeout"], (5, 30))

    @patch("product_search.models.text_embedding.requests.post")
    def test_query_returns_one_vector(self, post):
        post.return_value = response_for([1])
        self.assertEqual(self.model.generate_text_embedding("denim").shape, (DIMENSIONS,))
        post.return_value.raise_for_status.assert_called_once()

    @patch("product_search.models.text_embedding.requests.post")
    def test_empty_or_invalid_input_never_calls_api(self, post):
        for value in [" ", [], ["denim", ""], [None], 42, {"text": "denim"}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.model.generate_text_embedding(value)
        post.assert_not_called()

    @patch("product_search.models.text_embedding.requests.post")
    def test_invalid_response_never_reaches_index(self, post):
        valid = {"index": 0, "embedding": [1] * DIMENSIONS}
        invalid_batches = [[], [valid, valid], [{**valid, "index": 1}],
                           [{**valid, "index": "0"}], [{**valid, "index": True}], [None],
                           [{"index": 0}], [{**valid, "embedding": [1, 2]}],
                           [{**valid, "embedding": [float("nan")] * DIMENSIONS}],
                           [{**valid, "embedding": [0] * DIMENSIONS}]]
        for rows in invalid_batches:
            post.return_value = Mock(json=Mock(return_value={"data": rows}))
            with self.subTest(rows=str(rows)[:60]), self.assertRaises(ValueError):
                self.model.generate_text_embedding("denim")
        post.return_value = Mock(json=Mock(return_value={"data": [valid, valid]}))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.model.generate_text_embedding(["denim", "linen"])

    @patch("product_search.models.text_embedding.requests.post")
    def test_http_and_network_failures_are_not_fake_embeddings(self, post):
        post.return_value.raise_for_status.side_effect = requests.HTTPError("402 Insufficient credits")
        with self.assertRaises(requests.HTTPError):
            self.model.generate_text_embedding("denim")
        post.return_value.json.assert_not_called()
        post.side_effect = requests.Timeout("Timed out")
        with self.assertRaises(requests.Timeout):
            self.model.generate_text_embedding("denim")

    @patch("product_search.models.text_embedding.requests.post")
    def test_new_dimensions_build_reload_and_search(self, post):
        from product_search.search.indexes import load_or_build
        vectors = np.eye(2, DIMENSIONS, dtype=np.float32)
        post.side_effect = [
            Mock(json=Mock(return_value={"data": [
                {"index": i, "embedding": vector.tolist()} for i, vector in enumerate(vectors)
            ]})),
            Mock(json=Mock(return_value={"data": [{"index": 0, "embedding": vectors[1].tolist()}]})),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "catalog.csv"
            df = pd.DataFrame({"text_for_embedding": ["linen jacket", "denim jacket"]})
            df.to_csv(source, index=False)
            db, report = load_or_build(source, df, directory, "text", MODEL_ID, lambda: self.model)
            loaded, _ = load_or_build(source, df, directory, "text", MODEL_ID,
                                     Mock(side_effect=AssertionError("Rebuilt a saved index")))
            query = self.model.generate_text_embedding("denim jacket")
            self.assertEqual(loaded.search_by_text(query)[1].tolist(), [[1, 0]])
            self.assertEqual(db.index_text.d, DIMENSIONS)
            self.assertEqual(report["model"], MODEL_ID)
        self.assertEqual(post.call_count, 2)

    def test_key_is_required(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": " "}):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                TextEmbeddingGenerator()


if __name__ == "__main__":
    unittest.main()
