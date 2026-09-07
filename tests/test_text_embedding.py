import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.models.text_embedding import TextEmbeddingGenerator


class TextEmbeddingTests(unittest.TestCase):
    def test_document_batches_keep_order_and_task(self):
        model = TextEmbeddingGenerator.__new__(TextEmbeddingGenerator)
        model.client = Mock()
        model.client.models.embed_content.side_effect = [
            SimpleNamespace(embeddings=[SimpleNamespace(values=[i, 1]) for i in range(5)]),
            SimpleNamespace(embeddings=[SimpleNamespace(values=[5, 1])]),
        ]
        texts = [f"product {i}" for i in range(6)]
        vectors = model.generate_text_embedding(texts, task_type="RETRIEVAL_DOCUMENT")
        np.testing.assert_array_equal(vectors[:, 0], np.arange(6))
        calls = model.client.models.embed_content.call_args_list
        self.assertEqual(calls[0].kwargs["contents"], texts[:5])
        self.assertEqual(calls[1].kwargs["contents"], texts[5:])
        self.assertEqual(calls[0].kwargs["config"].task_type, "RETRIEVAL_DOCUMENT")
        self.assertFalse(calls[0].kwargs["config"].auto_truncate)

    def test_query_shape_and_incomplete_response(self):
        model = TextEmbeddingGenerator.__new__(TextEmbeddingGenerator)
        model.client = Mock()
        model.client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[1, 2])])
        self.assertEqual(model.generate_text_embedding("denim").shape, (2,))
        self.assertEqual(model.client.models.embed_content.call_args.kwargs["config"].task_type, "RETRIEVAL_QUERY")
        with self.assertRaises(ValueError):
            model.generate_text_embedding(["denim", "linen"])
        with self.assertRaises(ValueError):
            model.generate_text_embedding(" ")

    def test_project_is_required_without_using_another_account(self):
        with patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": ""}):
            with self.assertRaisesRegex(ValueError, "GOOGLE_CLOUD_PROJECT"):
                TextEmbeddingGenerator()


if __name__ == "__main__":
    unittest.main()
