import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

import embed_and_index


class RetrievalPipelineTests(unittest.TestCase):
    def test_retrieve_top_k_returns_expected_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            index_path = tmp_path / "test.faiss"
            metadata_path = tmp_path / "metadata.json"

            vectors = np.array(
                [[1.0, 0.0], [0.0, 1.0], [0.2, 0.8]],
                dtype="float32",
            )
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            faiss.write_index(index, str(index_path))

            metadata = [
                {"id": "chunk-a", "text": "alpha"},
                {"id": "chunk-b", "text": "beta"},
                {"id": "chunk-c", "text": "gamma"},
            ]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            query = np.array([[1.0, 0.0]], dtype="float32")
            results = embed_and_index.retrieve_top_k(index_path, metadata_path, query, top_k=2)

            self.assertEqual([item["id"] for item in results], ["chunk-a", "chunk-c"])


if __name__ == "__main__":
    unittest.main()
