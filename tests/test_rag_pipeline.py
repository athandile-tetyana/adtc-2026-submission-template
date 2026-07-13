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

    def test_rerank_prefers_keyword_matches(self):
        query = "maize soil preparation"
        candidates = [
            {"id": "chunk-a", "text": "tomato disease control"},
            {"id": "chunk-b", "text": "maize soil preparation and planting"},
            {"id": "chunk-c", "text": "soil preparation for vegetables"},
        ]

        reranked = embed_and_index.rerank_candidates(query, candidates, top_k=2)

        self.assertEqual([item["id"] for item in reranked], ["chunk-b", "chunk-c"])

    def test_rerank_uses_synonym_expansion(self):
        query = "maize soil prep"
        candidates = [
            {"id": "chunk-z", "text": "corn land preparation for planting"},
            {"id": "chunk-a", "text": "tomato disease control"},
        ]

        reranked = embed_and_index.rerank_candidates(query, candidates, top_k=1)

        self.assertEqual([item["id"] for item in reranked], ["chunk-z"])

    def test_rerank_combines_dense_and_lexical_scores(self):
        query = "maize soil prep"
        candidates = [
            {"id": "chunk-a", "text": "tomato disease control"},
            {"id": "chunk-b", "text": "maize soil preparation and planting"},
        ]

        reranked = embed_and_index.rerank_candidates(query, candidates, top_k=2, vector_scores=[0.95, 0.80])

        self.assertEqual([item["id"] for item in reranked], ["chunk-b", "chunk-a"])

    def test_rerank_boosts_matches_from_relevant_sources(self):
        query = "maize soil prep"
        candidates = [
            {"id": "chunk-a", "source": "vegetables.pdf", "text": "soil preparation for vegetables"},
            {"id": "chunk-b", "source": "maize production.pdf", "text": "soil preparation and planting"},
        ]

        reranked = embed_and_index.rerank_candidates(query, candidates, top_k=1)

        self.assertEqual([item["id"] for item in reranked], ["chunk-b"])

    def test_infer_query_metadata_extracts_explicit_fields(self):
        inferred = embed_and_index.infer_query_metadata("maize soil preparation in south africa")

        self.assertEqual(inferred["crop"], "maize")
        self.assertEqual(inferred["topic"], "soil preparation")
        self.assertEqual(inferred["region"], "south africa")

    def test_filter_candidates_by_metadata_prefers_matching_chunks(self):
        query = "maize soil preparation in south africa"
        candidates = [
            {"id": "chunk-a", "crop": "wheat", "topic": "pest control", "region": "kenya", "text": "wheat pest control"},
            {"id": "chunk-b", "crop": "maize", "topic": "soil preparation", "region": "south africa", "text": "maize soil preparation"},
        ]

        filtered = embed_and_index.filter_candidates_by_metadata(query, candidates, top_k=1)

        self.assertEqual([item["id"] for item in filtered], ["chunk-b"])


if __name__ == "__main__":
    unittest.main()
