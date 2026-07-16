import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_retrieve_top_k_reranks_full_pool_with_vector_scores(self):
        # A lexically strong chunk at FAISS rank 5 must survive into the
        # results: the pool is reranked before truncating to top_k, and the
        # FAISS scores (scaled 5x) feed into the reranker instead of being
        # discarded.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            index_path = tmp_path / "test.faiss"
            metadata_path = tmp_path / "metadata.json"

            vectors = np.array(
                [[1.0, 0.0], [0.98, 0.2], [0.95, 0.31], [0.9, 0.44], [0.8, 0.6]],
                dtype="float32",
            )
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            faiss.write_index(index, str(index_path))

            metadata = [
                {"id": "near-1", "source": "a.pdf", "page": 1, "text": "general farming notes"},
                {"id": "near-2", "source": "a.pdf", "page": 2, "text": "general farming notes"},
                {"id": "near-3", "source": "a.pdf", "page": 3, "text": "general farming notes"},
                {"id": "near-4", "source": "a.pdf", "page": 4, "text": "general farming notes"},
                {"id": "target", "source": "b.pdf", "page": 5, "text": "maize soil preparation planting guide"},
            ]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            query = np.array([[1.0, 0.0]], dtype="float32")
            results = embed_and_index.retrieve_top_k(
                index_path, metadata_path, query, top_k=2, query_text="maize soil preparation"
            )

            self.assertEqual(results[0]["id"], "target")
            self.assertEqual(results[1]["id"], "near-1")

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

    def test_evaluate_and_retrieve_agree_on_ranking(self):
        # evaluate_retrieval_variants exists to measure the production
        # retrieval path, so both functions must rank identically for the
        # same index, query vector, and query text.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            index_path = tmp_path / "test.faiss"
            metadata_path = tmp_path / "metadata.json"

            vectors = np.array(
                [[1.0, 0.0], [0.98, 0.2], [0.95, 0.31], [0.9, 0.44], [0.8, 0.6]],
                dtype="float32",
            )
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            faiss.write_index(index, str(index_path))

            metadata = [
                {"id": "near-1", "source": "a.pdf", "page": 1, "text": "general farming notes"},
                {"id": "near-2", "source": "a.pdf", "page": 2, "text": "general farming notes"},
                {"id": "near-3", "source": "a.pdf", "page": 3, "text": "general farming notes"},
                {"id": "near-4", "source": "a.pdf", "page": 4, "text": "general farming notes"},
                {"id": "target", "source": "b.pdf", "page": 5, "text": "maize soil preparation planting guide"},
            ]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            query_text = "maize soil preparation"
            query_vector = np.array([[1.0, 0.0]], dtype="float32")

            retrieved = embed_and_index.retrieve_top_k(
                index_path, metadata_path, query_vector, top_k=2, query_text=query_text
            )
            evaluated = embed_and_index.evaluate_retrieval_variants(
                query_text, index_path, metadata_path, query_vector=query_vector, top_k=2
            )

            self.assertEqual(
                [item["id"] for item in retrieved], evaluated["reranked_ids"]
            )
            # Guard that the shared ranking is a real reranking, not just
            # raw FAISS order coincidentally matching in both functions.
            self.assertNotEqual(evaluated["raw_ids"], evaluated["reranked_ids"])

    def test_filter_preserves_order_for_production_shaped_chunks(self):
        # Real corpus/chunks.json chunks only have id/source/page/text — no
        # crop/topic/region fields. The filter must not rescore and reorder
        # such candidates (a zero-match tie previously re-sorted them by id,
        # discarding FAISS's similarity ranking).
        query = "maize soil preparation"
        candidates = [
            {"id": "z_chunk", "source": "maize production.pdf", "page": 3, "text": "maize soil preparation"},
            {"id": "m_chunk", "source": "maize production.pdf", "page": 7, "text": "maize planting dates"},
            {"id": "a_chunk", "source": "vegprodnutshell-daff.pdf", "page": 1, "text": "vegetable production overview"},
        ]

        filtered = embed_and_index.filter_candidates_by_metadata(query, candidates, top_k=2)

        self.assertEqual([item["id"] for item in filtered], ["z_chunk", "m_chunk"])

    def test_evaluate_retrieval_variants_reports_raw_and_reranked_ids(self):
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
                {"id": "chunk-a", "crop": "potato", "topic": "disease", "region": "kenya", "text": "potato disease control"},
                {"id": "chunk-b", "crop": "maize", "topic": "soil preparation", "region": "south africa", "text": "maize soil preparation"},
                {"id": "chunk-c", "crop": "maize", "topic": "planting", "region": "south africa", "text": "maize planting guide"},
            ]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            query_vector = np.array([[0.95, 0.05]], dtype="float32")
            result = embed_and_index.evaluate_retrieval_variants(
                "maize soil prep",
                index_path,
                metadata_path,
                query_vector=query_vector,
                top_k=2,
            )

            self.assertEqual(result["raw_ids"][0], "chunk-a")
            self.assertEqual(result["reranked_ids"][0], "chunk-b")


class ServerEmbeddingTests(unittest.TestCase):
    @staticmethod
    def _response(payload):
        response = mock.Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    def test_get_embedding_mean_pools_per_token_server_response(self):
        # llama-server without --pooling returns one embedding per token; the
        # client must mean-pool them, not take the first (BOS) row.
        per_token = [{"embedding": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}]
        with mock.patch.object(embed_and_index.requests, "post", return_value=self._response(per_token)):
            embedding = embed_and_index.get_embedding("http://localhost:8080", "maize planting")

        self.assertEqual(embedding, [3.0, 4.0])
        self.assertNotEqual(embedding, [1.0, 2.0])

    def test_get_embedding_passes_flat_vector_through(self):
        flat = [{"embedding": [0.5, 0.6, 0.7]}]
        with mock.patch.object(embed_and_index.requests, "post", return_value=self._response(flat)):
            embedding = embed_and_index.get_embedding("http://localhost:8080", "maize planting")

        self.assertEqual(embedding, [0.5, 0.6, 0.7])

    def test_build_index_rejects_collapsed_server_embeddings(self):
        # A server returning the same vector for every text (the BOS-token
        # collapse) must fail the index build loudly, not write a broken index.
        constant = [{"embedding": [[0.1, 0.2, 0.3]] * 4}]
        chunks = [
            {"id": f"chunk-{i}", "source": "doc.pdf", "page": i, "text": f"text {i}"}
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with mock.patch.object(embed_and_index, "wait_for_server_ready"), \
                    mock.patch.object(embed_and_index.requests, "post", return_value=self._response(constant)):
                with self.assertRaisesRegex(RuntimeError, "Embedding collapse detected"):
                    embed_and_index.build_index_from_chunks(
                        chunks,
                        server_url="http://localhost:8080",
                        index_path=tmp_path / "index.faiss",
                        metadata_path=tmp_path / "metadata.json",
                    )
            self.assertFalse((tmp_path / "index.faiss").exists())


if __name__ == "__main__":
    unittest.main()
