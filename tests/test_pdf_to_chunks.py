import unittest

from pdf_to_chunks import MAX_CHUNK_CHARS, enforce_size_limits


class ChunkingTests(unittest.TestCase):
    def test_enforce_size_limits_keeps_chunks_within_target_size(self):
        oversized = "x" * (MAX_CHUNK_CHARS - 50)
        tiny = "y" * 150
        chunks = enforce_size_limits([oversized, tiny])

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= MAX_CHUNK_CHARS for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
