import unittest

from pdf_to_chunks import MAX_CHUNK_CHARS, clean_chunk_text, enforce_size_limits


class ChunkingTests(unittest.TestCase):
    def test_enforce_size_limits_keeps_chunks_within_target_size(self):
        oversized = "x" * (MAX_CHUNK_CHARS - 50)
        tiny = "y" * 150
        chunks = enforce_size_limits([oversized, tiny])

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= MAX_CHUNK_CHARS for chunk in chunks))

    def test_clean_chunk_text_removes_boilerplate_and_duplicates(self):
        text = "Maize Production\nDepartment: Agriculture\nRepublic of South Africa\n\nSoil preparation is important.\nSoil preparation is important.\nPage 2"

        cleaned = clean_chunk_text(text)

        self.assertIn("Soil preparation is important.", cleaned)
        self.assertNotIn("Department:", cleaned)
        self.assertNotIn("Republic of South Africa", cleaned)
        self.assertNotIn("Page 2", cleaned)

    def test_enforce_size_limits_drops_short_generic_sections(self):
        chunks = enforce_size_limits(["Introduction", "Soil preparation for maize is essential for stable yields."])

        self.assertEqual(len(chunks), 1)
        self.assertIn("Soil preparation for maize is essential for stable yields.", chunks[0])


if __name__ == "__main__":
    unittest.main()
