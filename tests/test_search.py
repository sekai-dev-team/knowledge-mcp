"""Tests for the Indexer hybrid search functionality."""


class TestSearch:
    """Tests for BM25, vector, RRF merge, snippet, and search options."""

    def test_bm25_search_returns_results(self, indexer):
        """BM25 search should return results for a relevant query."""
        indexer.full_index()

        results = indexer.search("machine learning")
        assert len(results) > 0
        # At least one result should have a positive BM25 score
        assert any(r["bm25_score"] > 0 for r in results)

    def test_vector_search_returns_results(self, indexer):
        """Vector search should return results with vec_score > 0."""
        indexer.full_index()

        results = indexer.search("async programming")
        assert len(results) > 0
        assert any(r["vec_score"] > 0 for r in results)

    def test_rrf_merges_results(self, indexer):
        """RRF merge should produce combined scores."""
        indexer.full_index()

        results = indexer.search("concurrency")
        assert len(results) > 0
        assert all(r["combined_score"] > 0 for r in results)

    def test_snippet_generation(self, indexer):
        """Snippet should contain query terms wrapped in <mark> tags."""
        indexer.full_index()

        results = indexer.search("supervised learning")
        assert len(results) > 0
        # At least one result should have a snippet with markers
        assert any("<mark>" in r["snippet"] for r in results)

    def test_search_empty_query(self, indexer):
        """Empty or whitespace-only query returns empty list."""
        indexer.full_index()

        assert indexer.search("") == []
        assert indexer.search("   ") == []

    def test_search_limit_respected(self, indexer):
        """Limit parameter caps the number of results."""
        indexer.full_index()

        results = indexer.search("learning", limit=2)
        assert len(results) <= 2

        results = indexer.search("learning", limit=1)
        assert len(results) <= 1
