"""
Unit tests for the retriever module.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import hybrid_retrieve, build_grounded_prompt


class TestHybridRetrieve:
    """Tests for hybrid_retrieve function."""
    
    def test_basic_retrieve(self):
        """Test basic retrieval returns a list."""
        query = "sample query unlikely to be in index"
        results = hybrid_retrieve(query, top_k_text=2, top_k_image=2)
        
        assert isinstance(results, list)
    
    def test_result_structure(self):
        """Test that results have required fields."""
        query = "test query"
        results = hybrid_retrieve(query, top_k_text=2, top_k_image=2)
        
        for r in results:
            assert 'id' in r, "Result should have 'id' field"
            assert 'document' in r, "Result should have 'document' field"
            assert 'fused_score' in r, "Result should have 'fused_score' field"
            assert 'modal' in r, "Result should have 'modal' field"
    
    def test_text_only_mode(self):
        """Test text-only retrieval."""
        query = "text query"
        results = hybrid_retrieve(
            query, 
            top_k_text=5, 
            top_k_image=0, 
            weight_text=1.0, 
            weight_image=0.0
        )
        
        assert isinstance(results, list)
        for r in results:
            if r.get('modal'):
                assert r['modal'] == 'text'
    
    def test_image_only_mode(self):
        """Test image-only retrieval."""
        query = "image query"
        results = hybrid_retrieve(
            query, 
            top_k_text=0, 
            top_k_image=5, 
            weight_text=0.0, 
            weight_image=1.0
        )
        
        assert isinstance(results, list)
        for r in results:
            if r.get('modal'):
                assert r['modal'] == 'image'
    
    def test_empty_query(self):
        """Test handling of empty-ish query."""
        query = "x"  # Very short query
        results = hybrid_retrieve(query, top_k_text=2, top_k_image=2)
        
        assert isinstance(results, list)


class TestBuildGroundedPrompt:
    """Tests for build_grounded_prompt function."""
    
    def test_basic_prompt(self):
        """Test basic prompt building."""
        retrieved = [
            {
                "id": "text::p1::0",
                "modal": "text",
                "document": "Short doc text about medical conditions.",
                "metadata": {"source": "p1.pdf"},
                "fused_score": 0.9
            },
            {
                "id": "image::fig1",
                "modal": "image",
                "document": "/tmp/fig1.png",
                "metadata": {"source": "p1.pdf", "caption": "an x-ray image"},
                "fused_score": 0.6
            }
        ]
        
        query = "What does this x-ray show?"
        prompt = build_grounded_prompt(query, retrieved, top_k=2)
        
        assert "Evidence" in prompt, "Prompt should contain 'Evidence'"
        assert query in prompt, "Prompt should contain the query"
        assert "text::p1::0" in prompt, "Prompt should contain text source ID"
        assert "image::fig1" in prompt, "Prompt should contain image source ID"
    
    def test_empty_retrieved(self):
        """Test prompt with no retrieved documents."""
        prompt = build_grounded_prompt("test query", [], top_k=5)
        
        assert "Evidence" in prompt
        assert "No relevant evidence" in prompt or "test query" in prompt
    
    def test_top_k_limit(self):
        """Test that top_k limits the evidence included."""
        retrieved = [
            {
                "id": f"text::doc{i}",
                "modal": "text",
                "document": f"Document {i} content",
                "metadata": {"source": f"doc{i}.txt"},
                "fused_score": 0.9 - (i * 0.1)
            }
            for i in range(10)
        ]
        
        prompt = build_grounded_prompt("query", retrieved, top_k=3)
        
        # Should only include first 3 documents
        assert "text::doc0" in prompt
        assert "text::doc1" in prompt
        assert "text::doc2" in prompt
        assert "text::doc5" not in prompt


class TestScoreNormalization:
    """Tests for score handling."""
    
    def test_scores_are_normalized(self):
        """Test that fused scores are in valid range."""
        query = "test"
        results = hybrid_retrieve(query, top_k_text=3, top_k_image=3)
        
        for r in results:
            score = r.get('fused_score', 0)
            assert 0 <= score <= 1, f"Score {score} should be between 0 and 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])