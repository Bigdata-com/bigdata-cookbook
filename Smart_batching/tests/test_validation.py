"""
Validation and integration tests.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock
import os

from search_function import (
    plan_search,
    execute_search,
    validate_date_range,
    validate_chunk_percentage,
    save_plan,
    load_plan
)


class TestInputValidation:
    """Tests for input validation."""
    
    def test_validate_date_format(self):
        """Test date format validation."""
        # Valid dates
        validate_date_range("2023-01-01", "2023-12-31")
        validate_date_range("2023-06-15", "2023-06-15")  # Same day is OK
        
        # Invalid formats
        with pytest.raises(ValueError, match="Invalid start date format"):
            validate_date_range("2023/01/01", "2023-12-31")
        
        with pytest.raises(ValueError, match="Invalid end date format"):
            validate_date_range("2023-01-01", "12/31/2023")
    
    def test_validate_date_range_order(self):
        """Test that start date must be before end date."""
        with pytest.raises(ValueError, match="Start date.*must be before"):
            validate_date_range("2023-12-31", "2023-01-01")
    
    def test_validate_chunk_percentage_bounds(self):
        """Test chunk percentage bounds validation."""
        # Valid percentages
        validate_chunk_percentage(0.0)
        validate_chunk_percentage(0.5)
        validate_chunk_percentage(1.0)
        
        # Invalid: too high
        with pytest.raises(ValueError, match="chunk_percentage must be between"):
            validate_chunk_percentage(1.5)
        
        # Invalid: negative
        with pytest.raises(ValueError, match="chunk_percentage must be between"):
            validate_chunk_percentage(-0.1)
        
        # Invalid: wrong type
        with pytest.raises(ValueError, match="chunk_percentage must be a number"):
            validate_chunk_percentage("0.5")
    
    def test_validate_csv_file_exists(self, tmp_path):
        """Test that CSV file existence is validated."""
        csv_file = tmp_path / "universe.csv"
        
        with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
            with pytest.raises(FileNotFoundError):
                plan_search(
                    text="test",
                    universe_csv_path=str(csv_file),
                    start_date="2023-01-01",
                    end_date="2023-12-31"
                )
    
    def test_validate_plan_structure(self):
        """Test that execute_search validates plan structure."""
        with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
            # Missing baskets
            with pytest.raises(ValueError, match="Invalid search plan"):
                execute_search({}, chunk_percentage=0.1)
            
            # Empty baskets
            with pytest.raises(ValueError, match="Invalid search plan"):
                execute_search({"baskets": []}, chunk_percentage=0.1)


class TestQueryStructure:
    """Tests for query structure validation."""
    
    def test_query_structure_matches_benchmark_format(self, tmp_path):
        """Test that query structure matches benchmark format."""
        csv_file = tmp_path / "universe.csv"
        csv_file.write_text("B8EF97\nBB07E4\n")
        
        with patch('search_function.get_smart_batching_planner') as mock_get_planner:
            mock_get_planner.return_value = None
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                plan = plan_search(
                    text="earnings revenue profit",
                    universe_csv_path=str(csv_file),
                    start_date="2023-01-01",
                    end_date="2023-12-31"
                )
        
        query = plan["baskets"][0]["query"]
        
        # Check required fields
        assert "auto_enrich_filters" in query
        assert "text" in query
        assert "filters" in query
        assert "ranking_params" in query
        assert "max_chunks" in query
        
        # Check filter structure
        assert "timestamp" in query["filters"]
        assert "entity" in query["filters"]
        assert "start" in query["filters"]["timestamp"]
        assert "end" in query["filters"]["timestamp"]
        assert "any_of" in query["filters"]["entity"]
        
        # Check ranking params
        assert "source_boost" in query["ranking_params"]
        assert "freshness_boost" in query["ranking_params"]
        assert "reranker" in query["ranking_params"]
        
        # Check text is embedded
        assert query["text"] == "earnings revenue profit"


class TestPlanPersistence:
    """Tests for plan save/load functionality."""
    
    def test_save_and_load_plan(self, tmp_path):
        """Test saving and loading plans."""
        plan = {
            "total_expected_chunks": 1000,
            "baskets": [
                {
                    "basket_id": "b1",
                    "expected_chunks": 1000,
                    "query": {"text": "test", "filters": {}}
                }
            ],
            "planning_metadata": {}
        }
        
        plan_file = tmp_path / "test_plan.json"
        save_plan(plan, str(plan_file))
        
        assert plan_file.exists()
        
        loaded_plan = load_plan(str(plan_file))
        assert loaded_plan["total_expected_chunks"] == 1000
        assert len(loaded_plan["baskets"]) == 1
        assert loaded_plan["baskets"][0]["query"]["text"] == "test"


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_complete_workflow(self, tmp_path):
        """Test complete workflow: plan -> execute."""
        csv_file = tmp_path / "universe.csv"
        csv_file.write_text("B8EF97\nBB07E4\n")
        
        # Mock API responses
        mock_search_response = {
            "results": {
                "chunks": [
                    {"text": "chunk1", "relevance": 0.9, "document_id": "doc1"},
                    {"text": "chunk2", "relevance": 0.8, "document_id": "doc2"}
                ]
            }
        }
        
        with patch('search_function.get_smart_batching_planner') as mock_get_planner:
            mock_get_planner.return_value = None
            
            with patch('search_function.make_search_request') as mock_api:
                mock_api.return_value = mock_search_response
                
                with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                    # Step 1: Plan
                    plan = plan_search(
                        text="earnings",
                        universe_csv_path=str(csv_file),
                        start_date="2023-01-01",
                        end_date="2023-12-31"
                    )
                    
                    # Step 2: Execute
                    results = execute_search(plan, chunk_percentage=0.1)
        
        # Validate results
        assert len(results) > 0
        assert "text" in results[0]
        assert "relevance" in results[0]
    
    def test_proportional_sampling_preserves_distribution(self, tmp_path):
        """Test that proportional sampling preserves distribution in end-to-end test."""
        csv_file = tmp_path / "universe.csv"
        csv_file.write_text("B8EF97\nBB07E4\n3461CF\n")
        
        # Create plan with multiple baskets
        plan = {
            "total_expected_chunks": 1000,
            "baskets": [
                {"basket_id": "b1", "expected_chunks": 600, "query": {"text": "test", "filters": {}}},
                {"basket_id": "b2", "expected_chunks": 400, "query": {"text": "test", "filters": {}}}
            ]
        }
        
        mock_response = {"results": {"chunks": [{"text": "chunk", "relevance": 0.8}]}}
        
        with patch('search_function.make_search_request') as mock_api:
            mock_api.return_value = mock_response
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                results = execute_search(plan, chunk_percentage=0.1)
        
        # Verify both baskets were queried with proportional limits
        assert mock_api.call_count == 2
        chunks_requested = [call[1]["query"]["max_chunks"] for call in mock_api.call_args_list]
        assert chunks_requested[0] == 60  # 600 * 0.1
        assert chunks_requested[1] == 40  # 400 * 0.1
