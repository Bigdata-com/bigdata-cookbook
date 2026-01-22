"""
Unit tests for plan_search() and execute_search() functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import os
from pathlib import Path

from search_function import (
    plan_search,
    execute_search,
    load_universe_from_csv,
    validate_date_range,
    validate_chunk_percentage,
    date_to_iso
)


class TestPlanSearch:
    """Tests for plan_search() function."""
    
    def test_plan_search_creates_valid_structure(self, tmp_path):
        """Test that plan_search returns valid structure with embedded queries."""
        # Create a test CSV file
        csv_file = tmp_path / "test_universe.csv"
        csv_file.write_text("B8EF97\nBB07E4\n3461CF\n")
        
        with patch('search_function.get_smart_batching_planner') as mock_get_planner:
            mock_get_planner.return_value = None  # Use simplified planner
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                plan = plan_search(
                    text="earnings",
                    universe_csv_path=str(csv_file),
                    start_date="2023-01-01",
                    end_date="2023-12-31"
                )
        
        # Validate structure
        assert "total_expected_chunks" in plan
        assert "baskets" in plan
        assert len(plan["baskets"]) > 0
        assert "query" in plan["baskets"][0]
        assert plan["baskets"][0]["query"]["text"] == "earnings"
        assert "filters" in plan["baskets"][0]["query"]
        assert "ranking_params" in plan["baskets"][0]["query"]
    
    def test_plan_search_with_smart_batching_planner(self, tmp_path):
        """Test plan_search with SmartBatchingPlanner."""
        csv_file = tmp_path / "test_universe.csv"
        csv_file.write_text("B8EF97\nBB07E4\n")
        
        # Mock SmartBatchingPlanner
        mock_planner = Mock()
        mock_planner.get_comention_volumes.return_value = (
            {"B8EF97": 100, "BB07E4": 50}, 1
        )
        mock_planner.create_baskets.return_value = [
            {
                "basket_id": "test_basket",
                "companies": ["B8EF97", "BB07E4"],
                "total_chunks": 150,
                "company_count": 2
            }
        ]
        
        with patch('search_function.get_smart_batching_planner') as mock_get_planner:
            mock_get_planner.return_value = mock_planner
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                plan = plan_search(
                    text="earnings",
                    universe_csv_path=str(csv_file),
                    start_date="2023-01-01",
                    end_date="2023-12-31"
                )
        
        assert plan["total_expected_chunks"] == 150
        assert len(plan["baskets"]) == 1
        assert plan["baskets"][0]["query"]["text"] == "earnings"
        assert plan["planning_metadata"]["uses_smart_batching"] is True
    
    def test_plan_search_validates_empty_text(self):
        """Test that plan_search raises error for empty text."""
        with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="text cannot be empty"):
                plan_search(
                    text="",
                    universe_csv_path="test.csv",
                    start_date="2023-01-01",
                    end_date="2023-12-31"
                )
    
    def test_plan_search_validates_dates(self):
        """Test that plan_search validates date formats."""
        with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="Invalid start date format"):
                plan_search(
                    text="earnings",
                    universe_csv_path="test.csv",
                    start_date="2023/01/01",  # Wrong format
                    end_date="2023-12-31"
                )
    
    def test_plan_search_validates_date_range(self):
        """Test that plan_search validates start < end."""
        with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="Start date.*must be before"):
                plan_search(
                    text="earnings",
                    universe_csv_path="test.csv",
                    start_date="2023-12-31",
                    end_date="2023-01-01"  # Start after end
                )
    
    def test_plan_search_validates_api_key(self):
        """Test that plan_search requires API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key must be provided"):
                plan_search(
                    text="earnings",
                    universe_csv_path="test.csv",
                    start_date="2023-01-01",
                    end_date="2023-12-31"
                )


class TestExecuteSearch:
    """Tests for execute_search() function."""
    
    def test_execute_search_proportional_sampling(self):
        """Test that execute_search applies proportional sampling correctly."""
        # Create mock plan
        plan = {
            "total_expected_chunks": 1000,
            "baskets": [
                {
                    "basket_id": "basket1",
                    "expected_chunks": 600,
                    "query": {
                        "text": "earnings",
                        "max_chunks": 600,
                        "filters": {
                            "timestamp": {"start": "2023-01-01T00:00:00Z", "end": "2023-12-31T23:59:59Z"},
                            "entity": {"any_of": ["B8EF97"]}
                        }
                    }
                },
                {
                    "basket_id": "basket2",
                    "expected_chunks": 400,
                    "query": {
                        "text": "earnings",
                        "max_chunks": 400,
                        "filters": {
                            "timestamp": {"start": "2023-01-01T00:00:00Z", "end": "2023-12-31T23:59:59Z"},
                            "entity": {"any_of": ["BB07E4"]}
                        }
                    }
                }
            ]
        }
        
        # Mock API response
        mock_response = {
            "results": {
                "chunks": [
                    {"text": "chunk1", "relevance": 0.9, "document_id": "doc1"}
                ]
            }
        }
        
        with patch('search_function.make_search_request') as mock_api:
            mock_api.return_value = mock_response
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                results = execute_search(plan, chunk_percentage=0.1)
        
        # Verify proportional sampling was applied
        assert mock_api.call_count == 2
        # First basket: 600 * 0.1 = 60 chunks
        call1_query = mock_api.call_args_list[0][1]["query"]
        assert call1_query["max_chunks"] == 60
        # Second basket: 400 * 0.1 = 40 chunks
        call2_query = mock_api.call_args_list[1][1]["query"]
        assert call2_query["max_chunks"] == 40
    
    def test_execute_search_minimum_chunks(self):
        """Test that baskets with small expected chunks get at least 1 chunk."""
        plan = {
            "total_expected_chunks": 100,
            "baskets": [
                {
                    "basket_id": "small_basket",
                    "expected_chunks": 5,  # Small basket
                    "query": {
                        "text": "earnings",
                        "max_chunks": 5,
                        "filters": {}
                    }
                }
            ]
        }
        
        mock_response = {"results": {"chunks": []}}
        
        with patch('search_function.make_search_request') as mock_api:
            mock_api.return_value = mock_response
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                execute_search(plan, chunk_percentage=0.1)
        
        # 5 * 0.1 = 0.5, should round to minimum of 1
        call_query = mock_api.call_args_list[0][1]["query"]
        assert call_query["max_chunks"] >= 1
    
    def test_execute_search_validates_percentage(self):
        """Test that execute_search validates chunk_percentage."""
        plan = {"baskets": []}
        
        with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="chunk_percentage must be between"):
                execute_search(plan, chunk_percentage=1.5)  # > 1.0
    
    def test_execute_search_validates_plan(self):
        """Test that execute_search validates plan structure."""
        with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="Invalid search plan"):
                execute_search({}, chunk_percentage=0.1)  # Missing baskets
    
    def test_execute_search_sorts_results(self):
        """Test that execute_search sorts results by relevance."""
        plan = {
            "total_expected_chunks": 100,
            "baskets": [
                {
                    "basket_id": "basket1",
                    "expected_chunks": 100,
                    "query": {
                        "text": "earnings",
                        "filters": {}
                    }
                }
            ]
        }
        
        mock_response = {
            "results": {
                "chunks": [
                    {"text": "chunk1", "relevance": 0.5, "document_id": "doc1"},
                    {"text": "chunk2", "relevance": 0.9, "document_id": "doc2"},
                    {"text": "chunk3", "relevance": 0.7, "document_id": "doc3"}
                ]
            }
        }
        
        with patch('search_function.make_search_request') as mock_api:
            mock_api.return_value = mock_response
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                results = execute_search(plan, chunk_percentage=0.1, sort_results=True)
        
        # Results should be sorted by relevance descending
        assert results[0]["relevance"] == 0.9
        assert results[1]["relevance"] == 0.7
        assert results[2]["relevance"] == 0.5


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_load_universe_from_csv(self, tmp_path):
        """Test loading universe from CSV."""
        csv_file = tmp_path / "universe.csv"
        csv_file.write_text("B8EF97\nBB07E4\n3461CF\n")
        
        companies = load_universe_from_csv(str(csv_file))
        assert len(companies) == 3
        assert "B8EF97" in companies
        assert "BB07E4" in companies
    
    def test_load_universe_file_not_found(self):
        """Test loading universe raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_universe_from_csv("nonexistent.csv")
    
    def test_validate_date_range(self):
        """Test date range validation."""
        # Valid range
        validate_date_range("2023-01-01", "2023-12-31")
        
        # Invalid: start after end
        with pytest.raises(ValueError, match="Start date.*must be before"):
            validate_date_range("2023-12-31", "2023-01-01")
    
    def test_validate_chunk_percentage(self):
        """Test chunk percentage validation."""
        # Valid percentages
        validate_chunk_percentage(0.0)
        validate_chunk_percentage(0.5)
        validate_chunk_percentage(1.0)
        
        # Invalid: too high
        with pytest.raises(ValueError):
            validate_chunk_percentage(1.5)
        
        # Invalid: negative
        with pytest.raises(ValueError):
            validate_chunk_percentage(-0.1)
    
    def test_date_to_iso(self):
        """Test date to ISO conversion."""
        assert date_to_iso("2023-01-01", is_start=True) == "2023-01-01T00:00:00Z"
        assert date_to_iso("2023-12-31", is_start=False) == "2023-12-31T23:59:59Z"
