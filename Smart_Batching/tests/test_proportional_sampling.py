"""
Unit tests for proportional sampling logic.
"""

import pytest
from search_function import execute_search
from unittest.mock import patch
import os


class TestProportionalSampling:
    """Tests for proportional sampling calculations."""
    
    def test_proportional_sampling_10_percent(self):
        """Test 10% sampling preserves distribution."""
        plan = {
            "total_expected_chunks": 1000,
            "baskets": [
                {"basket_id": "b1", "expected_chunks": 600, "query": {"text": "test", "filters": {}}},
                {"basket_id": "b2", "expected_chunks": 400, "query": {"text": "test", "filters": {}}}
            ]
        }
        
        with patch('search_function.make_search_request') as mock_api:
            mock_api.return_value = {"results": {"chunks": []}}
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                execute_search(plan, chunk_percentage=0.1)
        
        # Verify proportional distribution
        assert mock_api.call_count == 2
        # Access keyword arguments
        call1_query = mock_api.call_args_list[0][1]["query"]
        call2_query = mock_api.call_args_list[1][1]["query"]
        
        # 600 * 0.1 = 60, 400 * 0.1 = 40
        assert call1_query["max_chunks"] == 60
        assert call2_query["max_chunks"] == 40
        # Ratio preserved: 60/40 = 1.5, same as 600/400
    
    def test_proportional_sampling_50_percent(self):
        """Test 50% sampling."""
        plan = {
            "total_expected_chunks": 1000,
            "baskets": [
                {"basket_id": "b1", "expected_chunks": 1000, "query": {"text": "test", "filters": {}}}
            ]
        }
        
        with patch('search_function.make_search_request') as mock_api:
            mock_api.return_value = {"results": {"chunks": []}}
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                execute_search(plan, chunk_percentage=0.5)
        
        call_query = mock_api.call_args_list[0][1]["query"]
        assert call_query["max_chunks"] == 500  # 1000 * 0.5
    
    def test_proportional_sampling_minimum_chunks(self):
        """Test that small baskets get minimum 1 chunk."""
        plan = {
            "total_expected_chunks": 100,
            "baskets": [
                {"basket_id": "b1", "expected_chunks": 3, "query": {"text": "test", "filters": {}}},
                {"basket_id": "b2", "expected_chunks": 5, "query": {"text": "test", "filters": {}}}
            ]
        }
        
        with patch('search_function.make_search_request') as mock_api:
            mock_api.return_value = {"results": {"chunks": []}}
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                execute_search(plan, chunk_percentage=0.1)
        
        # 3 * 0.1 = 0.3 -> should be 1 (minimum)
        # 5 * 0.1 = 0.5 -> should be 1 (minimum)
        call1_query = mock_api.call_args_list[0][1]["query"]
        call2_query = mock_api.call_args_list[1][1]["query"]
        assert call1_query["max_chunks"] >= 1
        assert call2_query["max_chunks"] >= 1
    
    def test_proportional_sampling_zero_chunks(self):
        """Test that baskets with 0 expected chunks are skipped."""
        plan = {
            "total_expected_chunks": 100,
            "baskets": [
                {"basket_id": "b1", "expected_chunks": 0, "query": {"text": "test", "filters": {}}},
                {"basket_id": "b2", "expected_chunks": 100, "query": {"text": "test", "filters": {}}}
            ]
        }
        
        with patch('search_function.make_search_request') as mock_api:
            mock_api.return_value = {"results": {"chunks": []}}
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                execute_search(plan, chunk_percentage=0.1)
        
        # Only one basket should be searched (the one with chunks > 0)
        assert mock_api.call_count == 1
    
    def test_proportional_sampling_100_percent(self):
        """Test 100% sampling retrieves all expected chunks."""
        plan = {
            "total_expected_chunks": 1000,
            "baskets": [
                {"basket_id": "b1", "expected_chunks": 1000, "query": {"text": "test", "filters": {}}}
            ]
        }
        
        with patch('search_function.make_search_request') as mock_api:
            mock_api.return_value = {"results": {"chunks": []}}
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                execute_search(plan, chunk_percentage=1.0)
        
        call_query = mock_api.call_args_list[0][1]["query"]
        assert call_query["max_chunks"] == 1000  # 100% of 1000
    
    def test_proportional_sampling_preserves_ratios(self):
        """Test that proportional sampling preserves relative ratios."""
        plan = {
            "total_expected_chunks": 1000,
            "baskets": [
                {"basket_id": "b1", "expected_chunks": 100, "query": {"text": "test", "filters": {}}},
                {"basket_id": "b2", "expected_chunks": 200, "query": {"text": "test", "filters": {}}},
                {"basket_id": "b3", "expected_chunks": 700, "query": {"text": "test", "filters": {}}}
            ]
        }
        
        with patch('search_function.make_search_request') as mock_api:
            mock_api.return_value = {"results": {"chunks": []}}
            
            with patch.dict(os.environ, {'BIGDATA_API_KEY': 'test_key'}):
                execute_search(plan, chunk_percentage=0.1)
        
        calls = [call[1]["query"] for call in mock_api.call_args_list]
        chunks = [call["max_chunks"] for call in calls]
        
        # Ratios should be preserved: 100:200:700 = 10:20:70
        # Sort to handle parallel execution order
        chunks_sorted = sorted(chunks)
        assert chunks_sorted == [10, 20, 70]
        # Verify ratio: 10/20 = 0.5, same as 100/200
        assert chunks_sorted[0] / chunks_sorted[1] == 0.5  # 10/20 = 0.5
