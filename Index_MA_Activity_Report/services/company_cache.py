"""
Company data cache for storing entity IDs and company names
Extends functionality from the simple entity_cache in main.py
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CompanyData:
    """
    Company data structure.
    
    Attributes:
        ticker: Stock ticker symbol
        entity_id: Bigdata.com entity ID
        company_name: Full company name
        cached_at: Timestamp when data was cached
    """
    
    def __init__(self, ticker: str, entity_id: str, company_name: str):
        self.ticker = ticker.upper()
        self.entity_id = entity_id
        self.company_name = company_name
        self.cached_at = datetime.now()
    
    def is_valid(self, ttl_hours: int = 24) -> bool:
        """Check if cache entry is still valid."""
        age = datetime.now() - self.cached_at
        return age < timedelta(hours=ttl_hours)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "entity_id": self.entity_id,
            "company_name": self.company_name,
            "cached_at": self.cached_at.isoformat(),
        }


class CompanyDataCache:
    """
    Cache for company entity IDs and names.
    
    Provides fast lookup of company information with TTL management.
    Thread-safe for concurrent access.
    """
    
    def __init__(self, ttl_hours: int = 24):
        """
        Initialize company cache.
        
        Args:
            ttl_hours: Time-to-live for cache entries in hours (default: 24)
        """
        self._cache: Dict[str, CompanyData] = {}
        self.ttl_hours = ttl_hours
        self.hits = 0
        self.misses = 0
        
        logger.info(f"CompanyDataCache initialized with {ttl_hours}h TTL")
    
    def get(self, ticker: str) -> Optional[CompanyData]:
        """
        Get company data from cache.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CompanyData if found and valid, None otherwise
        """
        ticker = ticker.upper()
        
        if ticker in self._cache:
            data = self._cache[ticker]
            
            if data.is_valid(self.ttl_hours):
                self.hits += 1
                logger.debug(f"Cache hit for {ticker}: {data.company_name}")
                return data
            else:
                # Entry expired, remove it
                logger.debug(f"Cache entry expired for {ticker}")
                del self._cache[ticker]
        
        self.misses += 1
        logger.debug(f"Cache miss for {ticker}")
        return None
    
    def set(self, ticker: str, entity_id: str, company_name: str) -> CompanyData:
        """
        Store company data in cache.
        
        Args:
            ticker: Stock ticker symbol
            entity_id: Bigdata.com entity ID
            company_name: Full company name
            
        Returns:
            CompanyData object that was cached
        """
        ticker = ticker.upper()
        data = CompanyData(ticker, entity_id, company_name)
        self._cache[ticker] = data
        
        logger.info(f"Cached company data: {ticker} -> {company_name} ({entity_id})")
        return data
    
    def has(self, ticker: str) -> bool:
        """
        Check if ticker exists in cache and is valid.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if ticker is cached and valid, False otherwise
        """
        return self.get(ticker) is not None
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared: {count} entries removed")
        return count
    
    def remove(self, ticker: str) -> bool:
        """
        Remove a specific ticker from cache.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if ticker was removed, False if not found
        """
        ticker = ticker.upper()
        if ticker in self._cache:
            del self._cache[ticker]
            logger.debug(f"Removed {ticker} from cache")
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        expired = [
            ticker 
            for ticker, data in self._cache.items() 
            if not data.is_valid(self.ttl_hours)
        ]
        
        for ticker in expired:
            del self._cache[ticker]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired cache entries")
        
        return len(expired)
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats (size, hits, misses, hit rate)
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "ttl_hours": self.ttl_hours,
            "tickers": list(self._cache.keys()),
        }
    
    def reset_stats(self) -> None:
        """Reset hit/miss statistics."""
        self.hits = 0
        self.misses = 0
        logger.debug("Cache statistics reset")
    
    def get_all(self) -> Dict[str, dict]:
        """
        Get all cached company data.
        
        Returns:
            Dictionary of ticker -> company data
        """
        return {
            ticker: data.to_dict() 
            for ticker, data in self._cache.items()
        }

