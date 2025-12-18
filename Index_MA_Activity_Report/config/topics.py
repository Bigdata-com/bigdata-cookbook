"""
Topic templates and search configuration for news searches
Each topic is formatted with {company} placeholder for company name
"""

from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class SearchConfig:
    """Configuration for topic-based news searches"""
    
    # Search options
    baseline_enabled: bool = True   # Enable baseline company-only search
    topics_enabled: bool = True     # Enable topic-based searches
    
    # Time range
    days_lookback: int = 7  # How many days back to search
    
    # Sentiment filters
    sentiment_enabled: bool = True  # Enable sentiment filtering for topics
    sentiment_values: List[str] = None  # ["positive", "negative"] excludes neutral
    
    # Document types
    document_types: List[str] = None  # ["NEWS", "TRANSCRIPT", "PRESS_RELEASE", etc.]
    
    # Relevance settings
    min_relevance_baseline: float = 0.0  # Minimum relevance for baseline results
    min_relevance_topics: float = 0.0    # Minimum relevance for topic results
    rerank_threshold: Optional[float] = None  # Rerank threshold for API
    
    # Result limits
    max_chunks_baseline: int = 100  # Max chunks for baseline search
    max_chunks_topics: int = 10      # Max chunks per topic search
    max_results_per_topic: Optional[int] = None  # Limit results per topic
    
    # Parallel execution
    batch_size: int = 50  # Number of concurrent topic searches
    
    # Rate limiting
    rate_limit_rpm: int = 500  # Requests per minute
    
    # Caching
    cache_ttl_hours: int = 24  # Cache TTL for company data
    
    def __post_init__(self):
        """Set default values for None fields"""
        if self.sentiment_values is None:
            self.sentiment_values = ["positive", "negative"]
        if self.document_types is None:
            self.document_types = ["NEWS", "TRANSCRIPT"]


# Default configuration (current production settings)
DEFAULT_CONFIG = SearchConfig(
    baseline_enabled=False,       # Topics only by default
    topics_enabled=True,
    days_lookback=7,
    sentiment_enabled=True,
    sentiment_values=["positive", "negative"],
    document_types=["NEWS", "TRANSCRIPT"],
    min_relevance_baseline=0.0,
    min_relevance_topics=0.0,
    rerank_threshold=None,
    max_chunks_baseline=100,
    max_chunks_topics=10,
    batch_size=50,
    rate_limit_rpm=500,
    cache_ttl_hours=24,
)

# Aggressive configuration (stricter filters, higher quality)
AGGRESSIVE_CONFIG = SearchConfig(
    baseline_enabled=False,       # Topics only
    topics_enabled=True,
    days_lookback=7,
    sentiment_enabled=True,
    sentiment_values=["positive", "negative"],
    document_types=["NEWS", "TRANSCRIPT"],
    min_relevance_baseline=0.5,  # Only high-relevance baseline articles
    min_relevance_topics=0.6,     # Only very relevant topic articles
    rerank_threshold=0.3,         # Enable reranking
    max_chunks_baseline=50,       # Fewer chunks = faster
    max_chunks_topics=5,          # Very selective
    batch_size=50,
    rate_limit_rpm=500,
    cache_ttl_hours=24,
)

# Comprehensive configuration (more results, all sentiment)
COMPREHENSIVE_CONFIG = SearchConfig(
    baseline_enabled=False,       # Topics only
    topics_enabled=True,
    days_lookback=14,             # Look back 2 weeks
    sentiment_enabled=False,      # Include all sentiment (even neutral)
    sentiment_values=None,        # Not used when disabled
    document_types=["NEWS", "TRANSCRIPT", "PRESS_RELEASE"],
    min_relevance_baseline=0.0,
    min_relevance_topics=0.0,
    rerank_threshold=None,
    max_chunks_baseline=200,      # More chunks
    max_chunks_topics=20,         # More chunks per topic
    batch_size=50,
    rate_limit_rpm=500,
    cache_ttl_hours=24,
)

# Fast configuration (speed-optimized)
FAST_CONFIG = SearchConfig(
    baseline_enabled=False,       # Topics only
    topics_enabled=True,
    days_lookback=3,              # Only last 3 days
    sentiment_enabled=True,
    sentiment_values=["positive", "negative"],
    document_types=["NEWS"],      # News only (faster)
    min_relevance_baseline=0.3,   # Skip low-relevance
    min_relevance_topics=0.4,
    rerank_threshold=None,
    max_chunks_baseline=30,       # Fewer chunks
    max_chunks_topics=3,          # Very few per topic
    batch_size=50,
    rate_limit_rpm=500,
    cache_ttl_hours=24,
)

# Breaking news configuration (recent only, all sentiment)
BREAKING_NEWS_CONFIG = SearchConfig(
    baseline_enabled=False,       # Topics only
    topics_enabled=True,
    days_lookback=1,              # Only today
    sentiment_enabled=False,      # All sentiment (breaking news is breaking)
    sentiment_values=None,
    document_types=["NEWS", "PRESS_RELEASE"],
    min_relevance_baseline=0.0,
    min_relevance_topics=0.0,
    rerank_threshold=None,
    max_chunks_baseline=50,
    max_chunks_topics=5,
    batch_size=50,
    rate_limit_rpm=500,
    cache_ttl_hours=1,            # Short cache for breaking news
)

# Topics only configuration (no baseline search)
TOPICS_ONLY_CONFIG = SearchConfig(
    baseline_enabled=False,       # Skip baseline search
    topics_enabled=True,          # Only topic searches
    days_lookback=7,
    sentiment_enabled=True,
    sentiment_values=["positive", "negative"],
    document_types=["NEWS", "TRANSCRIPT"],
    min_relevance_baseline=0.0,
    min_relevance_topics=0.0,
    rerank_threshold=None,
    max_chunks_baseline=0,        # Not used
    max_chunks_topics=10,
    batch_size=50,
    rate_limit_rpm=500,
    cache_ttl_hours=24,
)


# Baseline only configuration (entity search only, no topics)
BASELINE_ONLY_CONFIG = SearchConfig(
    baseline_enabled=True,        # Run baseline search
    topics_enabled=False,         # Skip topic searches
    days_lookback=7,
    sentiment_enabled=False,      # No sentiment filter for basic search
    sentiment_values=None,
    document_types=["NEWS", "TRANSCRIPT"],
    min_relevance_baseline=0.0,
    min_relevance_topics=0.0,     # Not used
    rerank_threshold=None,
    max_chunks_baseline=100,      # All chunks for baseline
    max_chunks_topics=0,          # Not used
    batch_size=50,
    rate_limit_rpm=500,
    cache_ttl_hours=24,
)



STANDARD_TOPICS = [
    # Earnings & Financial Performance
    {"topic_name": "Financial Metrics", "topic_text": "What key takeaways emerged from {company}'s latest earnings report?"},
    {"topic_name": "Financial Metrics", "topic_text": "What notable changes in {company}'s financial performance metrics have been reported recently?"},
    {"topic_name": "Financial Metrics", "topic_text": "Has {company} revised its financial or operational guidance for upcoming periods?"},
    
    # Strategy & Business Development
    {"topic_name": "M&A", "topic_text": "What significant strategic initiatives or business pivots has {company} announced recently?"},
    {"topic_name": "M&A", "topic_text": "What material acquisition, merger, or divestiture activities involve {company} currently?"},
    
    # Leadership & Organization
    {"topic_name": "Leadership", "topic_text": "What executive leadership changes have been announced at {company} recently?"},
    
    # Commercial & Market Activity
    {"topic_name": "Competition", "topic_text": "What significant contract wins, losses, or renewals has {company} recently announced?"},
    {"topic_name": "Competition", "topic_text": "What notable market share shifts has {company} experienced recently?"},
    {"topic_name": "Competition", "topic_text": "How is {company} responding to new competitive threats or significant competitor actions?"},
    
    # Product & Innovation
    {"topic_name": "Products", "topic_text": "What significant new product launches or pipeline developments has {company} announced?"},
    
    # Operations & Supply Chain
    {"topic_name": "Supply Chain", "topic_text": "What material operational disruptions or capacity changes is {company} experiencing currently?"},
    {"topic_name": "Supply Chain", "topic_text": "How are supply chain conditions affecting {company}'s operations and outlook?"},
    {"topic_name": "Supply Chain", "topic_text": "What production milestones or efficiency improvements has {company} achieved recently?"},
    
    # Cost Management
    {"topic_name": "Costs", "topic_text": "What cost-cutting measures or expense management initiatives has {company} recently disclosed?"},
    
    # Regulatory & Legal
    {"topic_name": "Regulatory", "topic_text": "What specific regulatory developments are materially affecting {company}?"},
    {"topic_name": "Regulatory", "topic_text": "What material litigation developments involve {company} currently?"},
    
    # Macro & Industry Trends
    {"topic_name": "Industry", "topic_text": "How are current macroeconomic factors affecting {company}'s performance and outlook?"},
    {"topic_name": "Industry", "topic_text": "What industry-specific trends or disruptions are directly affecting {company}?"},
    
    # Capital Allocation & Financing
    {"topic_name": "Financing", "topic_text": "What significant capital allocation decisions has {company} announced recently?"},
    {"topic_name": "Financing", "topic_text": "What changes to dividends, buybacks, or other shareholder return programs has {company} announced?"},
    {"topic_name": "Financing", "topic_text": "What debt issuance, refinancing, or covenant changes has {company} recently announced?"},
    {"topic_name": "Financing", "topic_text": "Have there been any credit rating actions or outlook changes for {company} recently?"},
    
    # Market Sentiment & Events
    {"topic_name": "Markets", "topic_text": "What shifts in the prevailing narrative around {company} are emerging among influential investors?"},
    {"topic_name": "Markets", "topic_text": "What significant events could impact {company}'s performance in the near term?"},
    {"topic_name": "Markets", "topic_text": "What unexpected disclosures or unusual trading patterns has {company} experienced recently?"},
    {"topic_name": "Markets", "topic_text": "Is there any activist investor involvement or significant shareholder actions affecting {company}?"},
]

DEFAULT_MINI_TOPICS = [
    # Earnings & Financial Performance
    {"topic_name": "Financial Metrics", "topic_text": "What key takeaways emerged from {company}'s latest earnings report?"},
    
    # Commercial & Market Activity
    {"topic_name": "Competition", "topic_text": "What significant contract wins, losses, or renewals has {company} recently announced?"},
    {"topic_name": "Markets", "topic_text": "Is there any activist investor involvement or significant shareholder actions affecting {company}?"},
]

# M&A focused topics - subset for tracking acquisition activity
MA_TOPICS = [
    # Core M&A Activity
    {"topic_name": "M&A Activity", "topic_text": "What material acquisition, merger, or divestiture activities involve {company} currently?"},
    {"topic_name": "M&A Activity", "topic_text": "Is {company} being acquired or in merger discussions with another company?"},
    {"topic_name": "M&A Activity", "topic_text": "What acquisition offers or takeover bids has {company} received recently?"},
    {"topic_name": "M&A Activity", "topic_text": "What strategic alternatives or sale processes is {company} exploring?"},
    {"topic_name": "M&A Activity", "topic_text": "When did {company} announce or sign a definitive merger agreement or acquisition deal?"},
    {"topic_name": "M&A Activity", "topic_text": "What are the key timeline milestones and closing dates for {company}'s pending merger or acquisition transaction?"}

    
    
    # Deal Terms & Valuation
    #{"topic_name": "Deal Terms", "topic_text": "What are the deal terms, valuation, or price being discussed for {company}'s acquisition?"},
    #{"topic_name": "Deal Terms", "topic_text": "What premium to market price is being offered for {company} shares?"},
    
    # Regulatory & Approval
    #{"topic_name": "Regulatory Approval", "topic_text": "What regulatory approvals or antitrust reviews are pending for {company}'s deal?"},
    #{"topic_name": "Regulatory Approval", "topic_text": "What government or regulatory bodies are reviewing {company}'s merger or acquisition?"},
    
    # Activist & Shareholder Activity
    #{"topic_name": "Activist Involvement", "topic_text": "Is there activist investor involvement pushing for {company} to explore a sale or merger?"},
    #{"topic_name": "Shareholder Actions", "topic_text": "What significant shareholder actions or proxy contests are affecting {company}?"},
    
    # Strategic Rationale
    #{"topic_name": "Strategic Rationale", "topic_text": "What significant strategic initiatives or business pivots has {company} announced recently?"},
    #{"topic_name": "Strategic Rationale", "topic_text": "What synergies or strategic benefits are expected from {company}'s deal?"},
]

# Topic set registry - maps topic set names to their lists
TOPIC_SETS = {
    "default": {
        "name": "Default Topics",
        "description": "General portfolio analysis topics covering financial metrics, M&A, leadership, competition, and more",
        "topics": DEFAULT_MINI_TOPICS
    },

    "default_mini": {
        "name": "Standard Topics",
        "description": "General portfolio analysis topics covering financial metrics, M&A, leadership, competition, and more",
        "topics": STANDARD_TOPICS
    },
    "ma_topics": {
        "name": "M&A Topics",
        "description": "Topics focused on M&A activity, acquisitions, mergers, divestitures, deal terms, and regulatory approvals",
        "topics": MA_TOPICS
    }
}


def get_available_topic_sets() -> dict:
    """
    Get all available topic sets with metadata.
    
    Returns:
        Dictionary mapping topic set keys to metadata (name, description, topic_count)
    """
    return {
        key: {
            "name": value["name"],
            "description": value["description"],
            "topic_count": len(value["topics"])
        }
        for key, value in TOPIC_SETS.items()
    }


def get_topic_set_by_name(name: str) -> Optional[List[Dict[str, str]]]:
    """
    Get a topic set by its name/key.
    
    Args:
        name: Topic set key (default, data_ai, software_agents)
        
    Returns:
        List of topic dictionaries, or None if not found
    """
    topic_set = TOPIC_SETS.get(name.lower())
    if topic_set:
        # Return a copy to avoid modifying the original
        return [topic.copy() for topic in topic_set["topics"]]
    return None


def list_topic_set_names() -> List[str]:
    """
    Get list of all available topic set names/keys.
    
    Returns:
        List of topic set keys
    """
    return list(TOPIC_SETS.keys())


# Topic categories for grouping (optional, for future use)
TOPIC_CATEGORIES = {
    "earnings": [0, 1, 2],
    "strategy": [3, 4],
    "leadership": [5],
    "commercial": [6, 7, 8],
    "products": [9],
    "operations": [10, 11, 12],
    "costs": [13],
    "regulatory": [14, 15],
    "macro": [16, 17],
    "capital": [18, 19, 20, 21],
    "sentiment": [22, 23, 24, 25],
}

def get_topic_category(topic_index: int) -> str:
    """
    Get category name for a topic index.
    
    Args:
        topic_index: Index of topic in STANDARD_TOPICS list
        
    Returns:
        Category name or "uncategorized"
    """
    for category, indices in TOPIC_CATEGORIES.items():
        if topic_index in indices:
            return category
    return "uncategorized"


def get_config_by_name(name: str) -> SearchConfig:
    """
    Get a predefined configuration by name.
    
    Args:
        name: Configuration name (default, aggressive, comprehensive, fast, breaking_news, topics_only, baseline_only)
        
    Returns:
        SearchConfig instance
        
    Raises:
        ValueError: If configuration name not found
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
        "comprehensive": COMPREHENSIVE_CONFIG,
        "fast": FAST_CONFIG,
        "breaking_news": BREAKING_NEWS_CONFIG,
        "topics_only": TOPICS_ONLY_CONFIG,
        "baseline_only": BASELINE_ONLY_CONFIG,
    }
    
    name_lower = name.lower()
    if name_lower not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    
    return configs[name_lower]


def list_available_configs() -> dict:
    """
    Get all available configurations with descriptions.
    
    Returns:
        Dictionary of config_name -> description
    """
    return {
        "default": "Topic search (7 days, sentiment filtered, no baseline)",
        "aggressive": "Topic search with strict quality filters (high relevance only)",
        "comprehensive": "Topic search with max coverage (14 days, all sentiment)",
        "fast": "Topic search, speed-optimized (3 days, fewer chunks)",
        "breaking_news": "Topic search, recent only (1 day, all sentiment)",
        "topics_only": "Topic searches only (no baseline search)",
        "baseline_only": "Entity search only (no topic searches, no sentiment filter)",
    }

