"""Query router with intent classification."""

from enum import Enum
import re


class QueryIntent(Enum):
    NAVIGATION = "navigation"
    FACTUAL = "factual"
    COMPARISON = "comparison"
    LIST = "list"
    EXPLANATION = "explanation"


NAVIGATION_PATTERNS = [
    r"\bwhere\b",
    r"\bwhich\s+(section|part|chapter)\b",
    r"\bfind\b",
    r"\blocated\b",
    r"\blocated\s+in\b",
    r"\bdependencies\b",
    r"\bdepends\s+on\b",
]

FACTUAL_PATTERNS = [
    r"\bwhat\s+is\b",
    r"\bwhat\s+does\b",
    r"\bdefine\b",
    r"\bdefinition\b",
    r"\bmeaning\b",
    r"\bwhat\s+is\s+a\b",
    r"\bwhat\s+is\s+an\b",
]

COMPARISON_PATTERNS = [
    r"\bdifference\s+between\b",
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bbetter\s+than\b",
    r"\bdiffer\b",
    r"\bsame\s+as\b",
]

LIST_PATTERNS = [
    r"\blist\s+all\b",
    r"\blist\b",
    r"\bexamples\b",
    r"\btypes\b",
    r"\bkinds\b",
    r"\bcategories\b",
    r"\ballows\b",
]

EXPLANATION_PATTERNS = [
    r"\bhow\s+does\b",
    r"\bhow\s+to\b",
    r"\bwhy\b",
    r"\bwhy\s+does\b",
    r"\bexplain\b",
    r"\bexplain\s+how\b",
    r"\bdescribe\b",
]


def classify_intent(query: str) -> QueryIntent:
    """Classify query intent."""
    query_lower = query.lower()
    
    if any(re.search(p, query_lower) for p in NAVIGATION_PATTERNS):
        return QueryIntent.NAVIGATION
    
    if any(re.search(p, query_lower) for p in COMPARISON_PATTERNS):
        return QueryIntent.COMPARISON
    
    if any(re.search(p, query_lower) for p in LIST_PATTERNS):
        return QueryIntent.LIST
    
    if any(re.search(p, query_lower) for p in EXPLANATION_PATTERNS):
        return QueryIntent.EXPLANATION
    
    if any(re.search(p, query_lower) for p in FACTUAL_PATTERNS):
        return QueryIntent.FACTUAL
    
    return QueryIntent.FACTUAL


def get_retrieval_strategy(intent: QueryIntent) -> str:
    """Get retrieval strategy based on intent."""
    strategies = {
        QueryIntent.NAVIGATION: "graph_first",
        QueryIntent.COMPARISON: "hybrid_boosted",
        QueryIntent.LIST: "hybrid_broad",
        QueryIntent.EXPLANATION: "hybrid_deep",
        QueryIntent.FACTUAL: "hybrid_balanced",
    }
    return strategies.get(intent, "hybrid_balanced")