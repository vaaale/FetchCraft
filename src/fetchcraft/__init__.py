"""
Fetchcraft - A flexible RAG framework for building intelligent applications.
"""

# Export filter classes and functions for easy imports
from fetchcraft.filters import (
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    FieldFilter,
    CompositeFilter,
    EQ, NE, GT, LT, GTE, LTE, IN, CONTAINS,
    AND, OR, NOT
)

__all__ = [
    # Filter classes
    "FilterOperator",
    "FilterCondition",
    "MetadataFilter",
    "FieldFilter",
    "CompositeFilter",
    # Filter helper functions
    "EQ", "NE", "GT", "LT", "GTE", "LTE", "IN", "CONTAINS",
    "AND", "OR", "NOT",
]
