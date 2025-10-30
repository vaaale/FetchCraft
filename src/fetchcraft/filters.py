"""
Metadata filtering system for vector search.

Provides a flexible filter API that can be translated to native
vector store filter formats.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Union
from pydantic import BaseModel, Field


class FilterOperator(str, Enum):
    """Supported filter operators."""
    EQ = "=="  # Equal
    NE = "!="  # Not equal
    GT = ">"   # Greater than
    LT = "<"   # Less than
    GTE = ">=" # Greater than or equal
    LTE = "<=" # Less than or equal
    IN = "in"  # In list
    CONTAINS = "contains"  # String contains


class FilterCondition(str, Enum):
    """Logical conditions for combining filters."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class MetadataFilter(BaseModel, ABC):
    """Base class for metadata filters."""
    
    @abstractmethod
    def to_dict(self) -> dict:
        """Convert filter to dictionary representation."""
        pass


class FieldFilter(MetadataFilter):
    """
    Filter for a single metadata field.
    
    Example:
        ```python
        # Simple equality
        filter = FieldFilter(key="category", operator=FilterOperator.EQ, value="ai")
        
        # Numeric comparison
        filter = FieldFilter(key="year", operator=FilterOperator.GTE, value=2020)
        
        # In list
        filter = FieldFilter(key="status", operator=FilterOperator.IN, value=["active", "pending"])
        
        # String contains
        filter = FieldFilter(key="title", operator=FilterOperator.CONTAINS, value="machine learning")
        ```
    """
    key: str = Field(description="Metadata field key")
    operator: FilterOperator = Field(description="Comparison operator")
    value: Any = Field(description="Value to compare against")
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "type": "field",
            "key": self.key,
            "operator": self.operator.value,
            "value": self.value
        }
    
    def __repr__(self) -> str:
        return f"FieldFilter({self.key} {self.operator.value} {self.value})"


class CompositeFilter(MetadataFilter):
    """
    Composite filter combining multiple filters with logical conditions.
    
    Example:
        ```python
        # AND condition
        filter = CompositeFilter(
            condition=FilterCondition.AND,
            filters=[
                FieldFilter(key="category", operator=FilterOperator.EQ, value="ai"),
                FieldFilter(key="year", operator=FilterOperator.GTE, value=2020)
            ]
        )
        
        # OR condition
        filter = CompositeFilter(
            condition=FilterCondition.OR,
            filters=[
                FieldFilter(key="status", operator=FilterOperator.EQ, value="active"),
                FieldFilter(key="status", operator=FilterOperator.EQ, value="pending")
            ]
        )
        
        # NOT condition (negates a filter)
        filter = CompositeFilter(
            condition=FilterCondition.NOT,
            filters=[
                FieldFilter(key="archived", operator=FilterOperator.EQ, value=True)
            ]
        )
        
        # Nested conditions
        filter = CompositeFilter(
            condition=FilterCondition.AND,
            filters=[
                FieldFilter(key="category", operator=FilterOperator.EQ, value="ai"),
                CompositeFilter(
                    condition=FilterCondition.OR,
                    filters=[
                        FieldFilter(key="year", operator=FilterOperator.EQ, value=2023),
                        FieldFilter(key="year", operator=FilterOperator.EQ, value=2024)
                    ]
                )
            ]
        )
        ```
    """
    condition: FilterCondition = Field(description="Logical condition")
    filters: List[Union["CompositeFilter", FieldFilter]] = Field(
        description="List of filters to combine"
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "type": "composite",
            "condition": self.condition.value,
            "filters": [f.to_dict() for f in self.filters]
        }
    
    def __repr__(self) -> str:
        filters_repr = ", ".join(str(f) for f in self.filters)
        return f"CompositeFilter({self.condition.value}: [{filters_repr}])"


# Allow recursive type
CompositeFilter.model_rebuild()


# Convenience functions for creating filters
def EQ(key: str, value: Any) -> FieldFilter:
    """Create an equality filter."""
    return FieldFilter(key=key, operator=FilterOperator.EQ, value=value)


def NE(key: str, value: Any) -> FieldFilter:
    """Create a not-equal filter."""
    return FieldFilter(key=key, operator=FilterOperator.NE, value=value)


def GT(key: str, value: Any) -> FieldFilter:
    """Create a greater-than filter."""
    return FieldFilter(key=key, operator=FilterOperator.GT, value=value)


def LT(key: str, value: Any) -> FieldFilter:
    """Create a less-than filter."""
    return FieldFilter(key=key, operator=FilterOperator.LT, value=value)


def GTE(key: str, value: Any) -> FieldFilter:
    """Create a greater-than-or-equal filter."""
    return FieldFilter(key=key, operator=FilterOperator.GTE, value=value)


def LTE(key: str, value: Any) -> FieldFilter:
    """Create a less-than-or-equal filter."""
    return FieldFilter(key=key, operator=FilterOperator.LTE, value=value)


def IN(key: str, value: List[Any]) -> FieldFilter:
    """Create an in-list filter."""
    return FieldFilter(key=key, operator=FilterOperator.IN, value=value)


def CONTAINS(key: str, value: str) -> FieldFilter:
    """Create a string-contains filter."""
    return FieldFilter(key=key, operator=FilterOperator.CONTAINS, value=value)


def AND(*filters: Union[CompositeFilter, FieldFilter]) -> CompositeFilter:
    """Create an AND composite filter."""
    return CompositeFilter(condition=FilterCondition.AND, filters=list(filters))


def OR(*filters: Union[CompositeFilter, FieldFilter]) -> CompositeFilter:
    """Create an OR composite filter."""
    return CompositeFilter(condition=FilterCondition.OR, filters=list(filters))


def NOT(filter: Union[CompositeFilter, FieldFilter]) -> CompositeFilter:
    """Create a NOT composite filter."""
    return CompositeFilter(condition=FilterCondition.NOT, filters=[filter])
