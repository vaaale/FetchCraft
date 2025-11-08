"""
Qdrant filter translator - converts Fetchcraft filters to Qdrant format.
"""
from typing import Union, Dict, Any

from qdrant_client import models

from ..filters import (
    MetadataFilter,
    FieldFilter,
    CompositeFilter,
    FilterOperator,
    FilterCondition
)


class QdrantFilterTranslator:
    """Translates Fetchcraft filters to Qdrant filter format."""
    
    @staticmethod
    def translate(filter_obj: Union[MetadataFilter, Dict[str, Any]]) -> models.Filter:
        """
        Translate a MetadataFilter to Qdrant Filter format.
        
        Args:
            filter_obj: The filter to translate (FieldFilter, CompositeFilter, or raw dict)
            
        Returns:
            Qdrant Filter object
        """
        # If it's already a dict (raw Qdrant filter), assume it's in correct format
        if isinstance(filter_obj, dict):
            # For raw dicts, we can't easily convert to models.Filter
            # Return a simple filter wrapping the conditions
            return models.Filter(**filter_obj)
        
        # Translate FieldFilter
        if isinstance(filter_obj, FieldFilter):
            return QdrantFilterTranslator._translate_field_filter(filter_obj)
        
        # Translate CompositeFilter
        elif isinstance(filter_obj, CompositeFilter):
            return QdrantFilterTranslator._translate_composite_filter(filter_obj)
        
        else:
            raise ValueError(f"Unsupported filter type: {type(filter_obj)}")
    
    @staticmethod
    def _translate_field_filter(field_filter: FieldFilter) -> models.Filter:
        """Translate a FieldFilter to Qdrant format."""
        key = field_filter.key
        operator = field_filter.operator
        value = field_filter.value
        
        # Default condition for most operators
        condition = None
        
        if operator == FilterOperator.EQ:
            condition = models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value)
            )
        elif operator == FilterOperator.NE:
            # NOT equal is a negated match
            return models.Filter(
                must_not=[models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )]
            )
        elif operator == FilterOperator.GT:
            condition = models.FieldCondition(
                key=key,
                range=models.Range(gt=value)
            )
        elif operator == FilterOperator.GTE:
            condition = models.FieldCondition(
                key=key,
                range=models.Range(gte=value)
            )
        elif operator == FilterOperator.LT:
            condition = models.FieldCondition(
                key=key,
                range=models.Range(lt=value)
            )
        elif operator == FilterOperator.LTE:
            condition = models.FieldCondition(
                key=key,
                range=models.Range(lte=value)
            )
        elif operator == FilterOperator.IN:
            condition = models.FieldCondition(
                key=key,
                match=models.MatchAny(any=value)
            )
        elif operator == FilterOperator.CONTAINS:
            # Text contains - use MatchText for substring search
            condition = models.FieldCondition(
                key=key,
                match=models.MatchText(text=value)
            )
        else:
            raise ValueError(f"Unsupported operator: {operator}")
        
        # Wrap in Filter if we have a condition
        if condition:
            return models.Filter(must=[condition])
        
        raise ValueError(f"Failed to translate operator: {operator}")
    
    @staticmethod
    def _translate_composite_filter(composite: CompositeFilter) -> models.Filter:
        """Translate a CompositeFilter to Qdrant format."""
        condition = composite.condition
        filters = composite.filters
        
        if condition == FilterCondition.AND:
            # All filters must match
            must_conditions = []
            must_not_conditions = []
            should_conditions = []
            
            for f in filters:
                translated = QdrantFilterTranslator.translate(f)
                if translated.must:
                    must_conditions.extend(translated.must)
                if translated.must_not:
                    must_not_conditions.extend(translated.must_not)
                if translated.should:
                    should_conditions.extend(translated.should)
            
            filter_kwargs = {}
            if must_conditions:
                filter_kwargs['must'] = must_conditions
            if must_not_conditions:
                filter_kwargs['must_not'] = must_not_conditions
            if should_conditions:
                filter_kwargs['should'] = should_conditions
            
            return models.Filter(**filter_kwargs)
        
        elif condition == FilterCondition.OR:
            # Any filter must match
            should_conditions = []
            
            for f in filters:
                translated = QdrantFilterTranslator.translate(f)
                # Convert must/must_not to nested filters for OR
                if translated.must or translated.must_not or translated.should:
                    # Wrap the entire translated filter as a should condition
                    # Qdrant handles this by wrapping conditions
                    if translated.must:
                        should_conditions.extend(translated.must)
                    if translated.must_not:
                        # For must_not in OR, we need special handling
                        # Create a nested filter
                        should_conditions.append(
                            models.Filter(must_not=translated.must_not)
                        )
            
            return models.Filter(should=should_conditions)
        
        elif condition == FilterCondition.NOT:
            # Negate the filter(s)
            must_not_conditions = []
            
            for f in filters:
                translated = QdrantFilterTranslator.translate(f)
                # Invert: must becomes must_not, must_not becomes must
                if translated.must:
                    must_not_conditions.extend(translated.must)
                if translated.must_not:
                    # Double negative becomes positive (must)
                    # But we're in a NOT context, so keep as must_not
                    pass
            
            if must_not_conditions:
                return models.Filter(must_not=must_not_conditions)
            
            # If no must conditions, just negate everything
            if len(filters) == 1:
                translated = QdrantFilterTranslator.translate(filters[0])
                if translated.must:
                    return models.Filter(must_not=translated.must)
            
            raise ValueError("Complex NOT filter translation not yet supported")
        
        else:
            raise ValueError(f"Unsupported condition: {condition}")
