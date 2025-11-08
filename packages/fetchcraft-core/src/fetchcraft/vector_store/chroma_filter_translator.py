"""
ChromaDB filter translator - converts Fetchcraft filters to ChromaDB format.
"""
from typing import Union, Dict, Any

from ..filters import (
    MetadataFilter,
    FieldFilter,
    CompositeFilter,
    FilterOperator,
    FilterCondition
)


class ChromaFilterTranslator:
    """Translates Fetchcraft filters to ChromaDB where clause format."""
    
    @staticmethod
    def translate(filter_obj: Union[MetadataFilter, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Translate a MetadataFilter to ChromaDB where clause format.
        
        Args:
            filter_obj: The filter to translate (FieldFilter, CompositeFilter, or raw dict)
            
        Returns:
            ChromaDB where clause dictionary
        """
        # If it's already a dict (raw ChromaDB filter), return as-is
        if isinstance(filter_obj, dict):
            return filter_obj
        
        # Translate FieldFilter
        if isinstance(filter_obj, FieldFilter):
            return ChromaFilterTranslator._translate_field_filter(filter_obj)
        
        # Translate CompositeFilter
        elif isinstance(filter_obj, CompositeFilter):
            return ChromaFilterTranslator._translate_composite_filter(filter_obj)
        
        else:
            raise ValueError(f"Unsupported filter type: {type(filter_obj)}")
    
    @staticmethod
    def _translate_field_filter(field_filter: FieldFilter) -> Dict[str, Any]:
        """Translate a FieldFilter to ChromaDB format."""
        key = field_filter.key
        operator = field_filter.operator
        value = field_filter.value
        
        # Map operator to ChromaDB operator
        if operator == FilterOperator.EQ:
            return {key: {"$eq": value}}
        elif operator == FilterOperator.NE:
            return {key: {"$ne": value}}
        elif operator == FilterOperator.GT:
            return {key: {"$gt": value}}
        elif operator == FilterOperator.GTE:
            return {key: {"$gte": value}}
        elif operator == FilterOperator.LT:
            return {key: {"$lt": value}}
        elif operator == FilterOperator.LTE:
            return {key: {"$lte": value}}
        elif operator == FilterOperator.IN:
            return {key: {"$in": value}}
        elif operator == FilterOperator.CONTAINS:
            # ChromaDB supports $contains for substring search
            return {key: {"$contains": value}}
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    
    @staticmethod
    def _translate_composite_filter(composite: CompositeFilter) -> Dict[str, Any]:
        """Translate a CompositeFilter to ChromaDB format."""
        condition = composite.condition
        filters = composite.filters
        
        if condition == FilterCondition.AND:
            # All filters must match
            translated_filters = [ChromaFilterTranslator.translate(f) for f in filters]
            return {"$and": translated_filters}
        
        elif condition == FilterCondition.OR:
            # Any filter must match
            translated_filters = [ChromaFilterTranslator.translate(f) for f in filters]
            return {"$or": translated_filters}
        
        elif condition == FilterCondition.NOT:
            # ChromaDB doesn't support $not operator directly
            # We need to negate the logic by converting EQ to NE, etc.
            if len(filters) == 1 and isinstance(filters[0], FieldFilter):
                field_filter = filters[0]
                key = field_filter.key
                operator = field_filter.operator
                value = field_filter.value
                
                # Negate the operator
                if operator == FilterOperator.EQ:
                    return {key: {"$ne": value}}
                elif operator == FilterOperator.NE:
                    return {key: {"$eq": value}}
                elif operator == FilterOperator.GT:
                    return {key: {"$lte": value}}
                elif operator == FilterOperator.GTE:
                    return {key: {"$lt": value}}
                elif operator == FilterOperator.LT:
                    return {key: {"$gte": value}}
                elif operator == FilterOperator.LTE:
                    return {key: {"$gt": value}}
                else:
                    raise ValueError(f"Cannot negate operator {operator} in ChromaDB")
            else:
                raise ValueError("ChromaDB NOT filter only supports simple field filters (NOT(EQ(...)), NOT(GT(...)), etc.)")
        
        else:
            raise ValueError(f"Unsupported condition: {condition}")
