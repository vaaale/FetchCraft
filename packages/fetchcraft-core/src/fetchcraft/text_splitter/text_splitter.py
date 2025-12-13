from abc import ABC, abstractmethod
from typing import List, Tuple, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers used by text splitters."""
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        ...
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back into text."""
        ...


class CharacterTokenizer:
    """Default tokenizer that treats each character as a token."""
    
    def encode(self, text: str) -> List[int]:
        """Encode text - each character is a 'token'."""
        return list(range(len(text)))
    
    def decode(self, tokens: List[int]) -> str:
        """Decode is not meaningful for character tokenizer."""
        raise NotImplementedError("CharacterTokenizer does not support decode")


class TextSplitter(ABC):
    """Interface for text splitting strategies."""
    
    @abstractmethod
    def split(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of tuples (chunk_text, start_idx, end_idx)
        """
        pass
