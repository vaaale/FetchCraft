from typing import List, Tuple, Optional

from fetchcraft.text_splitter.text_splitter import TextSplitter, Tokenizer, CharacterTokenizer


class RecursiveTextSplitter(TextSplitter):
    """
    Text splitter that recursively splits text using multiple separators.
    
    Tries separators in order of preference, falling back to the next separator
    when the current one cannot achieve the target chunk size.
    
    Uses a tokenizer to measure chunk sizes in tokens rather than characters.
    """
    
    DEFAULT_SEPARATORS: List[str] = [
        "\n\n",  # Paragraph
        "\n",    # Line break
        ". ",    # Sentence (period)
        "? ",    # Sentence (question)
        "! ",    # Sentence (exclamation)
        "; ",    # Clause
    ]
    
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        self.tokenizer = tokenizer if tokenizer is not None else CharacterTokenizer()
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator
    
    def _token_count(self, text: str) -> int:
        """Get the token count for a piece of text."""
        return len(self.tokenizer.encode(text))
    
    def split(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks using recursive separator-based splitting.
        
        Args:
            text: The text to split
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of tuples (chunk_text, start_idx, end_idx)
        """
        return self._recursive_split(text, chunk_size, overlap, separator_index=0)
    
    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        separator_index: int = 0
    ) -> List[Tuple[str, int, int]]:
        """
        Recursively split text using multiple separators.
        
        Args:
            text: The text to split
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            separator_index: Current index in the separators list
            
        Returns:
            List of tuples (chunk_text, start_idx, end_idx)
        """
        text_tokens = self._token_count(text)
        
        if text_tokens <= chunk_size:
            return [(text, 0, len(text))]
        
        # If we've exhausted all separators, return text as-is
        if separator_index >= len(self.separators):
            return [(text, 0, len(text))]
        
        separator = self.separators[separator_index]
        splits = self._split_by_separator(text, separator)
        
        # If we can't split by this separator, try the next one only if text is too large
        if len(splits) == 1:
            if text_tokens > chunk_size:
                return self._recursive_split(text, chunk_size, overlap, separator_index + 1)
            else:
                return [(text, 0, len(text))]
        
        # Merge splits into chunks
        chunks = []
        current_chunk = ""
        current_start = 0
        current_chunk_tokens = 0
        
        for split_text, split_start, split_end in splits:
            split_tokens = self._token_count(split_text)
            
            # If adding this split would exceed chunk_size and we have content, save current chunk
            if current_chunk_tokens + split_tokens > chunk_size and current_chunk:
                chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                
                # Start new chunk with overlap (character-based for text slicing)
                overlap_start = max(0, current_start + len(current_chunk) - overlap)
                current_chunk = text[overlap_start:split_end]
                current_start = overlap_start
                current_chunk_tokens = self._token_count(current_chunk)
            else:
                # Add to current chunk
                if not current_chunk:
                    current_start = split_start
                current_chunk = text[current_start:split_end]
                current_chunk_tokens = self._token_count(current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
        
        # Recursively split any chunks that are still too large
        final_chunks = []
        for chunk_text, start_idx, end_idx in chunks:
            if self._token_count(chunk_text) > chunk_size:
                # Recursively split with next separator
                sub_chunks = self._recursive_split(
                    chunk_text,
                    chunk_size,
                    overlap,
                    separator_index + 1
                )
                # Adjust indices relative to original text
                for sub_text, sub_start, sub_end in sub_chunks:
                    final_chunks.append((sub_text, start_idx + sub_start, start_idx + sub_end))
            else:
                final_chunks.append((chunk_text, start_idx, end_idx))
        
        return final_chunks if final_chunks else [(text, 0, len(text))]
    
    def _split_by_separator(
        self,
        text: str,
        separator: str
    ) -> List[Tuple[str, int, int]]:
        """Split text by a separator, keeping track of positions."""
        if separator not in text:
            return [(text, 0, len(text))]
        
        splits = []
        current_pos = 0
        
        for part in text.split(separator):
            if self.keep_separator and splits:
                # Add separator to previous split
                prev_text, prev_start, prev_end = splits[-1]
                splits[-1] = (prev_text + separator, prev_start, prev_end + len(separator))
                current_pos += len(separator)
            
            if part:  # Only add non-empty parts
                end_pos = current_pos + len(part)
                splits.append((part, current_pos, end_pos))
                current_pos = end_pos
        
        return splits if splits else [(text, 0, len(text))]
