from typing import *

from fetchcraft.node import Chunk, Node, DocumentNode
from fetchcraft.node_parser.base import NodeParser


class SimpleNodeParser(NodeParser):
    chunk_size: int = 4096
    overlap: int = 200
    separator: str = " "
    keep_separator: bool = True

    # Recursive separators in order of preference
    separators: List[str] = [
        "\n\n",  # Paragraph
        "\n",  # Line break
        ". ",  # Sentence (period)
        "? ",  # Sentence (question)
        "! ",  # Sentence (exclamation)
        "; ",  # Clause
        ", ",  # Phrase
        " ",  # Word
    ]

    def __init__(self, chunk_size: int = 4096, overlap: int = 0, separators: Optional[List[str]] = None):
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        if separators is not None:
            self.separators = separators

    def get_nodes(self, documents: List[DocumentNode], metadata: Optional[Dict[str, Any]] = None) -> List[Node]:
        if metadata is None:
            metadata = {}
            
        chunk_nodes = []
        for document in documents:
            text = document.text
            chunk_tuples = self._recursive_split(text, )

            for idx, (chunk_text, start_idx, end_idx) in enumerate(chunk_tuples):
                chunk = Chunk.from_text(
                    text=chunk_text,
                    chunk_index=idx,
                    start_char_idx=start_idx,
                    end_char_idx=end_idx,
                    metadata={**metadata, "total_chunks": len(chunk_tuples), "chunk_strategy": "character"}
                )

                # Set doc_id from parent_node (but don't set parent relationship for first-level chunks)
                if document:
                    chunk.doc_id = document.doc_id

                # Link to previous chunk (sibling relationship)
                if chunk_nodes:
                    prev_chunk = chunk_nodes[-1]
                    chunk.previous_id = prev_chunk.id
                    prev_chunk.next_id = chunk.id

                chunk_nodes.append(chunk)

        return chunk_nodes

    def _recursive_split(
        self,
        text: str,
        separator_index: int = 0
    ) -> List[Tuple[str, int, int]]:
        """
        Recursively split text using multiple separators.

        Tries separators in order (paragraph -> line -> sentence -> space)
        and recursively splits chunks that are too large.

        Args:
            text: The text to split
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            separator_index: Current index in the separators list

        Returns:
            List of tuples (chunk_text, start_idx, end_idx)
        """
        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]

        # If we've exhausted all separators, fall back to character splitting
        if separator_index >= len(self.separators):
            return self._character_split(text)

        separator = self.separators[separator_index]
        splits = self._split_by_separator(text, separator)

        # If we can't split by this separator, try the next one
        if len(splits) == 1:
            return self._recursive_split(text, separator_index + 1)

        # Merge splits into chunks
        chunks = []
        current_chunk = ""
        current_start = 0

        for split_text, split_start, split_end in splits:
            # If adding this split would exceed chunk_size and we have content, save current chunk
            if len(current_chunk) + len(split_text) > self.chunk_size and current_chunk:
                chunks.append((current_chunk, current_start, current_start + len(current_chunk)))

                # Start new chunk with overlap
                overlap_start = max(0, current_start + len(current_chunk) - self.overlap)
                current_chunk = text[overlap_start:split_end]
                current_start = overlap_start
            else:
                # Add to current chunk
                if not current_chunk:
                    current_start = split_start
                current_chunk = text[current_start:split_end]

        # Add final chunk
        if current_chunk:
            chunks.append((current_chunk, current_start, current_start + len(current_chunk)))

        # Recursively split any chunks that are still too large
        final_chunks = []
        for chunk_text, start_idx, end_idx in chunks:
            if len(chunk_text) > self.chunk_size:
                # Recursively split with next separator
                sub_chunks = self._recursive_split(
                    chunk_text,
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
        """
        Split text by a separator, keeping track of positions.

        Args:
            text: Text to split
            separator: Separator string

        Returns:
            List of tuples (split_text, start_idx, end_idx)
        """
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

    def _character_split(
        self,
        text: str,
    ) -> List[Tuple[str, int, int]]:
        """
        Fall back to simple character-based splitting.

        Args:
            text: Text to split
            chunk_size: Target chunk size
            overlap: Overlap between chunks

        Returns:
            List of tuples (chunk_text, start_idx, end_idx)
        """
        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append((text[start:end], start, end))
            start = end - self.overlap

            # Ensure we make progress
            if start <= chunks[-1][1]:
                start = end

        return chunks

    def __repr__(self) -> str:
        return f"SimpleNodeParser(chunk_size={self.chunk_size}, overlap={self.overlap})"
