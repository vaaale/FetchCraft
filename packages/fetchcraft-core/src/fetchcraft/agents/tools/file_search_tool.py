"""
FileSearchTool for searching files on a filesystem using fsspec.
"""

import logging
from pathlib import Path
from typing import Optional, Callable, Any, List, Dict

try:
    from pydantic_ai import RunContext

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    RunContext = None

try:
    import fsspec

    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False
    fsspec = None

logger = logging.getLogger(__name__)


class FileSearchResult:
    """
    Result of a file search operation.
    """

    def __init__(
        self,
        path: str,
        size: Optional[int] = None,
        modified: Optional[float] = None,
        file_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a file search result.
        
        Args:
            path: File path
            size: File size in bytes
            modified: Modification timestamp
            file_type: File type/extension
            metadata: Additional metadata
        """
        self.path = path
        self.size = size
        self.modified = modified
        self.file_type = file_type or Path(path).suffix
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"FileSearchResult(path={self.path}, size={self.size}, type={self.file_type})"


class FileSearchTool:
    """
    Tool for searching files on a filesystem using fsspec.
    
    This tool provides controlled filesystem access for agents, allowing them
    to search for files within a specified root directory using glob patterns.
    
    Example:
        ```python
        # Create tool with local filesystem access
        tool = FileSearchTool(
            root_path="/data/documents",
            name="search_files",
            max_results=20
        )
        
        # Register with agent
        agent.register_tool(tool.get_tool_function())
        ```
    """

    def __init__(
        self,
        root_path: str = ".",
        fs: Optional[Any] = None,
        name: str = "find_files",
        description: Optional[str] = None,
        formatter: Optional[Callable] = None,
        max_results: int = 50,
        allowed_extensions: Optional[List[str]] = None,
        recursive: bool = True
    ):
        """
        Initialize the FileSearchTool.
        
        Args:
            root_path: Root directory to search within (default: current directory)
            fs: Optional fsspec filesystem instance (default: local filesystem)
            name: Name of the tool (default: "search_files")
            description: Tool description for the LLM
            formatter: Custom formatter for results (optional)
            max_results: Maximum number of results to return (default: 50)
            allowed_extensions: List of allowed file extensions (e.g., ['.txt', '.md'])
            recursive: Whether to search recursively (default: True)
        """
        if not FSSPEC_AVAILABLE:
            raise ImportError(
                "fsspec is required for FileSearchTool. "
                "Install it with: pip install fsspec"
            )

        self.root_path = str(root_path)
        self.fs = fs or fsspec.filesystem('dir', path=self.root_path)
        self.name = name
        self.formatter = formatter or self._default_formatter
        self.max_results = max_results
        self.allowed_extensions = allowed_extensions
        self.recursive = recursive

        # Default description
        if description is None:
            ext_info = ""
            if allowed_extensions:
                ext_info = f" Allowed extensions: {', '.join(allowed_extensions)}."

            description = f"""Search for files in the filesystem.

Args:
    pattern: Search pattern (supports glob syntax like *.txt, **/*.py, etc.)
    name_only: If True, search only by filename. If False, search full path (default: True)

Returns:
    List of matching files with their paths and metadata

Root directory: {self.root_path}
Recursive search: {self.recursive}
Max results: {self.max_results}{ext_info}

Examples:
    - "*.txt" - Find all .txt files
    - "report*.pdf" - Find PDFs starting with 'report'
    - "**/*.py" - Find all Python files (recursive)
    - "data/*/output.csv" - Find output.csv in subdirectories of data/
"""

        self.description = description

    @classmethod
    def from_local_path(
        cls,
        root_path: str,
        name: str = "search_files",
        description: Optional[str] = None,
        max_results: int = 50,
        allowed_extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> 'FileSearchTool':
        """
        Create a FileSearchTool for local filesystem access.
        
        Args:
            root_path: Root directory to search within
            name: Name of the tool
            description: Tool description for the LLM
            max_results: Maximum number of results
            allowed_extensions: List of allowed file extensions
            recursive: Whether to search recursively
            
        Returns:
            FileSearchTool instance
        """
        return cls(
            root_path=root_path,
            fs=fsspec.filesystem('file'),
            name=name,
            description=description,
            max_results=max_results,
            allowed_extensions=allowed_extensions,
            recursive=recursive
        )

    def _is_allowed_extension(self, path: str) -> bool:
        """
        Check if a file has an allowed extension.
        
        Args:
            path: File path
            
        Returns:
            True if allowed, False otherwise
        """
        if not self.allowed_extensions:
            return True

        ext = Path(path).suffix.lower()
        return ext in [e.lower() for e in self.allowed_extensions]

    def _normalize_path(self, path: str) -> str:
        """
        Normalize a path relative to root.
        
        Args:
            path: File path
            
        Returns:
            Normalized path
        """
        # Remove root_path prefix if present
        if path.startswith(self.root_path):
            path = path[len(self.root_path):].lstrip('/')
        return path

    def _search_files(self, pattern: str, name_only: bool = True) -> List[FileSearchResult]:
        """
        Search for files matching the pattern.
        
        Args:
            pattern: Glob pattern to match
            name_only: If True, match only filename. If False, match full path
            
        Returns:
            List of FileSearchResult objects
        """
        results = []

        if pattern.startswith("/"):
            pattern = pattern[1:]

        # Build search pattern
        if self.recursive and '**' not in pattern:
            # Add recursive search if not explicitly in pattern
            search_pattern = f"**/{pattern}" if name_only else f"**/{pattern}"
        else:
            search_pattern = f"{pattern}"

        logger.info(f"Searching with pattern: {search_pattern}")

        try:
            # Use fsspec glob to find files
            matched_paths = self.fs.glob(search_pattern)

            for path in matched_paths:
                # Skip directories
                if self.fs.isdir(path):
                    continue

                # Check extension filter
                if not self._is_allowed_extension(path):
                    continue

                # Get file info
                try:
                    info = self.fs.info(path)
                    size = info.get('size')
                    modified = info.get('mtime')
                except Exception as e:
                    logger.warning(f"Could not get info for {path}: {e}")
                    size = None
                    modified = None

                # Create result
                result = FileSearchResult(
                    path=self._normalize_path(path),
                    size=size,
                    modified=modified,
                    file_type=Path(path).suffix
                )
                results.append(result)

                # Limit results
                if len(results) >= self.max_results:
                    logger.info(f"Reached max results limit ({self.max_results})")
                    break

        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []

        return results

    def _default_formatter(self, results: List[FileSearchResult]) -> str:
        """
        Default formatter for file search results.
        
        Args:
            results: List of FileSearchResult objects
            
        Returns:
            Formatted string representation
        """
        if not results:
            return "No files found matching the pattern."

        formatted_results = [f"Found {len(results)} file(s):\n"]

        for i, result in enumerate(results, 1):
            size_str = f"{result.size:,} bytes" if result.size is not None else "unknown size"
            formatted_results.append(
                f"{i}. {result.path} ({size_str}, type: {result.file_type})"
            )

        return "\n".join(formatted_results)

    async def __call__(self, ctx, pattern: str, name_only: bool = True) -> str:
        """
        Execute the file search tool.
        
        Args:
            ctx: Context from pydantic-ai (RunContext)
            pattern: Search pattern (glob syntax)
            name_only: If True, search only by filename. If False, search full path
            
        Returns:
            Formatted search results
        """
        logger.info(f"Searching files with pattern: {pattern} (name_only={name_only})")

        # Perform search
        results = self._search_files(pattern, name_only)

        logger.info(f"Found {len(results)} matching files")

        # Format and return results
        return self.formatter(results)

    def get_tool_function(self):
        """
        Get the async function to be registered as a tool.
        
        Returns:
            Async function for tool registration
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "pydantic-ai is required for FileSearchTool. "
                "Install it with: pip install pydantic-ai"
            )

        async def search_files(ctx: RunContext[Any], pattern: str, name_only: bool = True) -> str:
            """Search for files in the filesystem."""
            return await self(ctx, pattern, name_only)

        # Set the docstring from description
        search_files.__doc__ = self.description
        search_files.__name__ = self.name

        return search_files

    def __repr__(self) -> str:
        return f"FileSearchTool(name={self.name}, root={self.root_path}, max_results={self.max_results})"
