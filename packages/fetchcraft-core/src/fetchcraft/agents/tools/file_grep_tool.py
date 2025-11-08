"""
Simplified FileGrepTool for searching file contents with fsspec.
"""

import re
from pathlib import Path
from typing import Optional, Any, List

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


class FileGrepTool:
    """
    Simplified tool for searching file contents using grep-like functionality.
    
    Example:
        tool = FileGrepTool(root_path="/data/documents")
        result = await tool(ctx, pattern="TODO", file_pattern="*.py")
    """

    def __init__(
        self,
        root_path: str = ".",
        fs: Optional[Any] = None,
        name: str = "grep_files",
        max_results: int = 50,
        max_matches_per_file: int = 10,
        context_lines: int = 2,
        allowed_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        max_file_size_mb: float = 10.0
    ):
        """Initialize the FileGrepTool."""
        if not FSSPEC_AVAILABLE:
            raise ImportError("fsspec is required. Install with: pip install fsspec")

        self.root_path = str(root_path)
        self.fs = fs or fsspec.filesystem('dir', path=self.root_path)
        self.name = name
        self.max_results = max_results
        self.max_matches_per_file = max_matches_per_file
        self.context_lines = context_lines
        self.allowed_extensions = allowed_extensions
        self.recursive = recursive
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

        # Build description
        self.description = f"""Search for text patterns within file contents (like grep).

Args:
    pattern: Text or regex pattern to search for
    file_pattern: Optional glob pattern to filter files (default: all files)
    case_sensitive: Whether search is case-sensitive (default: False)
    regex: Whether pattern is a regular expression (default: False)

Returns:
    Files containing matches with line numbers

Root directory: {self.root_path}
Max results: {self.max_results}
Context lines: {self.context_lines}
"""

    def _get_files(self, file_pattern: Optional[str] = None) -> List[str]:
        """Get list of files to search."""
        # Strip leading slash
        if file_pattern and file_pattern.startswith("/"):
            file_pattern = file_pattern[1:]

        # Build search pattern
        if file_pattern:
            pattern = f"**/{file_pattern}" if self.recursive and '**' not in file_pattern else file_pattern
        else:
            pattern = "**/*" if self.recursive else "*"

        # Get matching files
        files = []
        try:
            for path in self.fs.glob(pattern):
                if self.fs.isdir(path):
                    continue
                    
                # Check extension
                if self.allowed_extensions:
                    ext = Path(path).suffix.lower()
                    if ext not in [e.lower() for e in self.allowed_extensions]:
                        continue
                
                # Check size
                try:
                    size = self.fs.info(path).get('size', 0)
                    if size > self.max_file_size_bytes:
                        continue
                except:
                    continue
                    
                files.append(path)
        except:
            pass
            
        return files

    def _search_file(self, file_path: str, pattern: str, case_sensitive: bool, regex: bool) -> Optional[str]:
        """Search for pattern in a single file. Returns formatted result or None."""
        try:
            # Read file
            with self.fs.open(file_path, 'rb') as f:
                content = f.read()

            # Skip binary files (check for null bytes)
            if b'\x00' in content[:8192]:
                return None

            # Decode
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = content.decode('latin-1')
                except:
                    return None

            # Compile pattern
            if regex:
                try:
                    compiled = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
                except re.error:
                    return None
            else:
                compiled = re.compile(re.escape(pattern), 0 if case_sensitive else re.IGNORECASE)

            # Search lines
            lines = text.split('\n')
            matches = []
            
            for i, line in enumerate(lines, 1):
                if compiled.search(line):
                    # Get context
                    context_before = lines[max(0, i-1-self.context_lines):i-1] if self.context_lines > 0 else []
                    context_after = lines[i:min(len(lines), i+self.context_lines)] if self.context_lines > 0 else []
                    
                    matches.append({
                        'line_num': i,
                        'line': line.rstrip('\n\r'),
                        'before': [l.rstrip('\n\r') for l in context_before],
                        'after': [l.rstrip('\n\r') for l in context_after]
                    })
                    
                    if len(matches) >= self.max_matches_per_file:
                        break

            if not matches:
                return None

            # Format result
            normalized_path = file_path[len(self.root_path):].lstrip('/') if file_path.startswith(self.root_path) else file_path
            result = [f"\n📄 {normalized_path} ({len(matches)} matches):"]
            
            for match in matches[:5]:  # Show first 5
                result.append(f"  Line {match['line_num']}: {match['line']}")
                for ctx in match['before']:
                    result.append(f"    {ctx}")
                for ctx in match['after']:
                    result.append(f"    {ctx}")
            
            if len(matches) > 5:
                result.append(f"  ... and {len(matches) - 5} more matches")
                
            return "\n".join(result)
            
        except:
            return None

    async def __call__(
        self,
        ctx,
        pattern: str,
        file_pattern: Optional[str] = None,
        case_sensitive: bool = False,
        regex: bool = False
    ) -> str:
        """Execute the file grep tool."""
        # Get files
        files = self._get_files(file_pattern)
        
        # Search files
        results = []
        for file_path in files:
            result = self._search_file(file_path, pattern, case_sensitive, regex)
            if result:
                results.append(result)
                if len(results) >= self.max_results:
                    break

        if not results:
            return "No matches found."

        # Build output
        total_matches = len(results)
        header = f"Found matches in {total_matches} file(s):\n"
        return header + "\n".join(results)

    def get_tool_function(self):
        """Get the async function to be registered as a tool."""
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError("pydantic-ai is required. Install with: pip install pydantic-ai")

        async def grep_files(
            ctx: RunContext[Any],
            pattern: str,
            file_pattern: Optional[str] = None,
            case_sensitive: bool = False,
            regex: bool = False
        ) -> str:
            """Search for text patterns within files."""
            return await self(ctx, pattern, file_pattern, case_sensitive, regex)

        grep_files.__doc__ = self.description
        grep_files.__name__ = self.name
        return grep_files

    def __repr__(self) -> str:
        return f"FileGrepTool(name={self.name}, root={self.root_path}, max_results={self.max_results})"
