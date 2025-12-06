"""
Unit tests for FilesystemConnector and LocalFile
"""

import tempfile
from pathlib import Path

import pytest

from fetchcraft.connector.filesystem import FilesystemConnector, LocalFile


class TestLocalFile:
    """Test suite for LocalFile"""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test file content")
            temp_path = Path(f.name)
        yield temp_path
        temp_path.unlink(missing_ok=True)

    def test_local_file_init(self, temp_file):
        """Test LocalFile initialization"""
        local_file = LocalFile(path=temp_file)
        assert local_file.path == temp_file
        assert local_file.fs is not None

    def test_local_file_name(self, temp_file):
        """Test LocalFile.name() returns correct filename"""
        local_file = LocalFile(path=temp_file)
        assert local_file.name() == temp_file.name

    def test_local_file_permissions(self, temp_file):
        """Test LocalFile.permissions() returns permission roles"""
        local_file = LocalFile(path=temp_file)
        permissions = local_file.permissions()

        assert len(permissions) == 3
        role_names = [p["name"] for p in permissions]
        assert "user" in role_names
        assert "group" in role_names
        assert "other" in role_names

        for perm in permissions:
            assert "read" in perm
            assert "write" in perm
            assert "execute" in perm

    def test_local_file_metadata(self, temp_file):
        """Test LocalFile.metadata() returns file metadata"""
        local_file = LocalFile(path=temp_file)
        metadata = local_file.metadata()

        assert metadata["filename"] == temp_file.name
        assert metadata["source"] == str(temp_file)
        assert "size" in metadata
        assert "modified" in metadata
        assert "created" in metadata
        assert "owner" in metadata
        assert "group" in metadata
        assert "permissions" in metadata

    def test_local_file_mimetype_detection(self, temp_file):
        """Test that mimetype is correctly detected"""
        local_file = LocalFile(path=temp_file)
        assert local_file.mimetype == "text/plain"


class TestFilesystemConnector:
    """Test suite for FilesystemConnector"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            (tmp_path / "file1.txt").write_text("Content of file 1")
            (tmp_path / "file2.txt").write_text("Content of file 2")
            (tmp_path / "data.json").write_text('{"key": "value"}')

            subdir = tmp_path / "subdir"
            subdir.mkdir()
            (subdir / "file3.txt").write_text("Content of file 3 in subdirectory")

            nested_dir = subdir / "nested"
            nested_dir.mkdir()
            (nested_dir / "file4.txt").write_text("Content of file 4 in nested directory")

            yield tmp_path

    def test_filesystem_connector_init(self, temp_dir):
        """Test FilesystemConnector initialization"""
        connector = FilesystemConnector(path=temp_dir)
        assert connector.path == temp_dir
        assert connector.fs is not None
        assert connector.filter is None

    def test_filesystem_connector_get_name(self, temp_dir):
        """Test FilesystemConnector.get_name()"""
        connector = FilesystemConnector(path=temp_dir)
        assert connector.get_name() == "FilesystemConnector"

    @pytest.mark.asyncio
    async def test_filesystem_connector_list_directories(self, temp_dir):
        """Test FilesystemConnector.list_directories()"""
        connector = FilesystemConnector(path=temp_dir)
        directories = await connector.list_directories()

        assert len(directories) == 2
        dir_names = [Path(d).name for d in directories]
        assert "subdir" in dir_names
        assert "nested" in dir_names

    @pytest.mark.asyncio
    async def test_filesystem_connector_glob_all_files(self, temp_dir):
        """Test FilesystemConnector.glob() returns all files"""
        connector = FilesystemConnector(path=temp_dir)

        files = []
        async for file in connector.glob():
            files.append(file)

        assert len(files) == 5
        filenames = [f.name() for f in files]
        assert "file1.txt" in filenames
        assert "file2.txt" in filenames
        assert "data.json" in filenames
        assert "file3.txt" in filenames
        assert "file4.txt" in filenames

    @pytest.mark.asyncio
    async def test_filesystem_connector_glob_with_filter(self, temp_dir):
        """Test FilesystemConnector.glob() with filter function"""

        def txt_filter(file: LocalFile) -> bool:
            return file.path.suffix == ".txt"

        connector = FilesystemConnector(path=temp_dir, filter=txt_filter)

        files = []
        async for file in connector.glob():
            files.append(file)

        assert len(files) == 4
        for f in files:
            assert f.path.suffix == ".txt"

    @pytest.mark.asyncio
    async def test_filesystem_connector_glob_excludes_directories(self, temp_dir):
        """Test that glob() does not yield directories"""
        connector = FilesystemConnector(path=temp_dir)

        async for file in connector.glob():
            assert not file.path.is_dir()

    @pytest.mark.asyncio
    async def test_filesystem_connector_glob_returns_local_files(self, temp_dir):
        """Test that glob() yields LocalFile instances"""
        connector = FilesystemConnector(path=temp_dir)

        async for file in connector.glob():
            assert isinstance(file, LocalFile)

    @pytest.mark.asyncio
    async def test_filesystem_connector_empty_directory(self):
        """Test FilesystemConnector with empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            connector = FilesystemConnector(path=Path(tmpdir))

            files = []
            async for file in connector.glob():
                files.append(file)

            assert len(files) == 0

            directories = await connector.list_directories()
            assert len(directories) == 0

    @pytest.mark.asyncio
    async def test_filesystem_connector_filter_excludes_all(self, temp_dir):
        """Test filter that excludes all files"""

        def exclude_all(file: LocalFile) -> bool:
            return False

        connector = FilesystemConnector(path=temp_dir, filter=exclude_all)

        files = []
        async for file in connector.glob():
            files.append(file)

        assert len(files) == 0

    def test_filesystem_connector_with_custom_fs(self, temp_dir):
        """Test FilesystemConnector with custom filesystem"""
        import fsspec

        custom_fs = fsspec.filesystem("dir", path=temp_dir)
        connector = FilesystemConnector(path=temp_dir, fs=custom_fs)

        assert connector.fs is custom_fs
