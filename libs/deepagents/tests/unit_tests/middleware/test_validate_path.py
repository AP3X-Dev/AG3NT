"""Unit tests for _validate_path() and _supports_native_paths() functions."""

from unittest.mock import MagicMock

import pytest

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend
from deepagents.middleware.filesystem import _supports_native_paths, _validate_path


class TestValidatePath:
    """Test cases for path validation and normalization."""

    def test_relative_path_normalization(self):
        """Test that relative paths get normalized with leading slash."""
        assert _validate_path("foo/bar") == "/foo/bar"
        assert _validate_path("relative/path.txt") == "/relative/path.txt"

    def test_absolute_path_normalization(self):
        """Test that absolute virtual paths are preserved."""
        assert _validate_path("/workspace/file.txt") == "/workspace/file.txt"
        assert _validate_path("/output/report.csv") == "/output/report.csv"

    def test_path_normalization_removes_redundant_separators(self):
        """Test that redundant path separators are normalized."""
        assert _validate_path("/./foo//bar") == "/foo/bar"
        assert _validate_path("foo/./bar") == "/foo/bar"

    def test_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("foo/../../etc/passwd")

    def test_home_directory_expansion_rejected(self):
        """Test that home directory expansion is rejected."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("~/secret.txt")

    def test_windows_absolute_path_rejected_backslash(self):
        """Test that Windows absolute paths with backslashes are rejected."""
        with pytest.raises(ValueError, match="Windows absolute paths are not supported"):
            _validate_path("C:\\Users\\Documents\\file.txt")

        with pytest.raises(ValueError, match="Windows absolute paths are not supported"):
            _validate_path("F:\\git\\project\\file.txt")

    def test_windows_absolute_path_rejected_forward_slash(self):
        """Test that Windows absolute paths with forward slashes are rejected."""
        with pytest.raises(ValueError, match="Windows absolute paths are not supported"):
            _validate_path("C:/Users/Documents/file.txt")

        with pytest.raises(ValueError, match="Windows absolute paths are not supported"):
            _validate_path("D:/data/output.csv")

    def test_allowed_prefixes_enforcement(self):
        """Test that allowed_prefixes parameter is enforced."""
        # Should pass when prefix matches
        result = _validate_path("/workspace/file.txt", allowed_prefixes=["/workspace/"])
        assert result == "/workspace/file.txt"

        # Should fail when prefix doesn't match
        with pytest.raises(ValueError, match="Path must start with one of"):
            _validate_path("/etc/file.txt", allowed_prefixes=["/workspace/"])

    def test_backslash_normalization(self):
        """Test that backslashes in relative paths are normalized to forward slashes."""
        # Relative paths with backslashes should be normalized
        assert _validate_path("foo\\bar\\baz") == "/foo/bar/baz"


class TestValidatePathNativeAbsolute:
    """Test cases for allow_native_absolute parameter."""

    def test_windows_path_allowed_when_native_absolute_enabled(self):
        """Test that Windows paths are allowed when allow_native_absolute=True."""
        result = _validate_path("C:\\Users\\Documents\\file.txt", allow_native_absolute=True)
        assert result == "C:/Users/Documents/file.txt"

        result = _validate_path("D:/data/output.csv", allow_native_absolute=True)
        assert result == "D:/data/output.csv"

    def test_windows_path_rejected_when_native_absolute_disabled(self):
        """Test that Windows paths are still rejected when allow_native_absolute=False."""
        with pytest.raises(ValueError, match="Windows absolute paths are not supported"):
            _validate_path("C:\\Users\\file.txt", allow_native_absolute=False)

    def test_path_traversal_still_rejected_with_native_absolute(self):
        """Test that path traversal is still rejected even with allow_native_absolute=True."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("C:\\Users\\..\\etc\\passwd", allow_native_absolute=True)

        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("D:/data/../secret", allow_native_absolute=True)

    def test_home_directory_still_rejected_with_native_absolute(self):
        """Test that home directory expansion is still rejected with allow_native_absolute=True."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("~\\secret.txt", allow_native_absolute=True)

    def test_virtual_paths_still_work_with_native_absolute(self):
        """Test that virtual paths still work correctly with allow_native_absolute=True."""
        result = _validate_path("/workspace/file.txt", allow_native_absolute=True)
        assert result == "/workspace/file.txt"

        result = _validate_path("relative/path.txt", allow_native_absolute=True)
        assert result == "/relative/path.txt"

    def test_allowed_prefixes_not_applied_to_native_paths(self):
        """Test that allowed_prefixes are not applied to native absolute paths."""
        # Native paths should bypass prefix checks
        result = _validate_path(
            "C:\\Users\\file.txt",
            allowed_prefixes=["/workspace/"],
            allow_native_absolute=True,
        )
        assert result == "C:/Users/file.txt"

    def test_allowed_prefixes_still_applied_to_virtual_paths(self):
        """Test that allowed_prefixes are still applied to virtual paths."""
        with pytest.raises(ValueError, match="Path must start with one of"):
            _validate_path(
                "/etc/file.txt",
                allowed_prefixes=["/workspace/"],
                allow_native_absolute=True,
            )


class TestSupportsNativePaths:
    """Test cases for _supports_native_paths() helper function."""

    def test_filesystem_backend_default_supports_native(self):
        """Test that FilesystemBackend with virtual_mode=False supports native paths."""
        backend = FilesystemBackend()
        assert _supports_native_paths(backend) is True

    def test_filesystem_backend_virtual_mode_does_not_support_native(self):
        """Test that FilesystemBackend with virtual_mode=True does not support native paths."""
        backend = FilesystemBackend(virtual_mode=True)
        assert _supports_native_paths(backend) is False

    def test_state_backend_does_not_support_native(self):
        """Test that StateBackend does not support native paths."""
        mock_runtime = MagicMock()
        backend = StateBackend(runtime=mock_runtime)
        assert _supports_native_paths(backend) is False

    def test_composite_backend_inherits_from_default(self):
        """Test that CompositeBackend inherits native path support from its default backend."""
        # CompositeBackend with FilesystemBackend default
        fs_backend = FilesystemBackend()
        composite = CompositeBackend(default=fs_backend, routes={})
        assert _supports_native_paths(composite) is True

        # CompositeBackend with StateBackend default
        mock_runtime = MagicMock()
        state_backend = StateBackend(runtime=mock_runtime)
        composite = CompositeBackend(default=state_backend, routes={})
        assert _supports_native_paths(composite) is False

        # CompositeBackend with virtual FilesystemBackend default
        virtual_fs = FilesystemBackend(virtual_mode=True)
        composite = CompositeBackend(default=virtual_fs, routes={})
        assert _supports_native_paths(composite) is False
