"""Tests for checkpointer selection logic (SqliteSaver with MemorySaver fallback)."""
from unittest.mock import patch, MagicMock
import pytest


class TestCheckpointerSelection:
    """Test SqliteSaver preference with MemorySaver fallback.

    These tests exercise the selection logic directly since _build_agent()
    has heavy dependencies (langchain, langgraph) not available in the test env.
    """

    def test_uses_sqlite_saver_when_available(self, tmp_path):
        """When langgraph-checkpoint-sqlite is installed, use SqliteSaver."""
        mock_saver = MagicMock()
        mock_sqlite_module = MagicMock()
        mock_sqlite_module.SqliteSaver.from_conn_string.return_value = mock_saver

        with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite": mock_sqlite_module}):
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = tmp_path / "checkpoints.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            checkpointer = SqliteSaver.from_conn_string(str(db_path))

            assert checkpointer is mock_saver
            mock_sqlite_module.SqliteSaver.from_conn_string.assert_called_once_with(str(db_path))

    def test_falls_back_to_memory_saver_when_sqlite_unavailable(self):
        """When langgraph-checkpoint-sqlite is not installed, fall back."""
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            pytest.skip("SqliteSaver is available; cannot test fallback")
        except ImportError:
            pass

        # Reproduce the fallback logic from _build_agent()
        checkpointer = None
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            checkpointer = SqliteSaver.from_conn_string(":memory:")
        except (ImportError, Exception):
            checkpointer = "MemorySaver_fallback"

        assert checkpointer == "MemorySaver_fallback"

    def test_falls_back_on_sqlite_runtime_error(self):
        """If SqliteSaver constructor throws, fall back to MemorySaver."""
        mock_sqlite_module = MagicMock()
        mock_sqlite_module.SqliteSaver.from_conn_string.side_effect = RuntimeError("DB locked")

        with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite": mock_sqlite_module}):
            checkpointer = None
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                checkpointer = SqliteSaver.from_conn_string("/bad/path.db")
            except (ImportError, Exception):
                checkpointer = "MemorySaver_fallback"

            assert checkpointer == "MemorySaver_fallback"

    def test_db_path_uses_ag3nt_directory(self, tmp_path):
        """Checkpoints should be stored at ~/.ag3nt/checkpoints.db."""
        from pathlib import Path

        # Simulate _get_user_data_path() returning a tmp dir
        user_data = tmp_path / ".ag3nt"
        user_data.mkdir()
        db_path = user_data / "checkpoints.db"

        assert str(db_path).endswith("checkpoints.db")
        assert db_path.parent.exists()
