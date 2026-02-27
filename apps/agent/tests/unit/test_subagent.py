"""Unit tests for subagent configuration and monitoring.

Tests for:
- subagent_configs.py: SubagentConfig, registry, resource limits, resource manager
- subagent_monitor.py: SubagentExecution, SubagentMonitor

Updated to include 8 subagent types with enhanced token/turn limits
and ThinkingMode support to match/exceed Moltbot capabilities.

Also includes tests for:
- ContextPruningConfig: Context pruning for long-running sessions
- SubagentEventType: Lifecycle events for subagent execution
- Persistence: Save/load subagent runs to disk
"""

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from ag3nt_agent.subagent_configs import (
    ANALYST,
    BROWSER,
    CODER,
    CONTEXT_PRUNING_AGGRESSIVE,
    CONTEXT_PRUNING_OFF,
    CONTEXT_PRUNING_STANDARD,
    ContextPruningConfig,
    ContextPruningMode,
    MEMORY,
    PLANNER,
    RESEARCHER,
    REVIEWER,
    WRITER,
    SUBAGENT_REGISTRY,
    SubagentConfig,
    SubagentResourceLimits,
    SubagentResourceManager,
    ThinkingMode,
    get_subagent_config,
    list_subagent_types,
)
from ag3nt_agent.subagent_monitor import (
    SubagentEvent,
    SubagentEventType,
    SubagentExecution,
    SubagentMonitor,
)


# =============================================================================
# SubagentConfig Tests
# =============================================================================


class TestSubagentConfig:
    """Tests for SubagentConfig dataclass."""

    def test_subagent_config_creation(self):
        """Test creating a SubagentConfig."""
        config = SubagentConfig(
            name="test_agent",
            description="A test agent",
            system_prompt="You are a test agent.",
            tools=["tool1", "tool2"],
            max_tokens=2048,
            max_turns=5,
        )
        assert config.name == "test_agent"
        assert config.description == "A test agent"
        assert config.system_prompt == "You are a test agent."
        assert config.tools == ["tool1", "tool2"]
        assert config.max_tokens == 2048
        assert config.max_turns == 5

    def test_subagent_config_defaults(self):
        """Test SubagentConfig default values."""
        config = SubagentConfig(
            name="minimal",
            description="Minimal config",
            system_prompt="Minimal prompt",
        )
        assert config.tools == []
        assert config.max_tokens == 4096
        assert config.max_turns == 10
        # New fields added to match Moltbot capabilities
        assert config.model_override is None
        assert config.thinking_mode == ThinkingMode.MEDIUM
        assert config.allow_sandbox is True
        assert config.priority == 5
        # Context pruning defaults to OFF
        assert config.context_pruning.mode == ContextPruningMode.OFF


class TestPredefinedConfigs:
    """Tests for predefined subagent configurations.

    Note: Token and turn limits were tripled and doubled respectively
    to match/exceed Moltbot capabilities.
    """

    def test_researcher_config(self):
        """Test RESEARCHER configuration."""
        assert RESEARCHER.name == "researcher"
        assert "internet_search" in RESEARCHER.tools
        assert "fetch_url" in RESEARCHER.tools
        assert RESEARCHER.max_tokens == 12288  # 3x original (4096 * 3)
        assert RESEARCHER.max_turns == 20  # 2x original (10 * 2)

    def test_coder_config(self):
        """Test CODER configuration."""
        assert CODER.name == "coder"
        assert "read_file" in CODER.tools
        assert "write_file" in CODER.tools
        assert "shell" in CODER.tools
        assert CODER.max_tokens == 24576  # 3x original (8192 * 3)
        assert CODER.max_turns == 30  # 2x original (15 * 2)

    def test_reviewer_config(self):
        """Test REVIEWER configuration."""
        assert REVIEWER.name == "reviewer"
        assert "read_file" in REVIEWER.tools
        assert "git_diff" in REVIEWER.tools
        assert REVIEWER.max_tokens == 12288  # 3x original (4096 * 3)
        assert REVIEWER.max_turns == 20  # 2x original (10 * 2)

    def test_planner_config(self):
        """Test PLANNER configuration."""
        assert PLANNER.name == "planner"
        assert "write_todos" in PLANNER.tools
        assert "read_todos" in PLANNER.tools
        assert PLANNER.max_tokens == 6144  # 3x original (2048 * 3)
        assert PLANNER.max_turns == 16  # 2x original (8 * 2)

    def test_browser_config(self):
        """Test BROWSER configuration (new specialist subagent)."""
        assert BROWSER.name == "browser"
        assert "browser_navigate" in BROWSER.tools
        assert "browser_click" in BROWSER.tools
        assert BROWSER.max_tokens == 8192
        assert BROWSER.max_turns == 20
        assert BROWSER.thinking_mode == ThinkingMode.LOW
        assert BROWSER.priority == 6

    def test_analyst_config(self):
        """Test ANALYST configuration (new specialist subagent)."""
        assert ANALYST.name == "analyst"
        assert "read_file" in ANALYST.tools
        assert "shell" in ANALYST.tools
        assert ANALYST.max_tokens == 16384
        assert ANALYST.max_turns == 25
        assert ANALYST.thinking_mode == ThinkingMode.HIGH
        assert ANALYST.priority == 7

    def test_writer_config(self):
        """Test WRITER configuration (new specialist subagent)."""
        assert WRITER.name == "writer"
        assert "read_file" in WRITER.tools
        assert "write_file" in WRITER.tools
        assert "internet_search" in WRITER.tools
        assert WRITER.max_tokens == 16384
        assert WRITER.max_turns == 20
        assert WRITER.thinking_mode == ThinkingMode.MEDIUM
        assert WRITER.priority == 6

    def test_memory_config(self):
        """Test MEMORY configuration (new specialist subagent)."""
        assert MEMORY.name == "memory"
        assert "memory_search" in MEMORY.tools
        assert "memory_store" in MEMORY.tools
        assert MEMORY.max_tokens == 8192
        assert MEMORY.max_turns == 15
        assert MEMORY.thinking_mode == ThinkingMode.MEDIUM
        assert MEMORY.priority == 5


class TestThinkingMode:
    """Tests for ThinkingMode enum."""

    def test_thinking_mode_values(self):
        """Test all ThinkingMode values."""
        assert ThinkingMode.OFF.value == "off"
        assert ThinkingMode.MINIMAL.value == "minimal"
        assert ThinkingMode.LOW.value == "low"
        assert ThinkingMode.MEDIUM.value == "medium"
        assert ThinkingMode.HIGH.value == "high"
        assert ThinkingMode.XHIGH.value == "xhigh"

    def test_thinking_mode_is_string_enum(self):
        """Test that ThinkingMode is a string enum."""
        assert str(ThinkingMode.MEDIUM) == "ThinkingMode.MEDIUM"
        assert ThinkingMode.HIGH == "high"

    def test_subagent_thinking_modes(self):
        """Test thinking modes assigned to subagents."""
        assert RESEARCHER.thinking_mode == ThinkingMode.MEDIUM
        assert CODER.thinking_mode == ThinkingMode.HIGH
        assert REVIEWER.thinking_mode == ThinkingMode.HIGH
        assert PLANNER.thinking_mode == ThinkingMode.HIGH

    def test_subagent_priorities(self):
        """Test priorities assigned to subagents."""
        assert RESEARCHER.priority == 7
        assert CODER.priority == 9
        assert REVIEWER.priority == 8
        assert PLANNER.priority == 8


class TestSubagentRegistry:
    """Tests for subagent registry functions."""

    def test_registry_contains_all_types(self):
        """Test that registry contains all predefined types (8 subagents)."""
        # Original 4 subagents
        assert "researcher" in SUBAGENT_REGISTRY
        assert "coder" in SUBAGENT_REGISTRY
        assert "reviewer" in SUBAGENT_REGISTRY
        assert "planner" in SUBAGENT_REGISTRY
        # New 4 specialist subagents
        assert "browser" in SUBAGENT_REGISTRY
        assert "analyst" in SUBAGENT_REGISTRY
        assert "writer" in SUBAGENT_REGISTRY
        assert "memory" in SUBAGENT_REGISTRY

    def test_get_subagent_config_valid(self):
        """Test getting a valid subagent config."""
        config = get_subagent_config("researcher")
        assert config.name == "researcher"
        assert config is RESEARCHER

    def test_get_subagent_config_invalid(self):
        """Test getting an invalid subagent config."""
        with pytest.raises(ValueError) as exc_info:
            get_subagent_config("unknown_agent")
        assert "Unknown subagent: unknown_agent" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_list_subagent_types(self):
        """Test listing all subagent types (8 total)."""
        types = list_subagent_types()
        expected = {
            "researcher", "coder", "reviewer", "planner",
            "browser", "analyst", "writer", "memory",
        }
        assert set(types) == expected


# =============================================================================
# Dynamic SubagentRegistry Tests
# =============================================================================


class TestDynamicSubagentRegistry:
    """Tests for the dynamic SubagentRegistry singleton class."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry singleton before each test."""
        from ag3nt_agent.subagent_registry import SubagentRegistry
        SubagentRegistry.reset_instance()
        yield
        SubagentRegistry.reset_instance()

    def test_singleton_pattern(self):
        """Test that SubagentRegistry is a singleton."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry1 = SubagentRegistry.get_instance()
        registry2 = SubagentRegistry.get_instance()
        registry3 = SubagentRegistry()

        assert registry1 is registry2
        assert registry2 is registry3

    def test_builtin_subagents_loaded(self):
        """Test that builtin subagents are loaded on initialization."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        # All 8 builtin subagents should be loaded
        builtins = registry.list_by_source("builtin")
        assert len(builtins) == 8

        builtin_names = {c.name.lower() for c in builtins}
        assert builtin_names == {
            "researcher", "coder", "reviewer", "planner",
            "browser", "analyst", "writer", "memory",
        }

    def test_register_user_subagent(self):
        """Test registering a user-defined subagent."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        custom = SubagentConfig(
            name="DEBUGGER",
            description="Debugging specialist",
            system_prompt="You are a debugger.",
            tools=["execute", "read_file"],
            max_tokens=8000,
        )

        result = registry.register(custom, source="user")
        assert result is True

        # Should be retrievable
        retrieved = registry.get("debugger")
        assert retrieved is not None
        assert retrieved.name == "DEBUGGER"
        assert retrieved.description == "Debugging specialist"

        # Source should be "user"
        source = registry.get_source("debugger")
        assert source == "user"

    def test_register_duplicate_fails_without_overwrite(self):
        """Test that registering a duplicate name fails without overwrite."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        custom1 = SubagentConfig(
            name="CUSTOM",
            description="First version",
            system_prompt="First prompt",
        )
        custom2 = SubagentConfig(
            name="CUSTOM",
            description="Second version",
            system_prompt="Second prompt",
        )

        assert registry.register(custom1, source="user") is True
        assert registry.register(custom2, source="user") is False

        # Should still have the first version
        retrieved = registry.get("custom")
        assert retrieved.description == "First version"

    def test_register_with_overwrite(self):
        """Test that overwrite=True allows replacement."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        custom1 = SubagentConfig(
            name="CUSTOM",
            description="First version",
            system_prompt="First prompt",
        )
        custom2 = SubagentConfig(
            name="CUSTOM",
            description="Second version",
            system_prompt="Second prompt",
        )

        assert registry.register(custom1, source="user") is True
        assert registry.register(custom2, source="user", overwrite=True) is True

        # Should have the second version
        retrieved = registry.get("custom")
        assert retrieved.description == "Second version"

    def test_cannot_overwrite_builtin(self):
        """Test that builtin subagents cannot be overwritten."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        fake_researcher = SubagentConfig(
            name="researcher",
            description="Fake researcher",
            system_prompt="Fake prompt",
        )

        result = registry.register(fake_researcher, source="user", overwrite=True)
        assert result is False

        # Original should still be there
        retrieved = registry.get("researcher")
        assert "Fake" not in retrieved.description

    def test_unregister_user_subagent(self):
        """Test unregistering a user-defined subagent."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        custom = SubagentConfig(
            name="TEMPORARY",
            description="Temporary subagent",
            system_prompt="Temp prompt",
        )

        registry.register(custom, source="user")
        assert registry.get("temporary") is not None

        result = registry.unregister("temporary")
        assert result is True
        assert registry.get("temporary") is None

    def test_cannot_unregister_builtin(self):
        """Test that builtin subagents cannot be unregistered."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        result = registry.unregister("researcher")
        assert result is False

        # Should still exist
        assert registry.get("researcher") is not None

    def test_unregister_nonexistent(self):
        """Test unregistering a non-existent subagent."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_with_source(self):
        """Test get_with_source returns tuple of config and source."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        result = registry.get_with_source("researcher")
        assert result is not None
        config, source = result
        assert config.name.lower() == "researcher"
        assert source == "builtin"

    def test_get_nonexistent(self):
        """Test get returns None for nonexistent subagent."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        assert registry.get("nonexistent") is None
        assert registry.get_with_source("nonexistent") is None
        assert registry.get_source("nonexistent") is None

    def test_list_all(self):
        """Test listing all registered subagents."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        # Add a user subagent
        custom = SubagentConfig(
            name="CUSTOM",
            description="Custom",
            system_prompt="Custom prompt",
        )
        registry.register(custom, source="user")

        all_subagents = registry.list_all()
        assert len(all_subagents) == 9  # 8 builtin + 1 user

    def test_list_by_source(self):
        """Test listing subagents by source."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        # Add user subagents
        for i in range(3):
            custom = SubagentConfig(
                name=f"USER_{i}",
                description=f"User {i}",
                system_prompt=f"Prompt {i}",
            )
            registry.register(custom, source="user")

        # Add a plugin subagent
        plugin = SubagentConfig(
            name="PLUGIN_ONE",
            description="Plugin one",
            system_prompt="Plugin prompt",
        )
        registry.register(plugin, source="plugin")

        builtins = registry.list_by_source("builtin")
        users = registry.list_by_source("user")
        plugins = registry.list_by_source("plugin")

        assert len(builtins) == 8
        assert len(users) == 3
        assert len(plugins) == 1

    def test_list_names(self):
        """Test listing all subagent names."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        names = registry.list_names()
        assert "researcher" in names
        assert "coder" in names
        assert len(names) == 8  # Only builtins initially

    def test_on_change_callback(self):
        """Test that on_change callbacks are invoked."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        events = []

        def callback(event_type, name, source):
            events.append((event_type, name, source))

        registry.on_change(callback)

        custom = SubagentConfig(
            name="CALLBACK_TEST",
            description="Test",
            system_prompt="Test",
        )

        registry.register(custom, source="user")
        assert len(events) == 1
        assert events[0] == ("register", "callback_test", "user")

        registry.unregister("callback_test")
        assert len(events) == 2
        assert events[1] == ("unregister", "callback_test", "user")

    def test_remove_callback(self):
        """Test removing a callback."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        events = []

        def callback(event_type, name, source):
            events.append((event_type, name, source))

        registry.on_change(callback)

        custom1 = SubagentConfig(
            name="TEST1",
            description="Test 1",
            system_prompt="Test",
        )
        registry.register(custom1, source="user")
        assert len(events) == 1

        # Remove callback
        result = registry.remove_callback(callback)
        assert result is True

        custom2 = SubagentConfig(
            name="TEST2",
            description="Test 2",
            system_prompt="Test",
        )
        registry.register(custom2, source="user")
        # Should still be 1 since callback was removed
        assert len(events) == 1

    def test_callback_error_handling(self):
        """Test that callback errors don't break registry operations."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        good_events = []

        def bad_callback(event_type, name, source):
            raise RuntimeError("Callback error")

        def good_callback(event_type, name, source):
            good_events.append((event_type, name, source))

        registry.on_change(bad_callback)
        registry.on_change(good_callback)

        custom = SubagentConfig(
            name="ERROR_TEST",
            description="Test",
            system_prompt="Test",
        )

        # Should not raise despite bad_callback error
        registry.register(custom, source="user")
        assert len(good_events) == 1

    def test_load_from_yaml_file(self):
        """Test loading subagents from YAML file."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "subagents.yaml"
            yaml_content = """
subagents:
  - name: YAML_TEST
    description: Loaded from YAML
    system_prompt: YAML prompt
    tools:
      - read_file
    max_tokens: 4096
    max_turns: 5
"""
            yaml_path.write_text(yaml_content)

            loaded = registry.load_from_file(yaml_path, source="user")
            assert loaded == 1

            retrieved = registry.get("yaml_test")
            assert retrieved is not None
            assert retrieved.description == "Loaded from YAML"
            assert "read_file" in retrieved.tools

    def test_load_from_json_file(self):
        """Test loading subagents from JSON file."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "subagents.json"
            json_content = {
                "subagents": [
                    {
                        "name": "JSON_TEST",
                        "description": "Loaded from JSON",
                        "system_prompt": "JSON prompt",
                        "tools": ["write_file"],
                        "max_tokens": 4096,
                        "max_turns": 5,
                    }
                ]
            }
            json_path.write_text(json.dumps(json_content))

            loaded = registry.load_from_file(json_path, source="plugin")
            assert loaded == 1

            retrieved = registry.get("json_test")
            assert retrieved is not None
            assert retrieved.description == "Loaded from JSON"
            assert registry.get_source("json_test") == "plugin"

    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file returns 0."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        loaded = registry.load_from_file(Path("/nonexistent/path.yaml"))
        assert loaded == 0

    def test_save_to_file(self):
        """Test saving subagents to file."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        # Add user subagents
        for i in range(2):
            custom = SubagentConfig(
                name=f"SAVE_TEST_{i}",
                description=f"Save test {i}",
                system_prompt=f"Prompt {i}",
            )
            registry.register(custom, source="user")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "saved.yaml"

            saved = registry.save_to_file(save_path, source="user")
            assert saved == 2
            assert save_path.exists()

            # Verify content
            content = yaml.safe_load(save_path.read_text())
            assert len(content["subagents"]) == 2

    def test_save_single_config(self):
        """Test saving a single subagent config."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        custom = SubagentConfig(
            name="SINGLE_SAVE",
            description="Single save test",
            system_prompt="Single prompt",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            user_data_path = Path(tmpdir)

            result = registry.save_single_config(custom, user_data_path)
            assert result is True

            # Check file was created
            expected_path = user_data_path / "subagents" / "single_save.yaml"
            assert expected_path.exists()

    def test_load_user_configs(self):
        """Test loading user configs from ~/.ag3nt/subagents/."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        with tempfile.TemporaryDirectory() as tmpdir:
            user_data_path = Path(tmpdir)
            subagents_dir = user_data_path / "subagents"
            subagents_dir.mkdir()

            # Create a YAML file
            yaml_content = """
subagents:
  - name: USER_LOAD_1
    description: User load 1
    system_prompt: Prompt 1
"""
            (subagents_dir / "user1.yaml").write_text(yaml_content)

            # Create a JSON file
            json_content = {"subagents": [
                {"name": "USER_LOAD_2", "description": "User load 2", "system_prompt": "Prompt 2"}
            ]}
            (subagents_dir / "user2.json").write_text(json.dumps(json_content))

            loaded = registry.load_user_configs(user_data_path)
            assert loaded == 2

            assert registry.get("user_load_1") is not None
            assert registry.get("user_load_2") is not None

    def test_load_user_configs_creates_directory(self):
        """Test that load_user_configs creates directory if missing."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        with tempfile.TemporaryDirectory() as tmpdir:
            user_data_path = Path(tmpdir) / "newdir"

            loaded = registry.load_user_configs(user_data_path)
            assert loaded == 0

            # Directory should be created
            assert (user_data_path / "subagents").exists()

    def test_to_dict(self):
        """Test exporting registry as dictionary."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        custom = SubagentConfig(
            name="DICT_TEST",
            description="Dict test",
            system_prompt="Dict prompt",
        )
        registry.register(custom, source="user")

        result = registry.to_dict()

        assert "researcher" in result
        assert result["researcher"]["source"] == "builtin"

        assert "dict_test" in result
        assert result["dict_test"]["source"] == "user"
        assert result["dict_test"]["config"]["name"] == "DICT_TEST"

    def test_case_insensitive_lookup(self):
        """Test that subagent lookup is case-insensitive."""
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        custom = SubagentConfig(
            name="MiXeD_CaSe",
            description="Mixed case test",
            system_prompt="Mixed prompt",
        )
        registry.register(custom, source="user")

        # All these should work
        assert registry.get("mixed_case") is not None
        assert registry.get("MIXED_CASE") is not None
        assert registry.get("MiXeD_CaSe") is not None

    def test_thread_safety(self):
        """Test that registry operations are thread-safe."""
        import concurrent.futures
        from ag3nt_agent.subagent_registry import SubagentRegistry

        registry = SubagentRegistry.get_instance()

        results = []

        def register_subagent(index):
            config = SubagentConfig(
                name=f"THREAD_{index}",
                description=f"Thread {index}",
                system_prompt=f"Prompt {index}",
            )
            success = registry.register(config, source="user")
            results.append((index, success))
            return success

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_subagent, i) for i in range(20)]
            concurrent.futures.wait(futures)

        # All registrations should succeed (unique names)
        assert len([r for r in results if r[1]]) == 20

        # All should be retrievable
        for i in range(20):
            assert registry.get(f"thread_{i}") is not None


class TestBackwardCompatibilityFunctions:
    """Tests for backward compatibility functions in subagent_registry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry singleton before each test."""
        from ag3nt_agent.subagent_registry import SubagentRegistry
        SubagentRegistry.reset_instance()
        yield
        SubagentRegistry.reset_instance()

    def test_get_subagent_config_function(self):
        """Test the get_subagent_config convenience function."""
        from ag3nt_agent.subagent_registry import get_subagent_config

        config = get_subagent_config("researcher")
        assert config.name.lower() == "researcher"

    def test_get_subagent_config_raises_on_unknown(self):
        """Test that get_subagent_config raises ValueError for unknown types."""
        from ag3nt_agent.subagent_registry import get_subagent_config

        with pytest.raises(ValueError) as exc_info:
            get_subagent_config("unknown_type")

        assert "Unknown subagent: unknown_type" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_list_subagent_types_function(self):
        """Test the list_subagent_types convenience function."""
        from ag3nt_agent.subagent_registry import list_subagent_types

        types = list_subagent_types()
        assert "researcher" in types
        assert "coder" in types
        assert len(types) == 8


# =============================================================================
# SubagentResourceLimits Tests
# =============================================================================


class TestSubagentResourceLimits:
    """Tests for SubagentResourceLimits dataclass."""

    def test_default_limits(self):
        """Test default resource limits."""
        limits = SubagentResourceLimits()
        assert limits.max_execution_time_seconds == 120.0
        assert limits.max_turns == 10
        assert limits.max_tokens == 8192
        assert limits.max_tool_calls == 20
        assert limits.max_concurrent_subagents == 3
        assert limits.max_subagent_depth == 2

    def test_custom_limits(self):
        """Test custom resource limits."""
        limits = SubagentResourceLimits(
            max_execution_time_seconds=60.0,
            max_turns=5,
            max_tokens=4096,
            max_tool_calls=10,
            max_concurrent_subagents=2,
            max_subagent_depth=1,
        )
        assert limits.max_execution_time_seconds == 60.0
        assert limits.max_turns == 5
        assert limits.max_tokens == 4096
        assert limits.max_tool_calls == 10
        assert limits.max_concurrent_subagents == 2
        assert limits.max_subagent_depth == 1


# =============================================================================
# SubagentResourceManager Tests
# =============================================================================


class TestSubagentResourceManager:
    """Tests for SubagentResourceManager class."""

    def test_manager_initialization(self):
        """Test resource manager initialization."""
        manager = SubagentResourceManager()
        assert manager.active_count == 0
        assert manager.get_active_count() == 0
        assert manager.get_active_ids() == set()

    def test_manager_custom_limits(self):
        """Test manager with custom limits."""
        limits = SubagentResourceLimits(max_concurrent_subagents=5)
        manager = SubagentResourceManager(limits)
        assert manager.limits.max_concurrent_subagents == 5

    def test_can_spawn_when_empty(self):
        """Test can_spawn when no subagents are active."""
        manager = SubagentResourceManager()
        can_spawn, reason = manager.can_spawn()
        assert can_spawn is True
        assert reason is None

    def test_can_spawn_at_limit(self):
        """Test can_spawn when at concurrent limit."""
        limits = SubagentResourceLimits(max_concurrent_subagents=2)
        manager = SubagentResourceManager(limits)
        manager.acquire("exec1")
        manager.acquire("exec2")
        can_spawn, reason = manager.can_spawn()
        assert can_spawn is False
        assert "Max concurrent subagents reached" in reason

    def test_acquire_and_release(self):
        """Test acquiring and releasing subagent slots."""
        manager = SubagentResourceManager()
        assert manager.acquire("exec1") is True
        assert manager.get_active_count() == 1
        assert "exec1" in manager.get_active_ids()

        manager.release("exec1")
        assert manager.get_active_count() == 0
        assert "exec1" not in manager.get_active_ids()

    def test_acquire_fails_at_limit(self):
        """Test acquire fails when at limit."""
        limits = SubagentResourceLimits(max_concurrent_subagents=1)
        manager = SubagentResourceManager(limits)
        assert manager.acquire("exec1") is True
        assert manager.acquire("exec2") is False
        assert manager.get_active_count() == 1

    def test_release_unknown_id(self):
        """Test releasing an unknown execution ID."""
        manager = SubagentResourceManager()
        manager.release("unknown")  # Should not raise
        assert manager.get_active_count() == 0

    def test_check_limits_within_bounds(self):
        """Test check_limits when within bounds."""
        manager = SubagentResourceManager()
        within, reason = manager.check_limits(
            execution_time=30.0,
            turns=5,
            tokens=2000,
            tool_calls=10,
        )
        assert within is True
        assert reason is None

    def test_check_limits_time_exceeded(self):
        """Test check_limits when execution time exceeded."""
        manager = SubagentResourceManager()
        within, reason = manager.check_limits(
            execution_time=150.0,
            turns=5,
            tokens=2000,
            tool_calls=10,
        )
        assert within is False
        assert "Max execution time exceeded" in reason

    def test_check_limits_turns_exceeded(self):
        """Test check_limits when turns exceeded."""
        manager = SubagentResourceManager()
        within, reason = manager.check_limits(
            execution_time=30.0,
            turns=15,
            tokens=2000,
            tool_calls=10,
        )
        assert within is False
        assert "Max turns exceeded" in reason

    def test_check_limits_tokens_exceeded(self):
        """Test check_limits when tokens exceeded."""
        manager = SubagentResourceManager()
        within, reason = manager.check_limits(
            execution_time=30.0,
            turns=5,
            tokens=10000,
            tool_calls=10,
        )
        assert within is False
        assert "Max tokens exceeded" in reason

    def test_check_limits_tool_calls_exceeded(self):
        """Test check_limits when tool calls exceeded."""
        manager = SubagentResourceManager()
        within, reason = manager.check_limits(
            execution_time=30.0,
            turns=5,
            tokens=2000,
            tool_calls=25,
        )
        assert within is False
        assert "Max tool calls exceeded" in reason


# =============================================================================
# SubagentExecution Tests
# =============================================================================


class TestSubagentExecution:
    """Tests for SubagentExecution dataclass."""

    def test_execution_creation(self):
        """Test creating a SubagentExecution."""
        now = datetime.now()
        execution = SubagentExecution(
            id="exec1",
            parent_id="parent1",
            subagent_type="researcher",
            task="Find information",
            started_at=now,
        )
        assert execution.id == "exec1"
        assert execution.parent_id == "parent1"
        assert execution.subagent_type == "researcher"
        assert execution.task == "Find information"
        assert execution.started_at == now
        assert execution.ended_at is None
        assert execution.turns == 0
        assert execution.tool_calls == []
        assert execution.result is None
        assert execution.error is None
        assert execution.tokens_used == 0

    def test_execution_properties_incomplete(self):
        """Test execution properties when incomplete."""
        execution = SubagentExecution(
            id="exec1",
            parent_id="parent1",
            subagent_type="coder",
            task="Write code",
            started_at=datetime.now(),
        )
        assert execution.is_complete is False
        assert execution.is_success is False
        assert execution.duration_seconds is None

    def test_execution_properties_complete_success(self):
        """Test execution properties when complete and successful."""
        start = datetime.now()
        end = start + timedelta(seconds=10)
        execution = SubagentExecution(
            id="exec1",
            parent_id="parent1",
            subagent_type="reviewer",
            task="Review code",
            started_at=start,
            ended_at=end,
            result="Code looks good",
        )
        assert execution.is_complete is True
        assert execution.is_success is True
        assert execution.duration_seconds == 10.0

    def test_execution_properties_complete_with_error(self):
        """Test execution properties when complete with error."""
        start = datetime.now()
        end = start + timedelta(seconds=5)
        execution = SubagentExecution(
            id="exec1",
            parent_id="parent1",
            subagent_type="planner",
            task="Create plan",
            started_at=start,
            ended_at=end,
            error="Failed to create plan",
        )
        assert execution.is_complete is True
        assert execution.is_success is False
        assert execution.duration_seconds == 5.0

    def test_execution_to_dict(self):
        """Test converting execution to dictionary."""
        start = datetime.now()
        end = start + timedelta(seconds=15)
        execution = SubagentExecution(
            id="exec1",
            parent_id="parent1",
            subagent_type="researcher",
            task="Research topic",
            started_at=start,
            ended_at=end,
            turns=3,
            tool_calls=[{"tool": "internet_search", "args": {}}],
            result="Found info",
            tokens_used=500,
        )
        data = execution.to_dict()
        assert data["id"] == "exec1"
        assert data["parent_id"] == "parent1"
        assert data["subagent_type"] == "researcher"
        assert data["task"] == "Research topic"
        assert data["started_at"] == start.isoformat()
        assert data["ended_at"] == end.isoformat()
        assert data["turns"] == 3
        assert len(data["tool_calls"]) == 1
        assert data["result"] == "Found info"
        assert data["error"] is None
        assert data["tokens_used"] == 500
        assert data["duration_seconds"] == 15.0
        assert data["is_success"] is True


# =============================================================================
# SubagentMonitor Tests
# =============================================================================


class TestSubagentMonitor:
    """Tests for SubagentMonitor class."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = SubagentMonitor()
        assert len(monitor.executions) == 0
        assert monitor.max_history == 100
        assert monitor.get_active_count() == 0

    def test_monitor_custom_history(self):
        """Test monitor with custom max history."""
        monitor = SubagentMonitor(max_history=10)
        assert monitor.max_history == 10

    def test_start_execution(self):
        """Test starting an execution."""
        monitor = SubagentMonitor()
        execution = monitor.start_execution(
            parent_id="parent1",
            subagent_type="researcher",
            task="Research task",
        )
        assert execution.parent_id == "parent1"
        assert execution.subagent_type == "researcher"
        assert execution.task == "Research task"
        assert execution.id.startswith("subagent_")
        assert monitor.get_active_count() == 1

    def test_start_execution_custom_id(self):
        """Test starting an execution with custom ID."""
        monitor = SubagentMonitor()
        execution = monitor.start_execution(
            parent_id="parent1",
            subagent_type="coder",
            task="Code task",
            execution_id="custom_id",
        )
        assert execution.id == "custom_id"

    def test_record_turn(self):
        """Test recording turns."""
        monitor = SubagentMonitor()
        execution = monitor.start_execution(
            parent_id="parent1",
            subagent_type="reviewer",
            task="Review task",
        )
        assert execution.turns == 0
        monitor.record_turn(execution.id)
        assert execution.turns == 1
        monitor.record_turn(execution.id)
        assert execution.turns == 2

    def test_record_turn_unknown_id(self):
        """Test recording turn for unknown ID."""
        monitor = SubagentMonitor()
        monitor.record_turn("unknown")  # Should not raise

    def test_record_tool_call(self):
        """Test recording tool calls."""
        monitor = SubagentMonitor()
        execution = monitor.start_execution(
            parent_id="parent1",
            subagent_type="researcher",
            task="Research task",
        )
        monitor.record_tool_call(
            execution.id,
            tool_name="internet_search",
            args={"query": "test"},
            result="Found results",
        )
        assert len(execution.tool_calls) == 1
        assert execution.tool_calls[0]["tool"] == "internet_search"
        assert execution.tool_calls[0]["args"] == {"query": "test"}
        assert "timestamp" in execution.tool_calls[0]

    def test_record_tokens(self):
        """Test recording tokens."""
        monitor = SubagentMonitor()
        execution = monitor.start_execution(
            parent_id="parent1",
            subagent_type="coder",
            task="Code task",
        )
        monitor.record_tokens(execution.id, 100)
        assert execution.tokens_used == 100
        monitor.record_tokens(execution.id, 50)
        assert execution.tokens_used == 150

    def test_end_execution_success(self):
        """Test ending an execution successfully."""
        monitor = SubagentMonitor()
        execution = monitor.start_execution(
            parent_id="parent1",
            subagent_type="planner",
            task="Plan task",
        )
        exec_id = execution.id
        completed = monitor.end_execution(exec_id, result="Plan created")
        assert completed is not None
        assert completed.result == "Plan created"
        assert completed.error is None
        assert completed.is_complete is True
        assert completed.is_success is True
        assert monitor.get_active_count() == 0
        assert len(monitor.executions) == 1

    def test_end_execution_with_error(self):
        """Test ending an execution with error."""
        monitor = SubagentMonitor()
        execution = monitor.start_execution(
            parent_id="parent1",
            subagent_type="reviewer",
            task="Review task",
        )
        completed = monitor.end_execution(execution.id, error="Review failed")
        assert completed.error == "Review failed"
        assert completed.is_success is False

    def test_end_execution_unknown_id(self):
        """Test ending an unknown execution."""
        monitor = SubagentMonitor()
        result = monitor.end_execution("unknown")
        assert result is None

    def test_get_execution_active(self):
        """Test getting an active execution."""
        monitor = SubagentMonitor()
        execution = monitor.start_execution(
            parent_id="parent1",
            subagent_type="researcher",
            task="Research task",
        )
        found = monitor.get_execution(execution.id)
        assert found is execution

    def test_get_execution_completed(self):
        """Test getting a completed execution."""
        monitor = SubagentMonitor()
        execution = monitor.start_execution(
            parent_id="parent1",
            subagent_type="coder",
            task="Code task",
        )
        monitor.end_execution(execution.id, result="Done")
        found = monitor.get_execution(execution.id)
        assert found is not None
        assert found.result == "Done"

    def test_get_execution_not_found(self):
        """Test getting an execution that doesn't exist."""
        monitor = SubagentMonitor()
        found = monitor.get_execution("unknown")
        assert found is None

    def test_get_active_executions(self):
        """Test getting all active executions."""
        monitor = SubagentMonitor()
        exec1 = monitor.start_execution("p1", "researcher", "Task 1")
        exec2 = monitor.start_execution("p1", "coder", "Task 2")
        active = monitor.get_active_executions()
        assert len(active) == 2
        assert exec1 in active
        assert exec2 in active

    def test_get_recent_executions(self):
        """Test getting recent completed executions."""
        monitor = SubagentMonitor()
        for i in range(5):
            exec = monitor.start_execution("p1", "researcher", f"Task {i}")
            monitor.end_execution(exec.id, result=f"Result {i}")
            time.sleep(0.01)  # Ensure different timestamps
        recent = monitor.get_recent_executions(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].task == "Task 4"
        assert recent[1].task == "Task 3"
        assert recent[2].task == "Task 2"

    def test_get_statistics_empty(self):
        """Test statistics with no executions."""
        monitor = SubagentMonitor()
        stats = monitor.get_statistics()
        assert stats["total_executions"] == 0
        assert stats["active_count"] == 0

    def test_get_statistics_with_data(self):
        """Test statistics with completed executions."""
        monitor = SubagentMonitor()
        # Add successful execution
        exec1 = monitor.start_execution("p1", "researcher", "Task 1")
        monitor.record_turn(exec1.id)
        monitor.record_turn(exec1.id)
        monitor.record_tool_call(exec1.id, "search", {}, "result")
        monitor.end_execution(exec1.id, result="Done", tokens_used=500)

        # Add failed execution
        exec2 = monitor.start_execution("p1", "coder", "Task 2")
        monitor.end_execution(exec2.id, error="Failed", tokens_used=200)

        stats = monitor.get_statistics()
        assert stats["total_executions"] == 2
        assert stats["successful"] == 1
        assert stats["failed"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["total_tokens"] == 700
        assert stats["avg_turns"] == 1.0  # 2 turns / 2 executions
        assert stats["avg_tool_calls"] == 0.5  # 1 tool call / 2 executions
        assert "by_type" in stats
        assert stats["by_type"]["researcher"] == 1
        assert stats["by_type"]["coder"] == 1

    def test_history_trimming(self):
        """Test that history is trimmed to max_history."""
        monitor = SubagentMonitor(max_history=3)
        for i in range(5):
            exec = monitor.start_execution("p1", "researcher", f"Task {i}")
            monitor.end_execution(exec.id, result=f"Result {i}")
        assert len(monitor.executions) == 3
        # Only most recent 3 should remain
        tasks = [e.task for e in monitor.executions]
        assert tasks == ["Task 2", "Task 3", "Task 4"]

    def test_clear_history(self):
        """Test clearing execution history."""
        monitor = SubagentMonitor()
        for i in range(5):
            exec = monitor.start_execution("p1", "researcher", f"Task {i}")
            monitor.end_execution(exec.id, result=f"Result {i}")
        assert len(monitor.executions) == 5
        count = monitor.clear_history()
        assert count == 5
        assert len(monitor.executions) == 0

    def test_count_by_type(self):
        """Test counting executions by type."""
        monitor = SubagentMonitor()
        for _ in range(3):
            exec = monitor.start_execution("p1", "researcher", "Research")
            monitor.end_execution(exec.id, result="Done")
        for _ in range(2):
            exec = monitor.start_execution("p1", "coder", "Code")
            monitor.end_execution(exec.id, result="Done")
        stats = monitor.get_statistics()
        assert stats["by_type"]["researcher"] == 3
        assert stats["by_type"]["coder"] == 2

    def test_record_tool_call_unknown_id(self):
        """Test recording tool call for unknown ID."""
        monitor = SubagentMonitor()
        # Should not raise, just silently ignore
        monitor.record_tool_call("unknown", "tool", {}, "result")
        # Verify no side effects
        assert monitor.get_active_count() == 0

    def test_record_tokens_unknown_id(self):
        """Test recording tokens for unknown ID."""
        monitor = SubagentMonitor()
        # Should not raise, just silently ignore
        monitor.record_tokens("unknown", 100)
        # Verify no side effects
        assert monitor.get_active_count() == 0

    def test_get_execution_search_history(self):
        """Test getting execution from history (not found case)."""
        monitor = SubagentMonitor()
        # Add some executions to history
        exec1 = monitor.start_execution("p1", "researcher", "Task 1")
        monitor.end_execution(exec1.id, result="Done 1")
        exec2 = monitor.start_execution("p1", "coder", "Task 2")
        monitor.end_execution(exec2.id, result="Done 2")

        # Search for a non-existent ID in history
        found = monitor.get_execution("nonexistent_id")
        assert found is None

        # Search for first execution in history (iterates through loop)
        found = monitor.get_execution(exec1.id)
        assert found is not None
        assert found.id == exec1.id


# =============================================================================
# ContextPruningConfig Tests (Moltbot parity feature)
# =============================================================================


class TestContextPruningConfig:
    """Tests for ContextPruningConfig dataclass."""

    def test_context_pruning_mode_enum(self):
        """Test ContextPruningMode enum values."""
        assert ContextPruningMode.OFF == "off"
        assert ContextPruningMode.CACHE_TTL == "cache-ttl"
        assert ContextPruningMode.AGGRESSIVE == "aggressive"

    def test_context_pruning_config_defaults(self):
        """Test ContextPruningConfig default values."""
        config = ContextPruningConfig()
        assert config.mode == ContextPruningMode.OFF
        assert config.ttl_minutes == 30
        assert config.keep_last_assistants == 3
        assert config.soft_trim_ratio == 0.7
        assert config.hard_clear_ratio == 0.9

    def test_context_pruning_config_custom(self):
        """Test ContextPruningConfig with custom values."""
        config = ContextPruningConfig(
            mode=ContextPruningMode.AGGRESSIVE,
            ttl_minutes=10,
            keep_last_assistants=2,
            soft_trim_ratio=0.5,
            hard_clear_ratio=0.75,
        )
        assert config.mode == ContextPruningMode.AGGRESSIVE
        assert config.ttl_minutes == 10
        assert config.keep_last_assistants == 2
        assert config.soft_trim_ratio == 0.5
        assert config.hard_clear_ratio == 0.75

    def test_context_pruning_preset_off(self):
        """Test CONTEXT_PRUNING_OFF preset."""
        assert CONTEXT_PRUNING_OFF.mode == ContextPruningMode.OFF

    def test_context_pruning_preset_standard(self):
        """Test CONTEXT_PRUNING_STANDARD preset."""
        assert CONTEXT_PRUNING_STANDARD.mode == ContextPruningMode.CACHE_TTL
        assert CONTEXT_PRUNING_STANDARD.ttl_minutes == 30
        assert CONTEXT_PRUNING_STANDARD.soft_trim_ratio == 0.7

    def test_context_pruning_preset_aggressive(self):
        """Test CONTEXT_PRUNING_AGGRESSIVE preset."""
        assert CONTEXT_PRUNING_AGGRESSIVE.mode == ContextPruningMode.AGGRESSIVE
        assert CONTEXT_PRUNING_AGGRESSIVE.ttl_minutes == 10
        assert CONTEXT_PRUNING_AGGRESSIVE.soft_trim_ratio == 0.5

    def test_context_pruning_validation_soft_trim_ratio(self):
        """Test validation of soft_trim_ratio."""
        with pytest.raises(ValueError, match="soft_trim_ratio"):
            ContextPruningConfig(soft_trim_ratio=1.5)
        with pytest.raises(ValueError, match="soft_trim_ratio"):
            ContextPruningConfig(soft_trim_ratio=-0.1)

    def test_context_pruning_validation_hard_clear_ratio(self):
        """Test validation of hard_clear_ratio."""
        with pytest.raises(ValueError, match="hard_clear_ratio"):
            ContextPruningConfig(hard_clear_ratio=1.5)
        with pytest.raises(ValueError, match="hard_clear_ratio"):
            ContextPruningConfig(hard_clear_ratio=-0.1)

    def test_context_pruning_validation_ratio_ordering(self):
        """Test that soft_trim_ratio must be less than hard_clear_ratio."""
        with pytest.raises(ValueError, match="soft_trim_ratio.*must be less than"):
            ContextPruningConfig(soft_trim_ratio=0.9, hard_clear_ratio=0.7)
        with pytest.raises(ValueError, match="soft_trim_ratio.*must be less than"):
            ContextPruningConfig(soft_trim_ratio=0.8, hard_clear_ratio=0.8)

    def test_context_pruning_validation_ttl_minutes(self):
        """Test validation of ttl_minutes."""
        with pytest.raises(ValueError, match="ttl_minutes"):
            ContextPruningConfig(ttl_minutes=-1)

    def test_context_pruning_validation_keep_last_assistants(self):
        """Test validation of keep_last_assistants."""
        with pytest.raises(ValueError, match="keep_last_assistants"):
            ContextPruningConfig(keep_last_assistants=-1)

    def test_subagent_context_pruning_configs(self):
        """Test that subagents have appropriate context pruning settings."""
        # Long-running subagents should have pruning enabled
        assert RESEARCHER.context_pruning.mode == ContextPruningMode.CACHE_TTL
        assert CODER.context_pruning.mode == ContextPruningMode.CACHE_TTL
        assert REVIEWER.context_pruning.mode == ContextPruningMode.CACHE_TTL
        assert ANALYST.context_pruning.mode == ContextPruningMode.CACHE_TTL
        assert WRITER.context_pruning.mode == ContextPruningMode.CACHE_TTL
        # Browser sessions can be verbose - use aggressive
        assert BROWSER.context_pruning.mode == ContextPruningMode.AGGRESSIVE
        # Short-lived subagents use default (OFF)
        assert PLANNER.context_pruning.mode == ContextPruningMode.OFF
        assert MEMORY.context_pruning.mode == ContextPruningMode.OFF


# =============================================================================
# SubagentEventType and SubagentEvent Tests (Moltbot parity feature)
# =============================================================================


class TestSubagentEventType:
    """Tests for SubagentEventType enum."""

    def test_event_type_values(self):
        """Test SubagentEventType enum values."""
        assert SubagentEventType.STARTED == "started"
        assert SubagentEventType.TURN_COMPLETED == "turn_completed"
        assert SubagentEventType.TOOL_CALLED == "tool_called"
        assert SubagentEventType.COMPLETED == "completed"
        assert SubagentEventType.FAILED == "failed"
        assert SubagentEventType.TIMEOUT == "timeout"

    def test_event_type_is_string_enum(self):
        """Test that SubagentEventType values are strings."""
        for event_type in SubagentEventType:
            assert isinstance(event_type.value, str)


class TestSubagentEvent:
    """Tests for SubagentEvent dataclass."""

    def test_event_creation(self):
        """Test creating a SubagentEvent."""
        event = SubagentEvent(
            event_type=SubagentEventType.STARTED,
            execution_id="exec_123",
            subagent_type="researcher",
            timestamp=datetime.now(),
            data={"parent_id": "parent_1", "task": "Search for info"},
        )
        assert event.event_type == SubagentEventType.STARTED
        assert event.execution_id == "exec_123"
        assert event.subagent_type == "researcher"
        assert event.data["task"] == "Search for info"

    def test_event_to_dict(self):
        """Test converting SubagentEvent to dictionary."""
        now = datetime.now()
        event = SubagentEvent(
            event_type=SubagentEventType.COMPLETED,
            execution_id="exec_456",
            subagent_type="coder",
            timestamp=now,
            data={"result": "Done"},
        )
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "completed"
        assert event_dict["execution_id"] == "exec_456"
        assert event_dict["subagent_type"] == "coder"
        assert event_dict["timestamp"] == now.isoformat()
        assert event_dict["data"]["result"] == "Done"

    def test_event_default_data(self):
        """Test SubagentEvent with default empty data."""
        event = SubagentEvent(
            event_type=SubagentEventType.STARTED,
            execution_id="exec_789",
            subagent_type="planner",
            timestamp=datetime.now(),
        )
        assert event.data == {}


# =============================================================================
# SubagentMonitor Lifecycle Events Tests
# =============================================================================


class TestSubagentMonitorLifecycleEvents:
    """Tests for SubagentMonitor lifecycle event callbacks."""

    def test_on_event_registration(self):
        """Test registering event callbacks."""
        monitor = SubagentMonitor(auto_persist=False)
        received_events = []

        @monitor.on_event(SubagentEventType.STARTED)
        def on_started(event):
            received_events.append(event)

        exec = monitor.start_execution("p1", "researcher", "Task 1")
        assert len(received_events) == 1
        assert received_events[0].event_type == SubagentEventType.STARTED
        assert received_events[0].execution_id == exec.id

    def test_on_event_direct_registration(self):
        """Test registering callbacks directly (not as decorator)."""
        monitor = SubagentMonitor(auto_persist=False)
        received_events = []

        def callback(event):
            received_events.append(event)

        monitor.on_event(SubagentEventType.COMPLETED, callback)
        exec = monitor.start_execution("p1", "coder", "Code task")
        monitor.end_execution(exec.id, result="Done")

        # Should receive only the COMPLETED event
        completed_events = [e for e in received_events if e.event_type == SubagentEventType.COMPLETED]
        assert len(completed_events) == 1

    def test_global_callback_receives_all_events(self):
        """Test that global callbacks receive all event types."""
        monitor = SubagentMonitor(auto_persist=False)
        received_events = []

        @monitor.on_event()  # No specific event type = global
        def on_any(event):
            received_events.append(event)

        exec = monitor.start_execution("p1", "researcher", "Task")
        monitor.record_turn(exec.id)
        monitor.record_tool_call(exec.id, "search", {}, "result")
        monitor.end_execution(exec.id, result="Done")

        # Should receive: STARTED, TURN_COMPLETED, TOOL_CALLED, COMPLETED
        event_types = [e.event_type for e in received_events]
        assert SubagentEventType.STARTED in event_types
        assert SubagentEventType.TURN_COMPLETED in event_types
        assert SubagentEventType.TOOL_CALLED in event_types
        assert SubagentEventType.COMPLETED in event_types

    def test_remove_callback(self):
        """Test removing event callbacks."""
        monitor = SubagentMonitor(auto_persist=False)
        received_events = []

        def callback(event):
            received_events.append(event)

        monitor.on_event(SubagentEventType.STARTED, callback)
        exec1 = monitor.start_execution("p1", "researcher", "Task 1")
        assert len(received_events) == 1

        # Remove callback
        result = monitor.remove_callback(callback, SubagentEventType.STARTED)
        assert result is True

        # Start another execution - callback should not be called
        exec2 = monitor.start_execution("p1", "coder", "Task 2")
        assert len(received_events) == 1  # Still only 1

    def test_remove_callback_not_found(self):
        """Test removing callback that wasn't registered."""
        monitor = SubagentMonitor(auto_persist=False)

        def callback(event):
            pass

        result = monitor.remove_callback(callback, SubagentEventType.STARTED)
        assert result is False

    def test_failed_event_emitted(self):
        """Test that FAILED event is emitted on error."""
        monitor = SubagentMonitor(auto_persist=False)
        received_events = []

        @monitor.on_event(SubagentEventType.FAILED)
        def on_failed(event):
            received_events.append(event)

        exec = monitor.start_execution("p1", "researcher", "Task")
        monitor.end_execution(exec.id, error="Something went wrong")

        assert len(received_events) == 1
        assert received_events[0].data["error"] == "Something went wrong"

    def test_timeout_event_emitted(self):
        """Test that TIMEOUT event is emitted on timeout."""
        monitor = SubagentMonitor(auto_persist=False)
        received_events = []

        @monitor.on_event(SubagentEventType.TIMEOUT)
        def on_timeout(event):
            received_events.append(event)

        exec = monitor.start_execution("p1", "researcher", "Task")
        monitor.end_execution(exec.id, error="Timed out", timeout=True)

        assert len(received_events) == 1

    def test_callback_error_handling(self):
        """Test that callback errors don't break execution."""
        monitor = SubagentMonitor(auto_persist=False)
        good_events = []

        @monitor.on_event(SubagentEventType.STARTED)
        def bad_callback(event):
            raise RuntimeError("Callback error")

        @monitor.on_event(SubagentEventType.STARTED)
        def good_callback(event):
            good_events.append(event)

        # Should not raise, despite bad_callback throwing
        exec = monitor.start_execution("p1", "researcher", "Task")
        assert len(good_events) == 1


# =============================================================================
# SubagentMonitor Persistence Tests
# =============================================================================


class TestSubagentMonitorPersistence:
    """Tests for SubagentMonitor disk persistence."""

    def test_save_to_disk_creates_file(self):
        """Test that save_to_disk creates the persistence file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = Path(tmpdir) / "subagent_runs.json"
            monitor = SubagentMonitor(
                persistence_path=persistence_path,
                auto_persist=False,
            )
            exec = monitor.start_execution("p1", "researcher", "Task 1")
            monitor.end_execution(exec.id, result="Done")

            result = monitor.save_to_disk()
            assert result is True
            assert persistence_path.exists()

    def test_load_from_disk(self):
        """Test loading executions from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = Path(tmpdir) / "subagent_runs.json"

            # Create and save
            monitor1 = SubagentMonitor(
                persistence_path=persistence_path,
                auto_persist=False,
            )
            exec = monitor1.start_execution("p1", "researcher", "Task 1")
            monitor1.end_execution(exec.id, result="Done", tokens_used=100)
            monitor1.save_to_disk()

            # Load in new monitor
            monitor2 = SubagentMonitor(
                persistence_path=persistence_path,
                auto_persist=False,
            )
            count = monitor2.load_from_disk()
            assert count == 1
            assert len(monitor2.executions) == 1
            assert monitor2.executions[0].task == "Task 1"
            assert monitor2.executions[0].result == "Done"
            assert monitor2.executions[0].tokens_used == 100

    def test_load_from_disk_empty(self):
        """Test loading from non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = Path(tmpdir) / "nonexistent.json"
            monitor = SubagentMonitor(
                persistence_path=persistence_path,
                auto_persist=False,
            )
            count = monitor.load_from_disk()
            assert count == 0

    def test_delete_persistence_file(self):
        """Test deleting the persistence file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = Path(tmpdir) / "subagent_runs.json"
            monitor = SubagentMonitor(
                persistence_path=persistence_path,
                auto_persist=False,
            )
            exec = monitor.start_execution("p1", "researcher", "Task")
            monitor.end_execution(exec.id, result="Done")
            monitor.save_to_disk()

            assert persistence_path.exists()
            result = monitor.delete_persistence_file()
            assert result is True
            assert not persistence_path.exists()

    def test_delete_persistence_file_not_exists(self):
        """Test deleting non-existent persistence file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = Path(tmpdir) / "nonexistent.json"
            monitor = SubagentMonitor(
                persistence_path=persistence_path,
                auto_persist=False,
            )
            result = monitor.delete_persistence_file()
            assert result is True  # Succeeds even if file doesn't exist

    def test_auto_persist_on_end_execution(self):
        """Test that auto_persist saves after end_execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = Path(tmpdir) / "subagent_runs.json"
            monitor = SubagentMonitor(
                persistence_path=persistence_path,
                auto_persist=True,
            )
            exec = monitor.start_execution("p1", "researcher", "Task")
            # File should not exist yet (no completed executions)
            assert not persistence_path.exists()

            monitor.end_execution(exec.id, result="Done")
            # File should now exist due to auto_persist
            assert persistence_path.exists()

    def test_auto_persist_on_clear_history(self):
        """Test that auto_persist saves after clear_history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = Path(tmpdir) / "subagent_runs.json"
            monitor = SubagentMonitor(
                persistence_path=persistence_path,
                auto_persist=True,
            )
            exec = monitor.start_execution("p1", "researcher", "Task")
            monitor.end_execution(exec.id, result="Done")

            # Manually delete file to test clear_history saves
            persistence_path.unlink()
            assert not persistence_path.exists()

            monitor.clear_history()
            # File should be recreated (with empty executions)
            assert persistence_path.exists()

    def test_persistence_path_string(self):
        """Test that persistence_path accepts string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = str(Path(tmpdir) / "subagent_runs.json")
            monitor = SubagentMonitor(
                persistence_path=persistence_path,
                auto_persist=False,
            )
            assert monitor.persistence_path == Path(persistence_path)


# =============================================================================
# Enhanced Subagent Monitor: Announce Queue Tests
# =============================================================================

from ag3nt_agent.subagent_monitor import (
    AnnounceMessage,
    AnnouncePriority,
    AnnounceQueue,
    CrossSessionBus,
    DeliveryContext,
    DeliveryStatus,
    DeliveryTracker,
    SessionMessage,
    get_announce_queue,
    get_cross_session_bus,
    get_delivery_tracker,
    reset_global_instances,
)


class TestAnnouncePriority:
    """Tests for AnnouncePriority enum."""

    def test_priority_ordering(self):
        """Test that priority values are correctly ordered."""
        assert AnnouncePriority.LOW.value < AnnouncePriority.NORMAL.value
        assert AnnouncePriority.NORMAL.value < AnnouncePriority.HIGH.value
        assert AnnouncePriority.HIGH.value < AnnouncePriority.URGENT.value

    def test_priority_values(self):
        """Test specific priority values."""
        assert AnnouncePriority.LOW.value == 1
        assert AnnouncePriority.NORMAL.value == 5
        assert AnnouncePriority.HIGH.value == 10
        assert AnnouncePriority.URGENT.value == 20


class TestAnnounceMessage:
    """Tests for AnnounceMessage dataclass."""

    def test_message_creation(self):
        """Test creating an AnnounceMessage."""
        now = datetime.now()
        msg = AnnounceMessage(
            id="msg_001",
            source_id="subagent_1",
            source_session_id="session_123",
            priority=AnnouncePriority.HIGH,
            topic="findings",
            content={"key": "value"},
            created_at=now,
        )
        assert msg.id == "msg_001"
        assert msg.source_id == "subagent_1"
        assert msg.priority == AnnouncePriority.HIGH
        assert msg.topic == "findings"
        assert msg.content == {"key": "value"}

    def test_is_expired_no_expiry(self):
        """Test is_expired when no expiry set."""
        msg = AnnounceMessage(
            id="msg_001",
            source_id="subagent_1",
            source_session_id="session_123",
            priority=AnnouncePriority.NORMAL,
            topic="test",
            content="data",
            created_at=datetime.now(),
            expires_at=None,
        )
        assert msg.is_expired() is False

    def test_is_expired_future(self):
        """Test is_expired when expiry is in future."""
        msg = AnnounceMessage(
            id="msg_001",
            source_id="subagent_1",
            source_session_id="session_123",
            priority=AnnouncePriority.NORMAL,
            topic="test",
            content="data",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert msg.is_expired() is False

    def test_is_expired_past(self):
        """Test is_expired when expiry is in past."""
        msg = AnnounceMessage(
            id="msg_001",
            source_id="subagent_1",
            source_session_id="session_123",
            priority=AnnouncePriority.NORMAL,
            topic="test",
            content="data",
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert msg.is_expired() is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now()
        msg = AnnounceMessage(
            id="msg_001",
            source_id="subagent_1",
            source_session_id="session_123",
            priority=AnnouncePriority.HIGH,
            topic="findings",
            content={"key": "value"},
            created_at=now,
            metadata={"extra": "info"},
        )
        d = msg.to_dict()
        assert d["id"] == "msg_001"
        assert d["priority"] == 10  # HIGH.value
        assert d["metadata"] == {"extra": "info"}


class TestAnnounceQueue:
    """Tests for AnnounceQueue class."""

    def test_publish_and_poll(self):
        """Test basic publish and poll."""
        queue = AnnounceQueue()
        msg = queue.publish(
            source_id="agent_1",
            source_session_id="session_1",
            topic="results",
            content={"data": 123},
        )
        assert msg.id.startswith("announce_")
        assert msg.topic == "results"

        polled = queue.poll(topic="results", limit=1)
        assert len(polled) == 1
        assert polled[0].content == {"data": 123}

    def test_priority_ordering(self):
        """Test that messages are dequeued by priority."""
        queue = AnnounceQueue()
        queue.publish("a", "s", "t", "low", priority=AnnouncePriority.LOW)
        queue.publish("a", "s", "t", "normal", priority=AnnouncePriority.NORMAL)
        queue.publish("a", "s", "t", "high", priority=AnnouncePriority.HIGH)
        queue.publish("a", "s", "t", "urgent", priority=AnnouncePriority.URGENT)

        polled = queue.poll(limit=4)
        assert polled[0].content == "urgent"
        assert polled[1].content == "high"
        assert polled[2].content == "normal"
        assert polled[3].content == "low"

    def test_topic_filtering(self):
        """Test filtering by topic."""
        queue = AnnounceQueue()
        queue.publish("a", "s", "topic_a", "A1")
        queue.publish("a", "s", "topic_b", "B1")
        queue.publish("a", "s", "topic_a", "A2")

        polled_a = queue.poll(topic="topic_a")
        assert len(polled_a) == 2
        assert all(m.topic == "topic_a" for m in polled_a)

    def test_subscribe_and_get_subscribers(self):
        """Test topic subscription."""
        queue = AnnounceQueue()
        queue.subscribe("session_1", "events")
        queue.subscribe("session_2", "events")
        queue.subscribe("session_1", "logs")

        subs = queue.get_subscribers("events")
        assert subs == {"session_1", "session_2"}

        subs_logs = queue.get_subscribers("logs")
        assert subs_logs == {"session_1"}

    def test_unsubscribe_specific_topic(self):
        """Test unsubscribing from specific topic."""
        queue = AnnounceQueue()
        queue.subscribe("session_1", "events")
        queue.subscribe("session_1", "logs")

        queue.unsubscribe("session_1", "events")
        assert "session_1" not in queue.get_subscribers("events")
        assert "session_1" in queue.get_subscribers("logs")

    def test_unsubscribe_all_topics(self):
        """Test unsubscribing from all topics."""
        queue = AnnounceQueue()
        queue.subscribe("session_1", "events")
        queue.subscribe("session_1", "logs")

        queue.unsubscribe("session_1", None)
        assert "session_1" not in queue.get_subscribers("events")
        assert "session_1" not in queue.get_subscribers("logs")

    def test_peek_does_not_remove(self):
        """Test that peek returns messages without removing."""
        queue = AnnounceQueue()
        queue.publish("a", "s", "t", "content")

        peeked = queue.peek(limit=1)
        assert len(peeked) == 1

        # Message should still be in queue
        polled = queue.poll(limit=1)
        assert len(polled) == 1

    def test_count(self):
        """Test counting messages."""
        queue = AnnounceQueue()
        queue.publish("a", "s", "topic_a", "1")
        queue.publish("a", "s", "topic_b", "2")
        queue.publish("a", "s", "topic_a", "3")

        assert queue.count() == 3
        assert queue.count(topic="topic_a") == 2
        assert queue.count(topic="topic_b") == 1

    def test_clear(self):
        """Test clearing messages."""
        queue = AnnounceQueue()
        queue.publish("a", "s", "topic_a", "1")
        queue.publish("a", "s", "topic_b", "2")

        cleared = queue.clear(topic="topic_a")
        assert cleared == 1
        assert queue.count() == 1

        cleared_all = queue.clear()
        assert cleared_all == 1
        assert queue.count() == 0

    def test_max_size_trimming(self):
        """Test that queue trims when exceeding max size."""
        queue = AnnounceQueue(max_size=5)
        for i in range(10):
            queue.publish("a", "s", "t", f"msg_{i}")

        assert queue.count() <= 5


# =============================================================================
# Enhanced Subagent Monitor: Cross-Session Bus Tests
# =============================================================================


class TestSessionMessage:
    """Tests for SessionMessage dataclass."""

    def test_message_creation(self):
        """Test creating a SessionMessage."""
        now = datetime.now()
        msg = SessionMessage(
            id="msg_001",
            from_session="session_a",
            to_session="session_b",
            topic="task_result",
            payload={"status": "complete"},
            created_at=now,
        )
        assert msg.id == "msg_001"
        assert msg.from_session == "session_a"
        assert msg.to_session == "session_b"
        assert msg.acknowledged is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now()
        msg = SessionMessage(
            id="msg_001",
            from_session="session_a",
            to_session=None,  # broadcast
            topic="announcement",
            payload="hello",
            created_at=now,
        )
        d = msg.to_dict()
        assert d["id"] == "msg_001"
        assert d["to_session"] is None
        assert d["acknowledged"] is False


class TestCrossSessionBus:
    """Tests for CrossSessionBus class."""

    def test_send_and_get_messages(self):
        """Test sending and receiving direct messages."""
        bus = CrossSessionBus()
        msg = bus.send(
            from_session="session_a",
            to_session="session_b",
            topic="data",
            payload={"value": 42},
        )
        assert msg.id.startswith("msg_")

        messages = bus.get_messages("session_b")
        assert len(messages) == 1
        assert messages[0].payload == {"value": 42}

    def test_broadcast(self):
        """Test broadcast messages."""
        bus = CrossSessionBus()
        bus.subscribe_topic("session_b", "announcements")
        bus.subscribe_topic("session_c", "announcements")

        msg = bus.broadcast(
            from_session="session_a",
            topic="announcements",
            payload="Hello all!",
        )
        assert msg.id.startswith("bcast_")
        assert msg.to_session is None

        # Subscribers can see broadcast
        messages_b = bus.get_messages("session_b", topic="announcements")
        assert len(messages_b) == 1

    def test_topic_subscription(self):
        """Test topic subscription and filtering."""
        bus = CrossSessionBus()
        bus.subscribe_topic("session_a", "logs")
        bus.subscribe_topic("session_a", "errors")

        bus.broadcast("system", "logs", "Log message")
        bus.broadcast("system", "errors", "Error message")
        bus.broadcast("system", "metrics", "Metric message")

        # session_a subscribed to logs and errors, but not metrics
        messages = bus.get_messages("session_a")
        topics = {m.topic for m in messages}
        assert "logs" in topics
        assert "errors" in topics
        assert "metrics" not in topics

    def test_unsubscribe_topic(self):
        """Test unsubscribing from topics."""
        bus = CrossSessionBus()
        bus.subscribe_topic("session_a", "logs")
        bus.subscribe_topic("session_a", "errors")

        bus.unsubscribe_topic("session_a", "logs")
        bus.broadcast("system", "logs", "Log message")
        bus.broadcast("system", "errors", "Error message")

        messages = bus.get_messages("session_a")
        topics = {m.topic for m in messages}
        assert "logs" not in topics
        assert "errors" in topics

    def test_acknowledge(self):
        """Test message acknowledgment."""
        bus = CrossSessionBus()
        msg = bus.send("a", "b", "topic", "data")
        assert msg.acknowledged is False

        result = bus.acknowledge(msg.id)
        assert result is True

        messages = bus.get_messages("b", unacknowledged_only=True)
        assert len(messages) == 0

    def test_clear_session(self):
        """Test clearing session mailbox."""
        bus = CrossSessionBus()
        bus.send("a", "b", "t1", "m1")
        bus.send("a", "b", "t2", "m2")

        cleared = bus.clear_session("b")
        assert cleared == 2

        messages = bus.get_messages("b")
        assert len(messages) == 0

    def test_get_statistics(self):
        """Test bus statistics."""
        bus = CrossSessionBus()
        bus.send("a", "b", "t", "m1")
        bus.send("a", "c", "t", "m2")
        bus.broadcast("a", "t", "m3")

        stats = bus.get_statistics()
        assert stats["total_direct_messages"] == 2
        assert stats["total_broadcast_messages"] == 1
        assert stats["active_sessions"] == 2


# =============================================================================
# Enhanced Subagent Monitor: Delivery Tracker Tests
# =============================================================================


class TestDeliveryStatus:
    """Tests for DeliveryStatus enum."""

    def test_status_values(self):
        """Test delivery status values."""
        assert DeliveryStatus.PENDING.value == "pending"
        assert DeliveryStatus.DELIVERED.value == "delivered"
        assert DeliveryStatus.ACKNOWLEDGED.value == "acknowledged"
        assert DeliveryStatus.EXPIRED.value == "expired"
        assert DeliveryStatus.FAILED.value == "failed"


class TestDeliveryContext:
    """Tests for DeliveryContext dataclass."""

    def test_context_creation(self):
        """Test creating a DeliveryContext."""
        ctx = DeliveryContext(
            message_id="msg_001",
            recipient_id="session_123",
        )
        assert ctx.message_id == "msg_001"
        assert ctx.recipient_id == "session_123"
        assert ctx.status == DeliveryStatus.PENDING
        assert ctx.attempts == 0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ctx = DeliveryContext(
            message_id="msg_001",
            recipient_id="session_123",
            status=DeliveryStatus.DELIVERED,
            attempts=1,
        )
        d = ctx.to_dict()
        assert d["message_id"] == "msg_001"
        assert d["status"] == "delivered"
        assert d["attempts"] == 1


class TestDeliveryTracker:
    """Tests for DeliveryTracker class."""

    def test_track(self):
        """Test tracking a new delivery."""
        tracker = DeliveryTracker()
        ctx = tracker.track("msg_001", "session_1")

        assert ctx.message_id == "msg_001"
        assert ctx.recipient_id == "session_1"
        assert ctx.status == DeliveryStatus.PENDING

    def test_mark_delivered(self):
        """Test marking as delivered."""
        tracker = DeliveryTracker()
        tracker.track("msg_001", "session_1")

        result = tracker.mark_delivered("msg_001", "session_1")
        assert result is True

        ctx = tracker.get_context("msg_001", "session_1")
        assert ctx.status == DeliveryStatus.DELIVERED
        assert ctx.attempts == 1
        assert ctx.delivered_at is not None

    def test_acknowledge(self):
        """Test acknowledging delivery."""
        tracker = DeliveryTracker()
        tracker.track("msg_001", "session_1")
        tracker.mark_delivered("msg_001", "session_1")

        result = tracker.acknowledge("msg_001", "session_1")
        assert result is True

        ctx = tracker.get_context("msg_001", "session_1")
        assert ctx.status == DeliveryStatus.ACKNOWLEDGED
        assert ctx.acknowledged_at is not None

    def test_mark_failed(self):
        """Test marking as failed."""
        tracker = DeliveryTracker()
        tracker.track("msg_001", "session_1")

        result = tracker.mark_failed("msg_001", "session_1", "Connection refused")
        assert result is True

        ctx = tracker.get_context("msg_001", "session_1")
        assert ctx.status == DeliveryStatus.FAILED
        assert ctx.error == "Connection refused"
        assert ctx.attempts == 1

    def test_mark_expired(self):
        """Test marking as expired."""
        tracker = DeliveryTracker()
        tracker.track("msg_001", "session_1")

        result = tracker.mark_expired("msg_001", "session_1")
        assert result is True

        ctx = tracker.get_context("msg_001", "session_1")
        assert ctx.status == DeliveryStatus.EXPIRED

    def test_record_attempt(self):
        """Test recording delivery attempts."""
        tracker = DeliveryTracker()
        tracker.track("msg_001", "session_1")

        tracker.record_attempt("msg_001", "session_1")
        tracker.record_attempt("msg_001", "session_1")

        ctx = tracker.get_context("msg_001", "session_1")
        assert ctx.attempts == 2

    def test_get_pending(self):
        """Test getting pending deliveries."""
        tracker = DeliveryTracker()
        tracker.track("msg_001", "session_1")
        tracker.track("msg_002", "session_2")
        tracker.track("msg_003", "session_3")
        tracker.mark_delivered("msg_002", "session_2")

        pending = tracker.get_pending()
        assert len(pending) == 2
        ids = {p.message_id for p in pending}
        assert "msg_001" in ids
        assert "msg_003" in ids

    def test_get_pending_with_max_attempts(self):
        """Test filtering pending by max attempts."""
        tracker = DeliveryTracker()
        tracker.track("msg_001", "session_1")
        tracker.track("msg_002", "session_2")

        # Add attempts to msg_001
        tracker.record_attempt("msg_001", "session_1")
        tracker.record_attempt("msg_001", "session_1")
        tracker.record_attempt("msg_001", "session_1")

        pending = tracker.get_pending(max_attempts=3)
        assert len(pending) == 1
        assert pending[0].message_id == "msg_002"

    def test_get_failed(self):
        """Test getting failed deliveries."""
        tracker = DeliveryTracker()
        tracker.track("msg_001", "session_1")
        tracker.track("msg_002", "session_2")
        tracker.mark_failed("msg_001", "session_1", "Error")

        failed = tracker.get_failed()
        assert len(failed) == 1
        assert failed[0].message_id == "msg_001"

    def test_get_statistics(self):
        """Test delivery statistics."""
        tracker = DeliveryTracker()
        tracker.track("msg_001", "session_1")
        tracker.track("msg_002", "session_2")
        tracker.mark_delivered("msg_001", "session_1")
        tracker.acknowledge("msg_001", "session_1")

        stats = tracker.get_statistics()
        assert stats["total_tracked"] == 2
        assert stats["by_status"]["pending"] == 1
        assert stats["by_status"]["acknowledged"] == 1

    def test_clear(self):
        """Test clearing all contexts."""
        tracker = DeliveryTracker()
        tracker.track("msg_001", "session_1")
        tracker.track("msg_002", "session_2")

        cleared = tracker.clear()
        assert cleared == 2

        stats = tracker.get_statistics()
        assert stats["total_tracked"] == 0

    def test_context_not_found(self):
        """Test operations on non-existent context."""
        tracker = DeliveryTracker()

        assert tracker.mark_delivered("x", "y") is False
        assert tracker.acknowledge("x", "y") is False
        assert tracker.mark_failed("x", "y", "err") is False
        assert tracker.mark_expired("x", "y") is False
        assert tracker.record_attempt("x", "y") is False
        assert tracker.get_context("x", "y") is None


# =============================================================================
# Global Singleton Tests
# =============================================================================


class TestGlobalSingletons:
    """Tests for global singleton instances."""

    def test_get_announce_queue(self):
        """Test getting global announce queue."""
        reset_global_instances()
        q1 = get_announce_queue()
        q2 = get_announce_queue()
        assert q1 is q2

    def test_get_cross_session_bus(self):
        """Test getting global cross-session bus."""
        reset_global_instances()
        b1 = get_cross_session_bus()
        b2 = get_cross_session_bus()
        assert b1 is b2

    def test_get_delivery_tracker(self):
        """Test getting global delivery tracker."""
        reset_global_instances()
        t1 = get_delivery_tracker()
        t2 = get_delivery_tracker()
        assert t1 is t2

    def test_reset_global_instances(self):
        """Test resetting all global instances."""
        reset_global_instances()
        q1 = get_announce_queue()
        b1 = get_cross_session_bus()
        t1 = get_delivery_tracker()

        reset_global_instances()

        q2 = get_announce_queue()
        b2 = get_cross_session_bus()
        t2 = get_delivery_tracker()

        assert q1 is not q2
        assert b1 is not b2
        assert t1 is not t2
