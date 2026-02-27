"""Unit tests for subagent configuration.

Tests for:
- subagent_configs.py: SubagentConfig, registry, resource limits, resource manager

Updated to include 8 subagent types with enhanced token/turn limits
and ThinkingMode support to match/exceed Moltbot capabilities.

Also includes tests for:
- ContextPruningConfig: Context pruning for long-running sessions
"""

import json
import tempfile
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

