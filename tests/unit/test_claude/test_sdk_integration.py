"""Test Claude SDK integration."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_code_sdk import ClaudeCodeOptions

from src.claude.sdk_integration import ClaudeResponse, ClaudeSDKManager, StreamUpdate
from src.config.settings import Settings


class TestClaudeSDKManager:
    """Test Claude SDK manager."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test config without API key."""
        return Settings(
            telegram_bot_token="test:token",
            telegram_bot_username="testbot",
            approved_directory=tmp_path,
            use_sdk=True,
            claude_timeout_seconds=2,  # Short timeout for testing
        )

    @pytest.fixture
    def sdk_manager(self, config):
        """Create SDK manager."""
        return ClaudeSDKManager(config)

    async def test_sdk_manager_initialization_with_api_key(self, tmp_path):
        """Test SDK manager initialization with API key."""
        from src.config.settings import Settings

        # Test with API key provided
        config_with_key = Settings(
            telegram_bot_token="test:token",
            telegram_bot_username="testbot",
            approved_directory=tmp_path,
            anthropic_api_key="test-api-key",
            use_sdk=True,
            claude_timeout_seconds=2,
        )

        # Store original env var
        original_api_key = os.environ.get("ANTHROPIC_API_KEY")

        try:
            manager = ClaudeSDKManager(config_with_key)

            # Check that API key was set in environment
            assert os.environ.get("ANTHROPIC_API_KEY") == "test-api-key"
            assert manager.active_sessions == {}

        finally:
            # Restore original env var
            if original_api_key:
                os.environ["ANTHROPIC_API_KEY"] = original_api_key
            elif "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

    async def test_sdk_manager_initialization_without_api_key(self, config):
        """Test SDK manager initialization without API key (uses CLI auth)."""
        # Store original env var
        original_api_key = os.environ.get("ANTHROPIC_API_KEY")

        try:
            # Remove any existing API key
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

            manager = ClaudeSDKManager(config)

            # Check that no API key was set (should use CLI auth)
            assert config.anthropic_api_key_str is None
            assert manager.active_sessions == {}

        finally:
            # Restore original env var
            if original_api_key:
                os.environ["ANTHROPIC_API_KEY"] = original_api_key

    async def test_execute_command_success(self, sdk_manager):
        """Test successful command execution."""
        from claude_code_sdk.types import AssistantMessage, ResultMessage

        # Mock the claude-code-sdk query function
        async def mock_query(prompt, options):
            yield AssistantMessage(content="Test response")
            yield ResultMessage(
                subtype="success",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                session_id="test-session",
                total_cost_usd=0.05,
                result="Success",
            )

        with patch("src.claude.sdk_integration.query", side_effect=mock_query):
            response = await sdk_manager.execute_command(
                prompt="Test prompt",
                working_directory=Path("/test"),
                session_id="test-session",
            )

        # Verify response
        assert isinstance(response, ClaudeResponse)
        assert response.session_id == "test-session"
        assert response.duration_ms >= 0  # Can be 0 in tests
        assert not response.is_error
        assert response.cost == 0.05

    async def test_execute_command_with_streaming(self, sdk_manager):
        """Test command execution with streaming callback."""
        from claude_code_sdk.types import AssistantMessage, ResultMessage

        stream_updates = []

        async def stream_callback(update: StreamUpdate):
            stream_updates.append(update)

        # Mock the claude-code-sdk query function
        async def mock_query(prompt, options):
            yield AssistantMessage(content="Test response")
            yield ResultMessage(
                subtype="success",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                session_id="test-session",
                total_cost_usd=0.05,
                result="Success",
            )

        with patch("src.claude.sdk_integration.query", side_effect=mock_query):
            response = await sdk_manager.execute_command(
                prompt="Test prompt",
                working_directory=Path("/test"),
                stream_callback=stream_callback,
            )

        # Verify streaming was called
        assert len(stream_updates) > 0
        assert any(update.type == "assistant" for update in stream_updates)

    async def test_execute_command_timeout(self, sdk_manager):
        """Test command execution timeout."""
        import asyncio

        # Mock a hanging operation - return async generator that never yields
        async def mock_hanging_query(prompt, options):
            await asyncio.sleep(5)  # This should timeout (config has 2s timeout)
            yield  # This will never be reached

        from src.claude.exceptions import ClaudeTimeoutError

        with patch("src.claude.sdk_integration.query", side_effect=mock_hanging_query):
            with pytest.raises(ClaudeTimeoutError):
                await sdk_manager.execute_command(
                    prompt="Test prompt",
                    working_directory=Path("/test"),
                )

    async def test_session_management(self, sdk_manager):
        """Test session management."""
        from claude_code_sdk.types import AssistantMessage

        session_id = "test-session"
        messages = [AssistantMessage(content="test")]

        # Update session
        sdk_manager._update_session(session_id, messages)

        # Verify session was created
        assert session_id in sdk_manager.active_sessions
        session_data = sdk_manager.active_sessions[session_id]
        assert session_data["messages"] == messages

    async def test_kill_all_processes(self, sdk_manager):
        """Test killing all processes (clearing sessions)."""
        # Add some active sessions
        sdk_manager.active_sessions["session1"] = {"test": "data"}
        sdk_manager.active_sessions["session2"] = {"test": "data2"}

        assert len(sdk_manager.active_sessions) == 2

        # Kill all processes
        await sdk_manager.kill_all_processes()

        # Sessions should be cleared
        assert len(sdk_manager.active_sessions) == 0

    def test_get_active_process_count(self, sdk_manager):
        """Test getting active process count."""
        assert sdk_manager.get_active_process_count() == 0

        # Add sessions
        sdk_manager.active_sessions["session1"] = {"test": "data"}
        sdk_manager.active_sessions["session2"] = {"test": "data2"}

        assert sdk_manager.get_active_process_count() == 2
