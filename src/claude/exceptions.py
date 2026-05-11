"""Claude-specific exceptions."""


class ClaudeError(Exception):
    """Base Claude error."""


class ClaudeTimeoutError(ClaudeError):
    """Operation timed out."""


class ClaudeProcessError(ClaudeError):
    """Process execution failed."""


class ClaudeParsingError(ClaudeError):
    """Failed to parse output."""


class ClaudeSessionOverflowError(ClaudeParsingError):
    """Session context exceeded the Claude CLI JSON buffer limit (1 MB).

    Callers should clear the session and prompt the user to retry.
    """


class ClaudeSessionError(ClaudeError):
    """Session management error."""


class ClaudeMCPError(ClaudeError):
    """MCP server connection or configuration error."""

    def __init__(self, message: str, server_name: str = None):
        super().__init__(message)
        self.server_name = server_name
