"""Discover Claude Code skills from a project's .claude/skills/ directory.

Scans SKILL.md frontmatter for name, description, and argument-hint.
Returns a dict of skill name -> DiscoveredSkill. Project-agnostic —
works with any Claude Code project, not just a specific vault.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import structlog
import yaml

logger = structlog.get_logger(__name__)

# Commands that must not be overridden by skills
_BUILTIN_COMMANDS = frozenset({
    "start", "new", "status", "verbose", "repo", "tts",
    "help", "sync_threads",
})


@dataclass
class DiscoveredSkill:
    name: str
    description: str
    argument_hint: Optional[str] = None


def discover_skills(project_dir: Path) -> Dict[str, DiscoveredSkill]:
    """Scan {project_dir}/.claude/skills/*/SKILL.md for skill definitions.

    Returns a dict mapping command name (lowercase, no /) to DiscoveredSkill.
    Skips skills whose names clash with built-in bot commands.
    """
    skills_dir = project_dir / ".claude" / "skills"
    if not skills_dir.is_dir():
        return {}

    discovered: Dict[str, DiscoveredSkill] = {}

    for skill_md in skills_dir.glob("*/SKILL.md"):
        try:
            text = skill_md.read_text(encoding="utf-8")

            # Extract YAML frontmatter between --- delimiters
            match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
            if not match:
                continue

            meta = yaml.safe_load(match.group(1))
            if not isinstance(meta, dict) or "name" not in meta:
                continue

            # Normalize name to a valid Telegram command
            raw_name = str(meta["name"]).strip()
            cmd_name = raw_name.lower().replace(" ", "_").replace("-", "_")

            # Skip built-in command conflicts
            if cmd_name in _BUILTIN_COMMANDS:
                logger.debug("Skipping skill (conflicts with built-in)", skill=cmd_name)
                continue

            description = str(meta.get("description", "")).strip()
            if not description:
                description = f"Run /{cmd_name} skill"

            discovered[cmd_name] = DiscoveredSkill(
                name=cmd_name,
                description=description[:256],
                argument_hint=meta.get("argument-hint"),
            )
        except Exception as e:
            logger.warning(
                "Failed to parse skill frontmatter",
                path=str(skill_md),
                error=str(e),
            )

    if discovered:
        logger.info("Skills discovered", count=len(discovered), names=list(discovered.keys()))

    return discovered
