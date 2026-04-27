"""Promote archived media into the Obsidian attachments folder.

After Claude finishes a turn, this module scans .md files Claude touched
during the turn for ``![[name]]`` references whose target lives in the
media archive (``.media.telegram/`` by default). Each matched file is
copied into ``5-Attachments/<type>/YYYY-MM/<name>``. Both copies remain:

  * The archive copy stays (gitignored, raw record).
  * The attachments copy becomes what Obsidian renders.

Promotion is idempotent: if the attachment file already exists with the
same content, the copy is skipped.

The promoter does NOT rewrite wiki-links. Obsidian's link-resolver finds
attachments by name across the vault; once both copies exist with the
same name, the attachments folder takes precedence in vault-wide
rendering (configured via Obsidian's "Files & links" settings).
"""

import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import structlog

from src.config.settings import Settings

logger = structlog.get_logger(__name__)

# ![[target]] or ![[target|alias]] — Obsidian embed syntax. Path components
# are kept as-is so we can match against the archive's on-disk layout.
_EMBED_RE = re.compile(r"!\[\[([^\]\|]+?)(?:\|[^\]]*)?\]\]")

_TYPE_MAP = {
    # archive subdir → attachments subdir
    "images": "images",
    "pdfs": "pdfs",
    "documents": "documents",
    "audios": "audio",
}

_AUDIO_EXTENSIONS = {".ogg", ".oga", ".opus", ".mp3", ".wav", ".m4a"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
_PDF_EXTENSIONS = {".pdf"}


def collect_modified_md_paths(tool_log: Iterable[dict]) -> List[Path]:
    """Extract .md file paths from Edit / Write / MultiEdit tool calls."""
    seen: set[str] = set()
    out: List[Path] = []
    for entry in tool_log:
        name = entry.get("name", "")
        if name not in {"Edit", "Write", "MultiEdit"}:
            continue
        # input may be under .input, .tool_input, or .detail dict
        candidates = []
        for key in ("input", "tool_input", "detail"):
            val = entry.get(key)
            if isinstance(val, dict):
                candidates.append(val)
        for inp in candidates:
            file_path = inp.get("file_path") or inp.get("path")
            if isinstance(file_path, str) and file_path.endswith(".md"):
                if file_path not in seen:
                    seen.add(file_path)
                    out.append(Path(file_path))
    return out


def promote_referenced_media(
    config: Settings,
    modified_md_paths: Iterable[Path],
) -> List[Path]:
    """Scan ``modified_md_paths`` for embed references into the media archive.

    For every match found in the archive on-disk, copy it to the attachments
    folder under the appropriate type/YYYY-MM subdir. Returns the list of
    promoted destination paths.
    """
    if not config.attachment_promote_enabled:
        return []

    approved = Path(config.approved_directory).resolve()
    archive_root = (approved / config.media_archive_dir).resolve()
    attachments_root = (approved / config.attachment_dir).resolve()
    if not archive_root.exists():
        return []

    # Build a name → source-path map of archive contents (one-time scan).
    by_name: dict[str, Path] = {}
    for p in archive_root.rglob("*"):
        if p.is_file() and p.name not in by_name:
            by_name[p.name] = p

    if not by_name:
        return []

    promoted: List[Path] = []
    for md_path in modified_md_paths:
        try:
            md_path = Path(md_path)
            if not md_path.is_absolute():
                md_path = (approved / md_path).resolve()
            if not md_path.exists() or not md_path.is_file():
                continue
            text = md_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(
                "Failed reading modified markdown for promotion scan",
                path=str(md_path),
                error=str(e),
            )
            continue

        for match in _EMBED_RE.finditer(text):
            target = match.group(1).strip()
            # Use just the basename for archive lookup — Obsidian's resolver
            # is name-based and we save with unique chat_id_message_id names.
            base = Path(target).name
            source = by_name.get(base)
            if source is None:
                continue
            dest = _resolve_destination(source, archive_root, attachments_root)
            if dest is None:
                continue
            try:
                _copy_if_different(source, dest)
                if dest not in promoted:
                    promoted.append(dest)
            except Exception as e:
                logger.warning(
                    "Failed promoting archived media to attachments",
                    source=str(source),
                    dest=str(dest),
                    error=str(e),
                )

    if promoted:
        logger.info(
            "Promoted archived media to attachments folder",
            count=len(promoted),
            paths=[str(p) for p in promoted],
        )

    return promoted


def _resolve_destination(
    source: Path,
    archive_root: Path,
    attachments_root: Path,
) -> Optional[Path]:
    """Map archive/<kind>/.../<name> to attachments/<kind>/YYYY-MM/<name>."""
    try:
        rel = source.relative_to(archive_root)
    except ValueError:
        return None

    # First component is the kind (images, pdfs, audios, documents).
    parts = rel.parts
    if len(parts) < 2:
        return None
    kind = parts[0]
    out_kind = _TYPE_MAP.get(kind) or _infer_kind_from_extension(source.suffix)
    if out_kind is None:
        return None

    yyyy_mm = datetime.now().strftime("%Y-%m")
    dest_dir = attachments_root / out_kind / yyyy_mm
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / source.name


def _infer_kind_from_extension(ext: str) -> Optional[str]:
    """Fallback: map a file extension to an attachments-folder kind."""
    ext = ext.lower()
    if ext in _IMAGE_EXTENSIONS:
        return "images"
    if ext in _PDF_EXTENSIONS:
        return "pdfs"
    if ext in _AUDIO_EXTENSIONS:
        return "audio"
    return None


def _copy_if_different(source: Path, dest: Path) -> None:
    """Copy ``source`` to ``dest`` unless an identical file is already there."""
    if dest.exists() and _file_digest(dest) == _file_digest(source):
        return
    shutil.copy2(source, dest)


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
