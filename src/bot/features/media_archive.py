"""Persistent archive for Telegram-uploaded media.

Images, documents and voice messages received via Telegram are saved under
``approved_directory/<media_archive_dir>/`` (default ``.media.telegram/``)
in a flat per-type layout:

    <archive>/images/<chat_id>_<message_id>.<ext>
    <archive>/pdfs/<chat_id>_<message_id>_<original_name>
    <archive>/documents/<chat_id>_<message_id>_<original_name>
    <archive>/audios/<timestamp>/received.<ext>
    <archive>/audios/<timestamp>/received.txt
    <archive>/audios/<timestamp>/sent.<ext>
    <archive>/audios/<timestamp>/sent.txt

The archive is the staging area. Files referenced from Obsidian notes via
``![[name]]`` are then promoted into ``5-Attachments/<type>/YYYY-MM/`` by
the attachment_promoter at end-of-turn (see ``attachment_promoter.py``).

Both copies coexist after promotion: the archive remains as the raw
record (gitignored), and the promoted copy is what Obsidian renders from
the vault attachments folder.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import structlog

from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class MediaArchive:
    """Save Telegram uploads to the on-disk archive."""

    def __init__(self, config: Settings) -> None:
        self.config = config

    @property
    def root(self) -> Path:
        """Root directory of the archive (created on first access)."""
        path = Path(self.config.approved_directory) / self.config.media_archive_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_image(
        self,
        image_bytes: bytes,
        chat_id: int,
        message_id: int,
        ext: str = "png",
    ) -> Path:
        """Save image bytes; returns the absolute path."""
        target_dir = self.root / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = ext.lstrip(".") or "png"
        path = target_dir / f"{chat_id}_{message_id}.{ext}"
        path.write_bytes(bytes(image_bytes))
        logger.info(
            "Saved image to media archive",
            path=str(path),
            size_bytes=len(image_bytes),
        )
        return path

    def save_document(
        self,
        file_bytes: bytes,
        chat_id: int,
        message_id: int,
        original_filename: str,
        kind: str = "documents",
    ) -> Path:
        """Save a document (PDF or other). ``kind`` selects the subdir."""
        target_dir = self.root / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _sanitize_filename(original_filename) or "file"
        path = target_dir / f"{chat_id}_{message_id}_{safe_name}"
        path.write_bytes(bytes(file_bytes))
        logger.info(
            "Saved document to media archive",
            path=str(path),
            size_bytes=len(file_bytes),
            original_name=original_filename,
        )
        return path

    def make_audio_pair_dir(self) -> Path:
        """Create a fresh paired-audio directory for a voice exchange."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = self.root / "audios" / timestamp
        # If two voice messages land in the same second, append a counter
        suffix = 0
        path = target_dir
        while path.exists():
            suffix += 1
            path = target_dir.parent / f"{timestamp}_{suffix}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_received_audio(
        self,
        audio_bytes: bytes,
        pair_dir: Path,
        ext: str = "ogg",
        transcript: Optional[str] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """Persist the received voice audio + optional transcript inside ``pair_dir``."""
        ext = ext.lstrip(".") or "ogg"
        audio_path = pair_dir / f"received.{ext}"
        audio_path.write_bytes(bytes(audio_bytes))
        transcript_path: Optional[Path] = None
        if transcript:
            transcript_path = pair_dir / "received.txt"
            transcript_path.write_text(transcript, encoding="utf-8")
        logger.info(
            "Saved received audio to media archive",
            audio_path=str(audio_path),
            transcript_path=str(transcript_path) if transcript_path else None,
        )
        return audio_path, transcript_path


def _sanitize_filename(name: str) -> str:
    """Strip path separators / nullbytes / control chars from a Telegram filename.

    Keeps the original extension for downstream MIME handling. Falls back to
    a sanitized basename to avoid traversal.
    """
    if not name:
        return ""
    base = Path(name).name  # drops any directory component
    out = []
    for ch in base:
        if ch == "/" or ch == "\\" or ord(ch) < 32:
            continue
        out.append(ch)
    return "".join(out)[:255]
