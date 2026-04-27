"""Debounced buffer for Telegram media-group items.

When the user attaches multiple files to a single Telegram message, Telegram
delivers each item as a separate ``Update`` event sharing the same
``media_group_id``. The default agentic handlers fire one Claude session per
item and post one reply per item. This module collects the items and runs
Claude **once** over all of them.

Design notes:

* Keyed on ``(chat_id, media_group_id)`` because Telegram's media_group_id is
  unique only per chat / per "send".
* Pure debounce: the flush task is rescheduled on each new arrival so the bot
  always waits ``window_seconds`` after the *last* item before firing.
* Per-key ``asyncio.Lock`` guards the entry while it's being mutated, so a
  late item arriving exactly when the flush starts either joins the same
  flush (if before lock release) or lands in a fresh entry (if after).
* If items keep coming past ``max_files`` we force-flush early — Telegram
  itself caps a media group at 10 items, so this is a safety belt.
* On flush we delegate the actual Claude call + reply to a callable supplied
  by the orchestrator. The buffer never imports the orchestrator (avoids
  circular imports) and never touches Claude directly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import structlog
from telegram import Update
from telegram.ext import ContextTypes

logger = structlog.get_logger()


@dataclass
class BufferedFile:
    """One file in a media-group buffer entry."""

    relative_path: str
    file_type: str  # "image" or "pdf" / "document"
    caption: Optional[str]
    message_id: int


@dataclass
class _Entry:
    """Per-media-group buffered state."""

    files: List[BufferedFile] = field(default_factory=list)
    first_update: Optional[Update] = None
    context: Optional[ContextTypes.DEFAULT_TYPE] = None
    progress_msg: Any = None
    pending_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Async callback signature for the flush. Receives the orchestrator's update,
# context, list of buffered files, and the progress message to delete on
# completion. Return value ignored.
FlushCallback = Callable[
    [Update, ContextTypes.DEFAULT_TYPE, List[BufferedFile], Any],
    Awaitable[None],
]


class MediaGroupBuffer:
    """Debounced buffer keyed on ``(chat_id, media_group_id)``."""

    def __init__(
        self,
        window_seconds: float,
        max_files: int,
        on_flush: FlushCallback,
    ) -> None:
        self.window_seconds = window_seconds
        self.max_files = max_files
        self._on_flush = on_flush
        self._entries: Dict[Tuple[int, str], _Entry] = {}
        self._dict_lock = asyncio.Lock()

    async def add(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        buffered: BufferedFile,
        progress_text: str = "Working...",
    ) -> None:
        """Append *buffered* to its media group's entry and (re)schedule flush.

        Posts a single ``progress_text`` message on first arrival. Caller is
        responsible for downloading the file before calling this; *buffered*
        already references the saved-on-disk path.
        """
        chat_id = update.message.chat_id
        group_id = update.message.media_group_id
        if group_id is None:
            raise ValueError("MediaGroupBuffer.add requires a media_group_id")
        key = (chat_id, str(group_id))

        # Get-or-create the entry. The dict_lock protects the dict mutation.
        async with self._dict_lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry()
                self._entries[key] = entry

        async with entry.lock:
            entry.files.append(buffered)
            if entry.first_update is None:
                entry.first_update = update
                entry.context = context
                entry.progress_msg = await update.message.reply_text(progress_text)

            # Cap-driven force-flush: schedule an immediate flush.
            if len(entry.files) >= self.max_files:
                if entry.pending_task and not entry.pending_task.done():
                    entry.pending_task.cancel()
                entry.pending_task = asyncio.create_task(
                    self._scheduled_flush(key, delay=0.0)
                )
                return

            # Standard debounce: cancel + reschedule.
            if entry.pending_task and not entry.pending_task.done():
                entry.pending_task.cancel()
            entry.pending_task = asyncio.create_task(
                self._scheduled_flush(key, delay=self.window_seconds)
            )

    async def _scheduled_flush(self, key: Tuple[int, str], delay: float) -> None:
        """Wait *delay* seconds, then flush the entry under *key*."""
        try:
            if delay > 0:
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

        # Pop the entry under the dict_lock so a late arrival lands in a new
        # entry instead of mutating one we're about to flush.
        async with self._dict_lock:
            entry = self._entries.pop(key, None)
        if entry is None:
            return

        async with entry.lock:
            files = list(entry.files)
            first_update = entry.first_update
            context = entry.context
            progress_msg = entry.progress_msg

        if not files or first_update is None or context is None:
            return

        try:
            await self._on_flush(first_update, context, files, progress_msg)
        except Exception:
            logger.exception(
                "Media-group flush callback failed",
                chat_id=key[0],
                media_group_id=key[1],
                file_count=len(files),
            )
