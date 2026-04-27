"""Tests for MediaGroupBuffer — debounced batching of Telegram media-group items."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.features.media_group_buffer import BufferedFile, MediaGroupBuffer


def _make_update(chat_id: int, message_id: int, media_group_id):
    """Stand-in Update; only the attributes the buffer touches matter."""
    progress = SimpleNamespace()
    message = MagicMock()
    message.chat_id = chat_id
    message.message_id = message_id
    message.media_group_id = media_group_id
    message.reply_text = AsyncMock(return_value=progress)
    update = SimpleNamespace(message=message)
    return update, progress


def _make_file(name: str, caption: str = None) -> BufferedFile:
    return BufferedFile(
        relative_path=f".media.telegram/images/{name}",
        file_type="image",
        caption=caption,
        message_id=hash(name) & 0xFFFF,
    )


@pytest.mark.asyncio
async def test_two_items_same_group_flush_once() -> None:
    """Two items sharing media_group_id produce a single flush call."""
    flushed = []

    async def on_flush(update, context, files, progress_msg):
        flushed.append((files, progress_msg))

    buf = MediaGroupBuffer(window_seconds=0.05, max_files=10, on_flush=on_flush)
    update_a, progress = _make_update(chat_id=1, message_id=10, media_group_id="g1")
    update_b, _ = _make_update(chat_id=1, message_id=11, media_group_id="g1")
    context = MagicMock()

    await buf.add(update_a, context, _make_file("a.png"))
    await buf.add(update_b, context, _make_file("b.png", caption="hi"))

    # Wait past the window so the debounced task fires.
    await asyncio.sleep(0.2)

    assert len(flushed) == 1
    files, returned_progress = flushed[0]
    assert [f.relative_path for f in files] == [
        ".media.telegram/images/a.png",
        ".media.telegram/images/b.png",
    ]
    assert returned_progress is progress
    # Only the FIRST item posts the "Working..." progress message.
    assert update_a.message.reply_text.await_count == 1
    assert update_b.message.reply_text.await_count == 0


@pytest.mark.asyncio
async def test_separate_groups_flush_separately() -> None:
    """Items with different media_group_id produce independent flushes."""
    flushed = []

    async def on_flush(update, context, files, progress_msg):
        flushed.append(files)

    buf = MediaGroupBuffer(window_seconds=0.05, max_files=10, on_flush=on_flush)
    update_a, _ = _make_update(chat_id=1, message_id=10, media_group_id="g1")
    update_b, _ = _make_update(chat_id=1, message_id=11, media_group_id="g2")
    context = MagicMock()

    await buf.add(update_a, context, _make_file("a.png"))
    await buf.add(update_b, context, _make_file("b.png"))
    await asyncio.sleep(0.2)

    assert len(flushed) == 2
    assert {f[0].relative_path for f in flushed} == {
        ".media.telegram/images/a.png",
        ".media.telegram/images/b.png",
    }


@pytest.mark.asyncio
async def test_max_files_force_flush() -> None:
    """When max_files is hit, the buffer flushes immediately."""
    flushed = []

    async def on_flush(update, context, files, progress_msg):
        flushed.append(len(files))

    buf = MediaGroupBuffer(window_seconds=10.0, max_files=3, on_flush=on_flush)
    context = MagicMock()
    for i in range(3):
        update, _ = _make_update(chat_id=1, message_id=100 + i, media_group_id="big")
        await buf.add(update, context, _make_file(f"f{i}.png"))

    # The window is 10s but the cap is 3, so it should flush near-immediately.
    await asyncio.sleep(0.1)
    assert flushed == [3]


@pytest.mark.asyncio
async def test_debounce_resets_on_new_arrival() -> None:
    """The flush waits *window* after the LAST arrival, not the first."""
    flushed = []

    async def on_flush(update, context, files, progress_msg):
        flushed.append(len(files))

    buf = MediaGroupBuffer(window_seconds=0.15, max_files=10, on_flush=on_flush)
    context = MagicMock()
    update_a, _ = _make_update(chat_id=1, message_id=10, media_group_id="g")
    update_b, _ = _make_update(chat_id=1, message_id=11, media_group_id="g")

    await buf.add(update_a, context, _make_file("a.png"))
    await asyncio.sleep(0.05)  # < window; no flush yet
    assert flushed == []

    await buf.add(update_b, context, _make_file("b.png"))
    await asyncio.sleep(0.05)  # cumulative time is past first window but not second
    assert flushed == []

    await asyncio.sleep(0.2)  # past second window
    assert flushed == [2]


@pytest.mark.asyncio
async def test_add_without_media_group_id_raises() -> None:
    """Calling add() with media_group_id=None is a programming error."""
    buf = MediaGroupBuffer(window_seconds=0.1, max_files=10, on_flush=AsyncMock())
    update, _ = _make_update(chat_id=1, message_id=10, media_group_id=None)
    with pytest.raises(ValueError, match="media_group_id"):
        await buf.add(update, MagicMock(), _make_file("x.png"))
