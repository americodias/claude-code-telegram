"""Tests for the MediaArchive helper."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from src.bot.features.media_archive import MediaArchive
from src.config.settings import Settings


def _make_settings(approved: Path) -> Settings:
    return Settings(
        telegram_bot_token=SecretStr("dummy"),
        telegram_bot_username="dummy",
        approved_directory=approved,
        media_archive_dir=".media.telegram",
    )


@pytest.fixture()
def archive(tmp_path: Path) -> MediaArchive:
    return MediaArchive(_make_settings(tmp_path))


def test_save_image_writes_to_images_subdir(archive: MediaArchive, tmp_path: Path) -> None:
    path = archive.save_image(b"PNGDATA", chat_id=42, message_id=99, ext="png")
    assert path.parent == tmp_path / ".media.telegram" / "images"
    assert path.name == "42_99.png"
    assert path.read_bytes() == b"PNGDATA"


def test_save_image_strips_leading_dot(archive: MediaArchive) -> None:
    path = archive.save_image(b"x", chat_id=1, message_id=2, ext=".jpg")
    assert path.name == "1_2.jpg"


def test_save_document_uses_kind_subdir(archive: MediaArchive, tmp_path: Path) -> None:
    path = archive.save_document(
        b"%PDF-1.4",
        chat_id=10,
        message_id=20,
        original_filename="report.pdf",
        kind="pdfs",
    )
    assert path.parent == tmp_path / ".media.telegram" / "pdfs"
    assert path.name == "10_20_report.pdf"
    assert path.read_bytes() == b"%PDF-1.4"


def test_save_document_sanitizes_traversal(archive: MediaArchive) -> None:
    path = archive.save_document(
        b"data",
        chat_id=1,
        message_id=2,
        original_filename="../../escape.txt",
        kind="documents",
    )
    # No traversal escape; only basename retained.
    assert "escape.txt" in path.name
    assert ".." not in path.name


def test_make_audio_pair_dir_unique_when_collision(archive: MediaArchive) -> None:
    a = archive.make_audio_pair_dir()
    b = archive.make_audio_pair_dir()
    assert a != b
    assert a.exists() and b.exists()


def test_save_received_audio_writes_pair(archive: MediaArchive) -> None:
    pair_dir = archive.make_audio_pair_dir()
    audio_path, transcript_path = archive.save_received_audio(
        b"OGG_BYTES",
        pair_dir=pair_dir,
        ext="ogg",
        transcript="hello world",
    )
    assert audio_path.name == "received.ogg"
    assert audio_path.read_bytes() == b"OGG_BYTES"
    assert transcript_path is not None
    assert transcript_path.read_text(encoding="utf-8") == "hello world"


def test_save_received_audio_no_transcript(archive: MediaArchive) -> None:
    pair_dir = archive.make_audio_pair_dir()
    _, transcript_path = archive.save_received_audio(
        b"x", pair_dir=pair_dir, transcript=None
    )
    assert transcript_path is None
