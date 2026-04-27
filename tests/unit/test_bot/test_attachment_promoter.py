"""Tests for the attachment promoter."""

from datetime import datetime
from pathlib import Path

from pydantic import SecretStr

from src.bot.features.attachment_promoter import (
    collect_modified_md_paths,
    promote_referenced_media,
)
from src.config.settings import Settings


def _make_settings(approved: Path) -> Settings:
    return Settings(
        telegram_bot_token=SecretStr("dummy"),
        telegram_bot_username="dummy",
        approved_directory=approved,
        media_archive_dir=".media.telegram",
        attachment_dir="5-Attachments",
        attachment_promote_enabled=True,
    )


def test_collect_modified_md_paths_filters_to_md_writes() -> None:
    tool_log = [
        {"name": "Edit", "input": {"file_path": "/v/Daily/2026-04-27.md"}},
        {"name": "Read", "input": {"file_path": "/v/Daily/2026-04-27.md"}},
        {"name": "Write", "input": {"file_path": "/v/0-Inbox/note.md"}},
        {"name": "Bash", "input": {"command": "ls"}},
        {"name": "Edit", "input": {"file_path": "/v/scripts/foo.py"}},
        {"name": "MultiEdit", "input": {"file_path": "/v/Daily/2026-04-27.md"}},
    ]
    paths = collect_modified_md_paths(tool_log)
    assert paths == [
        Path("/v/Daily/2026-04-27.md"),
        Path("/v/0-Inbox/note.md"),
    ]


def test_collect_modified_md_paths_handles_alt_input_keys() -> None:
    tool_log = [
        {"name": "Edit", "tool_input": {"file_path": "/v/a.md"}},
        {"name": "Edit", "detail": {"file_path": "/v/b.md"}},
    ]
    assert collect_modified_md_paths(tool_log) == [Path("/v/a.md"), Path("/v/b.md")]


def test_promote_copies_archived_image_when_referenced(tmp_path: Path) -> None:
    archive = tmp_path / ".media.telegram" / "images"
    archive.mkdir(parents=True)
    src = archive / "100_42.png"
    src.write_bytes(b"FAKEPNG")

    note = tmp_path / "Daily" / "2026-04-27.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Today\n\n"
        "Reviewed thermostat photo: ![[100_42.png|thermostat]]\n",
        encoding="utf-8",
    )

    settings = _make_settings(tmp_path)
    promoted = promote_referenced_media(settings, [note])

    assert len(promoted) == 1
    yyyy_mm = datetime.now().strftime("%Y-%m")
    expected = tmp_path / "5-Attachments" / "images" / yyyy_mm / "100_42.png"
    assert expected.exists()
    assert expected.read_bytes() == b"FAKEPNG"


def test_promote_skips_files_not_in_archive(tmp_path: Path) -> None:
    note = tmp_path / "n.md"
    note.write_text("ref ![[unknown.png]]", encoding="utf-8")
    (tmp_path / ".media.telegram" / "images").mkdir(parents=True)
    settings = _make_settings(tmp_path)
    assert promote_referenced_media(settings, [note]) == []


def test_promote_disabled_returns_empty(tmp_path: Path) -> None:
    archive = tmp_path / ".media.telegram" / "images"
    archive.mkdir(parents=True)
    (archive / "x.png").write_bytes(b"x")
    note = tmp_path / "n.md"
    note.write_text("![[x.png]]", encoding="utf-8")
    settings = Settings(
        telegram_bot_token=SecretStr("dummy"),
        telegram_bot_username="dummy",
        approved_directory=tmp_path,
        media_archive_dir=".media.telegram",
        attachment_dir="5-Attachments",
        attachment_promote_enabled=False,
    )
    assert promote_referenced_media(settings, [note]) == []


def test_promote_idempotent_on_second_run(tmp_path: Path) -> None:
    archive = tmp_path / ".media.telegram" / "images"
    archive.mkdir(parents=True)
    (archive / "doc.png").write_bytes(b"D")
    note = tmp_path / "n.md"
    note.write_text("![[doc.png]]", encoding="utf-8")
    settings = _make_settings(tmp_path)

    first = promote_referenced_media(settings, [note])
    second = promote_referenced_media(settings, [note])
    assert first == second
    assert first[0].read_bytes() == b"D"


def test_promote_handles_pdf_kind(tmp_path: Path) -> None:
    pdfs = tmp_path / ".media.telegram" / "pdfs"
    pdfs.mkdir(parents=True)
    (pdfs / "5_9_report.pdf").write_bytes(b"%PDF")

    note = tmp_path / "n.md"
    note.write_text("![[5_9_report.pdf]]", encoding="utf-8")

    settings = _make_settings(tmp_path)
    promoted = promote_referenced_media(settings, [note])
    assert len(promoted) == 1
    assert "/pdfs/" in str(promoted[0])
