"""
Handle voice messages and audio files via OpenAI Whisper transcription.

Supports:
- Telegram VOICE messages (microphone button, OGG Opus)
- Telegram AUDIO messages (forwarded audio files, MP3/M4A/etc.)

Follows the same save-to-.tmp/ pattern as image_handler.py.
"""

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import structlog
from telegram import Audio, Voice

from src.config.settings import Settings

logger = structlog.get_logger(__name__)

# Whisper API hard limit is 25 MB; Telegram bot API limit is 20 MB for downloads.
_MAX_AUDIO_BYTES = 20 * 1024 * 1024


@dataclass
class ProcessedVoice:
    """Result of voice transcription."""

    prompt: str
    transcript: str
    duration_seconds: Optional[int]
    size_bytes: int
    audio_format: str
    saved_path: Optional[str] = None


class VoiceHandler:
    """Transcribe voice/audio messages via OpenAI Whisper and forward text to Claude."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self._client: Any = None  # lazy-init openai.AsyncOpenAI

    def _get_client(self) -> Any:
        """Lazy-initialise OpenAI async client on first use."""
        if self._client is None:
            import openai  # deferred — bot starts cleanly without the package

            api_key = self.config.openai_api_key_str
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Cannot transcribe voice messages."
                )
            self._client = openai.AsyncOpenAI(api_key=api_key)
        return self._client

    async def process_voice(
        self,
        attachment: Union[Voice, Audio],
        caption: Optional[str] = None,
        is_voice: bool = True,
    ) -> ProcessedVoice:
        """Download, save, transcribe, and return a Claude-ready prompt.

        Args:
            attachment: Telegram Voice or Audio object.
            caption: Optional user caption on the message.
            is_voice: True for microphone recordings, False for forwarded audio files.

        Returns:
            ProcessedVoice with the transcript embedded in the prompt.

        Raises:
            ValueError: If the audio file exceeds the size limit.
        """
        # 1. Size guard
        file_size = getattr(attachment, "file_size", 0) or 0
        if file_size > _MAX_AUDIO_BYTES:
            raise ValueError(
                f"Audio too large ({file_size / 1024 / 1024:.1f} MB). "
                f"Maximum is 20 MB."
            )

        # 2. Duration and format
        duration = getattr(attachment, "duration", None)
        mime = getattr(attachment, "mime_type", None) or "audio/ogg"
        audio_format = _mime_to_ext(mime)

        # 3. Download bytes from Telegram
        tg_file = await attachment.get_file()
        audio_bytes = await tg_file.download_as_bytearray()

        # 4. Save to .tmp/ (inside APPROVED_DIRECTORY — within Claude's sandbox)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        filename = f"voice_{timestamp}.{audio_format}"
        save_dir = Path(self.config.approved_directory) / ".tmp"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        save_path.write_bytes(audio_bytes)

        logger.info(
            "Audio saved",
            path=str(save_path),
            size_bytes=len(audio_bytes),
            duration=duration,
            format=audio_format,
        )

        # 5. Transcribe via Whisper API
        client = self._get_client()
        audio_buffer = io.BytesIO(bytes(audio_bytes))
        audio_buffer.name = filename  # Whisper uses the extension for format detection

        whisper_kwargs: dict[str, Any] = {
            "model": "whisper-1",
            "file": audio_buffer,
        }
        if self.config.whisper_language:
            whisper_kwargs["language"] = self.config.whisper_language

        transcription = await client.audio.transcriptions.create(**whisper_kwargs)
        transcript = transcription.text.strip()

        logger.info(
            "Voice transcribed",
            duration=duration,
            size_bytes=len(audio_bytes),
            transcript_length=len(transcript),
        )

        # 6. Build prompt for Claude
        label = "voice message" if is_voice else "audio file"
        duration_str = f" ({duration}s)" if duration else ""

        prompt = (
            f"The user sent a {label}{duration_str}. "
            f"I've transcribed it via Whisper:\n\n{transcript}"
        )
        if caption:
            prompt += f"\n\nCaption from user: {caption}"

        # When TTS is enabled, instruct Claude to reply in Portuguese
        # so the spoken reply sounds natural
        if self.config.whisper_language and self.config.tts_enabled:
            prompt += (
                "\n\nIMPORTANT: Your reply will be converted to speech. "
                "Respond in European Portuguese (pt-PT) — concise, conversational, "
                "no markdown, no code blocks, no bullet points."
            )

        return ProcessedVoice(
            prompt=prompt,
            transcript=transcript,
            duration_seconds=duration,
            size_bytes=len(audio_bytes),
            audio_format=audio_format,
            saved_path=str(save_path),
        )


def _mime_to_ext(mime: str) -> str:
    """Map MIME type to file extension for Whisper."""
    mapping = {
        "audio/ogg": "ogg",
        "audio/mpeg": "mp3",
        "audio/mp4": "mp4",
        "audio/x-m4a": "m4a",
        "audio/wav": "wav",
        "audio/webm": "webm",
        "audio/flac": "flac",
    }
    return mapping.get(mime.lower(), "ogg")
