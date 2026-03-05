"""
Convert text to speech via OpenAI, ElevenLabs, or Piper (Wyoming) TTS.

Provider is selected via TTS_PROVIDER setting (openai | elevenlabs | piper).
When pair_dir is provided (voice conversation), saves to the same timestamped
folder as the received audio. Otherwise saves to .media.telegram/audios/.
"""

import asyncio
import re
import struct
import wave
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import structlog

from src.config.settings import Settings

logger = structlog.get_logger(__name__)

_OPENAI_CHAR_LIMIT = 4096
_ELEVENLABS_CHAR_LIMIT = 5000


@dataclass
class TtsResult:
    """Result of a TTS synthesis."""

    audio_path: str  # absolute path to audio file
    transcript_path: str  # absolute path to transcript .txt
    chunks: int  # how many API calls were made


class TtsHandler:
    """Synthesise speech from text via configurable TTS provider."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self._openai_client: Any = None
        self._elevenlabs_client: Any = None

    def _get_openai_client(self) -> Any:
        """Lazy-initialise OpenAI async client on first use."""
        if self._openai_client is None:
            import openai

            api_key = self.config.openai_api_key_str
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set. Cannot synthesise speech.")
            self._openai_client = openai.AsyncOpenAI(api_key=api_key)
        return self._openai_client

    def _get_elevenlabs_client(self) -> Any:
        """Lazy-initialise ElevenLabs async client on first use."""
        if self._elevenlabs_client is None:
            from elevenlabs.client import AsyncElevenLabs

            api_key = self.config.elevenlabs_api_key_str
            if not api_key:
                raise RuntimeError("ELEVENLABS_API_KEY is not set. Cannot synthesise speech.")
            self._elevenlabs_client = AsyncElevenLabs(api_key=api_key)
        return self._elevenlabs_client

    async def synthesise(self, text: str, pair_dir: str | None = None) -> TtsResult:
        """Convert text to an audio file. Returns path to the file.

        Args:
            text: The text to synthesise (may contain markup — stripped internally).
            pair_dir: If provided, save into this directory as sent.* (paired with received.*).
        """
        plain = _strip_markup(text)
        if not plain:
            raise ValueError("No speakable text after stripping markup.")

        provider = self.config.tts_provider.lower()
        if provider == "elevenlabs":
            return await self._synthesise_elevenlabs(plain, pair_dir)
        if provider == "piper":
            return await self._synthesise_piper(plain, pair_dir)
        return await self._synthesise_openai(plain, pair_dir)

    async def _synthesise_openai(self, text: str, pair_dir: str | None = None) -> TtsResult:
        """Synthesise via OpenAI TTS API."""
        chunks = _split_text(text, _OPENAI_CHAR_LIMIT)
        client = self._get_openai_client()
        audio_parts: list[bytes] = []

        for chunk in chunks:
            response = await client.audio.speech.create(
                model=self.config.tts_model,
                voice=self.config.tts_voice,
                input=chunk,
                response_format="opus",
            )
            audio_parts.append(await response.aread())

        audio_bytes = b"".join(audio_parts)
        save_path, transcript_path = self._save_audio(audio_bytes, "ogg", text, pair_dir)

        logger.info(
            "TTS audio saved (openai)",
            path=str(save_path),
            size_bytes=len(audio_bytes),
            chunks=len(chunks),
        )
        return TtsResult(audio_path=str(save_path), transcript_path=str(transcript_path), chunks=len(chunks))

    async def _synthesise_elevenlabs(self, text: str, pair_dir: str | None = None) -> TtsResult:
        """Synthesise via ElevenLabs TTS API."""
        chunks = _split_text(text, _ELEVENLABS_CHAR_LIMIT)
        client = self._get_elevenlabs_client()
        audio_parts: list[bytes] = []

        for chunk in chunks:
            response = client.text_to_speech.convert(
                voice_id=self.config.elevenlabs_voice_id,
                text=chunk,
                model_id=self.config.elevenlabs_model,
                output_format="mp3_44100_128",
            )
            # response is an async generator of bytes chunks
            chunk_bytes = b""
            async for audio_chunk in response:
                chunk_bytes += audio_chunk
            audio_parts.append(chunk_bytes)

        audio_bytes = b"".join(audio_parts)
        save_path, transcript_path = self._save_audio(audio_bytes, "mp3", text, pair_dir)

        logger.info(
            "TTS audio saved (elevenlabs)",
            path=str(save_path),
            size_bytes=len(audio_bytes),
            chunks=len(chunks),
        )
        return TtsResult(audio_path=str(save_path), transcript_path=str(transcript_path), chunks=len(chunks))

    async def _synthesise_piper(self, text: str, pair_dir: str | None = None) -> TtsResult:
        """Synthesise via Piper TTS over Wyoming protocol."""
        from wyoming.audio import AudioChunk, AudioStart, AudioStop
        from wyoming.client import AsyncTcpClient
        from wyoming.tts import Synthesize

        client = AsyncTcpClient(self.config.piper_host, self.config.piper_port)
        await client.connect()

        try:
            synth = Synthesize(text=text, voice=None)
            await client.write_event(synth.event())

            pcm_bytes = b""
            rate = 22050
            width = 2
            channels = 1

            while True:
                event = await client.read_event()
                if event is None:
                    break
                if AudioStart.is_type(event.type):
                    start = AudioStart.from_event(event)
                    rate = start.rate
                    width = start.width
                    channels = start.channels
                elif AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    pcm_bytes += chunk.audio
                elif AudioStop.is_type(event.type):
                    break
        finally:
            await client.disconnect()

        if not pcm_bytes:
            raise RuntimeError("Piper returned no audio data.")

        # Convert raw PCM to WAV in memory, then to OGG Opus via ffmpeg
        wav_buf = BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(width)
            wf.setframerate(rate)
            wf.writeframes(pcm_bytes)
        wav_bytes = wav_buf.getvalue()

        # Use ffmpeg to convert WAV → OGG Opus (Telegram voice format)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0", "-c:a", "libopus",
            "-b:a", "64k", "-f", "ogg", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        ogg_bytes, stderr = await proc.communicate(wav_bytes)

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {stderr.decode()[:200]}")

        save_path, transcript_path = self._save_audio(ogg_bytes, "ogg", text, pair_dir)

        logger.info(
            "TTS audio saved (piper)",
            path=str(save_path),
            size_bytes=len(ogg_bytes),
            pcm_bytes=len(pcm_bytes),
            rate=rate,
        )
        return TtsResult(audio_path=str(save_path), transcript_path=str(transcript_path), chunks=1)

    def _save_audio(self, audio_bytes: bytes, ext: str, text: str, pair_dir: str | None = None) -> tuple[Path, Path]:
        """Save audio + transcript. Returns (audio_path, transcript_path)."""
        if pair_dir:
            save_dir = Path(pair_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(self.config.approved_directory) / ".media.telegram" / "audios" / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"sent.{ext}"
        save_path.write_bytes(audio_bytes)
        transcript_path = save_dir / "sent.txt"
        transcript_path.write_text(text, encoding="utf-8")
        return save_path, transcript_path


def _strip_markup(text: str) -> str:
    """Remove HTML tags and Markdown formatting before sending to TTS."""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Strip Markdown code fences and their content
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    # Strip inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Strip bold/italic markers
    text = re.sub(r"[*_]{1,3}", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_text(text: str, limit: int) -> list[str]:
    """Split text into chunks ≤ limit chars at sentence boundaries."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while len(text) > limit:
        # Find last sentence boundary before limit
        cut = text.rfind(". ", 0, limit)
        if cut == -1:
            cut = text.rfind("! ", 0, limit)
        if cut == -1:
            cut = text.rfind("? ", 0, limit)
        if cut == -1:
            cut = limit  # hard cut — no boundary found
        else:
            cut += 1  # include the punctuation
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()

    if text:
        chunks.append(text)
    return chunks
