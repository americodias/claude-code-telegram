"""
Convert text to speech via OpenAI, ElevenLabs, or Piper (Wyoming) TTS.

Supports a provider fallback chain: if the active provider fails (quota,
network, etc.), the next provider in the chain is tried automatically.
Provider can be switched at runtime without restarting the bot.
"""

import asyncio
import re
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.config.settings import Settings

logger = structlog.get_logger(__name__)

_OPENAI_CHAR_LIMIT = 4096
_ELEVENLABS_CHAR_LIMIT = 5000
_VALID_PROVIDERS = {"openai", "elevenlabs", "piper"}
_FAILURE_COOLDOWN_SECONDS = 300  # 5 min before retrying a failed provider


@dataclass
class TtsResult:
    """Result of a TTS synthesis."""

    audio_path: str  # absolute path to audio file
    transcript_path: str  # absolute path to transcript .txt
    chunks: int  # how many API calls were made
    provider: str  # which provider was used


class TtsHandler:
    """Synthesise speech from text via configurable TTS provider chain."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self._openai_client: Any = None
        self._elevenlabs_client: Any = None

        # Build provider chain
        chain_str = config.tts_provider_chain.strip()
        if chain_str:
            self._provider_chain = [
                p.strip().lower()
                for p in chain_str.split(",")
                if p.strip().lower() in _VALID_PROVIDERS
            ]
        else:
            self._provider_chain = [config.tts_provider.lower()]

        if not self._provider_chain:
            self._provider_chain = ["openai"]

        self._current_index = 0
        self._failed_providers: Dict[str, float] = {}  # provider -> failure timestamp

    # --- Public API ---

    @property
    def current_provider(self) -> str:
        """Return the currently active provider name."""
        return self._provider_chain[self._current_index]

    def get_status(self) -> Dict[str, Any]:
        """Return status dict for /tts command."""
        now = time.time()
        failed_info = {}
        for provider, fail_time in self._failed_providers.items():
            age = now - fail_time
            if age < _FAILURE_COOLDOWN_SECONDS:
                failed_info[provider] = f"failed {int(age)}s ago (cooldown {_FAILURE_COOLDOWN_SECONDS - int(age)}s)"
            else:
                failed_info[provider] = "cooldown expired, will retry"
        return {
            "current": self.current_provider,
            "chain": self._provider_chain,
            "failed": failed_info,
        }

    def switch_provider(self, name: str) -> str:
        """Switch to a named provider. Returns the new active provider."""
        name = name.lower()
        if name not in _VALID_PROVIDERS:
            raise ValueError(f"Unknown provider: {name}. Valid: {', '.join(sorted(_VALID_PROVIDERS))}")
        if name not in self._provider_chain:
            # Add to chain if not present
            self._provider_chain.append(name)
        self._current_index = self._provider_chain.index(name)
        # Clear failure status for this provider
        self._failed_providers.pop(name, None)
        return name

    def next_provider(self) -> str:
        """Switch to the next provider in the chain. Returns the new active provider."""
        self._current_index = (self._current_index + 1) % len(self._provider_chain)
        return self.current_provider

    async def synthesise(self, text: str, pair_dir: str | None = None) -> TtsResult:
        """Convert text to an audio file, falling back through the provider chain.

        Args:
            text: The text to synthesise (may contain markup — stripped internally).
            pair_dir: If provided, save into this directory as sent.* (paired with received.*).
        """
        plain = _strip_markup(text)
        if not plain:
            raise ValueError("No speakable text after stripping markup.")

        errors: List[tuple[str, Exception]] = []
        for provider in self._get_provider_order():
            try:
                if provider == "elevenlabs":
                    result = await self._synthesise_elevenlabs(plain, pair_dir)
                elif provider == "piper":
                    result = await self._synthesise_piper(plain, pair_dir)
                else:
                    result = await self._synthesise_openai(plain, pair_dir)

                # Success — if this wasn't the current provider, update index
                if provider != self.current_provider and provider in self._provider_chain:
                    self._current_index = self._provider_chain.index(provider)
                # Clear failure if it was previously failed
                self._failed_providers.pop(provider, None)
                return result
            except Exception as e:
                logger.warning(
                    "TTS provider failed, trying next",
                    provider=provider,
                    error=str(e),
                )
                self._failed_providers[provider] = time.time()
                errors.append((provider, e))

        raise RuntimeError(f"All TTS providers failed: {errors}")

    # --- Provider order ---

    def _get_provider_order(self) -> List[str]:
        """Return providers to try, starting from current, skipping recently failed."""
        now = time.time()
        ordered: List[str] = []

        # Start from current index, wrap around
        for i in range(len(self._provider_chain)):
            idx = (self._current_index + i) % len(self._provider_chain)
            provider = self._provider_chain[idx]

            # Skip if failed recently (within cooldown)
            fail_time = self._failed_providers.get(provider)
            if fail_time and (now - fail_time) < _FAILURE_COOLDOWN_SECONDS:
                continue
            ordered.append(provider)

        # If all are in cooldown, try all anyway (better than nothing)
        if not ordered:
            ordered = list(self._provider_chain)

        return ordered

    # --- Lazy client init ---

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

    # --- Provider implementations ---

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
        return TtsResult(
            audio_path=str(save_path),
            transcript_path=str(transcript_path),
            chunks=len(chunks),
            provider="openai",
        )

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
        return TtsResult(
            audio_path=str(save_path),
            transcript_path=str(transcript_path),
            chunks=len(chunks),
            provider="elevenlabs",
        )

    async def _synthesise_piper(self, text: str, pair_dir: str | None = None) -> TtsResult:
        """Synthesise via Piper TTS over Wyoming protocol."""
        from wyoming.audio import AudioChunk, AudioStart, AudioStop
        from wyoming.client import AsyncTcpClient
        from wyoming.tts import Synthesize

        _PIPER_TIMEOUT = 60  # seconds — avoid hanging forever

        client = AsyncTcpClient(self.config.piper_host, self.config.piper_port)
        await asyncio.wait_for(client.connect(), timeout=_PIPER_TIMEOUT)

        try:
            synth = Synthesize(text=text)
            ev = synth.event()
            if self.config.piper_voice:
                ev.data["voice"] = {"name": self.config.piper_voice}
            await client.write_event(ev)

            pcm_bytes = b""
            rate = 22050
            width = 2
            channels = 1

            async def _read_audio() -> None:
                nonlocal pcm_bytes, rate, width, channels
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

            await asyncio.wait_for(_read_audio(), timeout=_PIPER_TIMEOUT)
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
        return TtsResult(
            audio_path=str(save_path),
            transcript_path=str(transcript_path),
            chunks=1,
            provider="piper",
        )

    # --- Helpers ---

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
