"""A test agent for ``collapse_user_turns``, on OpenAI-compatible LLM, STT and TTS endpoints.

Say a number in pieces, pausing between them, before the agent answers: the frontend shows
one user bubble that keeps growing, and each LLM request (logged as ``llm.request``) holds
one user turn. Set ``COLLAPSE_USER_TURNS=false`` to compare with upstream behaviour.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
import weakref
from collections.abc import AsyncIterable

import aiohttp
import numpy as np
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    FlushSentinel,
    JobContext,
    JobProcess,
    ModelSettings,
    TurnHandlingOptions,
    cli,
    utils,
)
from livekit.agents.inference.eot.base import (
    DEFAULT_SAMPLE_RATE,
    TurnDetectorOptions,
    _BaseStreamingTurnDetector,
    _BaseStreamingTurnDetectorStream,
)
from livekit.agents.inference.eot.languages import ThresholdOptions
from livekit.agents.llm import ChatChunk, ChatContext, Tool
from livekit.plugins import openai, silero

logger = logging.getLogger("collapse-user-turns")

load_dotenv()


class _RemoteEOTTransport:
    """Posts the last eight seconds of the user's audio to the semantic VAD ``/predict``."""

    def __init__(self) -> None:
        self._url = f"{os.environ['EOT_URL'].rstrip('/').removesuffix('/predict')}/predict"
        self._window = utils.AudioArrayBuffer(
            buffer_size=int(8.0 * DEFAULT_SAMPLE_RATE), sample_rate=DEFAULT_SAMPLE_RATE
        )
        self._stream: weakref.ref[_BaseStreamingTurnDetectorStream] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def session_id(self) -> str | None:
        return None

    def attach(self, stream: _BaseStreamingTurnDetectorStream) -> None:
        self._stream = weakref.ref(stream)

    def detach(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._window.push_frame(frame)

    def flush(self) -> None:
        if len(self._window) > 0:
            self._window.shift(len(self._window))

    def run_inference(self, request_id: str) -> None:
        task = asyncio.create_task(self._predict(request_id, self._window.read()))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run(self) -> None:
        stream = self._stream() if self._stream is not None else None
        if stream is not None:
            await stream._drain_audio_channel()

    async def _predict(self, request_id: str, window: np.ndarray) -> None:
        started = time.monotonic()
        try:
            async with utils.http_context.http_session().post(
                self._url,
                json={
                    "audio": base64.b64encode(window.astype("<i2", copy=False).tobytes()).decode(
                        "ascii"
                    ),
                    "sample_rate": DEFAULT_SAMPLE_RATE,
                },
                headers={"Authorization": f"Bearer {os.environ['EOT_URL_API']}"},
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as resp:
                resp.raise_for_status()
                probability = float((await resp.json())["probability"])
        except Exception as exc:
            # a failed prediction leaves the turn to the endpointing timeout
            logger.warning("eot prediction failed: %s", type(exc).__name__)
            probability = 0.0

        stream = self._stream() if self._stream is not None else None
        if stream is not None:
            stream._resolve_prediction(
                request_id, probability, inference_duration=time.monotonic() - started
            )


class RemoteEOT(_BaseStreamingTurnDetector):
    def __init__(self) -> None:
        super().__init__(
            opts=TurnDetectorOptions(
                sample_rate=DEFAULT_SAMPLE_RATE,
                thresholds=ThresholdOptions("turn-detector-v1-mini", 0.5),
            )
        )

    @property
    def model(self) -> str:  # type: ignore[override]
        return "semantic-vad"

    @property
    def provider(self) -> str:
        return "self-hosted"

    async def supports_language(self, language: str | None) -> bool:
        return True

    def stream(self, *, conn_options: object = None) -> _BaseStreamingTurnDetectorStream:
        return _BaseStreamingTurnDetectorStream(
            detector=self, opts=self._opts, transport=_RemoteEOTTransport(), local_fallback=False
        )


class CollapseTestAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a phone support agent for an internet provider. Keep every "
            "reply to one short sentence. When the caller gives a phone or account number, read "
            "it back digit by digit and ask them to confirm it. Use no markdown or emojis.",
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the caller and ask for the phone number on their account."
        )

    def llm_node(
        self, chat_ctx: ChatContext, tools: list[Tool], model_settings: ModelSettings
    ) -> AsyncIterable[ChatChunk | str | FlushSentinel]:
        user_turns = [
            item.text_content
            for item in chat_ctx.items
            if item.type == "message" and item.role == "user"
        ]
        logger.info("llm.request user turns=%d %s", len(user_turns), user_turns)
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    collapse = os.environ.get("COLLAPSE_USER_TURNS", "true").lower() == "true"
    turn_handling = TurnHandlingOptions(
        turn_detection=RemoteEOT() if os.environ.get("EOT_URL") else "vad",
        # adaptive interruption would send the caller's audio to LiveKit Cloud
        interruption={"mode": "vad"},
        endpointing={"min_delay": 0.4, "max_delay": 1.5},
        collapse_user_turns=collapse,
    )
    session: AgentSession = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(
            model=os.environ["STT_MODEL"],
            base_url=os.environ["STT_URL"],
            api_key=os.environ["STT_URL_API"],
            language=os.environ.get("STT_LANGUAGE", "en"),
        ),
        llm=openai.LLM(
            model=os.environ["LLM_MODEL"],
            base_url=os.environ["LLM_URL"],
            api_key=os.environ["LLM_URL_API"],
            temperature=0.2,
        ),
        tts=openai.TTS(
            model=os.environ["TTS_MODEL"],
            voice=os.environ["TTS_SPEAKER"],
            base_url=os.environ["TTS_URL"],
            api_key=os.environ["TTS_URL_API"],
            response_format="pcm",
        ),
        turn_handling=turn_handling,
    )

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message":
            logger.info("chat.%s id=%s %r", ev.item.role, ev.item.id, ev.item.text_content)

    logger.info("collapse_user_turns=%s", collapse)
    await session.start(agent=CollapseTestAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
