from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from livekit.agents import Agent, ConversationItemAddedEvent, UserInputTranscribedEvent, utils
from livekit.agents.llm import ChatContext, ChatItem, LLMStream
from livekit.agents.voice.room_io.room_io import RoomIO, _EndOfUserTurn

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0


class _Agent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


def _texts(items: list[ChatItem], role: str) -> list[str | None]:
    return [item.text_content for item in items if item.type == "message" and item.role == role]


async def test_speech_before_the_reply_plays_extends_the_turn() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.", duration=1.0)
    actions.add_tts(10.0)
    actions.add_user_speech(3.0, 4.0, "Stop!", stt_delay=0.2)
    actions.add_llm("Okay, I will stop.", input="Tell me a story. Stop!")
    actions.add_tts(1.0)

    session = create_session(
        actions, can_pause_audio=True, turn_handling={"collapse_user_turns": True}
    )
    agent = _Agent()

    requests: list[ChatContext] = []
    fake_llm: Any = session.llm
    chat = fake_llm.chat

    def _recording_chat(**kwargs: Any) -> LLMStream:
        requests.append(kwargs["chat_ctx"].copy())
        return chat(**kwargs)

    fake_llm.chat = _recording_chat

    added: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", added.append)

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert _texts(agent.chat_ctx.items, "user") == ["Tell me a story. Stop!"]
    assert _texts(session.history.items, "user") == ["Tell me a story. Stop!"]
    assert _texts(agent.chat_ctx.items, "assistant") == ["Okay, I will stop."]
    # the model was asked once more, with one user turn holding both
    assert _texts(requests[-1].items, "user") == ["Tell me a story. Stop!"]

    # the combined turn is added again under the first turn's id
    user_items = [ev.item for ev in added if ev.item.type == "message" and ev.item.role == "user"]
    assert [item.text_content for item in user_items] == [
        "Tell me a story.",
        "Tell me a story. Stop!",
    ]
    assert user_items[0].id == user_items[1].id


async def test_a_turn_the_agent_answered_is_not_extended() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hi there.")
    actions.add_llm("Hello!", ttft=0.1, duration=0.2)
    actions.add_tts(0.5, ttfb=0.1, duration=0.2)
    actions.add_user_speech(4.0, 5.0, "What is my balance?")
    actions.add_llm("It is ten ringgit.")
    actions.add_tts(1.0)

    session = create_session(
        actions, can_pause_audio=True, turn_handling={"collapse_user_turns": True}
    )
    agent = _Agent()

    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    assert _texts(agent.chat_ctx.items, "user") == ["Hi there.", "What is my balance?"]
    assert _texts(agent.chat_ctx.items, "assistant") == ["Hello!", "It is ten ringgit."]


class _SegmentRecorder:
    """Starts a segment on the first capture after a flush, as the room output does."""

    def __init__(self) -> None:
        self.segments: dict[str, tuple[str, bool]] = {}
        self._current: str | None = None

    async def capture_text(self, text: str) -> None:
        if self._current is None:
            self._current = f"SG_{len(self.segments)}"
        self.segments[self._current] = (text, False)

    def flush(self) -> None:
        if self._current is not None:
            self.segments[self._current] = (self.segments[self._current][0], True)
            self._current = None


def _final(transcript: str) -> UserInputTranscribedEvent:
    return UserInputTranscribedEvent(transcript=transcript, is_final=True)


def _interim(transcript: str) -> UserInputTranscribedEvent:
    return UserInputTranscribedEvent(transcript=transcript, is_final=False)


async def _bubbles(
    events: list[UserInputTranscribedEvent | _EndOfUserTurn], *, collapse: bool = True
) -> list[tuple[str, bool]]:
    output = _SegmentRecorder()
    room_io: Any = SimpleNamespace(
        _user_tr_output=output,
        _agent_session=SimpleNamespace(options=SimpleNamespace(collapse_user_turns=collapse)),
    )
    event_ch = utils.aio.Chan[UserInputTranscribedEvent | _EndOfUserTurn]()
    for event in events:
        event_ch.send_nowait(event)
    event_ch.close()
    await RoomIO._forward_user_transcript(room_io, event_ch)
    return list(output.segments.values())


async def test_finals_of_an_unanswered_turn_share_one_segment() -> None:
    events: list[UserInputTranscribedEvent | _EndOfUserTurn] = [
        _final("zero one six"),
        _interim("two"),
        _final("two three four"),
        _EndOfUserTurn(),
        _final("yes"),
    ]

    assert await _bubbles(events) == [("zero one six two three four", True), ("yes", False)]
    assert await _bubbles(events, collapse=False) == [
        ("zero one six", True),
        ("two three four", True),
        ("yes", True),
    ]


async def test_an_answer_mid_utterance_moves_it_to_the_next_segment() -> None:
    assert await _bubbles(
        [
            _final("zero one six"),
            _interim("two"),
            _EndOfUserTurn(),
            _interim("two three"),
            _final("two three four"),
        ]
    ) == [("zero one six", True), ("two three four", False)]


async def test_an_answer_during_the_first_utterance_ends_its_segment_when_final() -> None:
    assert await _bubbles(
        [
            _interim("hello"),
            _EndOfUserTurn(),
            _final("hello there"),
            _final("how are you"),
        ]
    ) == [("hello there", True), ("how are you", False)]
