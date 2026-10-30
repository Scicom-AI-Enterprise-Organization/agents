# collapse_user_turns

A test harness for the fork's `collapse_user_turns` turn-handling option, with the
agent-starter-react frontend so it can be tried in a browser.

```python
AgentSession(turn_handling={"collapse_user_turns": True})  # default False
```

Until the agent is heard (its audio starts, an assistant message is added, or a tool runs),
anything more the user says extends their last turn instead of starting a new one:

- **LLM**: the chat context holds one user message, keeping the id of the first, with
  the pieces joined by spaces. `conversation_item_added` fires again for that id with the
  combined text, and `session.history` holds the combined message.
- **Transcript**: every STT final of the turn is published on one `lk.segment_id`, so the
  frontend shows one bubble that grows. The segment is marked final when the agent
  answers. A typed chat message also ends it.

## Run it

Needs `livekit-server` (`brew install livekit`), Node and pnpm.

```bash
# 1. LiveKit server, on ws://localhost:7880 with devkey / secret
livekit-server --dev

# 2. the agent, from the repository root
uv sync --all-extras --dev
cp examples/collapse_user_turns/.env.example examples/collapse_user_turns/.env   # fill it in
uv run examples/collapse_user_turns/agent.py dev

# 3. the frontend
cd examples/collapse_user_turns/frontend
cp .env.example .env.local   # LIVEKIT_URL=ws://localhost:7880, LIVEKIT_API_KEY=devkey,
                             # LIVEKIT_API_SECRET=secret, AGENT_NAME blank
pnpm install && pnpm dev     # http://localhost:3000
```

Start a call, open the transcript, and when asked for a number give it in pieces, pausing
about a second between them: "my number is zero one two" … "three four five" … "six seven
eight nine". Expect one user bubble that grows, and in the agent log one user turn per
request:

```
llm.request user turns=1 ['My number is 012.']
llm.request user turns=1 ['My number is 012. 345.']
llm.request user turns=1 ['My number is 012. 345. 6789.']
```

With `COLLAPSE_USER_TURNS=false` the same call gives three bubbles and
`user turns=3 ['My number is 012.', '345.', '6789.']`.

If the agent never joins the first call, the worker turned the job down under CPU load
(Next.js compiles the page on its first request). Start the call again.

## The frontend

`frontend/` is [agent-starter-react](https://github.com/livekit-examples/agent-starter-react)
at `9e1fc2f` (MIT, see `frontend/LICENSE`), unchanged. It needs no change for this option:
`@livekit/components-core` already replaces a bubble's text when a transcription stream
arrives with the same `lk.segment_id`.
