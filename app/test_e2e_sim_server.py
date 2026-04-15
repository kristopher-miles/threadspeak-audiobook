import json
import os

import requests

from e2e_sim.fixture_queue import ScriptedInteractionQueue
from e2e_sim.lmstudio_server import LMStudioSimServer


def test_scripted_queue_matches_partial_payload_and_tracks_pending():
    queue = ScriptedInteractionQueue(
        routes={
            "POST /v1/chat/completions": [
                {
                    "expect": {
                        "model": "sim-model",
                        "stream": False,
                    },
                    "response": {"ok": True},
                }
            ]
        },
        strict=True,
    )

    entry = queue.consume(
        "POST /v1/chat/completions",
        {
            "model": "sim-model",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert entry is not None
    assert entry["response"] == {"ok": True}
    assert queue.pending_counts() == {}
    queue.assert_all_consumed(context="unit test")


def test_lmstudio_sim_server_streams_scripted_events():
    fixture = os.path.join(
        os.path.dirname(__file__),
        "test_fixtures",
        "e2e_sim",
        "lmstudio_stream_tool.json",
    )

    with LMStudioSimServer(fixture) as server:
        response = requests.post(
            f"{server.base_url}/v1/chat/completions",
            json={
                "model": "sim-tool-model",
                "messages": [{"role": "user", "content": "who speaks"}],
                "stream": True,
            },
            timeout=20,
        )
        assert response.status_code == 200
        body = response.text
        assert "data: [DONE]" in body

        lines = [line[len("data: "):] for line in body.splitlines() if line.startswith("data: {")]
        assert len(lines) >= 2
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["object"] == "chat.completion.chunk"
        assert second["choices"][0]["finish_reason"] == "tool_calls"

        server.assert_all_consumed()
