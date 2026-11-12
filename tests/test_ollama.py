"""OllamaLLM against a tiny local fake server: checks the exact request we send."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from talewerk.backends import OllamaError, OllamaLLM


@pytest.fixture
def server():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if seen["body"]["model"] == "missing":
                out, code = b'{"error":"model \\"missing\\" not found"}', 404
            else:
                out, code = json.dumps({"message": {"content": '{"ok": true}'}}).encode(), 200
            self.send_response(code)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


def test_request_turns_thinking_off_and_forces_schema(server):
    url, seen = server
    llm = OllamaLLM(model="qwen3:8b", base_url=url + "/v1")  # /v1 is stripped
    out = llm.complete("sys", "hi", schema={"type": "object"}, temperature=0.2)
    assert out == '{"ok": true}'
    assert seen["path"] == "/api/chat"
    assert seen["body"]["think"] is False
    assert seen["body"]["format"] == {"type": "object"}
    assert seen["body"]["options"]["temperature"] == 0.2
    assert seen["body"]["stream"] is False


def test_missing_model_raises_clear_error(server):
    url, _ = server
    with pytest.raises(OllamaError, match="not found"):
        OllamaLLM(model="missing", base_url=url).complete("s", "u")
