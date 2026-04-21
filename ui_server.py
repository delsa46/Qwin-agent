#!/usr/bin/env python3
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from agent_session import DeviceAgentSession


HOST = "127.0.0.1"
PORT = 8008
ALLOWED_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}


def _resolve_cors_origin(handler: BaseHTTPRequestHandler) -> str:
    origin = handler.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS:
        return origin
    return "http://localhost:5173"


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", _resolve_cors_origin(handler))
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


class AgentApiHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", _resolve_cors_origin(self))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, 200, {"ok": True})
            return
        _json_response(self, 404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            _json_response(self, 404, {"ok": False, "error": "Not found"})
            return

        content_len = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            _json_response(self, 400, {"ok": False, "error": "Invalid JSON body"})
            return

        user_text = str(payload.get("message", "")).strip()
        history = payload.get("history", [])
        if not isinstance(history, list):
            _json_response(self, 400, {"ok": False, "error": "history must be a list"})
            return
        if not user_text:
            _json_response(self, 400, {"ok": False, "error": "message is required"})
            return

        try:
            response, updated_history = DeviceAgentSession.run_turn_stateless(user_text, history)
        except Exception as exc:  # pragma: no cover
            _json_response(self, 500, {"ok": False, "error": str(exc)})
            return

        _json_response(
            self,
            200,
            {
                "ok": True,
                "response": response,
                "history": updated_history,
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    server = HTTPServer((HOST, PORT), AgentApiHandler)
    print(f"Agent UI API is running on http://{HOST}:{PORT}")
    print(f"CORS origins: {', '.join(sorted(ALLOWED_ORIGINS))}")
    server.serve_forever()


if __name__ == "__main__":
    main()
