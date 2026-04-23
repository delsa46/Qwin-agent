# Development Guide

## Overview

This project is a device-management agent with:

- a Python backend agent layer
- a lightweight HTTP API server
- a React frontend
- an AI-driven message writer and tool-selection layer

The backend runtime intentionally uses Python standard-library modules as much as possible. The main external Python package currently used in development is `pytest` for tests.

## Important Files

- `agent_session.py`: main conversational orchestration for stateful chat turns
- `agent.py`: model calls, message-writer layer, and non-session agent flow
- `tool_search.py`: tool-selection/search layer used before tool execution
- `device_tools.py`: tool catalog plus actual backend tool execution
- `parser.py`: parsing `<tool>`, `<ask>`, and other structured model outputs
- `ui_server.py`: local HTTP API for the frontend
- `frontend/src/App.tsx`: main frontend chat UI

## Python Setup

Recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Start from:

```bash
cp .env.example .env
```

Key variables:

- `SENSOLIST_API_TOKEN`
- `SENSOLIST_BASE_URL`
- `MODEL_SERVER`
- `MODEL_NAME`
- `MODEL_API_KEY`
- `MODEL_REASONING_EFFORT`

## Running The Backend

CLI mode:

```bash
.venv/bin/python main.py
```

HTTP API mode for the frontend:

```bash
.venv/bin/python ui_server.py
```

Default API server address:

- `http://127.0.0.1:8008`

## Running The Frontend

```bash
cd frontend
npm install
npm run dev
```

Default frontend address:

- `http://127.0.0.1:5173`

## Testing

Run the full Python test suite:

```bash
.venv/bin/python -m pytest -q
```

Run a focused subset:

```bash
.venv/bin/python -m pytest -q test_agent.py test_regression_followup.py test_tool_search.py
```

## Architecture Notes

### 1. Message Generation

User-facing messages should not be hardcoded when the AI writer layer can generate them. The project routes final user-visible messages through the message-writer logic in `agent.py` and `agent_session.py`.

### 2. Tool Selection

Before a tool is executed, `tool_search.py` helps decide which tool is the best fit for the request. Tool metadata lives in `device_tools.py` inside `TOOL_CATALOG`.

### 3. Chat History

Chat history is currently passed between the frontend and backend per request for multi-turn continuity within the active UI session. There is no permanent persistence layer enabled in this stage.

## Development Conventions

- Prefer extending `TOOL_CATALOG` when adding a new tool.
- Keep tool-selection logic in `tool_search.py`, not scattered across the codebase.
- Keep user-facing text AI-driven when possible.
- Avoid introducing new runtime dependencies unless they are clearly needed.
- Add or update tests whenever behavior changes.

## Suggested Workflow For New Features

1. Update tool metadata if the feature affects tool behavior.
2. Update orchestration in `agent_session.py` or `agent.py`.
3. Add regression tests.
4. Verify the frontend contract if response payloads change.
