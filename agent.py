from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from parser import parse_agent_output
from device_tools import execute_tool
from tool_search import select_tool

ALLOWED_DEVICE_TYPES = ["sensor", "gateway", "controller", "processor", "ipcamera"]


def _load_dotenv(dotenv_path: str | Path | None = None) -> None:
    path = Path(dotenv_path or ".env")
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key in os.environ:
            continue
        os.environ[key] = value


_load_dotenv()


MODEL_SERVER = os.getenv(
    "MODEL_SERVER",
    "http://94.101.135.237:9000/v1/chat/completions",
)
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen")
API_KEY = os.getenv("MODEL_API_KEY", "tensorrt_llm")
MODEL_REASONING_EFFORT = os.getenv("MODEL_REASONING_EFFORT", "default")


def _normalize_model_server_url(url: str) -> str:
    base = (url or "").strip().rstrip("/")
    if not base:
        return "http://94.101.135.237:9000/v1/chat/completions"
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1") or base.endswith("/openai/v1"):
        return f"{base}/chat/completions"
    return base


MODEL_SERVER = _normalize_model_server_url(MODEL_SERVER)


SYSTEM_PROMPT = """You are a strict device CRUD agent.

You can do only one of the following:

1) Ask the user for missing information using exactly this format:

<ask>
tool=create_device
missing=field1,field2
question=Your question here
</ask>

2) Call exactly one tool using exactly this format:

<tool>
name=create_device
device_id=dev-1001
device_name=my-device
device_type=sensor
</tool>

3) After receiving tool result, respond with a short plain text answer only.

Available tools:
- create_device
- get_device
- update_device
- delete_device
- list_devices

Rules:
- Do not use JSON
- Do not use markdown
- Output only one block at a time
- Do not add extra text before or after a block
- If required information is missing, ask the user
- In ask blocks, include the intended tool name in the tool field
- If enough information exists, call exactly one tool
- For create_device required fields are: device_name, device_id, device_type
- Allowed device_type values: sensor, gateway, controller, processor, ipcamera
- For update_device required field is: device_id and at least one of device_name, device_type, role, description, group, save_data, status, tags, features
- For delete_device required field is: device_id
- For get_device required field is: device_id
- For list_devices no field is needed
"""

RESPONSE_WRITER_PROMPT = """You write concise end-user messages for a device CRUD assistant.

Rules:
- Return plain text only
- Do not use XML, JSON, or markdown
- Match the user's language when it is clear from the conversation
- Keep the message short, clear, and natural
- If information is missing, ask only for the missing fields
- If an operation succeeded, summarize the result naturally
- If an operation failed, explain the problem simply and helpfully
"""


def call_model(messages: List[Dict[str, str]]) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "reasoning_effort": MODEL_REASONING_EFFORT,
        "temperature": 0,
        "max_tokens": 256,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    request = urllib.request.Request(
        MODEL_SERVER,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise RuntimeError(
            f"Model server request failed ({exc.code}): {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Model server request failed: {exc.reason}") from exc
    except (http.client.RemoteDisconnected, ConnectionError, TimeoutError) as exc:
        raise RuntimeError(f"Model server connection failed: {exc}") from exc

    data = json.loads(response_body)
    return data["choices"][0]["message"]["content"]


def generate_ai_message(
    *,
    fallback_message: str,
    user_text: str = "",
    tool_name: Optional[str] = None,
    tool_result: Optional[dict[str, Any]] = None,
    missing_fields: Optional[List[str]] = None,
    current_data: Optional[dict[str, Any]] = None,
    intent: str = "final",
) -> str:
    prompt_payload = {
        "intent": intent,
        "user_text": user_text,
        "tool_name": tool_name,
        "tool_result": tool_result,
        "missing_fields": missing_fields or [],
        "current_data": current_data or {},
        "fallback_message": fallback_message,
    }

    try:
        message = call_model(
            [
                {"role": "system", "content": RESPONSE_WRITER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Write the assistant message for this situation.\n"
                        f"{json.dumps(prompt_payload, ensure_ascii=False)}"
                    ),
                },
            ]
        ).strip()
    except RuntimeError:
        return fallback_message

    parsed = parse_agent_output(message)
    if parsed.get("kind") != "final":
        return fallback_message

    return message or fallback_message


def build_fe_response(
    *,
    context: Dict[str, Any],
    message: str,
    actions: List[Dict[str, Any]],
    data: Optional[Any] = None,
    need_more_info: bool = False,
    missing_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload_data = data
    if payload_data is None:
        payload_data = context.get("data", {})
    return {
        "data": payload_data,
        "context": context,
        "message": message,
        "actions": actions,
        "need_more_info": need_more_info,
        "missing_fields": missing_fields or [],
    }


def _build_create_form_action(
    missing_fields: list[str],
    current_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    values = current_data or {}
    return {
        "type": "form",
        "tool": "create_device",
        "title": "Create New Device",
        "submit_label": "Create Device",
        "fields": [
            {
                "name": "device_name",
                "label": "Device Name",
                "input_type": "text",
                "required": True,
                "value": str(values.get("device_name", "")),
                "is_missing": "device_name" in missing_fields,
            },
            {
                "name": "device_id",
                "label": "Device ID",
                "input_type": "text",
                "required": True,
                "value": str(values.get("device_id", "")),
                "is_missing": "device_id" in missing_fields,
            },
            {
                "name": "device_type",
                "label": "Device Type",
                "input_type": "select",
                "required": True,
                "value": str(values.get("device_type", "")),
                "options": [{"label": v, "value": v} for v in ALLOWED_DEVICE_TYPES],
                "is_missing": "device_type" in missing_fields,
            },
        ],
    }


def build_missing_info_actions(
    *,
    tool_name: Optional[str],
    missing_fields: list[str],
    current_data: Optional[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    if tool_name == "create_device":
        return [_build_create_form_action(missing_fields, current_data)]
    return []


def map_tool_result_to_fe(
    tool_name: str,
    tool_result: dict,
    *,
    user_text: str = "",
) -> dict:
    if not tool_result.get("ok"):
        error_message = tool_result.get("error", "Tool execution failed.")
        if tool_name == "create_device" and "invalid device type" in str(error_message).lower():
            fallback_message = (
                "Invalid device type. Allowed values are: "
                + ", ".join(ALLOWED_DEVICE_TYPES)
            )
            return build_fe_response(
                context={"entity": "device", "data": {}},
                message=generate_ai_message(
                    fallback_message=fallback_message,
                    user_text=user_text,
                    tool_name=tool_name,
                    tool_result=tool_result,
                    missing_fields=["device_type"],
                    intent="missing_info",
                ),
                actions=build_missing_info_actions(
                    tool_name="create_device",
                    missing_fields=["device_type"],
                    current_data={},
                ),
                need_more_info=True,
                missing_fields=["device_type"],
            )
        return build_fe_response(
            context={
                "entity": "device",
                "data": tool_result,
            },
            data={"result": tool_result},
            message=generate_ai_message(
                fallback_message=error_message,
                user_text=user_text,
                tool_name=tool_name,
                tool_result=tool_result,
                intent="error",
            ),
            actions=[],
            need_more_info=False,
            missing_fields=[],
        )

    if tool_name == "create_device":
        device = tool_result["device"]
        fallback_message = "Device created successfully."
        return build_fe_response(
            context={
                "entity": "device",
                "data": device,
            },
            message=generate_ai_message(
                fallback_message=fallback_message,
                user_text=user_text,
                tool_name=tool_name,
                tool_result=tool_result,
                intent="success",
            ),
            actions=[
                {
                    "kind": "navigate",
                    "operationId": "GetDeviceById",
                    "destination": {
                        "screen": "resource_detail",
                        "resource": "device",
                        "idFrom": "data.device.deviceId",
                    },
                    "label": "Open device detail",
                    "variant": "success",
                },
                {
                    "kind": "show_list",
                    "resource": "device",
                    "label": "Back to devices",
                    "variant": "secondary",
                },
                {
                    "kind": "confirm",
                    "label": "Create connector",
                    "target": "CreateDeviceConnector",
                    "confirmKey": "create_device_connector",
                    "title": "Create connector",
                    "message": "Do you want to create a connector for this device now?",
                    "variant": "primary",
                }
            ],
            data={"device": device},
        )

    if tool_name == "get_device":
        device = tool_result["device"]
        fallback_message = "Device fetched successfully."
        return build_fe_response(
            context={
                "entity": "device",
                "data": device,
            },
            message=generate_ai_message(
                fallback_message=fallback_message,
                user_text=user_text,
                tool_name=tool_name,
                tool_result=tool_result,
                intent="success",
            ),
            actions=[
                {
                    "kind": "navigate",
                    "operationId": "UpdateDevice",
                    "destination": {
                        "screen": "resource_edit",
                        "resource": "device",
                        "idFrom": "data.device.deviceId",
                    },
                    "label": "Edit device",
                    "variant": "primary",
                },
                {
                    "kind": "show_list",
                    "resource": "device",
                    "label": "Back to devices",
                    "variant": "secondary",
                }
            ],
            data={"device": device},
        )

    if tool_name == "update_device":
        device = tool_result["device"]
        fallback_message = "Device updated successfully."
        return build_fe_response(
            context={
                "entity": "device",
                "data": device,
            },
            message=generate_ai_message(
                fallback_message=fallback_message,
                user_text=user_text,
                tool_name=tool_name,
                tool_result=tool_result,
                intent="success",
            ),
            actions=[
                {
                    "kind": "navigate",
                    "operationId": "GetDeviceById",
                    "destination": {
                        "screen": "resource_detail",
                        "resource": "device",
                        "idFrom": "data.device.deviceId",
                    },
                    "label": "Open updated device",
                    "variant": "success",
                },
                {
                    "kind": "show_list",
                    "resource": "device",
                    "label": "Back to devices",
                    "variant": "secondary",
                }
            ],
            data={"device": device},
        )

    if tool_name == "delete_device":
        device = tool_result["device"]
        fallback_message = "Device deleted successfully."
        return build_fe_response(
            context={
                "entity": "device",
                "data": device,
            },
            message=generate_ai_message(
                fallback_message=fallback_message,
                user_text=user_text,
                tool_name=tool_name,
                tool_result=tool_result,
                intent="success",
            ),
            actions=[
                {
                    "kind": "show_list",
                    "resource": "device",
                    "label": "Back to devices",
                    "variant": "primary",
                },
                {
                    "kind": "navigate",
                    "destination": {
                        "screen": "resource_create",
                        "resource": "device",
                    },
                    "label": "Create a new device",
                    "variant": "secondary",
                }
            ],
            data={"device": device},
        )

    if tool_name == "list_devices":
        devices = tool_result["devices"]
        fallback_message = f"{len(devices)} device(s) found."
        return build_fe_response(
            context={
                "entity": "device_list",
                "data": devices,
            },
            message=generate_ai_message(
                fallback_message=fallback_message,
                user_text=user_text,
                tool_name=tool_name,
                tool_result=tool_result,
                intent="success",
            ),
            actions=[
                {
                    "kind": "navigate",
                    "destination": {
                        "screen": "resource_create",
                        "resource": "device",
                    },
                    "label": "Create device",
                    "variant": "primary",
                }
            ],
            data={"devices": devices},
        )

    return build_fe_response(
        context={
            "entity": "device",
            "data": tool_result,
        },
        data={"result": tool_result},
        message=generate_ai_message(
            fallback_message="Operation completed.",
            user_text=user_text,
            tool_name=tool_name,
            tool_result=tool_result,
            intent="success",
        ),
        actions=[],
    )


def run_agent(user_text: str, max_turns: int = 3) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    for _ in range(max_turns):
        assistant_text = call_model(messages)
        parsed = parse_agent_output(assistant_text)

        if parsed["kind"] == "ask":
            question = parsed["data"].get("question", "Please provide more information.")
            ask_tool = parsed["data"].get("tool")
            missing_fields = parsed.get("missing_fields", [])
            return build_fe_response(
                context={
                    "entity": "device",
                    "data": {},
                },
                message=generate_ai_message(
                    fallback_message=question,
                    user_text=user_text,
                    tool_name=ask_tool,
                    missing_fields=missing_fields,
                    current_data={},
                    intent="missing_info",
                ),
                actions=build_missing_info_actions(
                    tool_name=ask_tool,
                    missing_fields=missing_fields,
                    current_data={},
                ),
                need_more_info=True,
                missing_fields=missing_fields,
            )

        if parsed["kind"] == "tool":
            tool_data = parsed["data"]
            selection = select_tool(
                user_text=user_text,
                candidate_tool=tool_data.get("name", ""),
                tool_data=tool_data,
            )
            tool_name = selection.tool_name or str(tool_data.get("name", "")).strip()
            tool_data["name"] = tool_name
            tool_result = execute_tool(tool_data)

            fe_response = map_tool_result_to_fe(tool_name, tool_result, user_text=user_text)
            return fe_response

        if parsed["kind"] == "final":
            return build_fe_response(
                context={
                    "entity": "device",
                    "data": {},
                },
                message=generate_ai_message(
                    fallback_message=parsed["data"]["message"],
                    user_text=user_text,
                    current_data={},
                    intent="final",
                ),
                actions=[],
                need_more_info=False,
                missing_fields=[],
            )

    return build_fe_response(
        context={"entity": "device", "data": {}},
        message=generate_ai_message(
            fallback_message="Max turns reached without a valid result.",
            user_text=user_text,
            current_data={},
            intent="error",
        ),
        actions=[],
        need_more_info=False,
        missing_fields=[],
    )
