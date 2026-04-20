import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from parser import parse_agent_output
from device_tools import execute_tool


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
device_name=my-device
device_type=sensor
role=temp
description=Edge temperature sensor
group=building-a
save_data=true
tags=critical,temperature
features=temperature|Temperature|sensors.temp|C|24.2
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
- For create_device required fields are: device_name, device_type
- Allowed device_type values: sensor, gateway, controller, processor, ipcamera
- Optional create_device field: device_id
- For update_device required field is: device_id and at least one of device_name, device_type, role, description, group, save_data, status, tags, features
- For delete_device required field is: device_id
- For get_device required field is: device_id
- For list_devices no field is needed
"""


def call_model(messages: List[Dict[str, str]]) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
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

    data = json.loads(response_body)
    return data["choices"][0]["message"]["content"]


def build_fe_response(
    *,
    context: Dict[str, Any],
    message: str,
    actions: List[Dict[str, Any]],
    need_more_info: bool = False,
    missing_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "context": context,
        "message": message,
        "actions": actions,
        "need_more_info": need_more_info,
        "missing_fields": missing_fields or [],
    }


def map_tool_result_to_fe(tool_name: str, tool_result: dict) -> dict:
    if not tool_result.get("ok"):
        return build_fe_response(
            context={
                "entity": "device",
                "data": tool_result,
            },
            message=tool_result.get("error", "Tool execution failed."),
            actions=[],
            need_more_info=False,
            missing_fields=[],
        )

    if tool_name == "create_device":
        device = tool_result["device"]
        device_target_id = device.get("deviceId") or device.get("id", "")
        return build_fe_response(
            context={
                "entity": "device",
                "data": device,
            },
            message="Device created successfully.",
            actions=[
                {
                    "type": "button",
                    "label": "Go To Device",
                    "variant": "success",
                    "target": f"/devices/{device_target_id}",
                }
            ],
        )

    if tool_name == "get_device":
        device = tool_result["device"]
        device_target_id = device.get("deviceId") or device.get("id", "")
        return build_fe_response(
            context={
                "entity": "device",
                "data": device,
            },
            message="Device fetched successfully.",
            actions=[
                {
                    "type": "button",
                    "label": "View Device",
                    "variant": "primary",
                    "target": f"/devices/{device_target_id}",
                }
            ],
        )

    if tool_name == "update_device":
        device = tool_result["device"]
        device_target_id = device.get("deviceId") or device.get("id", "")
        return build_fe_response(
            context={
                "entity": "device",
                "data": device,
            },
            message="Device updated successfully.",
            actions=[
                {
                    "type": "button",
                    "label": "Go To Device",
                    "variant": "success",
                    "target": f"/devices/{device_target_id}",
                }
            ],
        )

    if tool_name == "delete_device":
        device = tool_result["device"]
        return build_fe_response(
            context={
                "entity": "device",
                "data": device,
            },
            message="Device deleted successfully.",
            actions=[
                {
                    "type": "button",
                    "label": "Go To Devices",
                    "variant": "warning",
                    "target": "/devices",
                }
            ],
        )

    if tool_name == "list_devices":
        devices = tool_result["devices"]
        return build_fe_response(
            context={
                "entity": "device_list",
                "data": devices,
            },
            message=f"{len(devices)} device(s) found.",
            actions=[
                {
                    "type": "button",
                    "label": "Open Devices List",
                    "variant": "primary",
                    "target": "/devices",
                }
            ],
        )

    return build_fe_response(
        context={
            "entity": "device",
            "data": tool_result,
        },
        message="Operation completed.",
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
            return build_fe_response(
                context={
                    "entity": "device",
                    "data": {},
                },
                message=question,
                actions=[],
                need_more_info=True,
                missing_fields=parsed.get("missing_fields", []),
            )

        if parsed["kind"] == "tool":
            tool_data = parsed["data"]
            tool_name = tool_data.get("name", "")
            tool_result = execute_tool(tool_data)

            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool result:\n"
                        f"{json.dumps(tool_result, ensure_ascii=False)}\n\n"
                        "Now provide a short final answer in plain text only."
                    ),
                }
            )

            fe_response = map_tool_result_to_fe(tool_name, tool_result)
            return fe_response

        if parsed["kind"] == "final":
            return build_fe_response(
                context={
                    "entity": "device",
                    "data": {},
                },
                message=parsed["data"]["message"],
                actions=[],
                need_more_info=False,
                missing_fields=[],
            )

    return build_fe_response(
        context={"entity": "device", "data": {}},
        message="Max turns reached without a valid result.",
        actions=[],
        need_more_info=False,
        missing_fields=[],
    )
