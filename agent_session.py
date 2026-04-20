import re
from typing import Any, Optional

from agent import call_model, build_fe_response, map_tool_result_to_fe, SYSTEM_PROMPT
from parser import parse_agent_output
from device_tools import execute_tool


UPDATE_FIELDS = [
    "device_name",
    "device_type",
    "role",
    "description",
    "group",
    "save_data",
    "status",
    "tags",
    "features",
]

REQUIRED_FIELDS_BY_TOOL = {
    "create_device": ["device_name", "device_type"],
    "get_device": ["device_id"],
    "delete_device": ["device_id"],
}


def is_device_related(text: str) -> bool:
    t = text.lower()
    keywords = ["device", "devices"]
    return any(k in t for k in keywords)

class DeviceAgentSession:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.pending_tool: Optional[str] = None
        self.missing_fields: list[str] = []
        self.collected_data: dict[str, str] = {}

    def reset_pending(self) -> None:
        self.pending_tool = None
        self.missing_fields = []
        self.collected_data = {}

    def merge_followup_input(self, user_text: str) -> None:
        if not self.missing_fields:
            return

        clean_text = user_text.strip()
        if not clean_text:
            return

        # Preferred input style in follow-up is key=value pairs.
        # This keeps the parser stable and avoids relying on JSON.
        has_equals = "=" in clean_text
        for piece in clean_text.replace("\n", ",").split(","):
            part = piece.strip()
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = self._normalize_field_value(key, value.strip())
            if key in self.missing_fields and value:
                self.collected_data[key] = value

        remaining = [
            field for field in self.missing_fields
            if field not in self.collected_data or not self.collected_data[field].strip()
        ]

        if not remaining:
            return

        if len(remaining) == 1 and "=" not in clean_text:
            field = remaining[0]
            self.collected_data[field] = self._normalize_field_value(field, clean_text)
            return

        # If user explicitly used key=value, do not fallback to positional mapping.
        # This avoids accidental assignments like:
        #   device_type = "device_name=sensor-4"
        if has_equals:
            return

        if self.pending_tool == "update_device" and len(remaining) > 1 and "=" not in clean_text:
            # For update flows with many optional fields, positional mapping is ambiguous.
            # We force key=value style to avoid assigning values to wrong fields.
            detected_field = self._guess_field_from_free_text(clean_text, remaining)
            if detected_field:
                self.collected_data[detected_field] = self._normalize_field_value(
                    detected_field,
                    clean_text,
                )
            return

        parts = [p.strip() for p in clean_text.split(",") if p.strip()]
        for idx, field in enumerate(remaining):
            if idx < len(parts) and parts[idx]:
                self.collected_data[field] = self._normalize_field_value(field, parts[idx])

    def _has_update_payload(self, tool_data: dict[str, str]) -> bool:
        for field in UPDATE_FIELDS:
            value = tool_data.get(field)
            if value is not None and str(value).strip() != "":
                return True
        return False

    def _build_update_payload_question(self) -> dict[str, Any]:
        self.pending_tool = "update_device"
        self.missing_fields = UPDATE_FIELDS.copy()
        return build_fe_response(
            context={"entity": "device", "data": self.collected_data},
            message=(
                "Please provide at least one update field in key=value format. "
                "Example: status=inactive or device_name=sensor-2"
            ),
            actions=[],
            need_more_info=True,
            missing_fields=self.missing_fields,
        )

    def _normalize_missing_fields(self, tool_name: Optional[str], fields: list[str]) -> list[str]:
        if not tool_name:
            return fields

        required = REQUIRED_FIELDS_BY_TOOL.get(tool_name)
        if required is not None:
            normalized = [field for field in required if field in fields]
            if normalized:
                return normalized
            return required

        return fields

    def build_followup_prompt(self, user_text: str) -> str:
        if not self.pending_tool:
            return user_text

        self.merge_followup_input(user_text)

        still_missing = [
            field for field in self.missing_fields
            if field not in self.collected_data or not self.collected_data[field].strip()
        ]

        if self.pending_tool == "update_device":
            has_device_id = bool(self.collected_data.get("device_id", "").strip())
            has_payload = any(
                str(self.collected_data.get(field, "")).strip()
                for field in UPDATE_FIELDS
            )
            if has_device_id and has_payload:
                still_missing = []

        if still_missing:
            return (
                f"The user is continuing a previous request for tool '{self.pending_tool}'. "
                f"Collected data so far: {self.collected_data}. "
                f"Still missing: {still_missing}. "
                f"User message: {user_text}"
            )

        lines = [f"name={self.pending_tool}"]
        for key, value in self.collected_data.items():
            lines.append(f"{key}={value}")

        return "<tool>\n" + "\n".join(lines) + "\n</tool>"

    def run_turn(self, user_text: str) -> dict[str, Any]:
        if not self.pending_tool and not is_device_related(user_text):
            return build_fe_response(
                context={"entity": "device", "data": {}},
                message="I can only help with device CRUD operations.",
                actions=[],
                need_more_info=False,
                missing_fields=[],
            )

        if not self.pending_tool:
            direct_tool = self._try_direct_create_from_text(user_text)
            if direct_tool is not None:
                tool_result = execute_tool(direct_tool)
                return map_tool_result_to_fe("create_device", tool_result)

        if self.pending_tool:
            synthesized = self.build_followup_prompt(user_text)

            if synthesized.startswith("<tool>"):
                parsed = parse_agent_output(synthesized)
                tool_data = parsed["data"]
                tool_name = tool_data.get("name", "")
                if tool_name == "update_device" and not self._has_update_payload(tool_data):
                    return self._build_update_payload_question()
                tool_result = execute_tool(tool_data)
                if self._should_retry_device_id(tool_name, tool_result):
                    self.pending_tool = tool_name
                    self.missing_fields = ["device_id"]
                    self.collected_data = {}
                    return build_fe_response(
                        context={"entity": "device", "data": {}},
                        message=(
                            f"{tool_result.get('error')} "
                            "Please provide a valid device_id."
                        ),
                        actions=[],
                        need_more_info=True,
                        missing_fields=["device_id"],
                    )
                self.reset_pending()
                return map_tool_result_to_fe(tool_name, tool_result)

            still_missing = [
                field for field in self.missing_fields
                if field not in self.collected_data or not self.collected_data[field].strip()
            ]
            if still_missing:
                return build_fe_response(
                    context={"entity": "device", "data": self.collected_data},
                    message=f"Please provide: {', '.join(still_missing)}",
                    actions=[],
                    need_more_info=True,
                    missing_fields=still_missing,
                )

            self.messages.append({"role": "user", "content": synthesized})
        else:
            self.messages.append({"role": "user", "content": user_text})

        assistant_text = call_model(self.messages)
        parsed = parse_agent_output(assistant_text)

        self.messages.append({"role": "assistant", "content": assistant_text})

        if parsed["kind"] == "ask":
            self.pending_tool = parsed["data"].get("tool") or self.infer_pending_tool(user_text)
            self.missing_fields = self._normalize_missing_fields(
                self.pending_tool,
                parsed.get("missing_fields", []),
            )
            return build_fe_response(
                context={"entity": "device", "data": {}},
                message=parsed["data"].get("question", "Please provide more information."),
                actions=[],
                need_more_info=True,
                missing_fields=self.missing_fields,
            )

        if parsed["kind"] == "tool":
            tool_data = parsed["data"]
            tool_name = tool_data.get("name", "")
            if tool_name == "update_device" and not self._has_update_payload(tool_data):
                self.collected_data = {
                    "device_id": tool_data.get("device_id", "")
                } if tool_data.get("device_id") else {}
                return self._build_update_payload_question()
            tool_result = execute_tool(tool_data)
            if self._should_retry_device_id(tool_name, tool_result):
                self.pending_tool = tool_name
                self.missing_fields = ["device_id"]
                self.collected_data = {}
                return build_fe_response(
                    context={"entity": "device", "data": {}},
                    message=(
                        f"{tool_result.get('error')} "
                        "Please provide a valid device_id."
                    ),
                    actions=[],
                    need_more_info=True,
                    missing_fields=["device_id"],
                )
            self.reset_pending()
            return map_tool_result_to_fe(tool_name, tool_result)

        return build_fe_response(
            context={"entity": "device", "data": {}},
            message=parsed["data"]["message"],
            actions=[],
            need_more_info=False,
            missing_fields=[],
        )

    def infer_pending_tool(self, user_text: str) -> Optional[str]:
        text = user_text.lower()

        if any(k in text for k in ["create", "add", "new"]):
            return "create_device"
        if any(k in text for k in ["update", "edit", "change", "modify"]):
            return "update_device"
        if any(k in text for k in ["delete", "remove"]):
            return "delete_device"
        if any(k in text for k in ["get", "show", "find", "detail"]):
            return "get_device"
        if any(k in text for k in ["list", "all devices"]):
            return "list_devices"

        return None

    def _extract_value(self, text: str, patterns: list[str]) -> Optional[str]:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" .,!?:;")
                if value:
                    return value
        return None

    def _try_direct_create_from_text(self, user_text: str) -> Optional[dict[str, str]]:
        lowered = user_text.lower()
        if "create" not in lowered or "device" not in lowered:
            return None

        name = self._extract_value(
            user_text,
            [
                r"\bname\s+is\s+([a-zA-Z0-9._\-]+)",
                r"\bnamed\s+([a-zA-Z0-9._\-]+)",
                r"\bdevice\s+([a-zA-Z0-9._\-]+)",
            ],
        )
        device_type = self._extract_value(
            user_text,
            [
                r"\btype\s+is\s+([a-zA-Z0-9._\-]+)",
                r"\btypy\s+is\s+([a-zA-Z0-9._\-]+)",
                r"\bof\s+type\s+([a-zA-Z0-9._\-]+)",
            ],
        )

        if not name or not device_type:
            return None

        return {
            "name": "create_device",
            "device_name": name,
            "device_type": device_type,
        }

    def _normalize_field_value(self, field: str, value: str) -> str:
        text = value.strip()
        if not text:
            return text

        if field == "device_name":
            extracted = self._extract_value(
                text,
                [
                    r"^\s*device\s+name\s+is\s+(.+)$",
                    r"^\s*name\s+is\s+(.+)$",
                    r"^\s*named\s+(.+)$",
                ],
            )
            return extracted or text

        if field == "device_type":
            extracted = self._extract_value(
                text,
                [
                    r"^\s*device\s+type\s+is\s+(.+)$",
                    r"^\s*type\s+is\s+(.+)$",
                    r"^\s*typy\s+is\s+(.+)$",
                ],
            )
            return (extracted or text).lower()

        if field == "device_id":
            extracted = self._extract_value(
                text,
                [
                    r"\b(dev-[a-zA-Z0-9_-]+)\b",
                    r"\b(device-[a-zA-Z0-9_-]+)\b",
                    r"\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b",
                    r"^\s*id\s+is\s+([a-zA-Z0-9._\-]+)\s*$",
                ],
            )
            return extracted or text

        return text

    def _should_retry_device_id(self, tool_name: str, tool_result: dict[str, Any]) -> bool:
        if tool_name not in {"delete_device", "get_device", "update_device"}:
            return False
        if tool_result.get("ok"):
            return False
        error = str(tool_result.get("error", "")).lower()
        return "not found" in error and "device" in error

    def _guess_field_from_free_text(self, text: str, candidates: list[str]) -> Optional[str]:
        raw = text.strip()
        lower = raw.lower()

        if "device_id" in candidates:
            device_id = self._normalize_field_value("device_id", raw)
            if (
                device_id != raw
                or re.match(r"^\s*dev-[a-zA-Z0-9_-]+\s*$", raw, re.IGNORECASE)
                or re.match(r"^\s*device-[a-zA-Z0-9_-]+\s*$", raw, re.IGNORECASE)
                or re.match(
                    r"^\s*[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\s*$",
                    raw,
                )
                or lower.startswith("id is ")
                or lower.startswith("device id is ")
            ):
                return "device_id"

        if "device_name" in candidates and re.match(r"^\s*(device\s+name\s+is|name\s+is|named)\b", raw, re.IGNORECASE):
            return "device_name"

        if "device_type" in candidates and re.match(r"^\s*(device\s+type\s+is|type\s+is|typy\s+is)\b", raw, re.IGNORECASE):
            return "device_type"

        if "status" in candidates and re.match(r"^\s*(status\s+is|active|inactive|enabled|disabled|true|false)\b", raw, re.IGNORECASE):
            return "status"

        return None
