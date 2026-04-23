import re
import json
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Optional

from agent import (
    call_model,
    build_fe_response,
    build_missing_info_actions,
    generate_ai_message,
    map_tool_result_to_fe,
    SYSTEM_PROMPT,
)
from parser import parse_agent_output
from device_tools import _call_backend_api, _extract_device, execute_tool
from tool_search import select_tool


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
    "create_device": ["device_name", "device_id", "device_type"],
    "get_device": ["device_id"],
    "delete_device": ["device_id"],
}


def is_device_related(text: str) -> bool:
    t = text.lower()
    keywords = ["device", "devices"]
    if any(k in t for k in keywords):
        return True
    tokens = re.findall(r"[a-z]+", t)
    for keyword in keywords:
        if get_close_matches(keyword, tokens, n=1, cutoff=0.75):
            return True
    return False

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

    def _message(
        self,
        *,
        fallback_message: str,
        user_text: str = "",
        tool_name: Optional[str] = None,
        tool_result: Optional[dict[str, Any]] = None,
        missing_fields: Optional[list[str]] = None,
        intent: str = "final",
    ) -> str:
        return generate_ai_message(
            fallback_message=fallback_message,
            user_text=user_text,
            tool_name=tool_name,
            tool_result=tool_result,
            missing_fields=missing_fields,
            current_data=self.collected_data,
            intent=intent,
        )

    @staticmethod
    def _log_event(event: str, **payload: Any) -> None:
        try:
            line = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **payload,
            }
            with Path("agent_events.log").open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            return

    @staticmethod
    def _append_history(
        chat_history: list[dict[str, Any]],
        user_text: str,
        response: dict[str, Any],
    ) -> list[dict[str, Any]]:
        updated = list(chat_history)
        updated.append({"role": "user", "text": user_text})
        updated.append({"role": "assistant", "response": response})
        return updated

    def hydrate_from_history(self, chat_history: list[dict[str, Any]]) -> None:
        self.reset_pending()
        if not chat_history:
            return

        last_assistant_response: Optional[dict[str, Any]] = None
        last_user_text = ""
        for item in reversed(chat_history):
            if item.get("role") == "assistant" and isinstance(item.get("response"), dict):
                last_assistant_response = item["response"]
                break
        for item in reversed(chat_history):
            if item.get("role") == "user":
                last_user_text = str(item.get("text", ""))
                break

        if not last_assistant_response:
            return
        if not last_assistant_response.get("need_more_info"):
            return

        missing_fields = last_assistant_response.get("missing_fields", [])
        if not isinstance(missing_fields, list):
            missing_fields = []
        self.missing_fields = [str(field) for field in missing_fields if str(field).strip()]

        actions = last_assistant_response.get("actions", [])
        tool_name = None
        if isinstance(actions, list):
            for action in actions:
                if not isinstance(action, dict):
                    continue
                if action.get("tool"):
                    tool_name = str(action.get("tool"))
                    break
                target = str(action.get("target", "")).strip()
                if target == "CreateDevice":
                    tool_name = "create_device"
                    break
                if target == "GetDeviceById":
                    tool_name = "get_device"
                    break
                if target == "UpdateDevice":
                    tool_name = "update_device"
                    break
                if target == "DeleteDevice":
                    tool_name = "delete_device"
                    break
                if target == "GetAllDevices":
                    tool_name = "list_devices"
                    break
                confirm_key = str(action.get("confirmKey", "")).strip()
                if confirm_key == "delete_device":
                    tool_name = "delete_device"
                    break
                if confirm_key == "create_device_connector":
                    break
        if not tool_name:
            tool_name = self.infer_pending_tool(last_user_text)
        if not tool_name:
            for item in reversed(chat_history):
                if item.get("role") != "user":
                    continue
                candidate_text = str(item.get("text", "")).strip()
                if not candidate_text:
                    continue
                candidate_tool = self.infer_pending_tool(candidate_text)
                if candidate_tool:
                    tool_name = candidate_tool
                    break
        if not tool_name:
            tool_name = self._infer_tool_from_assistant_response(last_assistant_response)
        self.pending_tool = tool_name

        context = last_assistant_response.get("context", {})
        data = context.get("data", {}) if isinstance(context, dict) else {}
        if isinstance(data, dict):
            allowed_fields = set(REQUIRED_FIELDS_BY_TOOL.get("create_device", []))
            allowed_fields.update({"device_id", *UPDATE_FIELDS})
            self.collected_data = {
                str(k): str(v)
                for k, v in data.items()
                if str(k) in allowed_fields and str(v).strip()
            }
        self._log_event(
            "hydrate_from_history",
            pending_tool=self.pending_tool,
            missing_fields=self.missing_fields,
            collected_data=self.collected_data,
        )

    @classmethod
    def run_turn_stateless(
        cls,
        user_text: str,
        chat_history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        session = cls()
        session.hydrate_from_history(chat_history)
        response = session.run_turn(user_text)
        return response, cls._append_history(chat_history, user_text, response)

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
            key = self._canonicalize_field_key(key.strip())
            value = self._normalize_field_value(key, value.strip())
            if key in self.missing_fields and value:
                self.collected_data[key] = value

        remaining = [
            field for field in self.missing_fields
            if field not in self.collected_data or not self.collected_data[field].strip()
        ]

        extracted_from_text = self._extract_fields_from_free_text(clean_text, remaining)
        for field, value in extracted_from_text.items():
            if value:
                self.collected_data[field] = value

        remaining = [
            field for field in self.missing_fields
            if field not in self.collected_data or not self.collected_data[field].strip()
        ]

        if not remaining:
            return

        if len(remaining) == 1 and "=" not in clean_text:
            field = remaining[0]
            normalized_value = self._normalize_field_value(field, clean_text)
            if normalized_value == clean_text and " " in clean_text:
                detected_field = self._guess_field_from_free_text(
                    clean_text,
                    REQUIRED_FIELDS_BY_TOOL.get("create_device", []),
                )
                if detected_field and detected_field != field:
                    return
            self.collected_data[field] = normalized_value
            return

        # If user explicitly used key=value, do not fallback to positional mapping.
        # This avoids accidental assignments like:
        #   device_type = "device_name=sensor-4"
        if has_equals:
            return

        if len(remaining) > 1 and "=" not in clean_text:
            # For free-text follow-ups with multiple remaining fields, detect target field
            # from content first to avoid incorrect positional assignment.
            detected_field = self._guess_field_from_free_text(clean_text, remaining)
            if detected_field:
                self.collected_data[detected_field] = self._normalize_field_value(
                    detected_field,
                    clean_text,
                )
                return

        if self.pending_tool == "update_device" and len(remaining) > 1 and "=" not in clean_text:
            # For update flows with many optional fields, positional mapping is ambiguous.
            return

        if self.pending_tool == "create_device" and len(remaining) > 1 and "=" not in clean_text:
            # For create flow, avoid positional mapping when multiple fields are still
            # missing because free-text often contains mixed structured phrases.
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
        fallback_message = (
            "Please provide at least one update field. "
            "You can use key=value or free text. "
            "Examples: status=inactive, device_name=sensor-2, or device_type is gateway."
        )
        return build_fe_response(
            context={"entity": "device", "data": self.collected_data},
            message=self._message(
                fallback_message=fallback_message,
                tool_name="update_device",
                missing_fields=self.missing_fields,
                intent="missing_info",
            ),
            actions=build_missing_info_actions(
                tool_name=self.pending_tool,
                missing_fields=self.missing_fields,
                current_data=self.collected_data,
            ),
            need_more_info=True,
            missing_fields=self.missing_fields,
        )

    def _build_delete_confirmation(self, device: dict[str, Any], user_text: str = "") -> dict[str, Any]:
        device_id = str(device.get("deviceId") or device.get("id") or "").strip()
        device_name = str(device.get("name", "")).strip()
        device_type = str(device.get("type", "")).strip()

        self.pending_tool = "delete_device"
        self.missing_fields = []
        self.collected_data = {
            "device_id": device_id,
            "device_name": device_name,
            "device_type": device_type,
        }
        backend_device_id = str(device.get("id", "")).strip()
        if backend_device_id:
            self.collected_data["backend_device_id"] = backend_device_id

        details: list[str] = []
        if device_name:
            details.append(f"name={device_name}")
        if device_id:
            details.append(f"device_id={device_id}")
        if device_type:
            details.append(f"type={device_type}")
        detail_text = ", ".join(details) if details else "the selected device"
        fallback_message = f"Are you sure you want to delete this device? {detail_text}"

        return build_fe_response(
            context={"entity": "device", "data": device},
            message=self._message(
                fallback_message=fallback_message,
                user_text=user_text,
                tool_name="delete_device",
                tool_result={"ok": True, "device": device},
                intent="confirmation",
            ),
            actions=[
                {
                    "kind": "confirm",
                    "label": "Delete device",
                    "target": "DeleteDevice",
                    "confirmKey": "delete_device",
                    "title": "Delete device",
                    "message": fallback_message,
                    "variant": "danger",
                }
            ],
            data={"device": device},
            need_more_info=True,
            missing_fields=[],
        )

    def _ask_for_valid_delete_device_id(self, error_text: str = "") -> dict[str, Any]:
        self.pending_tool = "delete_device"
        self.missing_fields = ["device_id"]
        self.collected_data = {}
        fallback_message = self._retry_device_id_message("delete_device", error_text)
        return build_fe_response(
            context={"entity": "device", "data": {}},
            message=self._message(
                fallback_message=fallback_message,
                tool_name="delete_device",
                missing_fields=["device_id"],
                intent="missing_info",
            ),
            actions=[],
            need_more_info=True,
            missing_fields=["device_id"],
        )

    def _is_delete_confirmation_input(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        if normalized in {"__confirm__:delete_device", "__action__:deletedevice"}:
            return True
        if normalized in {"yes", "y", "confirm", "delete", "delete device", "do it"}:
            return True
        return False

    def _is_cancel_input(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        return normalized in {"no", "n", "cancel", "stop", "لغو", "نه"}

    def _lookup_device_for_delete_confirmation(self, device_id: str) -> dict[str, Any]:
        needle = str(device_id or "").strip()
        if not needle:
            return {"ok": False, "error": "Missing required field for delete_device: device_id"}

        list_lookup_attempt_failed = False
        try:
            listed = execute_tool({"name": "list_devices"})
        except Exception:
            listed = {"ok": False}
            list_lookup_attempt_failed = True
        if listed.get("ok"):
            for item in listed.get("devices", []):
                if not isinstance(item, dict):
                    continue
                candidate_device_id = str(item.get("deviceId", "")).strip()
                candidate_internal_id = str(item.get("id", "")).strip()
                if (
                    candidate_device_id == needle
                    or candidate_internal_id == needle
                    or candidate_device_id.lower() == needle.lower()
                    or candidate_internal_id.lower() == needle.lower()
                ):
                    return {"ok": True, "device": item}
            return {"ok": False, "error": f"Device '{needle}' not found."}

        if list_lookup_attempt_failed:
            direct = _call_backend_api("GET", f"/device/{needle}")
            if not direct.get("ok"):
                return direct
            device = _extract_device(direct.get("data"))
            if device:
                return {"ok": True, "device": device}
        return {"ok": False, "error": f"Device '{needle}' not found."}

    def _try_handle_delete_followup(self, user_text: str) -> Optional[dict[str, Any]]:
        if self.pending_tool != "delete_device":
            return None

        if self._is_cancel_input(user_text):
            self.reset_pending()
            return build_fe_response(
                context={"entity": "device", "data": {}},
                message=self._message(
                    fallback_message="Device deletion was canceled.",
                    user_text=user_text,
                    tool_name="delete_device",
                    intent="final",
                ),
                actions=[],
                need_more_info=False,
                missing_fields=[],
            )

        if self._is_delete_confirmation_input(user_text):
            device_id = str(self.collected_data.get("device_id", "")).strip()
            backend_device_id = str(self.collected_data.get("backend_device_id", "")).strip()
            if not device_id:
                return self._ask_for_valid_delete_device_id()
            delete_target = backend_device_id or device_id
            direct_delete = _call_backend_api("DELETE", f"/device/{delete_target}")
            if direct_delete.get("ok"):
                deleted_device = _extract_device(direct_delete.get("data"))
                if not deleted_device:
                    deleted_device = {"deviceId": device_id}
                    if backend_device_id:
                        deleted_device["id"] = backend_device_id
                tool_result = {
                    "ok": True,
                    "deleted": True,
                    "device": deleted_device,
                    "raw": direct_delete.get("data"),
                }
            else:
                tool_result = execute_tool({"name": "delete_device", "device_id": device_id})
            self._log_event(
                "tool_executed",
                tool="delete_device",
                tool_data={"name": "delete_device", "device_id": device_id},
                tool_result=tool_result,
                path="delete_confirmation",
            )
            if self._should_retry_device_id("delete_device", tool_result):
                return self._ask_for_valid_delete_device_id(str(tool_result.get("error", "")))
            self.reset_pending()
            return map_tool_result_to_fe("delete_device", tool_result, user_text=user_text)

        extracted = self._extract_fields_from_free_text(user_text, ["device_id"])
        candidate_device_id = str(
            extracted.get("device_id") or self.collected_data.get("device_id", "")
        ).strip()

        if not candidate_device_id and "=" in user_text:
            candidate_device_id = self._parse_key_value_fields(
                user_text,
                ["device_id"],
            ).get("device_id", "").strip()

        token = (user_text or "").strip()
        if not candidate_device_id and token and " " not in token and "=" not in token and "," not in token:
            candidate_device_id = self._normalize_field_value("device_id", token)

        if not candidate_device_id:
            return self._ask_for_valid_delete_device_id()

        device_lookup = self._lookup_device_for_delete_confirmation(candidate_device_id)
        if not device_lookup.get("ok"):
            return self._ask_for_valid_delete_device_id(str(device_lookup.get("error", "")))

        return self._build_delete_confirmation(device_lookup["device"], user_text)

    def _parse_key_value_fields(
        self,
        text: str,
        allowed_fields: list[str],
    ) -> dict[str, str]:
        parsed: dict[str, str] = {}
        clean_text = (text or "").strip()
        if not clean_text:
            return parsed

        allowed = set(allowed_fields)
        for piece in clean_text.replace("\n", ",").split(","):
            part = piece.strip()
            if "=" not in part:
                continue
            raw_key, raw_value = part.split("=", 1)
            key = self._canonicalize_field_key(raw_key.strip())
            if key not in allowed:
                continue
            value = self._normalize_field_value(key, raw_value.strip())
            if value:
                parsed[key] = value
        return parsed

    def _try_handle_update_followup(self, user_text: str) -> Optional[dict[str, Any]]:
        if self.pending_tool != "update_device":
            return None

        key_values = self._parse_key_value_fields(
            user_text,
            ["device_id", *UPDATE_FIELDS],
        )
        for key, value in key_values.items():
            self.collected_data[key] = value

        from_text = self._extract_fields_from_free_text(
            user_text,
            ["device_id", *UPDATE_FIELDS],
        )
        for field, value in from_text.items():
            if value:
                self.collected_data[field] = value

        if (
            not str(self.collected_data.get("device_id", "")).strip()
            and "device_id" in self.missing_fields
        ):
            token = (user_text or "").strip()
            if token and " " not in token and "=" not in token and "," not in token:
                self.collected_data["device_id"] = self._normalize_field_value("device_id", token)

        device_id = str(self.collected_data.get("device_id", "")).strip()
        has_payload = any(
            str(self.collected_data.get(field, "")).strip()
            for field in UPDATE_FIELDS
        )

        if not device_id:
            self.pending_tool = "update_device"
            self.missing_fields = ["device_id"]
            return build_fe_response(
                context={"entity": "device", "data": self.collected_data},
                message=self._message(
                    fallback_message="Please provide device_id.",
                    user_text=user_text,
                    tool_name="update_device",
                    missing_fields=["device_id"],
                    intent="missing_info",
                ),
                actions=[],
                need_more_info=True,
                missing_fields=["device_id"],
            )

        if not self._ensure_valid_update_device_id(device_id):
            return self._ask_for_valid_update_device_id()

        if not has_payload:
            return self._build_update_payload_question()

        tool_payload = {"name": "update_device", "device_id": device_id}
        for field in UPDATE_FIELDS:
            value = str(self.collected_data.get(field, "")).strip()
            if value:
                tool_payload[field] = value

        tool_result = execute_tool(tool_payload)
        self._log_event(
            "tool_executed",
            tool="update_device",
            tool_data=tool_payload,
            tool_result=tool_result,
            path="update_fast_followup",
        )
        if self._should_retry_device_id("update_device", tool_result):
            return self._ask_for_valid_update_device_id()
        self.reset_pending()
        return map_tool_result_to_fe("update_device", tool_result)

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
                message=self._message(
                    fallback_message="I can only help with device CRUD operations.",
                    user_text=user_text,
                    intent="error",
                ),
                actions=[],
                need_more_info=False,
                missing_fields=[],
            )

        if not self.pending_tool:
            direct_tool = self._try_direct_create_from_text(user_text)
            if direct_tool is not None:
                tool_result = execute_tool(direct_tool)
                return map_tool_result_to_fe("create_device", tool_result, user_text=user_text)
            inferred_tool = self.infer_pending_tool(user_text)
            if not inferred_tool and is_device_related(user_text):
                self._log_event("ambiguous_intent", user_text=user_text)
                fallback_message = (
                    "I can help with device operations: create, get, update, delete, list. "
                    "Please specify one operation."
                )
                return build_fe_response(
                    context={"entity": "device", "data": {}},
                    message=self._message(
                        fallback_message=fallback_message,
                        user_text=user_text,
                        intent="missing_info",
                    ),
                    actions=[],
                    need_more_info=True,
                    missing_fields=[],
                )
            if inferred_tool == "list_devices":
                return map_tool_result_to_fe(
                    "list_devices",
                    execute_tool({"name": "list_devices"}),
                    user_text=user_text,
                )
            if inferred_tool == "get_device":
                device_id = self._extract_fields_from_free_text(
                    user_text, ["device_id"]
                ).get("device_id", "")
                if device_id:
                    return map_tool_result_to_fe(
                        "get_device",
                        execute_tool({"name": "get_device", "device_id": device_id}),
                        user_text=user_text,
                    )
                self.pending_tool = "get_device"
                self.missing_fields = ["device_id"]
                self.collected_data = {}
                return build_fe_response(
                    context={"entity": "device", "data": {}},
                    message=self._message(
                        fallback_message="Please provide device_id.",
                        user_text=user_text,
                        tool_name="get_device",
                        missing_fields=["device_id"],
                        intent="missing_info",
                    ),
                    actions=[],
                    need_more_info=True,
                    missing_fields=["device_id"],
                )
            if inferred_tool == "delete_device":
                device_id = self._extract_fields_from_free_text(
                    user_text, ["device_id"]
                ).get("device_id", "")
                if device_id:
                    device_lookup = self._lookup_device_for_delete_confirmation(device_id)
                    if not device_lookup.get("ok"):
                        return self._ask_for_valid_delete_device_id(
                            str(device_lookup.get("error", ""))
                        )
                    return self._build_delete_confirmation(device_lookup["device"], user_text)
                self.pending_tool = "delete_device"
                self.missing_fields = ["device_id"]
                self.collected_data = {}
                return build_fe_response(
                    context={"entity": "device", "data": {}},
                    message=self._message(
                        fallback_message="Please provide device_id.",
                        user_text=user_text,
                        tool_name="delete_device",
                        missing_fields=["device_id"],
                        intent="missing_info",
                    ),
                    actions=[],
                    need_more_info=True,
                    missing_fields=["device_id"],
                )
            if inferred_tool == "update_device":
                update_data = self._collect_initial_update_data(user_text)
                device_id = update_data.get("device_id", "")
                has_payload = any(
                    str(update_data.get(field, "")).strip()
                    for field in UPDATE_FIELDS
                )
                if device_id and has_payload:
                    tool_payload = {"name": "update_device", **update_data}
                    return map_tool_result_to_fe(
                        "update_device",
                        execute_tool(tool_payload),
                        user_text=user_text,
                    )
                if not device_id:
                    self.pending_tool = "update_device"
                    self.missing_fields = ["device_id"]
                    self.collected_data = update_data
                    return build_fe_response(
                        context={"entity": "device", "data": self.collected_data},
                        message=self._message(
                            fallback_message="Please provide device_id.",
                            user_text=user_text,
                            tool_name="update_device",
                            missing_fields=["device_id"],
                            intent="missing_info",
                        ),
                        actions=[],
                        need_more_info=True,
                        missing_fields=["device_id"],
                    )
                if not self._ensure_valid_update_device_id(device_id):
                    return self._ask_for_valid_update_device_id()
                self.pending_tool = "update_device"
                self.collected_data = {"device_id": device_id}
                return self._build_update_payload_question()
            if inferred_tool == "create_device":
                self.pending_tool = "create_device"
                self.missing_fields = REQUIRED_FIELDS_BY_TOOL["create_device"].copy()
                self.collected_data = self._collect_initial_create_data(user_text)
                still_missing = [
                    field for field in self.missing_fields
                    if not str(self.collected_data.get(field, "")).strip()
                ]
                if still_missing:
                    self.missing_fields = still_missing
                    return build_fe_response(
                        context={"entity": "device", "data": self.collected_data},
                        message=self._message(
                            fallback_message=(
                                "To create a new device, please provide these fields: "
                                "name, deviceId, type."
                            ),
                            user_text=user_text,
                            tool_name="create_device",
                            missing_fields=self.missing_fields,
                            intent="missing_info",
                        ),
                        actions=build_missing_info_actions(
                            tool_name="create_device",
                            missing_fields=self.missing_fields,
                            current_data=self.collected_data,
                        ),
                        need_more_info=True,
                        missing_fields=self.missing_fields,
                    )

        if self.pending_tool:
            delete_followup_response = self._try_handle_delete_followup(user_text)
            if delete_followup_response is not None:
                return delete_followup_response

            fast_update_response = self._try_handle_update_followup(user_text)
            if fast_update_response is not None:
                return fast_update_response

            synthesized = self.build_followup_prompt(user_text)

            if synthesized.startswith("<tool>"):
                if self.pending_tool == "create_device" and not self._is_create_confirmation_input(user_text):
                    return build_fe_response(
                        context={"entity": "device", "data": self.collected_data},
                        message=self._message(
                            fallback_message="I have all required fields. Do you want me to create the device now?",
                            user_text=user_text,
                            tool_name="create_device",
                            intent="confirmation",
                        ),
                        actions=[
                            {
                                "type": "button",
                                "label": "Create Device",
                                "variant": "success",
                                "target": "CreateDevice",
                            }
                        ],
                        need_more_info=True,
                        missing_fields=[],
                    )
                parsed = parse_agent_output(synthesized)
                tool_data = parsed["data"]
                selection = select_tool(
                    user_text=user_text,
                    candidate_tool=tool_data.get("name", ""),
                    tool_data=tool_data,
                    pending_tool=self.pending_tool,
                )
                tool_name = selection.tool_name or str(tool_data.get("name", "")).strip()
                tool_data["name"] = tool_name
                if tool_name == "update_device" and not self._has_update_payload(tool_data):
                    if not self._ensure_valid_update_device_id(tool_data.get("device_id", "")):
                        return self._ask_for_valid_update_device_id()
                    return self._build_update_payload_question()
                tool_result = execute_tool(tool_data)
                self._log_event("tool_executed", tool=tool_name, tool_data=tool_data, tool_result=tool_result)
                if self._should_retry_create_fields(tool_name, tool_result):
                    missing_fields = self._missing_create_fields_from_error(
                        str(tool_result.get("error", ""))
                    )
                    self.pending_tool = "create_device"
                    self.missing_fields = missing_fields
                    self.collected_data = {
                        "device_name": str(tool_data.get("device_name", "")).strip(),
                        "device_id": str(tool_data.get("device_id", "")).strip(),
                        "device_type": str(tool_data.get("device_type", "")).strip(),
                    }
                    return build_fe_response(
                        context={"entity": "device", "data": self.collected_data},
                        message=self._message(
                            fallback_message=self._friendly_create_error_message(
                                str(tool_result.get("error", ""))
                            ),
                            user_text=user_text,
                            tool_name="create_device",
                            tool_result=tool_result,
                            missing_fields=missing_fields,
                            intent="missing_info",
                        ),
                        actions=build_missing_info_actions(
                            tool_name="create_device",
                            missing_fields=missing_fields,
                            current_data=self.collected_data,
                        ),
                        need_more_info=True,
                        missing_fields=missing_fields,
                    )
                if self._should_retry_device_id(tool_name, tool_result):
                    self.pending_tool = tool_name
                    self.missing_fields = ["device_id"]
                    self.collected_data = {}
                    return build_fe_response(
                        context={"entity": "device", "data": {}},
                        message=self._message(
                            fallback_message=self._retry_device_id_message(
                                tool_name,
                                str(tool_result.get("error", "")),
                            ),
                            user_text=user_text,
                            tool_name=tool_name,
                            tool_result=tool_result,
                            missing_fields=["device_id"],
                            intent="missing_info",
                        ),
                        actions=[],
                        need_more_info=True,
                        missing_fields=["device_id"],
                    )
                self.reset_pending()
                return map_tool_result_to_fe(tool_name, tool_result, user_text=user_text)

            still_missing = [
                field for field in self.missing_fields
                if field not in self.collected_data or not self.collected_data[field].strip()
            ]
            if still_missing:
                return build_fe_response(
                    context={"entity": "device", "data": self.collected_data},
                    message=self._message(
                        fallback_message=f"Please provide: {', '.join(still_missing)}",
                        user_text=user_text,
                        tool_name=self.pending_tool,
                        missing_fields=still_missing,
                        intent="missing_info",
                    ),
                    actions=build_missing_info_actions(
                        tool_name=self.pending_tool,
                        missing_fields=still_missing,
                        current_data=self.collected_data,
                    ),
                    need_more_info=True,
                    missing_fields=still_missing,
                )

            self.messages.append({"role": "user", "content": synthesized})
        else:
            self.messages.append({"role": "user", "content": user_text})

        try:
            assistant_text = call_model(self.messages)
        except RuntimeError as exc:
            self._log_event("model_error", user_text=user_text, error=str(exc))
            return build_fe_response(
                context={"entity": "device", "data": {}},
                message=self._message(
                    fallback_message=f"{exc}. Please try again.",
                    user_text=user_text,
                    intent="error",
                ),
                actions=[],
                need_more_info=False,
                missing_fields=[],
            )
        parsed = parse_agent_output(assistant_text)

        self.messages.append({"role": "assistant", "content": assistant_text})

        if parsed["kind"] == "ask":
            self.pending_tool = parsed["data"].get("tool") or self.infer_pending_tool(user_text)
            self.missing_fields = self._normalize_missing_fields(
                self.pending_tool,
                parsed.get("missing_fields", []),
            )
            fallback_message = parsed["data"].get("question", "Please provide more information.")
            return build_fe_response(
                context={"entity": "device", "data": {}},
                message=self._message(
                    fallback_message=fallback_message,
                    user_text=user_text,
                    tool_name=self.pending_tool,
                    missing_fields=self.missing_fields,
                    intent="missing_info",
                ),
                actions=build_missing_info_actions(
                    tool_name=self.pending_tool,
                    missing_fields=self.missing_fields,
                    current_data=self.collected_data,
                ),
                need_more_info=True,
                missing_fields=self.missing_fields,
            )

        if parsed["kind"] == "tool":
            tool_data = parsed["data"]
            selection = select_tool(
                user_text=user_text,
                candidate_tool=tool_data.get("name", ""),
                tool_data=tool_data,
                pending_tool=self.pending_tool,
            )
            tool_name = selection.tool_name or str(tool_data.get("name", "")).strip()
            tool_data["name"] = tool_name
            if tool_name == "update_device" and not self._has_update_payload(tool_data):
                if not self._ensure_valid_update_device_id(tool_data.get("device_id", "")):
                    return self._ask_for_valid_update_device_id()
                self.collected_data = {
                    "device_id": tool_data.get("device_id", "")
                } if tool_data.get("device_id") else {}
                return self._build_update_payload_question()
            tool_result = execute_tool(tool_data)
            self._log_event("tool_executed", tool=tool_name, tool_data=tool_data, tool_result=tool_result)
            if self._should_retry_create_fields(tool_name, tool_result):
                missing_fields = self._missing_create_fields_from_error(
                    str(tool_result.get("error", ""))
                )
                self.pending_tool = "create_device"
                self.missing_fields = missing_fields
                self.collected_data = {
                    "device_name": str(tool_data.get("device_name", "")).strip(),
                    "device_id": str(tool_data.get("device_id", "")).strip(),
                    "device_type": str(tool_data.get("device_type", "")).strip(),
                }
                return build_fe_response(
                    context={"entity": "device", "data": self.collected_data},
                    message=self._message(
                        fallback_message=self._friendly_create_error_message(
                            str(tool_result.get("error", ""))
                        ),
                        user_text=user_text,
                        tool_name="create_device",
                        tool_result=tool_result,
                        missing_fields=missing_fields,
                        intent="missing_info",
                    ),
                    actions=build_missing_info_actions(
                        tool_name="create_device",
                        missing_fields=missing_fields,
                        current_data=self.collected_data,
                    ),
                    need_more_info=True,
                    missing_fields=missing_fields,
                )
            if self._should_retry_device_id(tool_name, tool_result):
                self.pending_tool = tool_name
                self.missing_fields = ["device_id"]
                self.collected_data = {}
                return build_fe_response(
                    context={"entity": "device", "data": {}},
                    message=self._message(
                        fallback_message=self._retry_device_id_message(
                            tool_name,
                            str(tool_result.get("error", "")),
                        ),
                        user_text=user_text,
                        tool_name=tool_name,
                        tool_result=tool_result,
                        missing_fields=["device_id"],
                        intent="missing_info",
                    ),
                    actions=[],
                    need_more_info=True,
                    missing_fields=["device_id"],
                )
            self.reset_pending()
            return map_tool_result_to_fe(tool_name, tool_result, user_text=user_text)

        return build_fe_response(
            context={"entity": "device", "data": {}},
            message=self._message(
                fallback_message=parsed["data"]["message"],
                user_text=user_text,
                intent="final",
            ),
            actions=[],
            need_more_info=False,
            missing_fields=[],
        )

    def infer_pending_tool(self, user_text: str) -> Optional[str]:
        selection = select_tool(user_text=user_text)
        return selection.tool_name

    def _extract_value(self, text: str, patterns: list[str]) -> Optional[str]:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" .,!?:;")
                if value:
                    return value
        return None

    def _name_field_boundary(self) -> str:
        return (
            r"(?=,|"
            r"\s+(?:and\s+)?(?:device\s+id|id\s+is|device\s+type|type\s+is|typy\s+is|"
            r"status\s+is|group\s+is|role\s+is|description\s+is|save[_\s]?data\s+is|"
            r"tags\s+are|features\s+are)\b|$)"
        )

    def _try_direct_create_from_text(self, user_text: str) -> Optional[dict[str, str]]:
        lowered = user_text.lower()
        if "create" not in lowered or "device" not in lowered:
            return None

        name = self._extract_value(
            user_text,
            [
                rf"\bname\s+is\s+(.+?){self._name_field_boundary()}",
                rf"\bnamed\s+(.+?){self._name_field_boundary()}",
                rf"\bdevice\s+(.+?){self._name_field_boundary()}",
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
        device_id = self._extract_value(
            user_text,
            [
                r"\bdevice\s+id\s+is\s+([a-zA-Z0-9._\-]+)",
                r"\bid\s+is\s+([a-zA-Z0-9._\-]+)",
            ],
        )

        if not name or not device_type or not device_id:
            return None

        return {
            "name": "create_device",
            "device_name": name,
            "device_id": device_id,
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
                    r"^\s*device[_\s]?type\s+is\s+(.+)$",
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
                    r"^\s*((?:ai-)?dev-[a-zA-Z0-9._\-]+)\s*$",
                    r"^\s*device\s+id\s+is\s+([a-zA-Z0-9._\-]+)\s*$",
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
        return (
            ("not found" in error and "device" in error)
            or "invalid device id" in error
            or "not a valid objectid" in error
        )

    def _retry_device_id_message(self, tool_name: str, error_text: str) -> str:
        hint = {
            "get_device": "Please provide a valid device_id to get the device details.",
            "update_device": "Please provide a valid device_id to update the device.",
            "delete_device": "Please provide a valid device_id to delete the device.",
        }.get(tool_name, "Please provide a valid device_id.")
        return f"{error_text} {hint}".strip()

    def _ensure_valid_update_device_id(self, device_id: Any) -> bool:
        value = str(device_id or "").strip()
        if not value:
            return False
        check = execute_tool({"name": "get_device", "device_id": value})
        return bool(check.get("ok"))

    def _ask_for_valid_update_device_id(self) -> dict[str, Any]:
        self.pending_tool = "update_device"
        self.missing_fields = ["device_id"]
        self.collected_data = {}
        return build_fe_response(
            context={"entity": "device", "data": {}},
            message=self._message(
                fallback_message="Device not found or device_id is invalid. Please provide a valid device_id.",
                tool_name="update_device",
                missing_fields=["device_id"],
                intent="missing_info",
            ),
            actions=[],
            need_more_info=True,
            missing_fields=["device_id"],
        )

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

        if "device_type" in candidates and re.match(
            r"^\s*(device[_\s]?type\s+is|type\s+is|typy\s+is)\b",
            raw,
            re.IGNORECASE,
        ):
            return "device_type"

        if "status" in candidates and re.match(r"^\s*(status\s+is|active|inactive|enabled|disabled|true|false)\b", raw, re.IGNORECASE):
            return "status"

        return None

    def _should_retry_create_fields(self, tool_name: str, tool_result: dict[str, Any]) -> bool:
        if tool_name != "create_device":
            return False
        if tool_result.get("ok"):
            return False
        error = str(tool_result.get("error", "")).lower()
        return (
            "missing required fields for create_device" in error
            or "invalid device type" in error
        )

    def _missing_create_fields_from_error(self, error_message: str) -> list[str]:
        error = error_message.lower()
        missing: list[str] = []
        if "device_name" in error:
            missing.append("device_name")
        if "device_id" in error:
            missing.append("device_id")
        if "device_type" in error or "invalid device type" in error:
            missing.append("device_type")
        return missing or REQUIRED_FIELDS_BY_TOOL["create_device"].copy()

    def _friendly_create_error_message(self, error_message: str) -> str:
        lowered = error_message.lower()
        if "invalid device type" in lowered:
            return (
                "Invalid device type. Allowed values are: "
                "sensor, gateway, controller, processor, ipcamera."
            )
        if "missing required fields for create_device" in lowered:
            return (
                "To create a new device, please provide all required fields: "
                "device_name, device_id, device_type."
            )
        return error_message

    def _collect_initial_create_data(self, user_text: str) -> dict[str, str]:
        data: dict[str, str] = {}
        for piece in user_text.replace("\n", ",").split(","):
            part = piece.strip()
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            if key in REQUIRED_FIELDS_BY_TOOL["create_device"]:
                data[key] = self._normalize_field_value(key, value.strip())

        if "device_name" not in data:
            name = self._extract_fields_from_free_text(user_text, ["device_name"]).get(
                "device_name"
            )
            if name:
                data["device_name"] = name

        if "device_type" not in data:
            device_type = self._extract_fields_from_free_text(
                user_text, ["device_type"]
            ).get("device_type")
            if device_type:
                data["device_type"] = device_type.lower()

        if "device_id" not in data:
            device_id = self._extract_fields_from_free_text(user_text, ["device_id"]).get(
                "device_id"
            )
            if device_id:
                data["device_id"] = device_id

        return data

    def _extract_fields_from_free_text(
        self,
        text: str,
        candidate_fields: list[str],
    ) -> dict[str, str]:
        found: dict[str, str] = {}

        if "device_id" in candidate_fields:
            device_id = self._extract_value(
                text,
                [
                    r"\bdevice[_\s]?id\s*[:=]\s*([a-zA-Z0-9._\-]+)\b",
                    r"\bid\s*[:=]\s*([a-zA-Z0-9._\-]+)\b",
                    r"\bdevice\s+id\s+is\s+([a-zA-Z0-9._\-]+)\b",
                    r"\bdevice\s+id\s+([a-zA-Z0-9._\-]+)\b",
                    r"\bid\s+is\s+([a-zA-Z0-9._\-]+)\b",
                    r"\bid\s+([a-zA-Z0-9._\-]+)\b",
                    r"\b([a-zA-Z0-9._\-]+)\s+is\s+id\b",
                    r"^\s*((?:ai-)?dev-[a-zA-Z0-9._\-]+)\s*$",
                    r"^\s*(dev-[a-zA-Z0-9_-]+)\s*$",
                    r"^\s*(device-[a-zA-Z0-9_-]+)\s*$",
                    r"\b([0-9]{6,})\b",
                ],
            )
            if device_id:
                found["device_id"] = device_id

        if "device_type" in candidate_fields:
            device_type = self._extract_value(
                text,
                [
                    r"\bdevice[_\s]?type\s+is\s+([a-zA-Z0-9._\-]+)\b",
                    r"\bdevice\s+type\s+is\s+([a-zA-Z0-9._\-]+)\b",
                    r"\btype\s+is\s+([a-zA-Z0-9._\-]+)\b",
                    r"\b([a-zA-Z0-9._\-]+)\s+is\s+type\b",
                    r"\btypy\s+is\s+([a-zA-Z0-9._\-]+)\b",
                    r"\bof\s+type\s+([a-zA-Z0-9._\-]+)\b",
                    r"\bset\s+type\s+to\s+([a-zA-Z0-9._\-]+)\b",
                    r"\btype\s*[:=]\s*([a-zA-Z0-9._\-]+)\b",
                    r"\bdevice[_\s]?type\s+([a-zA-Z0-9._\-]+)\b(?!\s+is\b)",
                    r"\btype\s+([a-zA-Z0-9._\-]+)\b(?!\s+is\b)",
                ],
            )
            if device_type:
                found["device_type"] = device_type.lower()

        if "device_name" in candidate_fields:
            device_name = self._extract_value(
                text,
                [
                    rf"\bdevice\s+name\s+is\s+(.+?){self._name_field_boundary()}",
                    rf"\bname\s+is\s+(.+?){self._name_field_boundary()}",
                    rf"\bnam\s+is\s+(.+?){self._name_field_boundary()}",
                    rf"\bmame\s+is\s+(.+?){self._name_field_boundary()}",
                    rf"\bnamed\s+(.+?){self._name_field_boundary()}",
                    r"\b(.+?)\s+is\s+name\b",
                    rf"\bname\s+(.+?){self._name_field_boundary()}",
                ],
            )
            if device_name:
                found["device_name"] = device_name

        return found

    def _collect_initial_update_data(self, user_text: str) -> dict[str, str]:
        data: dict[str, str] = {}

        extracted = self._extract_fields_from_free_text(
            user_text,
            ["device_id", "device_name", "device_type"],
        )
        data.update(extracted)

        role = self._extract_value(
            user_text,
            [r"\brole\s+is\s+([a-zA-Z0-9._\-]+)\b", r"\bset\s+role\s+to\s+([a-zA-Z0-9._\-]+)\b"],
        )
        if role:
            data["role"] = role

        description = self._extract_value(
            user_text,
            [
                r"\bdescription\s+is\s+(.+)$",
                r"\bset\s+description\s+to\s+(.+)$",
            ],
        )
        if description:
            data["description"] = description

        group = self._extract_value(
            user_text,
            [r"\bgroup\s+is\s+([a-zA-Z0-9._\-]+)\b", r"\bset\s+group\s+to\s+([a-zA-Z0-9._\-]+)\b"],
        )
        if group:
            data["group"] = group

        save_data = self._extract_value(
            user_text,
            [
                r"\bsave[_\s]?data\s+is\s+(true|false|active|inactive|enabled|disabled|yes|no|1|0)\b",
                r"\bset\s+save[_\s]?data\s+to\s+(true|false|active|inactive|enabled|disabled|yes|no|1|0)\b",
            ],
        )
        if save_data:
            data["save_data"] = save_data

        status = self._extract_value(
            user_text,
            [
                r"\bstatus\s+is\s+(active|inactive|enabled|disabled|true|false|1|0)\b",
                r"\bset\s+status\s+to\s+(active|inactive|enabled|disabled|true|false|1|0)\b",
            ],
        )
        if status:
            data["status"] = status

        tags = self._extract_value(
            user_text,
            [
                r"\btags\s+are\s+([a-zA-Z0-9,._\-\s]+)\b",
                r"\bset\s+tags\s+to\s+([a-zA-Z0-9,._\-\s]+)\b",
            ],
        )
        if tags:
            data["tags"] = tags

        return data

    def _matches_intent_keywords(self, text: str, keywords: list[str]) -> bool:
        if any(keyword in text for keyword in keywords):
            return True

        tokens = re.findall(r"[a-z]+", text)
        for keyword in keywords:
            if " " in keyword:
                continue
            if get_close_matches(keyword, tokens, n=1, cutoff=0.75):
                return True
        return False

    def _is_create_confirmation_input(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        if normalized == "__action__:createdevice":
            return True
        if normalized in {"yes", "y", "confirm", "do it", "create", "create device"}:
            return True
        return False

    def _infer_tool_from_assistant_response(self, response: dict[str, Any]) -> Optional[str]:
        missing = response.get("missing_fields", [])
        if not isinstance(missing, list):
            missing = []
        missing_set = {str(item) for item in missing}
        update_set = set(UPDATE_FIELDS)
        create_set = set(REQUIRED_FIELDS_BY_TOOL["create_device"])

        if missing_set and missing_set.issubset(update_set):
            return "update_device"
        if missing_set == create_set or create_set.issubset(missing_set):
            return "create_device"
        if missing_set == {"device_id"}:
            msg = str(response.get("message", "")).lower()
            if "delete" in msg:
                return "delete_device"
            if "update" in msg:
                return "update_device"
            if "get" in msg or "detail" in msg or "show" in msg:
                return "get_device"
        return None

    def _canonicalize_field_key(self, raw_key: str) -> str:
        key = raw_key.strip().lower().replace("-", "_").replace(" ", "_")

        alias_map = {
            "name": "device_name",
            "nam": "device_name",
            "mame": "device_name",
            "device_name": "device_name",
            "device": "device_name",
            "id": "device_id",
            "deviceid": "device_id",
            "device_id": "device_id",
            "type": "device_type",
            "device_type": "device_type",
            "devicetype": "device_type",
        }
        if key in alias_map:
            return alias_map[key]

        known = [
            "device_name",
            "device_id",
            "device_type",
            "role",
            "description",
            "group",
            "save_data",
            "status",
            "tags",
            "features",
        ]
        matched = get_close_matches(key, known, n=1, cutoff=0.8)
        if matched:
            return matched[0]

        return key
