from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ALLOWED_DEVICE_TYPES = {"sensor", "gateway", "controller", "processor", "ipcamera"}
TOOL_CATALOG: dict[str, dict[str, Any]] = {
    "create_device": {
        "description": "Create a new device record.",
        "required_fields": ["device_name", "device_id", "device_type"],
        "keywords": ["create", "add", "new"],
    },
    "get_device": {
        "description": "Get one device by id.",
        "required_fields": ["device_id"],
        "keywords": ["get", "show", "find", "detail"],
    },
    "update_device": {
        "description": "Update an existing device.",
        "required_fields": ["device_id"],
        "optional_fields": [
            "device_name",
            "device_type",
            "role",
            "description",
            "group",
            "save_data",
            "status",
            "tags",
            "features",
        ],
        "keywords": ["update", "edit", "change", "modify"],
    },
    "delete_device": {
        "description": "Delete one device by id.",
        "required_fields": ["device_id"],
        "keywords": ["delete", "remove"],
    },
    "list_devices": {
        "description": "List all devices.",
        "required_fields": [],
        "keywords": ["list", "all devices"],
    },
}


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


class BackendRequestError(Exception):
    pass


def _log_tool_event(event: str, **payload: Any) -> None:
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


def _contract_error(operation: str, data: Any, hint: str) -> dict:
    if isinstance(data, dict):
        keys = list(data.keys())
    elif isinstance(data, list):
        keys = ["<list>"]
    else:
        keys = [str(type(data))]
    message = (
        f"Contract check failed for {operation}: {hint}. "
        f"Received keys/type: {keys}"
    )
    _log_tool_event("contract_error", operation=operation, hint=hint, received=keys)
    return {"ok": False, "error": message}


def _get_base_url() -> str:
    return os.getenv("SENSOLIST_BASE_URL", "http://be-dev.sensolist.com/api").rstrip("/")


def _get_api_token() -> str:
    token = os.getenv("SENSOLIST_API_TOKEN", "")
    if not token:
        raise BackendRequestError("Missing SENSOLIST_API_TOKEN environment variable")
    token = token.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def _http_request(method: str, path: str, payload: Optional[dict] = None) -> Any:
    url = f"{_get_base_url()}{path}"
    headers = {
        "Authorization": f"Bearer {_get_api_token()}",
        "Content-Type": "application/json",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {}
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(error_body)
            message = parsed.get("error") or parsed.get("message") or error_body
        except json.JSONDecodeError:
            message = error_body or exc.reason
        raise BackendRequestError(
            f"Backend request failed ({exc.code}): {message}"
        ) from exc
    except urllib.error.URLError as exc:
        raise BackendRequestError(f"Backend request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise BackendRequestError("Invalid JSON response from backend") from exc


def _call_backend_api(method: str, path: str, payload: Optional[dict] = None) -> dict:
    try:
        response = _http_request(method, path, payload)
        return {"ok": True, "data": response}
    except BackendRequestError as exc:
        _log_tool_event(
            "backend_error",
            method=method,
            path=path,
            payload=payload or {},
            error=str(exc),
        )
        return {"ok": False, "error": str(exc)}


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on", "active", "enabled"}:
        return True
    if text in {"false", "0", "no", "n", "off", "inactive", "disabled"}:
        return False
    return default


def _parse_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_features(raw: Any) -> tuple[list[dict[str, str]], Optional[str]]:
    if raw is None:
        return [], None

    if isinstance(raw, list):
        normalized: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                return [], "Each feature must be an object."
            normalized.append(
                {
                    "field": str(item.get("field", "")).strip(),
                    "label": str(item.get("label", "")).strip(),
                    "path": str(item.get("path", "")).strip(),
                    "unit": str(item.get("unit", "")).strip(),
                    "value": str(item.get("value", "")).strip(),
                }
            )
        return normalized, None

    text = str(raw).strip()
    if not text:
        return [], None

    features: list[dict[str, str]] = []
    for raw_feature in text.split(";"):
        part = raw_feature.strip()
        if not part:
            continue
        tokens = [token.strip() for token in part.split("|")]
        if len(tokens) != 5:
            return [], (
                "Invalid features format. Use: "
                "field|label|path|unit|value;field|label|path|unit|value"
            )
        features.append(
            {
                "field": tokens[0],
                "label": tokens[1],
                "path": tokens[2],
                "unit": tokens[3],
                "value": tokens[4],
            }
        )
    return features, None


def _build_group(group_name: str) -> dict[str, Any]:
    return {
        "id": f"group-{group_name}" if group_name else "",
        "name": group_name,
        "description": "",
        "owner": "",
        "workspace": "",
        "template": "",
        "roles": [],
        "features": [],
        "createdAt": "",
        "updatedAt": "",
    }


def _build_tag_objects(tag_names: list[str]) -> list[dict[str, str]]:
    return [
        {
            "id": f"tag-{idx + 1}",
            "name": tag_name,
            "description": "",
        }
        for idx, tag_name in enumerate(tag_names)
    ]


def _normalize_device_payload(device: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(device)

    group_value = normalized.get("group", "")
    if isinstance(group_value, str):
        normalized["group"] = _build_group(group_value)
    elif isinstance(group_value, dict):
        normalized_group = _build_group(str(group_value.get("name", "")))
        normalized_group.update(group_value)
        normalized["group"] = normalized_group
    else:
        normalized["group"] = _build_group("")

    tags_value = normalized.get("tags", [])
    if isinstance(tags_value, list) and tags_value:
        if all(isinstance(item, str) for item in tags_value):
            normalized["tags"] = _build_tag_objects([str(item) for item in tags_value])
        elif all(isinstance(item, dict) for item in tags_value):
            normalized["tags"] = [item for item in tags_value if isinstance(item, dict)]
        else:
            normalized["tags"] = []
    else:
        normalized["tags"] = []

    features_value = normalized.get("features", [])
    normalized["features"] = [
        item for item in features_value if isinstance(item, dict)
    ] if isinstance(features_value, list) else []

    return normalized


def _extract_device(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        if "device" in data and isinstance(data["device"], dict):
            return _normalize_device_payload(data["device"])
        if "data" in data and isinstance(data["data"], dict):
            return _normalize_device_payload(data["data"])
        if "item" in data and isinstance(data["item"], dict):
            return _normalize_device_payload(data["item"])
        if "result" in data and isinstance(data["result"], dict):
            return _normalize_device_payload(data["result"])
        if any(key in data for key in ("deviceId", "name", "type")):
            return _normalize_device_payload(data)
    return {}


def _extract_device_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [
            _normalize_device_payload(item) if isinstance(item, dict) else item
            for item in data
        ]

    if isinstance(data, dict):
        list_keys = ("devices", "data", "items", "results", "result", "List", "list", "docs")
        for key in list_keys:
            value = data.get(key)
            if isinstance(value, list):
                return [
                    _normalize_device_payload(item) if isinstance(item, dict) else item
                    for item in value
                ]
            if isinstance(value, dict):
                for nested_key in list_keys:
                    nested_value = value.get(nested_key)
                    if isinstance(nested_value, list):
                        return [
                            _normalize_device_payload(item) if isinstance(item, dict) else item
                            for item in nested_value
                        ]
    return []


def _is_object_id(value: str) -> bool:
    return re.fullmatch(r"[0-9a-fA-F]{24}", value or "") is not None


def _is_device_id_lookup_error(error_text: str) -> bool:
    text = str(error_text or "").lower()
    return (
        "not found" in text
        or "invalid device id" in text
        or "not a valid objectid" in text
    )


def _find_device_in_list(raw_device_id: str) -> Optional[dict[str, Any]]:
    listed = list_devices()
    if not listed.get("ok"):
        return None

    needle = str(raw_device_id or "").strip()
    if not needle:
        return None

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
            return _normalize_device_payload(item)
    return None


def _resolve_backend_device_identifier(device_id: str) -> str:
    raw = str(device_id or "").strip()
    if not raw:
        return raw

    if _is_object_id(raw):
        return raw

    listed = list_devices()
    if not listed.get("ok"):
        return raw

    for item in listed.get("devices", []):
        if not isinstance(item, dict):
            continue
        item_device_id = str(item.get("deviceId", "")).strip()
        item_internal_id = str(item.get("id", "")).strip()
        if item_device_id == raw or item_device_id.lower() == raw.lower():
            internal_id = str(item.get("id", "")).strip()
            return internal_id or raw
        if item_internal_id == raw or item_internal_id.lower() == raw.lower():
            return raw

    return raw


def create_device(
    *,
    name: str,
    type: str,
    device_id: str,
) -> dict:
    if not name or not type or not device_id:
        return {
            "ok": False,
            "error": "Missing required fields for create_device: name, device_id, type",
        }

    type_value = str(type).strip().lower()
    if type_value not in ALLOWED_DEVICE_TYPES:
        return {
            "ok": False,
            "error": (
                "Invalid device type. Allowed values are: "
                "sensor, gateway, controller, processor, ipcamera"
            ),
        }

    payload: dict[str, Any] = {
        "name": name,
        "deviceId": device_id,
        "type": type_value,
    }

    response = _call_backend_api("POST", "/device", payload)
    if not response["ok"]:
        return response

    device = _extract_device(response["data"])
    if not device:
        return _contract_error(
            "create_device",
            response["data"],
            "Expected a device object in one of: device, data, item, result",
        )

    return {"ok": True, "device": device}


def get_device(device_id: str) -> dict:
    if not device_id:
        return {"ok": False, "error": "Missing required field for get_device: device_id"}

    raw = str(device_id).strip()
    resolved = _resolve_backend_device_identifier(raw)
    response = _call_backend_api("GET", f"/device/{resolved}")
    if not response["ok"] and resolved != raw:
        response = _call_backend_api("GET", f"/device/{raw}")
    if not response["ok"]:
        if _is_device_id_lookup_error(response.get("error", "")):
            from_list = _find_device_in_list(raw)
            if from_list:
                return {"ok": True, "device": from_list}
        return response

    device = _extract_device(response["data"])
    if not device:
        return _contract_error(
            "get_device",
            response["data"],
            "Expected a device object in one of: device, data, item, result",
        )

    return {"ok": True, "device": device}


def update_device(
    *,
    device_id: str,
    name: Optional[str] = None,
    type: Optional[str] = None,
    role: Optional[str] = None,
    description: Optional[str] = None,
    group: Optional[str] = None,
    save_data: Optional[bool] = None,
    status: Optional[bool] = None,
    tags: Optional[list[str]] = None,
    features: Optional[list[dict[str, str]]] = None,
) -> dict:
    if not device_id:
        return {"ok": False, "error": "Missing required field for update_device: device_id"}

    payload: dict[str, Any] = {}
    if name is not None:
        payload["name"] = name
    if type is not None:
        payload["type"] = type
    if role is not None:
        payload["role"] = role
    if description is not None:
        payload["description"] = description
    if group is not None:
        payload["group"] = group
    if save_data is not None:
        payload["saveData"] = save_data
    if status is not None:
        payload["status"] = status
    if tags is not None:
        payload["tags"] = tags
    if features is not None:
        payload["features"] = features

    if not payload:
        return {
            "ok": False,
            "error": (
                "Missing update payload for update_device: provide at least one of "
                "name, type, role, description, group, saveData, status, tags, features"
            ),
        }

    resolved = _resolve_backend_device_identifier(device_id)
    response = _call_backend_api("PATCH", f"/device/{resolved}", payload)
    if not response["ok"] and resolved != device_id:
        response = _call_backend_api("PATCH", f"/device/{device_id}", payload)
    if not response["ok"]:
        return response

    device = _extract_device(response["data"])
    if not device:
        return _contract_error(
            "update_device",
            response["data"],
            "Expected a device object in one of: device, data, item, result",
        )

    return {"ok": True, "device": device}


def delete_device(device_id: str) -> dict:
    if not device_id:
        return {"ok": False, "error": "Missing required field for delete_device: device_id"}

    resolved = _resolve_backend_device_identifier(device_id)
    response = _call_backend_api("DELETE", f"/device/{resolved}")
    if not response["ok"] and resolved != device_id:
        response = _call_backend_api("DELETE", f"/device/{device_id}")
    if not response["ok"]:
        return response

    device = _extract_device(response["data"])
    if not device:
        # Some backends return only status/message on DELETE with no device payload.
        return {
            "ok": True,
            "deleted": True,
            "device": {"deviceId": device_id},
            "raw": response["data"],
        }

    return {"ok": True, "deleted": True, "device": device}


def list_devices() -> dict:
    response = _call_backend_api("GET", "/device")
    if not response["ok"]:
        return response

    devices = _extract_device_list(response["data"])
    if not isinstance(devices, list):
        return _contract_error(
            "list_devices",
            response["data"],
            "Expected a device list in one of: devices, data, items, results, result, List",
        )
    return {"ok": True, "devices": devices}


def execute_tool(tool_data: dict) -> dict:
    tool_name = tool_data.get("name", "").strip()

    if tool_name == "create_device":
        name = str(tool_data.get("device_name", "")).strip()
        type_ = str(tool_data.get("device_type", "")).strip()
        device_id = str(tool_data.get("device_id", "")).strip()
        if not name or not type_ or not device_id:
            return {
                "ok": False,
                "error": (
                    "Missing required fields for create_device: "
                    "device_name, device_id, device_type"
                ),
            }

        return create_device(
            name=name,
            type=type_,
            device_id=device_id,
        )

    if tool_name == "get_device":
        device_id = str(tool_data.get("device_id", "")).strip()
        if not device_id:
            return {"ok": False, "error": "Missing required field for get_device: device_id"}
        return get_device(device_id=device_id)

    if tool_name == "update_device":
        device_id = str(tool_data.get("device_id", "")).strip()
        if not device_id:
            return {"ok": False, "error": "Missing required field for update_device: device_id"}

        updates_present = any(
            key in tool_data
            for key in (
                "device_name",
                "device_type",
                "role",
                "description",
                "group",
                "save_data",
                "status",
                "tags",
                "features",
            )
        )
        if not updates_present:
            return {
                "ok": False,
                "error": (
                    "Missing update payload for update_device: provide at least one of "
                    "device_name, device_type, role, description, group, "
                    "save_data, status, tags, features"
                ),
            }

        features: Optional[list[dict[str, str]]] = None
        if "features" in tool_data:
            features, feature_error = _parse_features(tool_data.get("features"))
            if feature_error:
                return {"ok": False, "error": feature_error}

        tags: Optional[list[str]] = None
        if "tags" in tool_data:
            tags = _parse_tags(tool_data.get("tags"))

        return update_device(
            device_id=device_id,
            name=tool_data.get("device_name"),
            type=tool_data.get("device_type"),
            role=tool_data.get("role"),
            description=tool_data.get("description"),
            group=tool_data.get("group"),
            save_data=(
                _parse_bool(tool_data.get("save_data"), default=True)
                if "save_data" in tool_data
                else None
            ),
            status=(
                _parse_bool(tool_data.get("status"), default=True)
                if "status" in tool_data
                else None
            ),
            tags=tags,
            features=features,
        )

    if tool_name == "delete_device":
        device_id = str(tool_data.get("device_id", "")).strip()
        if not device_id:
            return {"ok": False, "error": "Missing required field for delete_device: device_id"}
        return delete_device(device_id=device_id)

    if tool_name == "list_devices":
        return list_devices()

    return {"ok": False, "error": f"Unknown tool: {tool_name}"}
