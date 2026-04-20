from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from schemas import Device


DEVICE_DB: dict[str, Device] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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

    # format: field|label|path|unit|value;field|label|path|unit|value
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
    now = _now_iso()
    return {
        "id": f"group-{group_name}" if group_name else "",
        "name": group_name,
        "description": "",
        "owner": "",
        "workspace": "",
        "template": "",
        "roles": [],
        "features": [],
        "createdAt": now,
        "updatedAt": now,
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


def _to_public_device(device: Device) -> dict[str, Any]:
    tag_names = device.get("tags", [])
    group_name = device.get("group", "")
    return {
        "id": device["id"],
        "deviceId": device["deviceId"],
        "name": device["name"],
        "type": device["type"],
        "role": device.get("role", ""),
        "description": device.get("description", ""),
        "saveData": device.get("saveData", True),
        "status": device.get("status", True),
        "features": device.get("features", []),
        "group": _build_group(group_name),
        "tags": _build_tag_objects(tag_names),
        "createdAt": device.get("createdAt", ""),
        "updatedAt": device.get("updatedAt", ""),
        "publishedAt": device.get("publishedAt", ""),
    }


def create_device(
    *,
    name: str,
    type: str,
    role: str = "",
    description: str = "",
    group: str = "",
    save_data: bool = True,
    tags: Optional[list[str]] = None,
    features: Optional[list[dict[str, str]]] = None,
    device_id: Optional[str] = None,
) -> dict:
    now = _now_iso()
    public_device_id = device_id or f"dev-{uuid4().hex[:10]}"
    if public_device_id in DEVICE_DB:
        return {
            "ok": False,
            "error": f"Device '{public_device_id}' already exists.",
        }

    normalized_type = (type or "").strip().lower()
    default_role = "network" if normalized_type == "gateway" else "sensor"
    resolved_role = role or default_role
    resolved_group = group or "building-a"
    resolved_description = description or f"{resolved_role.capitalize()} {normalized_type or 'device'} sensor"
    resolved_tags = tags or [resolved_role, normalized_type or "device"]
    resolved_features = features or (
        [
            {
                "field": "network",
                "label": "Network",
                "path": "sensors.net",
                "unit": "Mbps",
                "value": "1000",
            }
        ]
        if normalized_type == "gateway"
        else []
    )

    device: Device = {
        "id": str(uuid4()),
        "deviceId": public_device_id,
        "name": name,
        "type": type,
        "role": resolved_role,
        "description": resolved_description,
        "group": resolved_group,
        "saveData": save_data,
        "status": True,
        "tags": resolved_tags,
        "features": resolved_features,
        "createdAt": now,
        "updatedAt": now,
        "publishedAt": "",
    }
    DEVICE_DB[public_device_id] = device
    return {"ok": True, "device": _to_public_device(device)}


def get_device(device_id: str) -> dict:
    device = DEVICE_DB.get(device_id)
    if not device:
        return {
            "ok": False,
            "error": f"Device '{device_id}' not found.",
        }

    return {"ok": True, "device": _to_public_device(device)}


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
    device = DEVICE_DB.get(device_id)
    if not device:
        return {
            "ok": False,
            "error": f"Device '{device_id}' not found.",
        }

    if name is not None:
        device["name"] = name
    if type is not None:
        device["type"] = type
    if role is not None:
        device["role"] = role
    if description is not None:
        device["description"] = description
    if group is not None:
        device["group"] = group
    if save_data is not None:
        device["saveData"] = save_data
    if status is not None:
        device["status"] = status
    if tags is not None:
        device["tags"] = tags
    if features is not None:
        device["features"] = features

    device["updatedAt"] = _now_iso()
    return {"ok": True, "device": _to_public_device(device)}


def delete_device(device_id: str) -> dict:
    device = DEVICE_DB.get(device_id)
    if not device:
        return {
            "ok": False,
            "error": f"Device '{device_id}' not found.",
        }

    deleted = DEVICE_DB.pop(device_id)
    return {"ok": True, "deleted": True, "device": _to_public_device(deleted)}


def list_devices() -> dict:
    return {
        "ok": True,
        "devices": [_to_public_device(item) for item in DEVICE_DB.values()],
    }


def execute_tool(tool_data: dict) -> dict:
    tool_name = tool_data.get("name", "").strip()

    if tool_name == "create_device":
        name = tool_data.get("device_name")
        type_ = tool_data.get("device_type")
        if not name or not type_:
            return {
                "ok": False,
                "error": "Missing required fields for create_device: device_name, device_type",
            }

        features, feature_error = _parse_features(tool_data.get("features"))
        if feature_error:
            return {"ok": False, "error": feature_error}

        return create_device(
            name=name,
            type=type_,
            role=str(tool_data.get("role", "")).strip(),
            description=str(tool_data.get("description", "")).strip(),
            group=str(tool_data.get("group", "")).strip(),
            save_data=_parse_bool(tool_data.get("save_data"), default=True),
            tags=_parse_tags(tool_data.get("tags")),
            features=features,
            device_id=str(tool_data.get("device_id", "")).strip() or None,
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
