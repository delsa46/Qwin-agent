from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Optional

from device_tools import TOOL_CATALOG


@dataclass(frozen=True)
class ToolSelection:
    tool_name: Optional[str]
    confidence: float
    reason: str


def available_tools() -> list[str]:
    return list(TOOL_CATALOG.keys())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", (text or "").lower())


def _matches_keyword(text: str, keyword: str) -> bool:
    normalized = (text or "").lower()
    if keyword in normalized:
        return True

    if " " in keyword:
        return False

    tokens = _tokenize(normalized)
    return bool(get_close_matches(keyword, tokens, n=1, cutoff=0.75))


def _intent_score(text: str, tool_name: str) -> float:
    meta = TOOL_CATALOG.get(tool_name, {})
    keywords = meta.get("keywords", [])
    score = 0.0
    for keyword in keywords:
        if _matches_keyword(text, str(keyword)):
            score += 1.0
    return score


def _field_score(tool_name: str, tool_data: Optional[dict[str, Any]]) -> float:
    if not isinstance(tool_data, dict):
        return 0.0

    present_fields = {
        str(key).strip()
        for key, value in tool_data.items()
        if key != "name" and str(value).strip()
    }

    if tool_name == "create_device":
        needed = {"device_name", "device_id", "device_type"}
        return 3.0 if needed.issubset(present_fields) else 0.0

    if tool_name == "list_devices":
        return 2.0 if not present_fields else -1.0

    if tool_name in {"get_device", "delete_device"}:
        return 2.0 if present_fields == {"device_id"} else 0.0

    if tool_name == "update_device":
        if "device_id" not in present_fields:
            return 0.0
        has_update_payload = bool(
            present_fields.intersection(
                {
                    "device_name",
                    "device_type",
                    "role",
                    "description",
                    "group",
                    "save_data",
                    "status",
                    "tags",
                    "features",
                }
            )
        )
        return 3.0 if has_update_payload else 1.0

    return 0.0


def select_tool(
    *,
    user_text: str,
    candidate_tool: Optional[str] = None,
    tool_data: Optional[dict[str, Any]] = None,
    pending_tool: Optional[str] = None,
) -> ToolSelection:
    if pending_tool in TOOL_CATALOG:
        return ToolSelection(
            tool_name=pending_tool,
            confidence=1.0,
            reason="pending_tool",
        )

    best_tool: Optional[str] = None
    best_score = float("-inf")
    best_reason = "no_match"

    for tool_name in available_tools():
        score = 0.0
        reasons: list[str] = []

        if candidate_tool == tool_name:
            score += 2.5
            reasons.append("candidate_tool")

        intent = _intent_score(user_text, tool_name)
        if intent:
            score += intent
            reasons.append("intent")

        field = _field_score(tool_name, tool_data)
        if field:
            score += field
            reasons.append("fields")

        if score > best_score:
            best_tool = tool_name
            best_score = score
            best_reason = ",".join(reasons) if reasons else "no_match"

    if best_score <= 0:
        return ToolSelection(tool_name=None, confidence=0.0, reason="no_match")

    confidence = min(1.0, best_score / 6.0)
    return ToolSelection(tool_name=best_tool, confidence=confidence, reason=best_reason)
