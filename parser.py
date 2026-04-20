import re
from typing import Optional


def extract_block(tag: str, text: str) -> Optional[str]:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def is_pure_block(tag: str, text: str) -> bool:
    pattern = rf"^\s*<{tag}>\s*.*?\s*</{tag}>\s*$"
    return re.match(pattern, text, flags=re.DOTALL | re.IGNORECASE) is not None


def parse_key_value_block(text: str) -> dict:
    result = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()

    return result


def parse_agent_output(text: str) -> dict:
    tool_block = extract_block("tool", text)
    ask_block = extract_block("ask", text)

    if tool_block:
        return {
            "kind": "tool",
            "pure": is_pure_block("tool", text),
            "data": parse_key_value_block(tool_block),
            "raw": text,
        }

    if ask_block:
        parsed = parse_key_value_block(ask_block)
        missing = parsed.get("missing", "")
        missing_fields = [item.strip() for item in missing.split(",") if item.strip()]

        return {
            "kind": "ask",
            "pure": is_pure_block("ask", text),
            "data": parsed,
            "missing_fields": missing_fields,
            "raw": text,
        }

    return {
        "kind": "final",
        "pure": True,
        "data": {"message": text.strip()},
        "raw": text,
    }