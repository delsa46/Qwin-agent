from typing import Any, TypedDict, Literal


class DeviceFeature(TypedDict, total=False):
    field: str
    label: str
    path: str
    unit: str
    value: str


class Device(TypedDict, total=False):
    id: str
    deviceId: str
    name: str
    type: str
    role: str
    description: str
    group: str
    saveData: bool
    status: bool
    tags: list[str]
    features: list[DeviceFeature]
    createdAt: str
    updatedAt: str
    publishedAt: str


class FEAction(TypedDict):
    type: Literal["button"]
    label: str
    variant: Literal["success", "primary", "danger", "warning", "info"]
    target: str


class AgentResponse(TypedDict):
    context: dict[str, Any]
    message: str
    actions: list[FEAction]
    need_more_info: bool
    missing_fields: list[str]
