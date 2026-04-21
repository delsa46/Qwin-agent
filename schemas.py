from typing import Any, TypedDict, Literal, Union


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


class NavigateToResourceList(TypedDict):
    screen: Literal["resource_list"]
    resource: str


class NavigateToResourceDetail(TypedDict):
    screen: Literal["resource_detail"]
    resource: str
    idFrom: str


class NavigateToResourceCreate(TypedDict):
    screen: Literal["resource_create"]
    resource: str


class NavigateToResourceEdit(TypedDict):
    screen: Literal["resource_edit"]
    resource: str
    idFrom: str


NavigateDestination = Union[
    NavigateToResourceList,
    NavigateToResourceDetail,
    NavigateToResourceCreate,
    NavigateToResourceEdit,
]


class UiActionNavigate(TypedDict):
    kind: Literal["navigate"]
    destination: NavigateDestination
    label: str
    variant: Literal["primary", "secondary", "success", "danger"]


class UiActionShowList(TypedDict, total=False):
    kind: Literal["show_list"]
    resource: str
    label: str
    filtersFrom: str
    variant: Literal["primary", "secondary"]


class UiActionConfirm(TypedDict, total=False):
    kind: Literal["confirm"]
    label: str
    target: str
    confirmKey: str
    title: str
    message: str
    variant: Literal["primary", "danger"]


class UiActionAskInput(TypedDict, total=False):
    kind: Literal["ask_input"]
    label: str
    inputKey: str
    title: str
    message: str
    inputType: Literal["text", "number", "select", "textarea"]
    options: list[dict[str, str]]
    required: bool
    variant: Literal["primary", "secondary"]


class UiActionAskExplanation(TypedDict, total=False):
    kind: Literal["ask_explanation"]
    label: str
    explanationKey: str
    title: str
    message: str
    variant: Literal["secondary"]


FEAction = Union[
    UiActionNavigate,
    UiActionShowList,
    UiActionConfirm,
    UiActionAskInput,
    UiActionAskExplanation,
]


class AgentResponse(TypedDict):
    data: Any
    context: dict[str, Any]
    message: str
    actions: list[FEAction]
    need_more_info: bool
    missing_fields: list[str]
