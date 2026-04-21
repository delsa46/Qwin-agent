import pytest
from unittest.mock import patch

from agent_session import DeviceAgentSession


def _fake_http(method, path, payload=None):
    if method == "GET" and path == "/device":
        return {
            "devices": [
                {"id": "507f1f77bcf86cd799439101", "deviceId": "dev-1", "name": "n1", "type": "sensor"},
                {"id": "507f1f77bcf86cd799439102", "deviceId": "10101010", "name": "n2", "type": "gateway"},
            ]
        }
    if method == "GET" and path.startswith("/device/"):
        did = path.split("/")[-1]
        return {"device": {"deviceId": did, "name": "s", "type": "sensor"}}
    if method == "POST" and path == "/device":
        return {"device": {"deviceId": payload["deviceId"], "name": payload["name"], "type": payload["type"]}}
    if method == "PATCH" and path.startswith("/device/"):
        did = path.split("/")[-1]
        merged = {"deviceId": did, "name": "s", "type": "sensor"}
        if payload:
            if "name" in payload:
                merged["name"] = payload["name"]
            if "type" in payload:
                merged["type"] = payload["type"]
        return {"device": merged}
    if method == "DELETE" and path.startswith("/device/"):
        did = path.split("/")[-1]
        return {"device": {"deviceId": did, "name": "s", "type": "sensor"}}
    return {}


@pytest.mark.parametrize(
    "message,expect_entity,expect_need_more_info",
    [
        ("create devie", "device", True),
        ("create device id is a1 name is sensor and type is gateway", "device", False),
        ("list devices", "device_list", False),
        ("shwo device id is 10101010", "device", False),
        ("delet device id is dev-1", "device", False),
        ("updat device id is 10101010 status is inactive", "device", False),
        ("update device", "device", True),
        ("get device", "device", True),
        ("delete device", "device", True),
        ("nam is sensor type gateway id 12121212", "device", False),
        ("sensor is name gateway is type id is 12121212", "device", False),
    ],
)
@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_data_driven_user_inputs(mock_http, mock_call_model, message, expect_entity, expect_need_more_info):
    mock_http.side_effect = _fake_http
    mock_call_model.side_effect = RuntimeError("model should not be called")

    session = DeviceAgentSession()
    if message in {"nam is sensor type gateway id 12121212", "sensor is name gateway is type id is 12121212"}:
        session.run_turn("create device")
    response = session.run_turn(message)
    if response.get("actions") and response["actions"][0].get("target") == "CreateDevice":
        response = session.run_turn("__action__:CreateDevice")

    assert response["context"]["entity"] == expect_entity
    assert response["need_more_info"] is expect_need_more_info
