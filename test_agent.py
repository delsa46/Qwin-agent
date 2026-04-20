import pytest
from unittest.mock import MagicMock, patch

from parser import extract_block, is_pure_block, parse_key_value_block, parse_agent_output
from device_tools import BackendRequestError, execute_tool
from agent import run_agent
from agent_session import DeviceAgentSession


@pytest.fixture(autouse=True)
def default_backend_mock(monkeypatch):
    monkeypatch.setenv("SENSOLIST_API_TOKEN", "test-token")
    default_device = {
        "deviceId": "dev-default",
        "name": "sensor-default",
        "type": "temperature",
        "group": "building-a",
        "tags": ["default"],
    }
    mock_http = MagicMock(return_value={"device": default_device})
    monkeypatch.setattr("device_tools._http_request", mock_http)
    return mock_http


def setup_function():
    pass


def test_extract_tool_block():
    text = "<tool>\nname=create_device\ndevice_name=s1\ndevice_type=sensor\n</tool>"
    block = extract_block("tool", text)
    assert block is not None
    assert "name=create_device" in block


def test_is_pure_block_true():
    text = "<tool>\nname=create_device\n</tool>"
    assert is_pure_block("tool", text) is True


def test_is_pure_block_false():
    text = "<tool>\nname=create_device\n</tool>\nextra"
    assert is_pure_block("tool", text) is False


def test_parse_key_value_block():
    text = "name=create_device\ndevice_name=s1\ndevice_type=sensor"
    parsed = parse_key_value_block(text)
    assert parsed["name"] == "create_device"
    assert parsed["device_name"] == "s1"
    assert parsed["device_type"] == "sensor"


def test_parse_agent_output_tool():
    text = "<tool>\nname=create_device\ndevice_name=s1\ndevice_type=sensor\n</tool>"
    result = parse_agent_output(text)
    assert result["kind"] == "tool"
    assert result["data"]["name"] == "create_device"


def test_parse_agent_output_ask():
    text = "<ask>\nmissing=device_name,device_type\nquestion=Please provide name and type.\n</ask>"
    result = parse_agent_output(text)
    assert result["kind"] == "ask"
    assert result["missing_fields"] == ["device_name", "device_type"]


@patch("device_tools._http_request")
def test_execute_tool_create_device(mock_http):
    mock_http.return_value = {
        "device": {
            "deviceId": "device-001",
            "name": "sensor-a",
            "type": "temperature",
        }
    }

    result = execute_tool({
        "name": "create_device",
        "device_name": "sensor-a",
        "device_type": "temperature",
        "device_id": "device-001",
    })

    assert result["ok"] is True
    assert result["device"]["name"] == "sensor-a"
    assert result["device"]["deviceId"] == "device-001"
    assert result["device"]["saveData"] is True
    assert result["device"]["tags"][0]["name"] == "critical"


@patch("device_tools._http_request")
def test_execute_tool_create_backend_error(mock_http):
    mock_http.side_effect = BackendRequestError("Backend unreachable")

    result = execute_tool({
        "name": "create_device",
        "device_name": "sensor-a",
        "device_type": "temperature",
    })

    assert result["ok"] is False
    assert "Backend unreachable" in result["error"]


@patch("device_tools._http_request")
def test_execute_tool_get_device(mock_http):
    mock_http.return_value = {
        "deviceId": "device-002",
        "name": "sensor-b",
        "type": "humidity",
        "group": "floor-1",
        "tags": ["indoor"],
    }

    result = execute_tool({"name": "get_device", "device_id": "device-002"})

    assert result["ok"] is True
    assert result["device"]["deviceId"] == "device-002"
    assert result["device"]["group"]["name"] == "floor-1"


@patch("device_tools._http_request")
def test_execute_tool_update_device(mock_http):
    mock_http.return_value = {
        "deviceId": "device-003",
        "name": "sensor-c",
        "type": "temperature",
        "status": False,
    }

    result = execute_tool({
        "name": "update_device",
        "device_id": "device-003",
        "status": "false",
    })

    assert result["ok"] is True
    assert result["device"]["deviceId"] == "device-003"
    assert result["device"]["status"] is False


@patch("device_tools._http_request")
def test_execute_tool_delete_device(mock_http):
    mock_http.return_value = {
        "deviceId": "device-004",
        "name": "sensor-d",
        "type": "pressure",
    }

    result = execute_tool({"name": "delete_device", "device_id": "device-004"})

    assert result["ok"] is True
    assert result["deleted"] is True
    assert result["device"]["deviceId"] == "device-004"


@patch("device_tools._http_request")
def test_execute_tool_list_devices(mock_http):
    mock_http.return_value = {
        "devices": [
            {
                "deviceId": "device-005",
                "name": "sensor-e",
                "type": "temperature",
            }
        ]
    }

    result = execute_tool({"name": "list_devices"})

    assert result["ok"] is True
    assert len(result["devices"]) == 1
    assert result["devices"][0]["deviceId"] == "device-005"


@patch("device_tools._http_request")
@patch("agent.call_model")
def test_run_agent_create_device(mock_call_model, mock_http):
    mock_call_model.return_value = (
        "<tool>\n"
        "name=create_device\n"
        "device_id=device-001\n"
        "device_name=sensor-a\n"
        "device_type=temperature\n"
        "</tool>"
    )
    mock_http.return_value = {
        "device": {
            "deviceId": "device-001",
            "name": "sensor-a",
            "type": "temperature",
        }
    }

    result = run_agent("create a device named sensor-a of type temperature")

    assert result["message"] == "Device created successfully."
    assert result["context"]["entity"] == "device"
    assert result["context"]["data"]["name"] == "sensor-a"
    assert result["context"]["data"]["deviceId"] == "device-001"
    assert len(result["actions"]) == 1
    assert result["actions"][0]["label"] == "Go To Device"
    assert result["actions"][0]["target"] == "/devices/device-001"


@patch("agent.call_model")
def test_run_agent_ask_for_missing_fields(mock_call_model):
    mock_call_model.return_value = (
        "<ask>\n"
        "missing=device_name,device_type\n"
        "question=Please provide device name and device type.\n"
        "</ask>"
    )

    result = run_agent("create a device")

    assert result["need_more_info"] is True
    assert result["missing_fields"] == ["device_name", "device_type"]
    assert "Please provide" in result["message"]


@patch("device_tools._http_request")
@patch("agent.call_model")
def test_run_agent_list_devices(mock_call_model, mock_http):
    mock_call_model.return_value = (
        "<tool>\n"
        "name=list_devices\n"
        "</tool>"
    )
    mock_http.return_value = {
        "devices": [
            {
                "deviceId": "device-010",
                "name": "sensor-list",
                "type": "temperature",
            }
        ]
    }

    result = run_agent("list devices")

    assert result["context"]["entity"] == "device_list"
    assert "device(s) found" in result["message"]


@patch("agent.call_model")
def test_run_agent_invalid_plain_text(mock_call_model):
    mock_call_model.return_value = "I need more details."

    result = run_agent("create a device")
    assert result["message"] == "I need more details."
    assert result["need_more_info"] is False


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_session_followup_create_with_missing_fields(mock_http, mock_call_model):
    mock_call_model.return_value = (
        "<ask>\n"
        "tool=create_device\n"
        "missing=device_name,device_type\n"
        "question=Please provide device_name and device_type.\n"
        "</ask>"
    )
    mock_http.return_value = {
        "device": {
            "deviceId": "device-006",
            "name": "sensor-a",
            "type": "temperature",
            "group": "building-a",
            "tags": ["critical"],
        }
    }

    session = DeviceAgentSession()
    first = session.run_turn("create a device")
    assert first["need_more_info"] is True
    assert first["missing_fields"] == ["device_name", "device_type"]

    second = session.run_turn("sensor-a,temperature")
    assert second["message"] == "Device created successfully."
    assert second["context"]["entity"] == "device"
    assert second["context"]["data"]["name"] == "sensor-a"
    assert second["context"]["data"]["type"] == "temperature"


@patch("device_tools._http_request")
def test_session_create_does_not_require_optional_fields(mock_http):
    mock_http.return_value = {
        "device": {
            "deviceId": "device-007",
            "name": "sensor-1",
            "type": "gateway",
            "role": "network",
            "group": "building-a",
            "tags": ["network"],
        }
    }

    session = DeviceAgentSession()
    result = session.run_turn("create device name is sensor-1 and typy is gateway")

    assert result["need_more_info"] is False
    assert result["message"] == "Device created successfully."


@patch("device_tools._http_request")
def test_session_direct_create_from_plain_text(mock_http):
    mock_http.return_value = {
        "device": {
            "deviceId": "dev-1234567890",
            "name": "sensor-1",
            "type": "gateway",
            "role": "network",
            "group": "building-a",
            "tags": ["network"],
        }
    }

    session = DeviceAgentSession()
    result = session.run_turn("create device name is sensor-1 and typy is gateway")

    assert result["need_more_info"] is False
    assert result["message"] == "Device created successfully."
    assert result["context"]["entity"] == "device"
    assert result["context"]["data"]["name"] == "sensor-1"
    assert result["context"]["data"]["type"] == "gateway"
    assert result["context"]["data"]["role"] == "network"
    assert result["context"]["data"]["group"]["name"] == "building-a"
    assert result["context"]["data"]["tags"][0]["name"] == "network"
    assert result["actions"][0]["target"].startswith("/devices/dev-")


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_session_get_flow_with_followup(mock_http, mock_call_model):
    mock_call_model.return_value = (
        "<ask>\n"
        "tool=get_device\n"
        "missing=device_id\n"
        "question=Please provide device_id.\n"
        "</ask>"
    )
    mock_http.side_effect = [
        {
            "device": {
                "deviceId": "device-008",
                "name": "sensor-r1",
                "type": "temp",
                "group": "floor-1",
            }
        },
        {
            "device": {
                "deviceId": "device-008",
                "name": "sensor-r1",
                "type": "temp",
                "group": "floor-1",
            }
        },
    ]

    created = execute_tool({
        "name": "create_device",
        "device_name": "sensor-r1",
        "device_type": "temp",
    })
    device_id = created["device"]["deviceId"]

    session = DeviceAgentSession()
    first = session.run_turn("get device details")
    assert first["need_more_info"] is True
    assert first["missing_fields"] == ["device_id"]

    second = session.run_turn(device_id)
    assert second["message"] == "Device fetched successfully."
    assert second["context"]["data"]["deviceId"] == device_id


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_session_update_flow_with_followup(mock_http, mock_call_model):
    mock_call_model.return_value = (
        "<ask>\n"
        "tool=update_device\n"
        "missing=device_id,status\n"
        "question=Please provide device_id and status.\n"
        "</ask>"
    )
    mock_http.side_effect = [
        {
            "device": {
                "deviceId": "device-009",
                "name": "sensor-u1",
                "type": "temp",
            }
        },
        {
            "device": {
                "deviceId": "device-009",
                "name": "sensor-u1",
                "type": "temp",
                "status": False,
            }
        },
    ]

    created = execute_tool({
        "name": "create_device",
        "device_name": "sensor-u1",
        "device_type": "temp",
    })
    device_id = created["device"]["deviceId"]

    session = DeviceAgentSession()
    first = session.run_turn("update device")
    assert first["need_more_info"] is True
    assert first["missing_fields"] == ["device_id", "status"]

    second = session.run_turn(f"device_id={device_id},status=inactive")
    assert second["message"] == "Device updated successfully."
    assert second["context"]["data"]["deviceId"] == device_id
    assert second["context"]["data"]["status"] is False


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_session_update_requires_payload_after_device_id(mock_http, mock_call_model):
    mock_call_model.return_value = (
        "<ask>\n"
        "tool=update_device\n"
        "missing=device_id\n"
        "question=Please provide the device ID to update\n"
        "</ask>"
    )
    mock_http.side_effect = [
        {
            "device": {
                "deviceId": "device-up-1",
                "name": "sensor-up-1",
                "type": "gateway",
            }
        },
        {
            "device": {
                "deviceId": "device-up-1",
                "name": "sensor-up-1",
                "type": "gateway",
                "status": False,
            }
        },
    ]

    created = execute_tool({
        "name": "create_device",
        "device_name": "sensor-up-1",
        "device_type": "gateway",
    })
    device_id = created["device"]["deviceId"]

    session = DeviceAgentSession()
    first = session.run_turn("update device")
    assert first["need_more_info"] is True
    assert first["missing_fields"] == ["device_id"]

    second = session.run_turn(device_id)
    assert second["need_more_info"] is True
    assert "at least one update field" in second["message"]
    assert "status" in second["missing_fields"]

    third = session.run_turn("status=inactive")
    assert third["need_more_info"] is False
    assert third["message"] == "Device updated successfully."
    assert third["context"]["data"]["status"] is False


@patch("agent_session.call_model")
def test_session_delete_flow_with_followup(mock_call_model):
    mock_call_model.return_value = (
        "<ask>\n"
        "tool=delete_device\n"
        "missing=device_id\n"
        "question=Please provide device_id.\n"
        "</ask>"
    )

    created = execute_tool({
        "name": "create_device",
        "device_name": "sensor-d1",
        "device_type": "temp",
    })
    device_id = created["device"]["deviceId"]

    session = DeviceAgentSession()
    first = session.run_turn("delete device")
    assert first["need_more_info"] is True
    assert first["missing_fields"] == ["device_id"]

    second = session.run_turn(device_id)
    assert second["message"] == "Device deleted successfully."
    assert second["context"]["data"]["deviceId"] == device_id


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_session_delete_invalid_id_then_retry_success(mock_http, mock_call_model):
    mock_call_model.return_value = (
        "<ask>\n"
        "tool=delete_device\n"
        "missing=device_id\n"
        "question=Please provide device_id.\n"
        "</ask>"
    )
    mock_http.side_effect = [
        {
            "device": {
                "deviceId": "device-del-1",
                "name": "sensor-del-1",
                "type": "gateway",
            }
        },
        BackendRequestError("Backend request failed (404): Device 'dev-not-exists' not found."),
        {
            "device": {
                "deviceId": "device-del-1",
                "name": "sensor-del-1",
                "type": "gateway",
            }
        },
    ]

    created = execute_tool({
        "name": "create_device",
        "device_name": "sensor-del-1",
        "device_type": "gateway",
    })
    valid_device_id = created["device"]["deviceId"]

    session = DeviceAgentSession()
    first = session.run_turn("delete device")
    assert first["need_more_info"] is True
    assert first["missing_fields"] == ["device_id"]

    second = session.run_turn("dev-not-exists")
    assert second["need_more_info"] is True
    assert second["missing_fields"] == ["device_id"]
    assert "Please provide a valid device_id." in second["message"]

    third = session.run_turn(valid_device_id)
    assert third["need_more_info"] is False
    assert third["message"] == "Device deleted successfully."


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_session_create_followup_normalizes_name_and_type(mock_http, mock_call_model):
    mock_call_model.return_value = (
        "<ask>\n"
        "tool=create_device\n"
        "missing=device_name,device_type\n"
        "question=Please provide the device name and type\n"
        "</ask>"
    )
    mock_http.return_value = {
        "device": {
            "deviceId": "device-2",
            "name": "sensor-2",
            "type": "gateway",
        }
    }

    session = DeviceAgentSession()
    first = session.run_turn("create device")
    assert first["need_more_info"] is True
    assert first["missing_fields"] == ["device_name", "device_type"]

    second = session.run_turn("name is sensor-2")
    assert second["need_more_info"] is True
    assert second["context"]["data"]["device_name"] == "sensor-2"

    third = session.run_turn("type is gateway")
    assert third["need_more_info"] is False
    assert third["message"] == "Device created successfully."
    assert third["context"]["data"]["name"] == "sensor-2"
    assert third["context"]["data"]["type"] == "gateway"


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_session_update_name_does_not_change_type(mock_http, mock_call_model):
    mock_call_model.return_value = (
        "<ask>\n"
        "tool=update_device\n"
        "missing=device_id,device_name\n"
        "question=Please provide device_id and new name\n"
        "</ask>"
    )
    mock_http.side_effect = [
        {
            "device": {
                "deviceId": "device-old",
                "name": "sensor-old",
                "type": "gateway",
            }
        },
        {
            "device": {
                "deviceId": "device-old",
                "name": "sensor-new",
                "type": "gateway",
            }
        },
    ]

    created = execute_tool({
        "name": "create_device",
        "device_name": "sensor-old",
        "device_type": "gateway",
    })
    device_id = created["device"]["deviceId"]

    session = DeviceAgentSession()
    first = session.run_turn("update device name")
    assert first["need_more_info"] is True

    second = session.run_turn(device_id)
    assert second["need_more_info"] is True

    third = session.run_turn("name is sensor-new")
    assert third["need_more_info"] is False
    assert third["message"] == "Device updated successfully."
    assert third["context"]["data"]["name"] == "sensor-new"
    assert third["context"]["data"]["type"] == "gateway"


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_session_update_name_with_key_value_does_not_override_type(mock_http, mock_call_model):
    mock_call_model.return_value = (
        "<ask>\n"
        "tool=update_device\n"
        "missing=device_id,device_name,device_type\n"
        "question=Please provide device_id and update fields\n"
        "</ask>"
    )
    mock_http.side_effect = [
        {
            "device": {
                "deviceId": "device-3",
                "name": "sensor-3",
                "type": "gateway",
            }
        },
        {
            "device": {
                "deviceId": "device-3",
                "name": "sensor-4",
                "type": "gateway",
            }
        },
    ]

    created = execute_tool({
        "name": "create_device",
        "device_name": "sensor-3",
        "device_type": "gateway",
    })
    device_id = created["device"]["deviceId"]

    session = DeviceAgentSession()
    first = session.run_turn("update device")
    assert first["need_more_info"] is True

    second = session.run_turn(device_id)
    assert second["need_more_info"] is True

    third = session.run_turn("device_name=sensor-4")
    assert third["need_more_info"] is False
    assert third["message"] == "Device updated successfully."
    assert third["context"]["data"]["name"] == "sensor-4"
    assert third["context"]["data"]["type"] == "gateway"
