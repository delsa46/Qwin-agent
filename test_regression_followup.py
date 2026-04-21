from unittest.mock import patch

from agent import map_tool_result_to_fe
from agent_session import DeviceAgentSession
from device_tools import BackendRequestError
from device_tools import execute_tool


@patch("device_tools._http_request")
def test_create_followup_device_id_text_is_normalized(mock_http):
    session = DeviceAgentSession()
    first = session.run_turn("create a device")
    assert first["need_more_info"] is True

    second = session.run_turn("name is sensor")
    assert second["need_more_info"] is True
    assert second["missing_fields"] == ["device_id", "device_type"]

    third = session.run_turn("device id is 12121212")
    assert third["need_more_info"] is True
    assert third["context"]["data"]["device_id"] == "12121212"
    assert third["context"]["data"]["device_name"] == "sensor"
    assert third["missing_fields"] == ["device_type"]
    assert mock_http.call_count == 0


@patch("device_tools._http_request")
def test_create_final_field_submits_and_succeeds(mock_http):
    mock_http.return_value = {
        "device": {
            "deviceId": "12121212",
            "name": "sensor",
            "type": "processor",
        }
    }

    session = DeviceAgentSession()
    session.run_turn("create a device")
    session.run_turn("name is sensor")
    session.run_turn("device id is 12121212")
    confirm = session.run_turn("type is processor")
    assert confirm["need_more_info"] is True
    assert confirm["actions"][0]["type"] == "button"
    assert confirm["actions"][0]["target"] == "CreateDevice"

    result = session.run_turn("__action__:CreateDevice")

    assert result["need_more_info"] is False
    assert "created successfully" in result["message"].lower()
    assert result["context"]["data"]["deviceId"] == "12121212"
    assert result["context"]["data"]["type"] == "processor"


@patch("device_tools._http_request")
def test_create_does_not_positional_map_wrong_field(mock_http):
    session = DeviceAgentSession()
    session.run_turn("create a device")
    result = session.run_turn("device id is 545463543")

    assert result["need_more_info"] is True
    assert result["context"]["data"]["device_id"] == "545463543"
    assert "device_name" not in result["context"]["data"]
    assert result["missing_fields"] == ["device_name", "device_type"]
    assert mock_http.call_count == 0


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_device_list_does_not_call_model(mock_http, mock_call_model):
    mock_http.return_value = {
        "devices": [{"deviceId": "dev-1", "name": "n1", "type": "sensor"}]
    }
    mock_call_model.side_effect = RuntimeError("model should not be called")

    session = DeviceAgentSession()
    result = session.run_turn("device list")

    assert result["need_more_info"] is False
    assert "device(s) found" in result["message"]
    assert result["context"]["entity"] == "device_list"
    assert len(result["context"]["data"]) == 1


@patch("agent_session.call_model")
def test_typo_in_create_is_understood_without_model_call(mock_call_model):
    mock_call_model.side_effect = RuntimeError("model should not be called")

    session = DeviceAgentSession()
    result = session.run_turn("crete device")

    assert result["need_more_info"] is True
    assert set(result["missing_fields"]) == {"device_name", "device_id", "device_type"}
    assert result["actions"][0]["type"] == "form"


@patch("device_tools._http_request")
def test_create_sentence_with_all_fields_is_parsed_and_created(mock_http):
    mock_http.return_value = {
        "device": {
            "deviceId": "333444555",
            "name": "sensor",
            "type": "gateway",
        }
    }

    session = DeviceAgentSession()
    session.run_turn("create a device")
    confirm = session.run_turn(
        "create device id is 333444555 name is sensor and type is gateway"
    )
    assert confirm["need_more_info"] is True
    assert confirm["actions"][0]["target"] == "CreateDevice"

    result = session.run_turn("__action__:CreateDevice")

    assert result["need_more_info"] is False
    assert "created successfully" in result["message"].lower()
    assert result["context"]["data"]["deviceId"] == "333444555"
    assert result["context"]["data"]["name"] == "sensor"
    assert result["context"]["data"]["type"] == "gateway"


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_get_with_typo_intent_and_free_text_id(mock_http, mock_call_model):
    mock_call_model.side_effect = RuntimeError("model should not be called")
    mock_http.return_value = {
        "device": {"deviceId": "111222333", "name": "sensor-x", "type": "sensor"}
    }

    session = DeviceAgentSession()
    result = session.run_turn("shwo device id is 111222333")

    assert result["need_more_info"] is False
    assert "fetched successfully" in result["message"].lower()
    assert result["context"]["data"]["deviceId"] == "111222333"


@patch("agent_session.call_model")
def test_get_without_id_asks_without_model(mock_call_model):
    mock_call_model.side_effect = RuntimeError("model should not be called")
    session = DeviceAgentSession()
    result = session.run_turn("get device")

    assert result["need_more_info"] is True
    assert result["missing_fields"] == ["device_id"]
    assert "provide device_id" in result["message"].lower()


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_delete_with_typo_intent_and_free_text_id(mock_http, mock_call_model):
    mock_call_model.side_effect = RuntimeError("model should not be called")
    mock_http.return_value = {
        "device": {"deviceId": "444555666", "name": "sensor-z", "type": "gateway"}
    }

    session = DeviceAgentSession()
    result = session.run_turn("delet device id is 444555666")

    assert result["need_more_info"] is False
    assert "deleted successfully" in result["message"].lower()
    assert result["context"]["data"]["deviceId"] == "444555666"


@patch("agent_session.call_model")
def test_delete_without_id_asks_without_model(mock_call_model):
    mock_call_model.side_effect = RuntimeError("model should not be called")
    session = DeviceAgentSession()
    result = session.run_turn("delete device")

    assert result["need_more_info"] is True
    assert result["missing_fields"] == ["device_id"]
    assert "provide device_id" in result["message"].lower()


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_update_with_typo_intent_and_free_text_fields(mock_http, mock_call_model):
    mock_call_model.side_effect = RuntimeError("model should not be called")
    mock_http.return_value = {
        "device": {"deviceId": "777888999", "name": "sensor-u", "type": "gateway", "status": False}
    }

    session = DeviceAgentSession()
    result = session.run_turn("updat device id is 777888999 status is inactive")

    assert result["need_more_info"] is False
    assert "updated successfully" in result["message"].lower()
    assert result["context"]["data"]["deviceId"] == "777888999"
    assert result["context"]["data"]["status"] is False


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_update_with_only_id_asks_payload_without_model(mock_http, mock_call_model):
    mock_http.return_value = {
        "device": {"deviceId": "777888999", "name": "sensor-u", "type": "gateway"}
    }
    mock_call_model.side_effect = RuntimeError("model should not be called")
    session = DeviceAgentSession()
    result = session.run_turn("update device id is 777888999")

    assert result["need_more_info"] is True
    assert "at least one update field" in result["message"].lower()
    assert "status" in result["missing_fields"]


@patch("device_tools._http_request")
def test_create_with_device_typo_still_starts_create_flow(mock_http):
    session = DeviceAgentSession()
    result = session.run_turn("create devie")

    assert result["need_more_info"] is True
    assert set(result["missing_fields"]) == {"device_name", "device_id", "device_type"}
    assert result["actions"][0]["type"] == "form"
    assert mock_http.call_count == 0


@patch("device_tools._http_request")
def test_update_invalid_id_is_reasked_before_payload(mock_http):
    def fake_http(method, path, payload=None):
        if method == "GET" and path == "/device/565fgf":
            raise BackendRequestError("Backend request failed (404): Device not found")
        return {}

    mock_http.side_effect = fake_http
    session = DeviceAgentSession()
    first = session.run_turn("device update")
    assert first["need_more_info"] is True
    assert first["missing_fields"] == ["device_id"]

    second = session.run_turn("565fgf")
    assert second["need_more_info"] is True
    assert second["missing_fields"] == ["device_id"]
    assert "please provide a valid device_id" in second["message"].lower()


@patch("device_tools._http_request")
def test_update_invalid_numeric_id_is_reasked_and_does_not_accept_payload(mock_http):
    def fake_http(method, path, payload=None):
        if method == "GET" and path == "/device/99996666":
            raise BackendRequestError("Backend request failed (400): Invalid device ID")
        return {}

    mock_http.side_effect = fake_http
    session = DeviceAgentSession()
    session.run_turn("device update")
    second = session.run_turn("device id is 99996666")

    assert second["need_more_info"] is True
    assert second["missing_fields"] == ["device_id"]


@patch("device_tools._http_request")
def test_get_device_falls_back_to_list_when_direct_lookup_404(mock_http):
    internal_id = "507f1f77bcf86cd799439099"

    def fake_http(method, path, payload=None):
        if method == "GET" and path == "/device":
            return {
                "data": {
                    "items": [
                        {
                            "id": internal_id,
                            "deviceId": "real-device-b",
                            "name": "Real Device B",
                            "type": "gateway",
                        }
                    ]
                }
            }
        if method == "GET" and path in {
            "/device/real-device-b",
            f"/device/{internal_id}",
        }:
            raise BackendRequestError("Backend request failed (404): Device not found")
        raise AssertionError(f"Unexpected request: {(method, path)}")

    mock_http.side_effect = fake_http
    session = DeviceAgentSession()
    session.run_turn("get a device")
    result = session.run_turn("real-device-b")

    assert result["need_more_info"] is False
    assert "fetched successfully" in result["message"].lower()
    assert result["context"]["data"]["deviceId"] == "real-device-b"


@patch("device_tools._http_request")
def test_stateless_get_retry_keeps_intent_after_invalid_device_id(mock_http):
    def fake_http(method, path, payload=None):
        if method == "GET" and path == "/device":
            return {
                "devices": [
                    {
                        "id": "507f1f77bcf86cd799439211",
                        "deviceId": "14141414",
                        "name": "sensor-1414",
                        "type": "sensor",
                    }
                ]
            }
        if method == "GET" and path == "/device/real-device-b":
            raise BackendRequestError("Backend request failed (404): Device not found")
        if method == "GET" and path in {"/device/14141414", "/device/507f1f77bcf86cd799439211"}:
            return {
                "device": {
                    "id": "507f1f77bcf86cd799439211",
                    "deviceId": "14141414",
                    "name": "sensor-1414",
                    "type": "sensor",
                }
            }
        raise AssertionError(f"Unexpected request: {(method, path)}")

    mock_http.side_effect = fake_http
    history: list[dict] = []

    r1, history = DeviceAgentSession.run_turn_stateless("get a device", history)
    assert r1["need_more_info"] is True
    assert r1["missing_fields"] == ["device_id"]

    r2, history = DeviceAgentSession.run_turn_stateless("real-device-b", history)
    assert r2["need_more_info"] is True
    assert "get the device details" in r2["message"].lower()
    assert r2["missing_fields"] == ["device_id"]

    r3, history = DeviceAgentSession.run_turn_stateless("14141414", history)
    assert r3["need_more_info"] is False
    assert "fetched successfully" in r3["message"].lower()
    assert r3["context"]["data"]["deviceId"] == "14141414"


@patch("device_tools._http_request")
def test_delete_by_text_device_id_resolves_to_internal_object_id(mock_http):
    internal_id = "507f1f77bcf86cd799439011"
    public_id = "ai-dev-1776705850"
    called = []

    def fake_http(method, path, payload=None):
        called.append((method, path))
        if method == "GET" and path == "/device":
            return {
                "devices": [
                    {
                        "id": internal_id,
                        "deviceId": public_id,
                        "name": "sensor-x",
                        "type": "sensor",
                    }
                ]
            }
        if method == "DELETE" and path == f"/device/{internal_id}":
            return {
                "device": {
                    "id": internal_id,
                    "deviceId": public_id,
                    "name": "sensor-x",
                    "type": "sensor",
                }
            }
        raise AssertionError(f"Unexpected request: {(method, path)}")

    mock_http.side_effect = fake_http
    result = execute_tool({"name": "delete_device", "device_id": public_id})

    assert result["ok"] is True
    assert ("DELETE", f"/device/{internal_id}") in called


def test_actions_use_ui_action_contract_for_crud_results():
    create = map_tool_result_to_fe(
        "create_device",
        {"ok": True, "device": {"deviceId": "d1", "name": "n1", "type": "sensor"}},
    )
    get_one = map_tool_result_to_fe(
        "get_device",
        {"ok": True, "device": {"deviceId": "d1", "name": "n1", "type": "sensor"}},
    )
    update = map_tool_result_to_fe(
        "update_device",
        {"ok": True, "device": {"deviceId": "d1", "name": "n1", "type": "sensor"}},
    )
    delete = map_tool_result_to_fe(
        "delete_device",
        {"ok": True, "device": {"deviceId": "d1", "name": "n1", "type": "sensor"}},
    )
    list_all = map_tool_result_to_fe(
        "list_devices",
        {"ok": True, "devices": [{"deviceId": "d1", "name": "n1", "type": "sensor"}]},
    )

    assert create["data"]["device"]["deviceId"] == "d1"
    assert create["actions"][0]["kind"] == "navigate"
    assert create["actions"][0]["destination"]["screen"] == "resource_detail"
    assert create["actions"][0]["destination"]["idFrom"] == "data.device.id"
    assert create["actions"][1]["kind"] == "show_list"
    assert create["actions"][2]["kind"] == "confirm"

    assert get_one["actions"][0]["kind"] == "navigate"
    assert get_one["actions"][0]["destination"]["screen"] == "resource_edit"
    assert get_one["actions"][0]["destination"]["idFrom"] == "data.device.id"
    assert get_one["actions"][1]["kind"] == "show_list"

    assert update["actions"][0]["kind"] == "navigate"
    assert update["actions"][0]["destination"]["screen"] == "resource_detail"
    assert update["actions"][0]["destination"]["idFrom"] == "data.device.id"
    assert update["actions"][1]["kind"] == "show_list"

    assert delete["actions"][0]["kind"] == "show_list"
    assert delete["actions"][1]["kind"] == "navigate"
    assert delete["actions"][1]["destination"]["screen"] == "resource_create"

    assert list_all["actions"][0]["kind"] == "navigate"
    assert list_all["actions"][0]["destination"]["screen"] == "resource_create"


@patch("device_tools._http_request")
def test_session_delete_keeps_full_ai_dev_identifier_and_resolves(mock_http):
    internal_id = "507f1f77bcf86cd799439011"
    public_id = "ai-dev-1776705850"

    def fake_http(method, path, payload=None):
        if method == "GET" and path == "/device":
            return {
                "devices": [
                    {
                        "id": internal_id,
                        "deviceId": public_id,
                        "name": "sensor-x",
                        "type": "sensor",
                    }
                ]
            }
        if method == "DELETE" and path == f"/device/{internal_id}":
            return {
                "device": {
                    "id": internal_id,
                    "deviceId": public_id,
                    "name": "sensor-x",
                    "type": "sensor",
                }
            }
        raise AssertionError(f"Unexpected request: {(method, path)}")

    mock_http.side_effect = fake_http
    session = DeviceAgentSession()
    session.run_turn("delete device")
    result = session.run_turn(public_id)

    assert result["need_more_info"] is False
    assert "deleted successfully" in result["message"].lower()


@patch("device_tools._http_request")
def test_update_accepts_device_type_is_with_underscore_notation(mock_http):
    def fake_http(method, path, payload=None):
        if method == "GET" and path == "/device/1919191919":
            return {"device": {"deviceId": "1919191919", "name": "sensor-1", "type": "sensor"}}
        if method == "PATCH" and path == "/device/1919191919":
            assert payload["type"] == "gateway"
            return {"device": {"deviceId": "1919191919", "name": "sensor-1", "type": "gateway"}}
        return {}

    mock_http.side_effect = fake_http
    session = DeviceAgentSession()
    first = session.run_turn("update device")
    assert first["need_more_info"] is True
    second = session.run_turn("id is 1919191919")
    assert second["need_more_info"] is True
    third = session.run_turn("device_type is gateway")

    assert third["need_more_info"] is False
    assert "updated successfully" in third["message"].lower()
    assert third["context"]["data"]["type"] == "gateway"


@patch("device_tools._http_request")
def test_delete_success_when_backend_returns_no_device_payload(mock_http):
    def fake_http(method, path, payload=None):
        if method == "GET" and path == "/device":
            return {
                "devices": [
                    {"id": "507f1f77bcf86cd799439012", "deviceId": "1919191919", "name": "x", "type": "sensor"}
                ]
            }
        if method == "DELETE" and path == "/device/507f1f77bcf86cd799439012":
            return {"success": True, "message": "deleted"}
        return {}

    mock_http.side_effect = fake_http
    session = DeviceAgentSession()
    session.run_turn("delete device")
    result = session.run_turn("1919191919")

    assert result["need_more_info"] is False
    assert "deleted successfully" in result["message"].lower()
    assert result["context"]["data"]["deviceId"] == "1919191919"


@patch("device_tools._http_request")
def test_create_accepts_device_type_is_with_underscore_notation(mock_http):
    mock_http.return_value = {
        "device": {
            "deviceId": "1010101010",
            "name": "sensor5",
            "type": "gateway",
        }
    }
    session = DeviceAgentSession()
    session.run_turn("create device")
    confirm = session.run_turn("type is gateway, name is sensor5 , id is 1010101010")
    assert confirm["need_more_info"] is True
    assert confirm["actions"][0]["target"] == "CreateDevice"

    result = session.run_turn("__action__:CreateDevice")

    assert result["need_more_info"] is False
    assert "created successfully" in result["message"].lower()
    assert result["context"]["data"]["type"] == "gateway"


@patch("device_tools._http_request")
def test_stateless_history_continues_create_flow(mock_http):
    mock_http.return_value = {
        "device": {"deviceId": "st-1001", "name": "sensor-st", "type": "gateway"}
    }

    history: list[dict] = []
    r1, history = DeviceAgentSession.run_turn_stateless("create device", history)
    assert r1["need_more_info"] is True

    r2, history = DeviceAgentSession.run_turn_stateless("name is sensor-st", history)
    assert r2["need_more_info"] is True
    assert "device_id" in r2["missing_fields"]

    r3, history = DeviceAgentSession.run_turn_stateless("id is st-1001", history)
    assert r3["need_more_info"] is True
    assert "device_type" in r3["missing_fields"]

    r4, history = DeviceAgentSession.run_turn_stateless("type is gateway", history)
    assert r4["need_more_info"] is True
    assert r4["actions"][0]["target"] == "CreateDevice"

    r5, history = DeviceAgentSession.run_turn_stateless("__action__:CreateDevice", history)
    assert r5["need_more_info"] is False
    assert "created successfully" in r5["message"].lower()
    assert r5["context"]["data"]["deviceId"] == "st-1001"
    assert len(history) == 10


@patch("device_tools._http_request")
def test_create_understands_misspelled_name_and_type_alias(mock_http):
    mock_http.return_value = {
        "device": {"deviceId": "x-1", "name": "sensor", "type": "gateway"}
    }

    session = DeviceAgentSession()
    session.run_turn("create device")
    confirm = session.run_turn("mame is sensor , type = gateway, id = x-1")
    assert confirm["need_more_info"] is True
    assert confirm["actions"][0]["target"] == "CreateDevice"

    result = session.run_turn("__action__:CreateDevice")

    assert result["need_more_info"] is False
    assert "created successfully" in result["message"].lower()
    assert result["context"]["data"]["name"] == "sensor"
    assert result["context"]["data"]["type"] == "gateway"


@patch("device_tools._http_request")
def test_create_understands_nam_type_id_compact_sentence(mock_http):
    mock_http.return_value = {
        "device": {"deviceId": "12121212", "name": "sensor", "type": "gateway"}
    }
    session = DeviceAgentSession()
    session.run_turn("create device")
    confirm = session.run_turn("nam is sensor type gateway id 12121212")

    assert confirm["need_more_info"] is True
    assert confirm["actions"][0]["target"] == "CreateDevice"

    result = session.run_turn("__action__:CreateDevice")
    assert result["need_more_info"] is False
    assert result["context"]["data"]["deviceId"] == "12121212"
    assert result["context"]["data"]["name"] == "sensor"
    assert result["context"]["data"]["type"] == "gateway"


@patch("device_tools._http_request")
def test_create_understands_inverted_name_and_type_sentence(mock_http):
    mock_http.return_value = {
        "device": {"deviceId": "12121212", "name": "sensor", "type": "gateway"}
    }
    session = DeviceAgentSession()
    session.run_turn("create device")
    session.run_turn("id is 12121212")
    confirm = session.run_turn("sensor is name gateway is type")

    assert confirm["need_more_info"] is True
    assert confirm["actions"][0]["target"] == "CreateDevice"

    result = session.run_turn("__action__:CreateDevice")
    assert result["need_more_info"] is False
    assert result["context"]["data"]["name"] == "sensor"
    assert result["context"]["data"]["type"] == "gateway"


@patch("agent_session.call_model")
@patch("device_tools._http_request")
def test_stateless_update_followup_does_not_fall_back_to_model(mock_http, mock_call_model):
    mock_call_model.side_effect = RuntimeError("model should not be called")

    def fake_http(method, path, payload=None):
        if method == "GET" and path == "/device/10101010":
            return {"device": {"deviceId": "10101010", "name": "sensor-1", "type": "sensor"}}
        if method == "PATCH" and path == "/device/10101010":
            return {"device": {"deviceId": "10101010", "name": "sensor-10", "type": "sensor"}}
        return {}

    mock_http.side_effect = fake_http
    history: list[dict] = []
    r1, history = DeviceAgentSession.run_turn_stateless("device update", history)
    assert r1["need_more_info"] is True
    r2, history = DeviceAgentSession.run_turn_stateless("id is 10101010", history)
    assert r2["need_more_info"] is True
    r3, history = DeviceAgentSession.run_turn_stateless("device_name=sensor-10", history)
    assert r3["need_more_info"] is False
    assert "updated successfully" in r3["message"].lower()
