from tool_search import select_tool


def test_select_tool_picks_create_from_intent_and_fields():
    selection = select_tool(
        user_text="create a new device",
        candidate_tool="get_device",
        tool_data={
            "name": "get_device",
            "device_name": "sensor-a",
            "device_id": "dev-1",
            "device_type": "sensor",
        },
    )

    assert selection.tool_name == "create_device"
    assert selection.confidence > 0


def test_select_tool_picks_update_from_fields():
    selection = select_tool(
        user_text="device changes",
        candidate_tool="get_device",
        tool_data={
            "name": "get_device",
            "device_id": "dev-2",
            "status": "inactive",
        },
    )

    assert selection.tool_name == "update_device"


def test_select_tool_returns_none_for_unrelated_text():
    selection = select_tool(user_text="hello there")

    assert selection.tool_name is None
    assert selection.confidence == 0.0
