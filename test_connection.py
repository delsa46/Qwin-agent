#!/usr/bin/env python3

import json
from unittest.mock import patch
from device_tools import list_devices, create_device


def test_with_mock():
    print("Testing with mock backend data and verifying listed devices...")

    created_devices: list[dict[str, str]] = []

    def mock_http_request(method, path, payload=None):
        if method == "GET" and path == "/device":
            base_devices = [
                {
                    "deviceId": "dev-001",
                    "name": "Temperature Sensor 1",
                    "type": "temperature",
                },
                {
                    "deviceId": "dev-002",
                    "name": "Humidity Sensor",
                    "type": "humidity",
                },
            ]
            return {"devices": base_devices + created_devices}

        if method == "POST" and path == "/device":
            device = {
                "deviceId": payload.get("deviceId", "dev-new-000"),
                "name": payload.get("name", "new-device"),
                "type": payload.get("type", "sensor"),
            }
            created_devices.append(device)
            return {"device": device}

        raise Exception("Mock not implemented for this request")

    with patch('device_tools._http_request', side_effect=mock_http_request):
        print("\n1. Creating device A")
        create_a = create_device(name="device-a", type="sensor", device_id="dev-a")
        print(json.dumps(create_a, ensure_ascii=False, indent=2))

        print("\n2. Creating device B")
        create_b = create_device(name="device-b", type="gateway", device_id="dev-b")
        print(json.dumps(create_b, ensure_ascii=False, indent=2))

        print("\n3. Listing devices after creation")
        result = list_devices()
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if result.get("ok"):
            found_ids = {device.get("deviceId") for device in result["devices"]}
            print("\nFound device ids in list:", sorted(found_ids))
            assert "dev-a" in found_ids and "dev-b" in found_ids, "Created devices not found in list"
            print("\nBoth created devices are visible in the returned list.")
        else:
            print("\nFailed to list devices")


def test_real_connection():
    print("Testing real connection to Sensolist backend...")

    print("\n1. Listing devices:")
    result = list_devices()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not result.get("ok"):
        print("Failed to list devices - this may be due to network or backend issues")
        return

    print("\n2. Creating device A in real backend:")
    payload_a = {
        "name": "real-device-a",
        "type": "sensor",
        "device_id": "real-dev-a",
    }
    create_result_a = create_device(**payload_a)
    print(json.dumps(create_result_a, ensure_ascii=False, indent=2))

    print("\n3. Creating device B in real backend:")
    payload_b = {
        "name": "real-device-b",
        "type": "gateway",
        "device_id": "real-dev-b",
    }
    create_result_b = create_device(**payload_b)
    print(json.dumps(create_result_b, ensure_ascii=False, indent=2))

    if not create_result_a.get("ok") or not create_result_b.get("ok"):
        print("\nFailed to create one or both real devices")
        return

    print("\n4. Verifying real devices are present in list:")
    verify_result = list_devices()
    print(json.dumps(verify_result, ensure_ascii=False, indent=2))

    if verify_result.get("ok"):
        found_ids = {device.get("deviceId") for device in verify_result["devices"]}
        print("\nFound device ids in list:", sorted(found_ids))
        if "real-dev-a" in found_ids and "real-dev-b" in found_ids:
            print("\nReal devices were created and are visible in the backend list.")
        else:
            print("\nCreated real devices were not found in the backend list.")
    else:
        print("\nFailed to list devices during verification")


if __name__ == "__main__":
    test_with_mock()
    print("\n" + "="*50)
    test_real_connection()
