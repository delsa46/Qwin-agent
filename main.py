# import json
# from agent import run_agent


# if __name__ == "__main__":
#     examples = [
#         "create a device named sensor-a of type temperature",
#         "create a device",
#         "list devices",
#     ]

#     for text in examples:
#         print("=" * 80)
#         print("USER:", text)
#         result = run_agent(text)
#         print(json.dumps(result, ensure_ascii=False, indent=2))

import json
from agent_session import DeviceAgentSession


if __name__ == "__main__":
    session = DeviceAgentSession()
    while True:
        user_text = input("\nUSER: ").strip()

        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        result = session.run_turn(user_text)
        print("\nAGENT RESPONSE:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
