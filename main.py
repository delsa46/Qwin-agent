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
