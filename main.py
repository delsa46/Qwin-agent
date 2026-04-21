import json
from agent_session import DeviceAgentSession


if __name__ == "__main__":
    chat_history: list[dict] = []
    while True:
        user_text = input("\nUSER: ").strip()

        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        result, chat_history = DeviceAgentSession.run_turn_stateless(user_text, chat_history)
        print("\nAGENT RESPONSE:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
