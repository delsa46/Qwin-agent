import { FormEvent, useMemo, useState } from "react";
import { AgentAction, AgentResponse, ChatHistoryItem } from "./types";

const API_BASE =
  (((import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_API_BASE as
    | string
    | undefined) || "");

function isAssistantItem(item: ChatHistoryItem): item is Extract<ChatHistoryItem, { role: "assistant" }> {
  return item.role === "assistant";
}

function pretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function isLegacyFormAction(action: AgentAction): action is Extract<AgentAction, { type: "form" | "button" }> {
  return "type" in action;
}

function isUiAction(action: AgentAction): action is Extract<AgentAction, { kind: string }> {
  return "kind" in action;
}

function buildMessageFromForm(action: Extract<AgentAction, { type: "form" | "button" }>, formState: Record<string, string>): string {
  const fields = action.fields ?? [];
  return fields
    .map((f) => `${f.name}=${(formState[f.name] ?? "").trim()}`)
    .filter((line) => !line.endsWith("="))
    .join(",");
}

function extractDevices(response: AgentResponse): Array<{ id?: string; deviceId?: string; name?: string; type?: string }> {
  const data = response.data as Record<string, unknown> | undefined;
  if (data && Array.isArray(data.devices)) {
    return data.devices as Array<{ id?: string; deviceId?: string; name?: string; type?: string }>;
  }
  const contextData = response.context?.data;
  if (Array.isArray(contextData)) {
    return contextData as Array<{ id?: string; deviceId?: string; name?: string; type?: string }>;
  }
  return [];
}

export default function App() {
  const [history, setHistory] = useState<ChatHistoryItem[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [formState, setFormState] = useState<Record<string, string>>({});

  const latestAssistant = useMemo(() => {
    for (let i = history.length - 1; i >= 0; i -= 1) {
      const item = history[i];
      if (isAssistantItem(item)) return item.response;
    }
    return null;
  }, [history]);

  async function sendMessage(message: string) {
    const trimmed = message.trim();
    if (!trimmed || isLoading) return;
    setIsLoading(true);
    setError("");
    try {
      const apiUrl = API_BASE ? `${API_BASE}/api/chat` : "/api/chat";
      const res = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          history
        })
      });
      const data = (await res.json()) as { ok: boolean; error?: string; history?: ChatHistoryItem[] };
      if (!res.ok || !data.ok || !data.history) {
        throw new Error(data.error || "API request failed");
      }
      setHistory(data.history);
      setInput("");
      setFormState({});
    } catch (err) {
      const baseMsg = err instanceof Error ? err.message : "Unexpected error";
      setError(
        `${baseMsg}. If this persists, ensure ui_server.py is running on port 8008.`
      );
    } finally {
      setIsLoading(false);
    }
  }

  function onSend(e: FormEvent) {
    e.preventDefault();
    void sendMessage(input);
  }

  function onFormSubmit(action: AgentAction) {
    if (!isLegacyFormAction(action)) return;
    const payload = buildMessageFromForm(action, formState);
    void sendMessage(payload);
  }

  function renderUiActionButton(action: Extract<AgentAction, { kind: string }>, idx: number) {
    let title: string | undefined;
    if (action.kind === "navigate") {
      title = `${action.destination.screen} • ${action.destination.resource}`;
    } else if (action.kind === "show_list") {
      title = `Show ${action.resource} list`;
    } else if (action.kind === "confirm") {
      title = action.message || action.title;
    } else if (action.kind === "ask_input") {
      title = action.message || action.title;
    } else if (action.kind === "ask_explanation") {
      title = action.message || action.title;
    }

    return (
      <button
        key={idx}
        className="action-btn"
        type="button"
        title={title}
        onClick={() => {
          if (action.kind === "show_list") {
            void sendMessage(`${action.resource} list`);
            return;
          }
          if (action.kind === "confirm") {
            void sendMessage(`__confirm__:${action.confirmKey}`);
            return;
          }
          if (action.kind === "ask_input") {
            setInput(`${action.inputKey}=`);
            return;
          }
          if (action.kind === "ask_explanation") {
            setInput(`${action.explanationKey}: `);
            return;
          }
        }}
      >
        {action.label}
      </button>
    );
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>Device Agent Console</h1>
        <p>Test multi-turn CRUD flows with structured AI responses.</p>
      </header>

      <main className="layout">
        <section className="panel chat">
          <div className="panel-header">Conversation</div>
          <div className="messages">
            {history.length === 0 ? (
              <div className="empty">Start with something like: create device</div>
            ) : (
              history.map((item, idx) => (
                <div key={idx} className={`message ${item.role}`}>
                  {!isAssistantItem(item) ? (
                    <>
                      <div className="bubble-title">User</div>
                      <div>{item.text}</div>
                    </>
                  ) : (
                    <>
                      <div className="bubble-title">Agent</div>
                      <div>{item.response.message}</div>
                      {extractDevices(item.response).length > 0 ? (
                        <div className="device-list">
                          <div className="device-list-title">Devices</div>
                          <ul>
                            {extractDevices(item.response).map((device, deviceIdx) => (
                              <li key={`${device.deviceId || device.id || "device"}-${deviceIdx}`}>
                                <strong>{device.name || "-"}</strong>
                                <span>ID: {device.deviceId || device.id || "-"}</span>
                                <span>Type: {device.type || "-"}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                      <div className="meta">
                        <span>need_more_info: {String(item.response.need_more_info)}</span>
                        <span>missing: {item.response.missing_fields.join(", ") || "-"}</span>
                      </div>
                    </>
                  )}
                </div>
              ))
            )}
          </div>
          <form className="composer" onSubmit={onSend}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={isLoading}
            />
            <button disabled={isLoading || !input.trim()} type="submit">
              {isLoading ? "Sending..." : "Send"}
            </button>
          </form>
          {error ? <div className="error">{error}</div> : null}
        </section>

        <section className="panel response">
          <div className="panel-header">Structured Response</div>
          {!latestAssistant ? (
            <div className="empty">No response yet.</div>
          ) : (
            <>
              <pre className="json-block">{pretty(latestAssistant)}</pre>
              {latestAssistant.actions?.map((action: AgentAction, i: number) => (
                <div className="action-block" key={i}>
                  {isUiAction(action) ? (
                    renderUiActionButton(action, i)
                  ) : action.type === "button" ? (
                    <button
                      className="action-btn"
                      type="button"
                      onClick={() => {
                        if (action.target) {
                          void sendMessage(`__action__:${action.target}`);
                        }
                      }}
                    >
                      {action.label || "Action"}
                    </button>
                  ) : (
                    <div className="dynamic-form">
                      <div className="form-title">{action.title || "Follow-up Form"}</div>
                      {(action.fields || []).map((field: NonNullable<typeof action.fields>[number]) => (
                        <label key={field.name} className="field">
                          <span>
                            {field.label}
                            {field.required ? " *" : ""}
                          </span>
                          {field.input_type === "select" ? (
                            <select
                              value={formState[field.name] ?? field.value ?? ""}
                              onChange={(e) =>
                                setFormState((prev) => ({ ...prev, [field.name]: e.target.value }))
                              }
                            >
                              <option value="">Select...</option>
                              {(field.options || []).map((opt: { label: string; value: string }) => (
                                <option key={opt.value} value={opt.value}>
                                  {opt.label}
                                </option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="text"
                              value={formState[field.name] ?? field.value ?? ""}
                              onChange={(e) =>
                                setFormState((prev) => ({ ...prev, [field.name]: e.target.value }))
                              }
                            />
                          )}
                        </label>
                      ))}
                      <button
                        type="button"
                        className="submit-form"
                        onClick={() => onFormSubmit(action)}
                        disabled={isLoading}
                      >
                        {action.submit_label || "Submit"}
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
