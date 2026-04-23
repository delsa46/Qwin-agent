export type LegacyFormAction = {
  type: "button" | "form";
  label?: string;
  variant?: "success" | "primary" | "danger" | "warning" | "info";
  target?: string;
  tool?: string;
  title?: string;
  submit_label?: string;
  fields?: Array<{
    name: string;
    label: string;
    input_type: "text" | "select";
    required?: boolean;
    value?: string;
    options?: Array<{ label: string; value: string }>;
    is_missing?: boolean;
  }>;
};

export type UiAction =
  | {
      kind: "navigate";
      operationId?: string;
      destination:
        | { screen: "resource_list"; resource: string }
        | { screen: "resource_detail"; resource: string; idFrom: string }
        | { screen: "resource_create"; resource: string }
        | { screen: "resource_edit"; resource: string; idFrom: string };
      label: string;
      variant?: "primary" | "secondary" | "success" | "danger";
    }
  | {
      kind: "show_list";
      resource: string;
      label: string;
      filtersFrom?: string;
      variant?: "primary" | "secondary";
    }
  | {
      kind: "confirm";
      label: string;
      target?: string;
      confirmKey: string;
      title?: string;
      message?: string;
      variant?: "primary" | "danger";
    }
  | {
      kind: "ask_input";
      label: string;
      inputKey: string;
      title?: string;
      message?: string;
      inputType?: "text" | "number" | "select" | "textarea";
      options?: Array<{ label: string; value: string }>;
      required?: boolean;
      variant?: "primary" | "secondary";
    }
  | {
      kind: "ask_explanation";
      label: string;
      explanationKey: string;
      title?: string;
      message?: string;
      variant?: "secondary";
    };

export type AgentAction = LegacyFormAction | UiAction;

export type AgentResponse = {
  data?: unknown;
  context: {
    entity: string;
    data: unknown;
  };
  message: string;
  actions: AgentAction[];
  need_more_info: boolean;
  missing_fields: string[];
};

export type ChatHistoryItem =
  | { role: "user"; text: string }
  | { role: "assistant"; response: AgentResponse };
