export interface TeachingPlan {
  topic: string;
  steps: string[];
  diagrams_needed: string[];
}

export interface Diagram {
  id: string;
  image_url: string;
  title: string;
  description: string;
  teaches: string;
  source_name: string;
  source_title: string;
  source_url: string;
  score: number;
  query: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  plan?: TeachingPlan;
  diagrams?: Diagram[];
  isStreaming?: boolean;
  wasStopped?: boolean;
  // The answer stopped early (connection lost or a server error)
  wasInterrupted?: boolean;
}

export interface StreamEvent {
  event: "token" | "reset" | "plan" | "diagram" | "done" | "error";
  data: string;
}
