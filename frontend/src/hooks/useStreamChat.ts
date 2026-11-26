import { useState, useCallback, useRef } from "react";
import type { Message, TeachingPlan, Diagram } from "../types";

interface UseStreamChatOptions {
  onPlan?: (plan: TeachingPlan) => void;
  onDiagram?: (diagram: Diagram) => void;
  // Called when the server has no valid session (expired, or the server's signing key changed),
  // with the message that could not be sent
  onUnauthorized?: (unsentMessage: string) => void;
}

// A conversation's ID is its only key on the server, so it must be unguessable.
// crypto.getRandomValues works on plain http too, unlike crypto.randomUUID.
function newThreadId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return "thread-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export function useStreamChat(options: UseStreamChatOptions = {}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const threadIdRef = useRef<string>(newThreadId());

  const sendMessage = useCallback(
    async (content: string) => {
      if (isStreaming || !content.trim()) return;

      // Add user message
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        role: "user",
        content: content.trim(),
      };

      // Add placeholder assistant message
      const assistantId = `assistant-${Date.now()}`;
      const assistantMessage: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsStreaming(true);

      // Create abort controller
      abortControllerRef.current = new AbortController();

      try {
        const response = await fetch("/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: content.trim(),
            thread_id: threadIdRef.current,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (response.status === 401) {
          // No valid session: show the login over the chat. The server never got this
          // question, so drop it from the transcript; it goes back in the input box
          // and the user resends it after signing in.
          setMessages((prev) =>
            prev.filter((msg) => msg.id !== userMessage.id && msg.id !== assistantId)
          );
          options.onUnauthorized?.(content.trim());
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";
        let accumulatedContent = "";
        let plan: TeachingPlan | undefined;
        const diagrams: Map<string, Diagram> = new Map();
        // An "event:" line and its "data:" line can arrive in different reads
        let eventType = "";
        let finished = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Parse SSE events
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7);
            } else if (line.startsWith("data: ") && eventType) {
              const data = line.slice(6);
              try {
                if (eventType === "token") {
                  const token = JSON.parse(data) as string;
                  accumulatedContent += token;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    )
                  );
                } else if (eventType === "reset") {
                  // The server restarted the answer (model fallback, or a second
                  // call after an extra search): drop the text streamed so far
                  accumulatedContent = "";
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId ? { ...msg, content: "" } : msg
                    )
                  );
                } else if (eventType === "plan") {
                  plan = JSON.parse(data) as TeachingPlan;
                  options.onPlan?.(plan);
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId ? { ...msg, plan } : msg
                    )
                  );
                } else if (eventType === "diagram") {
                  const diagram = JSON.parse(data) as Diagram;
                  diagrams.set(diagram.id, diagram);
                  options.onDiagram?.(diagram);
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId
                        ? { ...msg, diagrams: Array.from(diagrams.values()) }
                        : msg
                    )
                  );
                } else if (eventType === "done") {
                  finished = true;
                  const doneData = JSON.parse(data) as { diagrams: Diagram[] };
                  // Update with final diagrams (only referenced ones)
                  const finalDiagrams = doneData.diagrams;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId
                        ? {
                            ...msg,
                            diagrams: finalDiagrams,
                            isStreaming: false,
                          }
                        : msg
                    )
                  );
                } else if (eventType === "error") {
                  // Show the server's message; keep any text that already arrived
                  const errorData = JSON.parse(data) as { error: string };
                  finished = true;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId
                        ? {
                            ...msg,
                            content: msg.content || errorData.error,
                            isStreaming: false,
                            wasInterrupted: Boolean(msg.content),
                    diagrams: [], // not saved by the server, so later turns could not refer to them
                          }
                        : msg
                    )
                  );
                }
              } catch (e) {
                if (e instanceof SyntaxError) {
                  console.warn("Failed to parse SSE data:", data);
                } else {
                  throw e;
                }
              }
              eventType = "";
            }
          }
        }

        if (!finished) {
          // The connection closed early (proxy timeout, redeploy): keep any partial
          // text, stop the spinner and say what happened
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    content:
                      msg.content ||
                      "The connection was interrupted before an answer arrived. Please try again.",
                    isStreaming: false,
                    wasInterrupted: Boolean(msg.content),
                    diagrams: [], // not saved by the server, so later turns could not refer to them
                  }
                : msg
            )
          );
        }
      } catch (error) {
        if ((error as Error).name === "AbortError") {
          console.log("Request aborted");
          // Mark message as stopped
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, isStreaming: false, wasStopped: true, diagrams: [] }
                : msg
            )
          );
        } else {
          console.error("Stream error:", error);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    content:
                      msg.content ||
                      "Sorry, an error occurred. Please try again.",
                    isStreaming: false,
                    wasInterrupted: Boolean(msg.content),
                    diagrams: [], // not saved by the server, so later turns could not refer to them
                  }
                : msg
            )
          );
        }
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [isStreaming, options]
  );

  const stopStreaming = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    threadIdRef.current = newThreadId();
  }, []);

  return {
    messages,
    isStreaming,
    sendMessage,
    stopStreaming,
    clearMessages,
  };
}
