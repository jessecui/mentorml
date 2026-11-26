import { useState } from "react";
import { Moon, Plus, Sun } from "lucide-react";
import { useStreamChat } from "../hooks/useStreamChat";
import { useTheme } from "../lib/theme";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { LogoMark } from "./LogoMark";

const headerButton =
  "flex h-9 cursor-pointer items-center gap-1.5 rounded-md px-2.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-muted-foreground";

interface ChatProps {
  onUnauthorized?: () => void;
}

export function Chat({ onUnauthorized }: ChatProps) {
  const [input, setInput] = useState("");
  const { messages, isStreaming, sendMessage, stopStreaming, clearMessages } =
    useStreamChat({
      onUnauthorized: (unsentMessage) => {
        setInput(unsentMessage); // put the question back so it can be resent
        onUnauthorized?.();
      },
    });
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="flex h-dvh flex-col bg-background">
      {/* Header */}
      <header className="flex shrink-0 items-center gap-3 border-b border-border bg-surface px-4 py-2.5">
        <LogoMark className="h-8 w-8" />
        <div className="min-w-0">
          <h1 className="text-sm font-semibold leading-tight">MentorML</h1>
          <p className="truncate text-xs text-muted-foreground">
            Machine learning, explained with diagrams
          </p>
        </div>
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={clearMessages}
            disabled={messages.length === 0 || isStreaming}
            className={headerButton}
            title="Start a new conversation"
          >
            <Plus className="h-4 w-4" aria-hidden="true" />
            <span className="hidden sm:inline">New chat</span>
          </button>
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            className={headerButton}
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4" aria-hidden="true" />
            ) : (
              <Moon className="h-4 w-4" aria-hidden="true" />
            )}
          </button>
        </div>
      </header>

      {/* Messages */}
      <MessageList messages={messages} onFillSuggestion={setInput} />

      {/* Input */}
      <ChatInput
        input={input}
        setInput={setInput}
        onSend={sendMessage}
        onStop={stopStreaming}
        isStreaming={isStreaming}
      />
    </div>
  );
}
