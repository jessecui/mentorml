import { useRef, useEffect } from "react";
import { Message } from "./Message";
import { LogoMark } from "./LogoMark";
import type { Message as MessageType } from "../types";

interface MessageListProps {
  messages: MessageType[];
  onFillSuggestion?: (message: string) => void;
}

const SUGGESTIONS = [
  "How does self-attention work in a transformer?",
  "How does an LSTM remember long-range context?",
  "How does RLHF train a language model?",
  "How do diffusion models generate images?",
];

export function MessageList({ messages, onFillSuggestion }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center p-8 text-center">
        <LogoMark className="mb-5 h-12 w-12" />
        <h2 className="mb-2 text-2xl font-semibold tracking-tight">What do you want to understand?</h2>
        <p className="mb-6 max-w-md text-sm text-muted-foreground">
          Ask about any machine learning concept. Answers come with a diagram
          chosen from a curated library, linked to its source.
        </p>
        <div className="grid w-full max-w-xl gap-2 text-sm sm:grid-cols-2">
          {SUGGESTIONS.map((text) => (
            <button
              key={text}
              type="button"
              className="cursor-pointer rounded-md border border-border bg-surface px-3 py-2.5 text-left text-muted-foreground hover:border-accent hover:text-foreground"
              onClick={() => onFillSuggestion?.(text)}
            >
              {text}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto [contain:strict]">
      <div className="mx-auto max-w-3xl px-4 pb-12 pt-6">
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
