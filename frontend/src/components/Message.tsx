import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import type { Message as MessageType, Diagram, TeachingPlan } from "../types";
import { DiagramCard } from "./DiagramCard";
import { Loader2, ChevronDown, ChevronRight, CornerDownRight, StopCircle } from "lucide-react";
import { useState, type ReactNode } from "react";

// Matches MAX_DIAGRAMS_PER_ANSWER on the server
const MAX_DIAGRAMS = 2;

interface MessageProps {
  message: MessageType;
  // Set on the latest answer only: sends a suggested follow-up question
  onFollowup?: (question: string) => void;
}

// The agent's teaching plan: one quiet line that expands
function PlanSection({ plan }: { plan: TeachingPlan }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="mb-3 text-sm">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        className="flex max-w-full cursor-pointer items-center gap-1.5 rounded-md py-0.5 text-left text-muted-foreground hover:text-foreground"
      >
        {isExpanded ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        )}
        <span className="shrink-0 font-medium">Plan</span>
        <span className="truncate">
          {plan.topic} · {plan.steps.length} {plan.steps.length === 1 ? "step" : "steps"}
        </span>
      </button>
      {isExpanded && (
        <div className="ml-1.5 mt-2 space-y-3 border-l-2 border-border pl-4 text-muted-foreground">
          {plan.steps.length > 0 && (
            <ol className="ml-4 list-decimal space-y-1">
              {plan.steps.map((step, i) => (
                <li key={i}>{step}</li>
              ))}
            </ol>
          )}
          {plan.diagrams_needed.length > 0 && (
            <div>
              <div className="mb-1 font-medium text-foreground">Diagrams searched for</div>
              <ul className="ml-4 list-disc space-y-1">
                {plan.diagrams_needed.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function Message({ message, onFollowup }: MessageProps) {
  const isUser = message.role === "user";

  // Build a map of diagrams by ID for quick lookup
  const diagramMap = new Map<string, Diagram>();
  message.diagrams?.forEach((d) => diagramMap.set(d.id, d));

  // Custom renderer to replace [diagram: <id>] with actual diagrams
  const renderContent = (content: string) => {
    // Split by diagram references
    const parts = content.split(/\[diagram:\s*([A-Za-z0-9_-]+)\]/g);
    const result: ReactNode[] = [];
    const shown = new Set<string>();

    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 0) {
        // Text part
        if (parts[i]) {
          result.push(
            <ReactMarkdown
              key={`text-${i}`}
              remarkPlugins={[[remarkGfm, { singleTilde: false }], remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={{
                a: ({ children, href }) => (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent-text underline underline-offset-2"
                  >
                    {children}
                  </a>
                ),
                hr: () => <hr className="my-6 border-border" />,
                h1: ({ children }) => (
                  <h1 className="mb-4 mt-6 text-2xl font-semibold tracking-tight first:mt-0">
                    {children}
                  </h1>
                ),
                h2: ({ children }) => (
                  <h2 className="mb-3 mt-5 text-xl font-semibold first:mt-0">
                    {children}
                  </h2>
                ),
                h3: ({ children }) => (
                  <h3 className="mb-2 mt-4 text-lg font-semibold first:mt-0">
                    {children}
                  </h3>
                ),
                h4: ({ children }) => (
                  <h4 className="mb-2 mt-4 font-semibold first:mt-0">{children}</h4>
                ),
                p: ({ children }) => <p className="mb-3 leading-relaxed last:mb-0">{children}</p>,
                ul: ({ children }) => (
                  <ul className="mb-3 ml-6 list-disc space-y-1">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="mb-3 ml-6 list-decimal space-y-1">{children}</ol>
                ),
                li: ({ children }) => <li>{children}</li>,
                strong: ({ children }) => (
                  <strong className="font-semibold">{children}</strong>
                ),
                em: ({ children }) => <em className="italic">{children}</em>,
                code: ({ children, className }) => {
                  const isBlock = className?.includes("language-");
                  return isBlock ? (
                    <pre className="my-3 overflow-x-auto rounded-md bg-code-bg p-4 text-sm text-code-fg">
                      <code>{children}</code>
                    </pre>
                  ) : (
                    <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.9em]">
                      {children}
                    </code>
                  );
                },
                table: ({ children }) => (
                  <div className="my-4 overflow-x-auto rounded-md border border-border bg-surface">
                    <table className="w-full border-collapse text-sm">{children}</table>
                  </div>
                ),
                thead: ({ children }) => <thead className="bg-muted">{children}</thead>,
                tr: ({ children }) => (
                  <tr className="border-b border-border last:border-b-0">{children}</tr>
                ),
                th: ({ children, style }) => (
                  <th className="px-3 py-2 text-left font-semibold" style={style}>
                    {children}
                  </th>
                ),
                td: ({ children, style }) => (
                  <td className="px-3 py-2 align-top" style={style}>
                    {children}
                  </td>
                ),
                blockquote: ({ children }) => (
                  <blockquote className="my-3 border-l-2 border-accent pl-4 text-muted-foreground">
                    {children}
                  </blockquote>
                ),
              }}
            >
              {parts[i]}
            </ReactMarkdown>
          );
        }
      } else {
        // Diagram reference: each diagram once, at most two, as on the server
        const diagramId = parts[i];
        const diagram = diagramMap.get(diagramId);
        if (diagram && !shown.has(diagramId) && shown.size < MAX_DIAGRAMS) {
          shown.add(diagramId);
          result.push(<DiagramCard key={`diagram-${i}-${diagramId}`} diagram={diagram} />);
        }
      }
    }

    return result;
  };

  if (isUser) {
    return (
      <div className="mb-6 flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-lg bg-muted px-4 py-2.5">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <article className="mb-10">
      {/* Show the teaching plan once it arrives */}
      {message.plan && <PlanSection plan={message.plan} />}

      <div className="max-w-none">
        {message.content ? (
          renderContent(message.content)
        ) : message.isStreaming ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            <span>
              {!message.plan
                ? "Planning…"
                : message.plan.diagrams_needed.length > 0
                  ? "Finding diagrams…"
                  : "Writing…"}
            </span>
          </div>
        ) : null}

        {/* Show stopped indicator */}
        {message.wasInterrupted && (
          <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
            <StopCircle className="h-4 w-4" aria-hidden="true" />
            <span>Something went wrong: this answer may be incomplete</span>
          </div>
        )}

        {message.wasStopped && (
          <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
            <StopCircle className="h-4 w-4" aria-hidden="true" />
            <span>Response stopped</span>
          </div>
        )}

        {/* Ways to go deeper, once the answer has finished cleanly */}
        {onFollowup &&
          !message.isStreaming &&
          !message.wasStopped &&
          !message.wasInterrupted &&
          message.content &&
          (message.plan?.followups?.length ?? 0) > 0 && (
            <div className="mt-5 flex flex-col items-start gap-2">
              {message.plan!.followups!.map((question) => (
                <button
                  key={question}
                  type="button"
                  onClick={() => onFollowup(question)}
                  className="flex max-w-full cursor-pointer items-center gap-2 rounded-md border border-border bg-surface px-3 py-1.5 text-left text-sm text-muted-foreground hover:border-accent hover:text-foreground"
                >
                  <CornerDownRight className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                  <span>{question}</span>
                </button>
              ))}
            </div>
          )}
      </div>
    </article>
  );
}
