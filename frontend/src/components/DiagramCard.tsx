import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { Diagram } from "../types";
import { ExternalLink, Maximize2, X } from "lucide-react";

interface DiagramCardProps {
  diagram: Diagram;
}

export function DiagramCard({ diagram }: DiagramCardProps) {
  const [isZoomed, setIsZoomed] = useState(false);

  // Close the enlarged view with Escape
  useEffect(() => {
    if (!isZoomed) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsZoomed(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isZoomed]);

  const sourceLabel = [diagram.source_name, diagram.source_title]
    .filter(Boolean)
    .join(": ");

  return (
    <figure className="my-5 overflow-hidden rounded-lg border border-border bg-surface">
      {/* Diagrams are drawn for a light background, so the image area stays white in dark mode */}
      <button
        type="button"
        onClick={() => setIsZoomed(true)}
        className="group relative block w-full cursor-zoom-in bg-white p-3"
        title="Enlarge"
      >
        <img
          src={diagram.image_url}
          alt={diagram.description || diagram.title}
          className="mx-auto max-h-[28rem] w-auto max-w-full object-contain"
          loading="lazy"
        />
        <span className="absolute right-2 top-2 rounded-md bg-black/60 p-1.5 text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
          <Maximize2 className="h-3.5 w-3.5" aria-hidden="true" />
        </span>
      </button>
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-t border-border px-3 py-2 text-xs">
        <span className="font-medium text-foreground">{diagram.title}</span>
        {diagram.source_url && (
          <a
            href={diagram.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex min-w-0 items-center gap-1 text-muted-foreground hover:text-accent-text"
          >
            <span className="truncate">{sourceLabel || "Source"}</span>
            <ExternalLink className="h-3 w-3 shrink-0" aria-hidden="true" />
          </a>
        )}
      </figcaption>

      {/* Rendered into <body>: the message list is a containment context, which
          would otherwise trap a fixed overlay inside the scrolled content */}
      {isZoomed && createPortal(
        <div
          role="dialog"
          aria-modal="true"
          aria-label={diagram.title}
          className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-black/80 p-4 sm:p-8"
          onClick={() => setIsZoomed(false)}
        >
          <button
            type="button"
            onClick={() => setIsZoomed(false)}
            aria-label="Close"
            className="absolute right-4 top-4 cursor-pointer rounded-md bg-white/10 p-2 text-white hover:bg-white/20"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
          <img
            src={diagram.image_url}
            alt={diagram.description || diagram.title}
            className="max-h-full max-w-full rounded-md bg-white object-contain p-3"
          />
        </div>,
        document.body,
      )}
    </figure>
  );
}
