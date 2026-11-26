import { cn } from "../lib/utils";

const NODES: [number, number][] = [
  [8.5, 22.5],
  [8.5, 9.5],
  [16, 18],
  [23.5, 9.5],
  [23.5, 22.5],
];

// The MentorML mark: an "M" drawn as a small neural network, five white nodes
// joined by the letter's strokes, on a flat amber tile. Same drawing as
// public/favicon.svg.
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className={cn("h-7 w-7 shrink-0", className)}>
      <rect width="32" height="32" rx="7" fill="#f59e0b" />
      <path
        d="M8.5 22.5V9.5l7.5 8.5 7.5-8.5v13"
        fill="none"
        stroke="#ffffff"
        strokeOpacity="0.72"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {NODES.map(([cx, cy]) => (
        <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="2.3" fill="#ffffff" />
      ))}
    </svg>
  );
}
