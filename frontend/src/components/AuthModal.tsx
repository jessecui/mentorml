import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { LogoMark } from "./LogoMark";

interface AuthModalProps {
  onAuthenticated: () => void;
}

export function AuthModal({ onAuthenticated }: AuthModalProps) {
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const inputRef = useRef<HTMLInputElement>(null);

  // The input is disabled while verifying, which drops focus; give it back
  // so a mistyped name can be retyped straight away
  useEffect(() => {
    if (!isLoading) inputRef.current?.focus();
  }, [isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    let response: Response;
    try {
      response = await fetch("/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer: password }),
      });
    } catch {
      setError("Network error. Please try again.");
      setIsLoading(false);
      return;
    }

    if (response.ok) {
      // The server set its HttpOnly session cookie in this response
      onAuthenticated();
    } else {
      // A wrong name (401), too many attempts (429) or a server problem: show the server's message
      const body = await response.json().catch(() => null);
      setError(typeof body?.detail === "string" ? body.detail : `Request failed (HTTP ${response.status})`);
      if (response.status === 401) setPassword("");
    }
    setIsLoading(false);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(17,24,39,0.45)" }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-title"
        className="w-full max-w-md rounded-lg border border-border bg-surface p-6 shadow-xl"
      >
        <h2 id="auth-title" className="flex items-center gap-2 text-lg font-semibold text-foreground">
          <LogoMark />
          MentorML
        </h2>
        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              An AI tutor that explains machine learning concepts with diagrams
            </p>
            <label
              htmlFor="creator-question"
              className="block pt-2 text-sm font-medium text-foreground"
            >
              To continue, enter the name of this project's creator.
            </label>
            <input
              ref={inputRef}
              id="creator-question"
              name="creator-question"
              type="text"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={isLoading}
              autoComplete="one-time-code"
              placeholder="Full name"
              autoFocus
              className="h-10 w-full rounded-md border border-border bg-surface px-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30 disabled:bg-muted"
            />
          </div>

          {error && (
            <p role="alert" className="text-sm text-danger">{error}</p>
          )}

          <button
            type="submit"
            disabled={isLoading || !password.trim()}
            className="flex h-10 w-full cursor-pointer items-center justify-center gap-2 rounded-md bg-primary text-sm font-medium text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Verifying…
              </>
            ) : (
              "Continue"
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
