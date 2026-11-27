import { useEffect } from "react";

/**
 * Keeps --app-height equal to the part of the screen that is actually visible.
 *
 * iOS Safari does not shrink the page when the on-screen keyboard opens: it lays
 * the keyboard over the bottom of it, so a full-height layout leaves its input
 * hidden until the user scrolls. The visual viewport does shrink, so the layout
 * follows that instead (see .h-app in index.css).
 */
export function useVisibleHeight() {
  useEffect(() => {
    const viewport = window.visualViewport;
    if (!viewport) return;
    const root = document.documentElement;

    const update = () => {
      // Pinch-zoom also shrinks the visual viewport; only follow it at normal zoom
      if (Math.abs(viewport.scale - 1) > 0.01) {
        root.style.removeProperty("--app-height");
        return;
      }
      root.style.setProperty("--app-height", `${viewport.height}px`);
      // Safari may have scrolled the page up to reveal the field; the layout now fits, so undo that
      if (viewport.offsetTop > 0 || window.scrollY > 0) window.scrollTo(0, 0);
    };

    update();
    viewport.addEventListener("resize", update);
    viewport.addEventListener("scroll", update);
    return () => {
      viewport.removeEventListener("resize", update);
      viewport.removeEventListener("scroll", update);
      root.style.removeProperty("--app-height");
    };
  }, []);
}
