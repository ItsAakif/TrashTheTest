import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { deleteNote, type NotesListItem } from "@/lib/api-services";
import { SpinnerIcon } from "@/components/icons";

const MUTED = "#acabaa";
const INK = "#e7e5e4";
const PANEL = "#131313";
const TILE = "#1f2020";
const DANGER = "#ee7d77";

/** Human date for a list row; never a raw "Invalid Date". */
function formatDate(value?: string | null): string | null {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toISOString().slice(0, 10);
}

/**
 * Metadata line for one note: mode · pages · topics · coverage · date.
 * Every field comes from the stored artifact; missing fields are omitted
 * rather than shown as zeros.
 */
export function noteMetaLine(item: NotesListItem): string {
  const parts: Array<string | null> = [
    item.mode ? String(item.mode) : null,
    item.pages !== undefined && item.pages !== null ? `${item.pages} pages` : null,
    item.topics !== undefined && item.topics !== null ? `${item.topics} topics` : null,
    item.coverage && item.coverage.topic_coverage !== undefined && item.coverage.topic_coverage !== null
      ? `coverage ${Math.round(Number(item.coverage.topic_coverage) * 100)}%`
      : null,
    formatDate(item.createdAt ?? item.created_at),
  ];
  return parts.filter(Boolean).join(" · ");
}

interface NotesArtifactListProps {
  projectId: string;
  items: NotesListItem[];
  emptyMessage?: string;
  /** Notified after a successful delete so the page can refresh other reads. */
  onChanged?: () => void;
}

/**
 * The notes LIST: every generated note for the project, not just the newest.
 *
 * Open streams that note's stored PDF into the EXISTING viewer
 * (/documents/study-notes-{projectId}?version=N), so a specific note is
 * reachable instead of only the latest one. Delete asks for confirmation,
 * calls the ownership-scoped DELETE, and refetches the notes + PDF metadata
 * so the list and the header can never disagree.
 */
export function NotesArtifactList({
  projectId,
  items,
  emptyMessage = "No notes generated yet for this project.",
  onChanged,
}: NotesArtifactListProps) {
  const [, navigate] = useLocation();
  const queryClient = useQueryClient();
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);

  const deleteMutation = useMutation({
    mutationFn: (noteId: string) => deleteNote(projectId, noteId),
    onSuccess: () => {
      setPendingDelete(null);
      void queryClient.invalidateQueries({ queryKey: ["notes", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["study-notes-pdf", projectId] });
      onChanged?.();
    },
  });

  if (!items || items.length === 0) {
    return (
      <p className="text-sm" style={{ color: MUTED }}>
        {emptyMessage}
      </p>
    );
  }

  return (
    <ul className="flex flex-col gap-3" data-testid="notes-artifact-list">
      {items.map((item) => {
        const meta = noteMetaLine(item);
        const version = item.version ?? null;
        const target = version
          ? `/documents/study-notes-${projectId}?version=${version}`
          : `/documents/study-notes-${projectId}`;
        const confirming = pendingDelete === item.id;
        const busy = deleteMutation.isPending && deleteMutation.variables === item.id;
        return (
          <li
            key={item.id}
            className="p-4 rounded-2xl flex flex-col sm:flex-row sm:items-center gap-3"
            style={{ backgroundColor: PANEL }}
          >
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: TILE }}
            >
              <span className="material-symbols-outlined" style={{ color: MUTED }}>
                article
              </span>
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold truncate" style={{ color: INK }}>
                {item.title || "Study notes"}
              </p>
              <p
                className="text-xs"
                style={{ color: MUTED, fontFamily: "Space Grotesk, sans-serif" }}
              >
                {meta || "No metadata recorded"}
              </p>
              {confirming && (
                <p className="text-xs mt-2" style={{ color: DANGER }}>
                  Delete this note and its stored PDF? This cannot be undone.
                </p>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => navigate(target)}
                className="px-5 py-2 rounded-xl text-xs font-semibold uppercase tracking-widest transition-all hover:opacity-90 whitespace-nowrap"
                style={{ color: INK, border: "1px solid rgba(72,72,72,0.3)" }}
              >
                Open
              </button>
              {confirming ? (
                <>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => deleteMutation.mutate(item.id)}
                    className="px-5 py-2 rounded-xl text-xs font-semibold uppercase tracking-widest transition-all hover:opacity-90 disabled:opacity-50 whitespace-nowrap"
                    style={{ backgroundColor: DANGER, color: "#131313" }}
                  >
                    {busy ? "Deleting..." : "Confirm delete"}
                  </button>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => setPendingDelete(null)}
                    className="px-5 py-2 rounded-xl text-xs font-semibold uppercase tracking-widest transition-all hover:opacity-90 disabled:opacity-50 whitespace-nowrap"
                    style={{ color: INK, border: "1px solid rgba(72,72,72,0.3)" }}
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={() => setPendingDelete(item.id)}
                  className="px-5 py-2 rounded-xl text-xs font-semibold uppercase tracking-widest transition-all hover:opacity-90 whitespace-nowrap"
                  style={{ color: DANGER, border: "1px solid rgba(72,72,72,0.3)" }}
                >
                  Delete
                </button>
              )}
            </div>
          </li>
        );
      })}
      {deleteMutation.isError && (
        <li className="text-xs" role="alert" style={{ color: DANGER }}>
          {(deleteMutation.error as Error)?.message ||
            "Could not delete this note. Check your connection and try again."}
        </li>
      )}
      {deleteMutation.isPending && (
        <li className="text-xs flex items-center gap-2" style={{ color: MUTED }}>
          <SpinnerIcon size={14} color={MUTED} /> Deleting note...
        </li>
      )}
    </ul>
  );
}

export default NotesArtifactList;
