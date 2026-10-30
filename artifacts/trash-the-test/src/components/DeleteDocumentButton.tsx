import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteDocument } from "@/lib/api-services";

const INK = "#e7e5e4";
const DANGER = "#ee7d77";

interface DeleteDocumentButtonProps {
  documentId: string;
  /** Shown in the confirmation copy so the row never reads as a generic action. */
  fileName?: string | null;
  /** Notified after the server confirmed the delete (job/file/pages gone). */
  onDeleted?: (documentId: string) => void;
  /** Compact sizing for dense rows (smaller padding, same two-step flow). */
  compact?: boolean;
  className?: string;
}

/**
 * Per-document delete (uploaded material removal).
 *
 * Flow: two-step confirm -> DELETE /api/documents/{id} -> refetch the cached
 * document reads -> optional `onDeleted` hook (the Upload page uses it to
 * stop its progress poller and cancel a job still running for that file).
 *
 * Ownership is enforced SERVER-SIDE by `require_document_access`: another
 * user's document answers 404 and never changes state here, so a stale list
 * row cannot delete anything the caller does not own. The endpoint also
 * cancels any active job and removes pages/chunks/stored bytes for us.
 */
export function DeleteDocumentButton({
  documentId,
  fileName,
  onDeleted,
  compact = false,
  className,
}: DeleteDocumentButtonProps) {
  const [confirming, setConfirming] = useState(false);
  const queryClient = useQueryClient();

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteDocument(id),
    onSuccess: (_result, id) => {
      setConfirming(false);
      // Every document list in the app reads ["documents", projectId]:
      // prefix invalidation refetches the Upload, Dashboard and Study lists
      // together so the row disappears everywhere at once.
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      // The open viewer (and any other single-document reader) must not keep
      // showing a document that no longer exists.
      void queryClient.invalidateQueries({ queryKey: ["document", id] });
      // Project-switcher document counts.
      void queryClient.invalidateQueries({ queryKey: ["project-doc-counts"] });
      onDeleted?.(id);
    },
  });

  const padding = compact ? "px-4 py-1.5" : "px-5 py-2";
  const busy = deleteMutation.isPending;

  return (
    <span
      className={className}
      style={{ display: "inline-flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "center" }}
      data-testid={`delete-document-${documentId}`}
    >
      {confirming && (
        <span className="text-xs" style={{ flexBasis: "100%", color: DANGER }}>
          Delete {fileName ? `"${fileName}"` : "this file"} from this project, together with
          its indexed pages and stored file? This cannot be undone.
        </span>
      )}
      {confirming ? (
        <>
          <button
            type="button"
            disabled={busy}
            onClick={() => deleteMutation.mutate(documentId)}
            className={`${padding} rounded-xl text-xs font-semibold uppercase tracking-widest transition-all hover:opacity-90 disabled:opacity-50 whitespace-nowrap`}
            style={{ backgroundColor: DANGER, color: "#131313" }}
            aria-label={fileName ? `Confirm delete ${fileName}` : "Confirm delete document"}
          >
            {busy ? "Deleting..." : "Confirm delete"}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              deleteMutation.reset();
              setConfirming(false);
            }}
            className={`${padding} rounded-xl text-xs font-semibold uppercase tracking-widest transition-all hover:opacity-90 disabled:opacity-50 whitespace-nowrap`}
            style={{ color: INK, border: "1px solid rgba(72,72,72,0.3)" }}
          >
            Cancel
          </button>
        </>
      ) : (
        <button
          type="button"
          onClick={() => {
            deleteMutation.reset();
            setConfirming(true);
          }}
          className={`${padding} rounded-xl text-xs font-semibold uppercase tracking-widest transition-all hover:opacity-90 whitespace-nowrap`}
          style={{ color: DANGER, border: "1px solid rgba(238,125,119,0.3)" }}
          aria-label={fileName ? `Delete ${fileName}` : "Delete document"}
          title="Delete document"
        >
          Delete
        </button>
      )}
      {deleteMutation.isError && (
        <span role="alert" className="text-xs" style={{ flexBasis: "100%", color: DANGER }}>
          {(deleteMutation.error as Error)?.message ||
            "Could not delete this document. Check your connection and try again."}
        </span>
      )}
    </span>
  );
}

export default DeleteDocumentButton;
