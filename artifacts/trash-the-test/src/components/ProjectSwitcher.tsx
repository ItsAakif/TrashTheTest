import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import CreateProjectDialog from "@/components/CreateProjectDialog";
import {
  deleteProject,
  getProjectDocuments,
  type Project,
} from "@/lib/api-services";
import { useActiveProject } from "@/lib/use-active-project";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

/** Window event pages fire to open the switcher (single creation path). */
export const OPEN_SWITCHER_EVENT = "ttt:open-project-switcher";

export function requestOpenSwitcher(create = false) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent<boolean>(OPEN_SWITCHER_EVENT, { detail: create }),
  );
}

const STARRED_KEY = "ttt_starred_projects";

function readStarred(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STARRED_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed)
      ? parsed.filter((v): v is string => typeof v === "string")
      : [];
  } catch {
    return [];
  }
}

function writeStarred(ids: string[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STARRED_KEY, JSON.stringify(ids));
  } catch {
    // Star pref only; never break the page.
  }
}

type SwitcherTab = "recent" | "starred" | "all";

interface ProjectSwitcherProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Increment to auto-open the nested New-project dialog (event flow). */
  createSignal?: number;
}

function isUnauthorizedMessage(message: string | null): boolean {
  return (
    !!message && /401|unauthorized|unauthenticated|token|forbidden|403/i.test(message)
  );
}

/**
 * GCP-style project switcher. Single creation path (nested
 * CreateProjectDialog over the real POST /api/projects) and real
 * project statuses: live document counts per project plus the server
 * project status/progress. Delete calls the real
 * DELETE /api/projects/{id} (server enforces JWT ownership; foreign
 * ids 404 and stay hidden). 401 surfaces a sign-in nudge.
 */
export default function ProjectSwitcher({
  open,
  onOpenChange,
  createSignal = 0,
}: ProjectSwitcherProps) {
  const [, navigate] = useLocation();
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<SwitcherTab>("recent");
  const [search, setSearch] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [starred, setStarred] = useState<string[]>(() => readStarred());

  const {
    projects,
    projectsLoading,
    projectsError,
    activeProjectId,
    selectProject,
  } = useActiveProject();

  // Event-driven auto-open of the nested create dialog (pages fire
  // requestOpenSwitcher(true); Layout owns this dialog, so there is
  // exactly one creation path).
  const [lastSignal, setLastSignal] = useState(createSignal);
  useEffect(() => {
    if (createSignal !== lastSignal) {
      setLastSignal(createSignal);
      if (open) setCreateOpen(true);
    }
  }, [createSignal, lastSignal, open]);

  // Reset transient UI state whenever the modal closes.
  useEffect(() => {
    if (!open) {
      setSearch("");
      setTab("recent");
      setConfirmDeleteId(null);
      setDeleteError(null);
      setCreateOpen(false);
    }
  }, [open ]);

  // Real per-project statuses: live document counts + processing states
  // from GET /api/documents/project/{id}. Fetched only while open so
  // background pages never pay for N project queries.
  const projectIds = useMemo(() => projects.map((p) => p.id), [projects]);
  const { data: docCounts } = useQuery({
    queryKey: ["project-doc-counts", projectIds],
    queryFn: async (): Promise<
      Record<string, { total: number; completed: number; processing: number; failed: number }>
    > => {
      const entries = await Promise.all(
        projectIds.map(async (id) => {
          try {
            const docs = await getProjectDocuments(id);
            // "Partial" is a usable index, not an error and not still
            // processing: count it as completed so a project with a
            // partially readable document is not shown as unfinished.
            const isIndexed = (d: (typeof docs)[number]) => {
              const state = (d.indexingState ?? "").toLowerCase();
              return (
                state === "completed" ||
                state === "partial" ||
                state === "indexed" ||
                d.processingStatus === "completed"
              );
            };
            return [
              id,
              {
                total: docs.length,
                completed: docs.filter(isIndexed).length,
                processing: docs.filter(
                  (d) =>
                    !isIndexed(d) &&
                    (d.processingStatus === "processing" ||
                      d.processingStatus === "pending"),
                ).length,
                failed: docs.filter(
                  (d) =>
                    !isIndexed(d) &&
                    (d.processingStatus === "failed" ||
                      (d.indexingState ?? "").toLowerCase() === "failed"),
                ).length,
              },
            ] as const;
          } catch {
            // A single project's docs failing must not hide the row;
            // the project-level status still renders.
            return [
              id,
              { total: 0, completed: 0, processing: 0, failed: 0 },
            ] as const;
          }
        }),
      );
      return Object.fromEntries(entries);
    },
    enabled: open && projectIds.length > 0,
    staleTime: 30_000,
  });

  const deleteMutation = useMutation({
    mutationFn: (projectId: string) => deleteProject(projectId),
    onSuccess: (_ok, deletedId) => {
      setConfirmDeleteId(null);
      setDeleteError(null);
      setStarred((prev) => {
        const next = prev.filter((id) => id !== deletedId);
        writeStarred(next);
        return next;
      });
      // Active-deleted: pick the next surviving project (recency order)
      // or clear the stored id so pages fall back to onboarding.
      const survivors = projects.filter((p) => p.id !== deletedId);
      if (deletedId === activeProjectId) {
        if (survivors.length > 0) {
          selectProject(survivors[0].id);
        } else {
          try {
            window.localStorage.removeItem("trash_the_test_last_project_id");
          } catch {
            // UI pref only.
          }
        }
      }
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
      void queryClient.invalidateQueries({ queryKey: ["project-doc-counts"] });
    },
    onError: (error: Error) => {
      const message = error.message || "Delete failed. Please try again.";
      setDeleteError(message);
      if (isUnauthorizedMessage(message)) navigate("/login");
    },
  });

  const toggleStar = (projectId: string) => {
    setStarred((prev) => {
      const next = prev.includes(projectId)
        ? prev.filter((id) => id !== projectId)
        : [...prev, projectId];
      writeStarred(next);
      return next;
    });
  };

  const statusLabel = (project: Project): string => {
    const counts = docCounts?.[project.id];
    if (counts && counts.processing > 0)
      return `Indexing ${counts.processing}/${counts.total}`;
    if (counts && counts.total > 0 && counts.completed === counts.total)
      return `Indexed · ${counts.total} doc${counts.total === 1 ? "" : "s"}`;
    if (counts && counts.total > 0)
      return `${counts.completed}/${counts.total} ready`;
    if (project.status === "ready")
      return counts ? `Ready · ${counts.total} docs` : "Ready";
    if (project.status === "failed") return "Failed";
    return counts ? `Processing · ${counts.total} docs` : "Processing";
  };

  const ordered = useMemo(() => {
    // Recency: last-selected first, then updatedAt desc.
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem("trash_the_test_last_project_id");
    } catch {
      stored = null;
    }
    return [...projects].sort((a, b) => {
      if (stored) {
        if (a.id === stored && b.id !== stored) return -1;
        if (b.id === stored && a.id !== stored) return 1;
      }
      return (b.updatedAt || "").localeCompare(a.updatedAt || "");
    });
  }, [projects]);

  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    let list = ordered;
    if (tab === "starred") list = list.filter((p) => starred.includes(p.id));
    if (q)
      list = list.filter(
        (p) =>
          p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q),
      );
    return list;
  }, [ordered, tab, starred, search]);

  const handleSelect = (projectId: string) => {
    selectProject(projectId);
    onOpenChange(false);
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent
          className="sm:max-w-2xl"
          style={{
            backgroundColor: "#131313",
            border: "1px solid rgba(72,72,72,0.2)",
          }}
        >
          <DialogHeader>
            <div className="flex items-center justify-between gap-4">
              <div>
                <DialogTitle style={{ color: "#e7e5e4" }}>
                  Select a project
                </DialogTitle>
                <DialogDescription style={{ color: "#acabaa" }}>
                  Switch the active study workspace. Ownership is enforced —
                  only your projects appear.
                </DialogDescription>
              </div>
              <button
                onClick={() => setCreateOpen(true)}
                className="px-4 py-2 rounded-xl text-sm font-semibold transition-all hover:opacity-90 whitespace-nowrap"
                style={{ backgroundColor: "#edfee9", color: "#131313" }}
              >
                + New project
              </button>
            </div>
          </DialogHeader>

          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search projects"
            aria-label="Search projects"
            className="w-full px-4 py-3 rounded-xl text-sm focus:outline-none"
            style={{
              backgroundColor: "#1f2020",
              color: "#e7e5e4",
              border: "1px solid rgba(72,72,72,0.2)",
            }}
          />

          <div className="flex items-center gap-2" role="tablist" aria-label="Project lists">
            {(
              [
                { id: "recent", label: "Recent" },
                { id: "starred", label: "Starred" },
                { id: "all", label: "All" },
              ] as Array<{ id: SwitcherTab; label: string }>
            ).map((t) => (
              <button
                key={t.id}
                role="tab"
                aria-selected={tab === t.id}
                onClick={() => {
                  setTab(t.id);
                  setConfirmDeleteId(null);
                }}
                className="px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-widest transition-all"
                style={{
                  fontFamily: "Space Grotesk, sans-serif",
                  backgroundColor: tab === t.id ? "#2a3e46" : "transparent",
                  color: tab === t.id ? "#e7e5e4" : "#acabaa",
                }}
              >
                {t.label}
              </button>
            ))}
          </div>

          {projectsLoading && (
            <p className="text-sm text-center py-8" style={{ color: "#acabaa" }}>
              Loading projects...
            </p>
          )}

          {!projectsLoading && projectsError && (
            <div
              className="p-4 rounded-xl text-center"
              style={{
                backgroundColor: "rgba(255,80,80,0.08)",
                border: "1px solid rgba(255,80,80,0.25)",
              }}
            >
              <p className="text-sm" style={{ color: "#ff9d9d" }}>
                Could not load projects. Check your connection and try again.
              </p>
              <button
                onClick={() => navigate("/login")}
                className="mt-2 text-xs underline"
                style={{ color: "#ff9d9d" }}
              >
                Go to login
              </button>
            </div>
          )}

          {!projectsLoading && !projectsError && visible.length === 0 && (
            <div className="text-center py-8">
              <p className="text-sm mb-4" style={{ color: "#acabaa" }}>
                {projects.length === 0
                  ? "No projects yet. Create one to start studying."
                  : tab === "starred"
                    ? "No starred projects. Star a project to pin it here."
                    : "No projects match your search."}
              </p>
              {projects.length === 0 && (
                <button
                  onClick={() => setCreateOpen(true)}
                  className="px-6 py-3 rounded-xl text-sm font-semibold transition-all hover:opacity-90"
                  style={{ backgroundColor: "#edfee9", color: "#131313" }}
                >
                  Create project
                </button>
              )}
            </div>
          )}

          {!projectsLoading && !projectsError && visible.length > 0 && (
            <div
              className="rounded-xl overflow-hidden"
              style={{ border: "1px solid rgba(72,72,72,0.2)" }}
            >
              <div
                className="hidden sm:grid grid-cols-[1fr_90px_130px_24px_24px] gap-3 px-4 py-2 uppercase tracking-widest"
                style={{
                  fontFamily: "Space Grotesk, sans-serif",
                  fontSize: "10px",
                  color: "#484848",
                  backgroundColor: "#1a1a1a",
                }}
              >
                <span>Name</span>
                <span>Type</span>
                <span>ID</span>
                <span aria-hidden="true" />
                <span aria-hidden="true" />
              </div>
              <ul className="max-h-80 overflow-y-auto">
                {visible.map((project) => {
                  const isActive = project.id === activeProjectId;
                  const isStarring = starred.includes(project.id);
                  const confirming = confirmDeleteId === project.id;
                  return (
                    <li
                      key={project.id}
                      className="px-4 py-3 transition-colors"
                      style={{
                        backgroundColor: isActive ? "#1f2020" : "transparent",
                        borderTop: "1px solid rgba(72,72,72,0.12)",
                      }}
                    >
                      <div className="grid grid-cols-[1fr_24px_24px] sm:grid-cols-[1fr_90px_130px_24px_24px] gap-3 items-center">
                        <button
                          onClick={() => handleSelect(project.id)}
                          className="text-left min-w-0"
                          title={`Switch to ${project.name}`}
                        >
                          <span
                            className="block text-sm font-semibold truncate"
                            style={{ color: "#e7e5e4" }}
                          >
                            {project.name}
                            {isActive && (
                              <span
                                className="ml-2 px-2 py-0.5 rounded text-xs uppercase tracking-widest"
                                style={{
                                  fontFamily: "Space Grotesk, sans-serif",
                                  fontSize: "9px",
                                  backgroundColor: "rgba(237,254,233,0.12)",
                                  color: "#edfee9",
                                }}
                              >
                                Active
                              </span>
                            )}
                          </span>
                          <span
                            className="block text-xs truncate"
                            style={{ color: "#acabaa" }}
                          >
                            {statusLabel(project)}
                          </span>
                          <span
                            className="sm:hidden block text-xs truncate"
                            style={{
                              color: "#484848",
                              fontFamily: "Space Grotesk, sans-serif",
                            }}
                          >
                            {project.id.slice(0, 8)}...
                          </span>
                        </button>
                        <span
                          className="hidden sm:block text-xs"
                          style={{ color: "#acabaa" }}
                        >
                          Study
                        </span>
                        <span
                          className="hidden sm:block text-xs truncate"
                          title={project.id}
                          style={{
                            color: "#484848",
                            fontFamily: "Space Grotesk, sans-serif",
                          }}
                        >
                          {project.id.slice(0, 8)}...
                        </span>
                        <button
                          onClick={() => toggleStar(project.id)}
                          aria-label={isStarring ? `Unstar ${project.name}` : `Star ${project.name}`}
                          title={isStarring ? "Unstar" : "Star"}
                          className="transition-all hover:opacity-80"
                        >
                          <span
                            className="material-symbols-outlined"
                            style={{
                              fontSize: "18px",
                              color: isStarring ? "#edfee9" : "#484848",
                              fontVariationSettings: isStarring ? "'FILL' 1" : "'FILL' 0",
                            }}
                          >
                            star
                          </span>
                        </button>
                        <button
                          onClick={() =>
                            setConfirmDeleteId(confirming ? null : project.id)
                          }
                          aria-label={`Delete ${project.name}`}
                          title="Delete project"
                          className="transition-all hover:opacity-80"
                        >
                          <span
                            className="material-symbols-outlined"
                            style={{ fontSize: "18px", color: "#ee7d77" }}
                          >
                            delete
                          </span>
                        </button>
                      </div>
                      {confirming && (
                        <div
                          className="mt-3 p-3 rounded-xl flex flex-col sm:flex-row sm:items-center gap-3"
                          style={{
                            backgroundColor: "rgba(255,80,80,0.06)",
                            border: "1px solid rgba(255,80,80,0.25)",
                          }}
                        >
                          <p className="text-sm flex-1" style={{ color: "#ff9d9d" }}>
                            Delete &ldquo;{project.name}&rdquo; and all its
                            documents? This cannot be undone.
                          </p>
                          <div className="flex gap-2">
                            <button
                              onClick={() => setConfirmDeleteId(null)}
                              disabled={deleteMutation.isPending}
                              className="px-4 py-2 rounded-lg text-xs font-semibold disabled:opacity-50"
                              style={{
                                backgroundColor: "#252626",
                                color: "#e7e5e4",
                              }}
                            >
                              Cancel
                            </button>
                            <button
                              onClick={() => deleteMutation.mutate(project.id)}
                              disabled={deleteMutation.isPending}
                              className="px-4 py-2 rounded-lg text-xs font-semibold disabled:opacity-50"
                              style={{
                                backgroundColor: "#ee7d77",
                                color: "#131313",
                              }}
                            >
                              {deleteMutation.isPending
                                ? "Deleting..."
                                : "Confirm delete"}
                            </button>
                          </div>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          {deleteError && (
            <p className="text-sm" style={{ color: "#ff9d9d" }}>
              {deleteError}
            </p>
          )}
        </DialogContent>
      </Dialog>

      <CreateProjectDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onCreated={(project) => {
          selectProject(project.id);
          setCreateOpen(false);
        }}
      />
    </>
  );
}
