import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createProject,
  defaultProjectNameFor,
  getCurrentUser,
  getProjects,
  type Project,
} from "@/lib/api-services";

const LAST_PROJECT_KEY = "trash_the_test_last_project_id";

export function readLastProjectId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(LAST_PROJECT_KEY);
  } catch {
    return null;
  }
}

export function writeLastProjectId(projectId: string | null) {
  if (typeof window === "undefined") return;
  try {
    if (projectId) window.localStorage.setItem(LAST_PROJECT_KEY, projectId);
    else window.localStorage.removeItem(LAST_PROJECT_KEY);
  } catch {
    // UI pref only; never let storage failure break the page.
  }
}

export interface ActiveProjectSelection {
  projects: Project[];
  projectsLoading: boolean;
  projectsError: boolean;
  activeProject: Project | null;
  activeProjectId: string | null;
  selectProject: (projectId: string) => void;
  /** True while the first-run default project is being created. */
  defaultCreating: boolean;
  /** True when the default-project create attempt failed (show onboarding). */
  defaultFailed: boolean;
  /** True when there is no valid session (caller redirects to /login). */
  authRequired: boolean;
}

/**
 * Single project-context source for every page (replaces the fragile
 * `projects[0]` pattern). The last-selected project id persists as a UI
 * pref; an explicit dropdown selection wins, then the stored id, then
 * the first project. Matches the Website page selector pattern.
 *
 * First-run default project: when the authenticated user owns zero
 * projects, exactly one "<firstName>'s project" is created via the real
 * POST /api/projects (name from GET /api/auth/me, fallback
 * "My First Project"). Idempotency: the list is re-checked right before
 * the POST, a module-level in-flight guard means concurrent hook
 * instances (navbar + page) share one attempt, and a 409/race simply
 * refetches the list instead of creating a duplicate.
 */
let defaultCreateInFlight: Promise<void> | null = null;

export function useActiveProject(explicitId?: string | null): ActiveProjectSelection {
  const queryClient = useQueryClient();
  const { data: projects = [], isLoading, isError } = useQuery({
    queryKey: ["projects"],
    queryFn: getProjects,
    staleTime: 30_000,
  });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [defaultCreating, setDefaultCreating] = useState(false);
  const [defaultFailed, setDefaultFailed] = useState(false);
  const [authRequired, setAuthRequired] = useState(false);
  const ensureAttemptedRef = useRef(false);

  // Stale stored id (e.g. the active project was deleted in another tab
  // or via the switcher): drop it so every page falls back to the next
  // surviving project or the empty onboarding state.
  useEffect(() => {
    if (isLoading || projects.length === 0) return;
    const stored = readLastProjectId();
    if (stored && !projects.some((p) => p.id === stored)) {
      writeLastProjectId(null);
      setSelectedId((prev) => (prev === stored ? null : prev));
    }
  }, [isLoading, projects]);

  // Default project auto-creation (once per session, authenticated only).
  useEffect(() => {
    if (isLoading || isError) return;
    if (projects.length > 0) return;
    if (ensureAttemptedRef.current) return;
    ensureAttemptedRef.current = true;

    const attempt = async (): Promise<void> => {
      // A sibling hook instance (navbar + page mount together) may
      // already own the attempt: await it, then re-read the list.
      if (defaultCreateInFlight) {
        await defaultCreateInFlight.catch(() => undefined);
        await queryClient.invalidateQueries({ queryKey: ["projects"] });
        return;
      }
      const task = (async (): Promise<void> => {
        // Authenticated users only: no session -> login, never create.
        const me = await getCurrentUser().catch(() => null);
        if (!me) {
          setAuthRequired(true);
          return;
        }
        // Re-check the list first: never duplicate on refresh or race.
        const latest = await getProjects().catch(() => null);
        if (latest && latest.length > 0) return;
        try {
          const created = await createProject({
            name: defaultProjectNameFor(me),
          });
          writeLastProjectId(created.id);
          setSelectedId(created.id);
        } catch (err) {
          const status = (err as { status?: number })?.status;
          if (status === 401 || status === 403) {
            setAuthRequired(true);
            return;
          }
          // 409/race or any other failure: the refetched list owns the
          // truth. If it is still empty the page shows onboarding.
          setDefaultFailed(true);
        } finally {
          await queryClient.invalidateQueries({ queryKey: ["projects"] });
        }
      })();
      defaultCreateInFlight = task;
      try {
        await task;
      } finally {
        defaultCreateInFlight = null;
      }
    };

    setDefaultCreating(true);
    void attempt().finally(() => setDefaultCreating(false));
  }, [isLoading, isError, projects, queryClient]);

  const storedId = readLastProjectId();
  const activeProject =
    (explicitId ? projects.find((p) => p.id === explicitId) : undefined) ??
    (selectedId ? projects.find((p) => p.id === selectedId) : undefined) ??
    (storedId ? projects.find((p) => p.id === storedId) : undefined) ??
    (projects.length > 0 ? projects[0] : null) ??
    null;

  const selectProject = (projectId: string) => {
    setSelectedId(projectId);
    writeLastProjectId(projectId);
  };

  return {
    projects,
    projectsLoading: isLoading,
    projectsError: isError,
    activeProject,
    activeProjectId: activeProject?.id ?? null,
    selectProject,
    defaultCreating,
    defaultFailed,
    authRequired,
  };
}

/** Small project dropdown used by pages with an explicit selector. */
export function ProjectSelector({
  projects,
  activeProjectId,
  onSelect,
  id,
}: {
  projects: Project[];
  activeProjectId: string | null;
  onSelect: (projectId: string) => void;
  id?: string;
}) {
  if (projects.length === 0) return null;
  return (
    <label className="flex items-center gap-2 text-sm" style={{ color: "#acabaa" }}>
      <span
        className="uppercase tracking-widest"
        style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px" }}
      >
        Project
      </span>
      <select
        id={id}
        value={activeProjectId ?? ""}
        onChange={(e) => onSelect(e.target.value)}
        className="px-3 py-2 rounded-xl text-sm focus:outline-none"
        style={{
          backgroundColor: "#1f2020",
          color: "#e7e5e4",
          border: "1px solid rgba(72,72,72,0.2)",
          fontFamily: "Inter, sans-serif",
        }}
      >
        {projects.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>
    </label>
  );
}
