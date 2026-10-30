import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { createProject, type Project } from "@/lib/api-services";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface CreateProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called with the created project after the list refetch is triggered. */
  onCreated?: (project: Project) => void;
}

/**
 * New-project form (name + description) over the real POST /api/projects.
 * No hardcoded IDs, no demo fallback: success invalidates ["projects"] so
 * every consumer refetches the authoritative list.
 */
export default function CreateProjectDialog({
  open,
  onOpenChange,
  onCreated,
}: CreateProjectDialogProps) {
  const [, navigate] = useLocation();
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const mutation = useMutation({
    mutationFn: () => createProject({ name: name.trim(), description: description.trim() || undefined }),
    onSuccess: (project) => {
      // Newest-first ordering makes the created project the active one;
      // refetch (not cache patching) so the UI reflects server state.
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
      setName("");
      setDescription("");
      onOpenChange(false);
      onCreated?.(project);
    },
  });

  const errorMessage =
    mutation.error instanceof Error ? mutation.error.message : null;
  const isUnauthorized =
    !!errorMessage && /401|unauthorized|unauthenticated|token/i.test(errorMessage);
  const trimmed = name.trim();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!trimmed || mutation.isPending) return;
    if (isUnauthorized) {
      navigate("/login");
      return;
    }
    mutation.mutate();
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!mutation.isPending) {
          if (!next) mutation.reset();
          onOpenChange(next);
        }
      }}
    >
      <DialogContent
        className="sm:max-w-md"
        style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.2)" }}
      >
        <DialogHeader>
          <DialogTitle style={{ color: "#e7e5e4" }}>New Project</DialogTitle>
          <DialogDescription style={{ color: "#acabaa" }}>
            Name your project, then upload documents into it.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <label className="flex flex-col gap-2">
            <span
              className="uppercase tracking-widest"
              style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
            >
              Project name (required)
            </span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Data Structures"
              maxLength={255}
              disabled={mutation.isPending}
              autoFocus
              className="w-full px-4 py-3 rounded-xl text-sm focus:outline-none"
              style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.2)" }}
            />
          </label>
          <label className="flex flex-col gap-2">
            <span
              className="uppercase tracking-widest"
              style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
            >
              Description (optional)
            </span>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What is this project about?"
              maxLength={1000}
              rows={3}
              disabled={mutation.isPending}
              className="w-full px-4 py-3 rounded-xl text-sm focus:outline-none resize-none"
              style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.2)" }}
            />
          </label>
          {errorMessage && (
            <div
              className="p-3 rounded-xl"
              style={{ backgroundColor: "rgba(255,80,80,0.08)", border: "1px solid rgba(255,80,80,0.25)" }}
            >
              <p className="text-sm" style={{ color: "#ff9d9d" }}>
                {isUnauthorized ? "Session expired. Please sign in again." : errorMessage}
              </p>
              {isUnauthorized && (
                <button
                  type="button"
                  onClick={() => navigate("/login")}
                  className="mt-2 text-xs underline"
                  style={{ color: "#ff9d9d" }}
                >
                  Go to login
                </button>
              )}
            </div>
          )}
          <DialogFooter>
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              disabled={mutation.isPending}
              className="px-6 py-3 rounded-xl text-sm font-semibold transition-all hover:opacity-90 disabled:opacity-50"
              style={{
                backgroundColor: "#252626",
                color: "#e7e5e4",
                border: "1px solid rgba(72,72,72,0.2)",
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!trimmed || mutation.isPending}
              className="px-6 py-3 rounded-xl text-sm font-semibold transition-all hover:opacity-90 disabled:opacity-50"
              style={{ backgroundColor: "#edfee9", color: "#131313" }}
            >
              {mutation.isPending ? "Creating..." : "Create Project"}
            </button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
