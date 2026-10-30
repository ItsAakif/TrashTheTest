import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Layout from "@/components/Layout";
import MagicCard from "@/components/MagicCard";
import { SpinnerIcon } from "@/components/icons";
import { useLocation } from "wouter";
import {
  getProjectDocuments,
  generateStudyNotes,
  readHierarchicalNotes,
  generateFlashcards,
  generateQuiz,
  generateExplanation,
  createStudyWebsite,
  listWebsites,
  previewWebsiteHtml,
  publishWebsite,
  unpublishWebsite,
  setWebsiteVisibility,
  deleteWebsite,
  publicWebsiteUrl,
  DEFAULT_GENERATION_TIMEOUT_MS,
  DEFAULT_WEBSITE_TIMEOUT_MS,
  type Project,
  type Document,
  type WebsiteContentType,
} from "@/lib/api-services";
import { useActiveProject } from "@/lib/use-active-project";

// ADR-008 (T014) named step budgets. Every LLM-bound step gets its own
// deadline so one stalled call can be named in the error instead of
// hanging the whole build. Explanation is best-effort and shortest.
const NOTES_STEP_TIMEOUT_MS = DEFAULT_GENERATION_TIMEOUT_MS;
const FLASHCARDS_STEP_TIMEOUT_MS = DEFAULT_GENERATION_TIMEOUT_MS;
const QUIZ_STEP_TIMEOUT_MS = DEFAULT_GENERATION_TIMEOUT_MS;
const EXPLANATION_STEP_TIMEOUT_MS = 60_000;
const WEBSITE_STEP_TIMEOUT_MS = DEFAULT_WEBSITE_TIMEOUT_MS;

export default function Website() {
  const [, navigate] = useLocation();
  const queryClient = useQueryClient();
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [contentType, setContentType] = useState<WebsiteContentType>("mixed");
  const [websiteTitle, setWebsiteTitle] = useState("");
  const [websiteDescription, setWebsiteDescription] = useState("");
  const [visibility, setVisibility] = useState<"private" | "public">("private");
  const [includeNotes, setIncludeNotes] = useState(true);
  const [includeFlashcards, setIncludeFlashcards] = useState(true);
  const [includeQuiz, setIncludeQuiz] = useState(true);
  const [generatedSlug, setGeneratedSlug] = useState<string | null>(null);
  const [generatedHtml, setGeneratedHtml] = useState<string | null>(null);
  const [shareCopied, setShareCopied] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  // Shared project context (last-selected project, never a blind [0]).
  // The dropdown below keeps the Website selector pattern and persists
  // the choice for the other pages.
  const {
    projects,
    projectsLoading,
    activeProject,
    activeProjectId,
    selectProject,
  } = useActiveProject(selectedProject);

  const projectId = activeProjectId || null;

  // Fetch documents for the active project
  const { data: documents = [] } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => getProjectDocuments(projectId!),
    enabled: !!projectId,
  });

  // Existing websites for this project (manage list with badges).
  const { data: websites = [], isLoading: websitesLoading } = useQuery({
    queryKey: ["websites", projectId],
    queryFn: () => listWebsites(projectId!),
    enabled: !!projectId,
  });

  // Check if we have content to publish
  const hasDocuments = documents.length > 0;
  const hasProcessedDocuments = documents.some(
    (d: Document) => d.processingStatus === "completed",
  );

  const refreshWebsites = () => queryClient.invalidateQueries({ queryKey: ["websites", projectId] });

  // Website generation mutation: assembles included resources from the
  // generation APIs, then POST /api/website/generate with
  // content_type/title/structured_content (the correct endpoint).
  //
  // ADR-008 (T014): the four LLM-bound steps run concurrently, each with
  // its own deadline. A stalled step is reported by name instead of hanging
  // the whole build, and only notes/flashcards/quiz are allowed to abort it.
  const generateMutation = useMutation({
    mutationFn: async () => {
      if (!projectId) throw new Error("No project selected");
      const structured: Record<string, unknown> = { sections: [] };
      const title = websiteTitle || activeProject?.name || "Study Website";

      // Notes path: hierarchical job -> read the artifact it produced.
      const notesPromise = includeNotes
        ? (async () => {
            const generated = await generateStudyNotes({
              projectId,
              timeoutMs: NOTES_STEP_TIMEOUT_MS,
            });
            const artifact = await readHierarchicalNotes(projectId).catch(
              () => null,
            );
            const legacyContent = (generated as { content?: unknown })?.content;
            const artifactSections: unknown =
              artifact && Array.isArray(artifact.sections)
                ? artifact.sections
                : null;
            const legacySections: unknown =
              legacyContent &&
              typeof legacyContent === "object" &&
              Array.isArray((legacyContent as { sections?: unknown }).sections)
                ? (legacyContent as { sections: unknown }).sections
                : null;
            return {
              artifact,
              legacyContent,
              sections: artifactSections ?? legacySections,
            };
          })()
        : Promise.resolve(null);

      const flashcardsPromise = includeFlashcards
        ? generateFlashcards({ projectId, timeoutMs: FLASHCARDS_STEP_TIMEOUT_MS })
        : Promise.resolve(null);

      const quizPromise = includeQuiz
        ? generateQuiz({ projectId, timeoutMs: QUIZ_STEP_TIMEOUT_MS })
        : Promise.resolve(null);

      // Explanations are query-bound and best-effort: 60s, never fatal.
      const explanationPromise = generateExplanation({
        projectId,
        query: websiteTitle || activeProject?.name || "overview",
        timeoutMs: EXPLANATION_STEP_TIMEOUT_MS,
      }).catch(() => null);

      const [notesRes, flashcardsRes, quizRes, explanationRes] =
        await Promise.allSettled([
          notesPromise,
          flashcardsPromise,
          quizPromise,
          explanationPromise,
        ]);

      const stepErrors: Partial<
        Record<"notes" | "flashcards" | "quiz" | "explanation", string>
      > = {};
      const reasonOf = (r: PromiseRejectedResult): string =>
        r.reason instanceof Error ? r.reason.message : "unknown";
      if (notesRes.status === "rejected") stepErrors.notes = reasonOf(notesRes);
      if (flashcardsRes.status === "rejected")
        stepErrors.flashcards = reasonOf(flashcardsRes);
      if (quizRes.status === "rejected") stepErrors.quiz = reasonOf(quizRes);
      if (explanationRes.status === "rejected")
        stepErrors.explanation = reasonOf(explanationRes);

      const required = [
        ...(includeNotes ? (["notes"] as const) : []),
        ...(includeFlashcards ? (["flashcards"] as const) : []),
        ...(includeQuiz ? (["quiz"] as const) : []),
      ].filter((key) => {
        const res =
          key === "notes"
            ? notesRes
            : key === "flashcards"
              ? flashcardsRes
              : quizRes;
        return res.status !== "fulfilled";
      });
      if (required.length) {
        throw new Error(
          `Cannot build website: ${required
            .map((key) => `${key} step failed: ${stepErrors[key] ?? "unknown"}`)
            .join("; ")}`,
        );
      }

      if (notesRes.status === "fulfilled" && notesRes.value) {
        const notes = notesRes.value;
        structured.notes = notes.artifact ?? notes.legacyContent;
        if (notes.sections) structured.sections = notes.sections;
      }
      if (flashcardsRes.status === "fulfilled" && flashcardsRes.value) {
        structured.flashcards = flashcardsRes.value.content;
      }
      if (quizRes.status === "fulfilled" && quizRes.value) {
        structured.quiz = quizRes.value.content;
        const questions = (quizRes.value.content as { questions?: unknown })
          ?.questions;
        if (Array.isArray(questions)) structured.questions = questions;
      }
      if (explanationRes.status === "fulfilled" && explanationRes.value) {
        structured.explanation = explanationRes.value.content;
      }

      return createStudyWebsite({
        projectId,
        contentType,
        title,
        description: websiteDescription || undefined,
        structuredContent: structured,
        timeoutMs: WEBSITE_STEP_TIMEOUT_MS,
      });
    },
    onSuccess: async (data) => {
      setGeneratedSlug(data.slug);
      setGeneratedHtml(data.html_content);
      setShareCopied(false);
      setActionError(null);
      // New sites start as draft/private; apply visibility choice explicitly.
      if (visibility === "public") {
        try {
          await publishWebsite(data.slug);
        } catch (err) {
          setActionError(err instanceof Error ? err.message : "Auto-publish failed.");
        }
      }
      refreshWebsites();
    },
    onError: (error: Error) => {
      const detail = error.message || "Website generation failed.";
      setActionError(detail);
    },
  });

  const publishMutation = useMutation({
    mutationFn: (slug: string) => publishWebsite(slug),
    onSuccess: () => { setActionError(null); refreshWebsites(); },
    onError: (error: Error) => setActionError(error.message),
  });

  const unpublishMutation = useMutation({
    mutationFn: (slug: string) => unpublishWebsite(slug),
    onSuccess: () => { setActionError(null); refreshWebsites(); },
    onError: (error: Error) => setActionError(error.message),
  });

  const visibilityMutation = useMutation({
    mutationFn: ({ slug, vis }: { slug: string; vis: "public" | "private" }) =>
      setWebsiteVisibility(slug, vis),
    onSuccess: () => { setActionError(null); refreshWebsites(); },
    onError: (error: Error) => setActionError(error.message),
  });

  const deleteMutation = useMutation({
    mutationFn: (slug: string) => deleteWebsite(slug),
    onSuccess: (_ok, slug) => {
      setActionError(null);
      if (generatedSlug === slug) {
        setGeneratedSlug(null);
        setGeneratedHtml(null);
      }
      refreshWebsites();
    },
    onError: (error: Error) => setActionError(error.message),
  });

  const handleGenerate = () => {
    generateMutation.mutate();
  };

  // Owner preview (drafts AND published): loads the persisted sanitised
  // HTML into the preview pane. Drafts never link the public route (it
  // 404s for unpublished slugs); published rows get an "Open" link too.
  const [previewLoading, setPreviewLoading] = useState<string | null>(null);

  const handlePreview = async (slug: string) => {
    setPreviewLoading(slug);
    setActionError(null);
    try {
      const html = await previewWebsiteHtml(slug);
      setGeneratedSlug(slug);
      setGeneratedHtml(html);
      setShareCopied(false);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Preview failed.");
    } finally {
      setPreviewLoading(null);
    }
  };

  const copyShareLink = async (slug: string) => {
    const url = `${window.location.origin}/website/${slug}`;
    try {
      await navigator.clipboard.writeText(url);
      setShareCopied(true);
    } catch {
      setActionError("Could not copy to clipboard.");
    }
  };

  const publishedSite = generatedSlug ? websites.find((w) => w.slug === generatedSlug) : undefined;
  const isPublished = publishedSite?.is_published ?? publishedSite?.status === "published";

  return (
    <Layout activeNav="publish">
      <div className="pb-20 md:pb-8 px-6 md:px-12 max-w-7xl mx-auto pt-10">
        {/* Header */}
        <header className="mb-12">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
            <div className="max-w-2xl">
              <div
                className="px-2 py-1 rounded text-xs uppercase tracking-widest inline-block mb-4"
                style={{
                  fontFamily: "Space Grotesk, sans-serif",
                  fontSize: "10px",
                  color: "#acabaa",
                  backgroundColor: "#1f2020",
                }}
              >
                Publish
              </div>
              <h1
                className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4"
                style={{
                  fontFamily: "Inter, sans-serif",
                  color: "#e7e5e4",
                  lineHeight: "1.1",
                }}
              >
                Publish Study Website
              </h1>
              <p className="text-lg mb-6" style={{ color: "#acabaa" }}>
                Publish your study materials as a beautiful, shareable website.
                Turn your notes, flashcards, and quizzes into a hosted page.
              </p>
            </div>
          </div>
        </header>

        {/* Empty State - No Projects */}
        {projects.length === 0 && !projectsLoading && (
          <MagicCard className="p-8 mb-8">
            <div className="text-center py-12">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6"
                style={{ backgroundColor: "#1f2020" }}
              >
                <span
                  className="material-symbols-outlined"
                  style={{ fontSize: "32px", color: "#acabaa" }}
                >
                  public
                </span>
              </div>
              <h3
                className="text-xl font-bold mb-3"
                style={{ color: "#e7e5e4" }}
              >
                Start by Uploading Documents
              </h3>
              <p className="mb-6 max-w-md mx-auto" style={{ color: "#acabaa" }}>
                To create a website, you first need to upload and process study
                materials. Begin your journey by adding documents.
              </p>
              <button
                onClick={() => navigate("/upload")}
                className="px-6 py-3 rounded-xl font-semibold transition-all hover:opacity-90"
                style={{ backgroundColor: "#edfee9", color: "#131313" }}
              >
                Upload Documents
              </button>
            </div>
          </MagicCard>
        )}

        {/* Empty State - Documents Processing */}
        {projects.length > 0 && hasDocuments && !hasProcessedDocuments && (
          <MagicCard className="p-8 mb-8">
            <div className="text-center py-12">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6"
                style={{ backgroundColor: "#1f2020" }}
              >
                <span
                  className="material-symbols-outlined"
                  style={{ fontSize: "32px", color: "#acabaa" }}
                >
                  pending
                </span>
              </div>
              <h3
                className="text-xl font-bold mb-3"
                style={{ color: "#e7e5e4" }}
              >
                Documents Still Processing
              </h3>
              <p className="mb-6 max-w-md mx-auto" style={{ color: "#acabaa" }}>
                Your documents are being processed. Once complete, you can
                generate study materials and publish them as a website.
              </p>
              <button
                onClick={() => navigate("/upload")}
                className="px-6 py-3 rounded-xl font-semibold transition-all hover:opacity-90"
                style={{
                  backgroundColor: "#1f2020",
                  color: "#e7e5e4",
                  border: "1px solid rgba(72,72,72,0.3)",
                }}
              >
                Check Upload Status
              </button>
            </div>
          </MagicCard>
        )}

        {/* Website Generator Form */}
        {projects.length > 0 && hasProcessedDocuments && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Configuration Panel */}
            <div className="lg:col-span-1">
              <MagicCard className="p-6">
                <h3
                  className="text-lg font-bold mb-6"
                  style={{ color: "#e7e5e4" }}
                >
                  Website Settings
                </h3>

                {/* Project Selection */}
                <div className="mb-6">
                  <label
                    className="block text-xs uppercase tracking-widest mb-3"
                    style={{
                      fontFamily: "Space Grotesk, sans-serif",
                      color: "#acabaa",
                    }}
                  >
                    Select Project
                  </label>
                  <select
                    value={projectId || ""}
                    onChange={(e) => {
                      const id = e.target.value;
                      setSelectedProject(id || null);
                      if (id) selectProject(id);
                    }}
                    className="w-full px-4 py-3 rounded-xl text-sm"
                    style={{
                      backgroundColor: "#1f2020",
                      color: "#e7e5e4",
                      border: "1px solid rgba(72,72,72,0.3)",
                    }}
                  >
                    {projects.map((project: Project) => (
                      <option key={project.id} value={project.id}>
                        {project.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Content Type */}
                <div className="mb-6">
                  <label
                    className="block text-xs uppercase tracking-widest mb-3"
                    style={{
                      fontFamily: "Space Grotesk, sans-serif",
                      color: "#acabaa",
                    }}
                  >
                    Content Type
                  </label>
                  <div className="space-y-2">
                    {[
                      {
                        value: "mixed",
                        label: "Mixed Content",
                        desc: "Notes, flashcards & quizzes",
                      },
                      {
                        value: "notes",
                        label: "Notes Only",
                        desc: "Study notes and summaries",
                      },
                      {
                        value: "flashcards",
                        label: "Flashcards",
                        desc: "Flashcard deck format",
                      },
                      {
                        value: "quiz",
                        label: "Quiz",
                        desc: "Interactive quiz format",
                      },
                    ].map((option) => (
                      <button
                        key={option.value}
                        onClick={() =>
                          setContentType(option.value as typeof contentType)
                        }
                        className="w-full p-3 rounded-xl text-left transition-all"
                        style={{
                          backgroundColor:
                            contentType === option.value
                              ? "#2a3e46"
                              : "#1f2020",
                          border: `1px solid ${contentType === option.value ? "#454747" : "rgba(72,72,72,0.3)"}`,
                        }}
                      >
                        <div
                          className="text-sm font-medium"
                          style={{ color: "#e7e5e4" }}
                        >
                          {option.label}
                        </div>
                        <div className="text-xs" style={{ color: "#acabaa" }}>
                          {option.desc}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Title Input */}
                <div className="mb-6">
                  <label
                    className="block text-xs uppercase tracking-widest mb-3"
                    style={{
                      fontFamily: "Space Grotesk, sans-serif",
                      color: "#acabaa",
                    }}
                  >
                    Website Title
                  </label>
                  <input
                    type="text"
                    placeholder={activeProject?.name || "My Study Guide"}
                    value={websiteTitle}
                    onChange={(e) => setWebsiteTitle(e.target.value)}
                    className="w-full px-4 py-3 rounded-xl text-sm"
                    style={{
                      backgroundColor: "#1f2020",
                      color: "#e7e5e4",
                      border: "1px solid rgba(72,72,72,0.3)",
                    }}
                  />
                </div>

                {/* Description Input */}
                <div className="mb-6">
                  <label
                    className="block text-xs uppercase tracking-widest mb-3"
                    style={{
                      fontFamily: "Space Grotesk, sans-serif",
                      color: "#acabaa",
                    }}
                  >
                    Description (Optional)
                  </label>
                  <textarea
                    placeholder="What is this study website about?"
                    value={websiteDescription}
                    onChange={(e) => setWebsiteDescription(e.target.value)}
                    rows={2}
                    className="w-full px-4 py-3 rounded-xl text-sm resize-none"
                    style={{
                      backgroundColor: "#1f2020",
                      color: "#e7e5e4",
                      border: "1px solid rgba(72,72,72,0.3)",
                    }}
                  />
                </div>

                {/* Included resources */}
                <div className="mb-6">
                  <label
                    className="block text-xs uppercase tracking-widest mb-3"
                    style={{
                      fontFamily: "Space Grotesk, sans-serif",
                      color: "#acabaa",
                    }}
                  >
                    Included Resources
                  </label>
                  {[
                    { id: "notes", label: "Notes", checked: includeNotes, set: setIncludeNotes },
                    { id: "flashcards", label: "Flashcards", checked: includeFlashcards, set: setIncludeFlashcards },
                    { id: "quiz", label: "Quiz / MCQs", checked: includeQuiz, set: setIncludeQuiz },
                  ].map((r) => (
                    <label key={r.id} className="flex items-center gap-3 cursor-pointer mb-2">
                      <input
                        type="checkbox"
                        checked={r.checked}
                        onChange={(e) => r.set(e.target.checked)}
                        className="w-4 h-4"
                      />
                      <span className="text-sm" style={{ color: "#e7e5e4" }}>{r.label}</span>
                    </label>
                  ))}
                  <p className="text-xs mt-1" style={{ color: "#484848" }}>
                    Explanations are bundled automatically (best-effort).
                  </p>
                </div>

                {/* Visibility */}
                <div className="mb-6">
                  <label
                    className="block text-xs uppercase tracking-widest mb-3"
                    style={{
                      fontFamily: "Space Grotesk, sans-serif",
                      color: "#acabaa",
                    }}
                  >
                    Visibility
                  </label>
                  <div className="flex gap-2">
                    {(["private", "public"] as const).map((v) => (
                      <button
                        key={v}
                        onClick={() => setVisibility(v)}
                        className="flex-1 py-2 rounded-xl text-sm font-semibold transition-all"
                        style={{
                          backgroundColor: visibility === v ? "#2a3e46" : "#1f2020",
                          color: "#e7e5e4",
                          border: `1px solid ${visibility === v ? "#454747" : "rgba(72,72,72,0.3)"}`,
                        }}
                      >
                        {v === "private" ? "Private draft" : "Public"}
                      </button>
                    ))}
                  </div>
                  <p className="text-xs mt-1" style={{ color: "#484848" }}>
                    Private drafts are never publicly resolvable — only published sites get a share link.
                  </p>
                </div>

                {/* Generate Button */}
                <button
                  onClick={handleGenerate}
                  disabled={generateMutation.isPending}
                  className="w-full py-3 rounded-xl font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                  style={{ backgroundColor: "#edfee9", color: "#131313" }}
                >
                  {generateMutation.isPending ? (
                    <span className="flex items-center justify-center gap-2">
                      <SpinnerIcon size={18} color="#131313" />
                      Generating...
                    </span>
                  ) : (
                    "Generate Website"
                  )}
                </button>
                {actionError && (
                  <p className="mt-3 text-sm" style={{ color: "#ee7d77" }}>{actionError}</p>
                )}
              </MagicCard>
            </div>

            {/* Preview Panel */}
            <div className="lg:col-span-2">
              <MagicCard className="p-6 h-full">
                <div className="flex items-center justify-between mb-6">
                  <h3
                    className="text-lg font-bold"
                    style={{ color: "#e7e5e4" }}
                  >
                    Preview
                  </h3>
                  {publishedSite && (
                    <span
                      className="px-2 py-0.5 rounded text-xs uppercase tracking-widest"
                      style={{
                        fontFamily: "Space Grotesk, sans-serif",
                        backgroundColor: isPublished ? "rgba(237,254,233,0.1)" : "rgba(140,161,170,0.1)",
                        color: isPublished ? "#edfee9" : "#8ca1aa",
                      }}
                    >
                      {isPublished ? "published · public" : "draft · private"}
                    </span>
                  )}
                </div>

                {!generatedHtml && !generateMutation.isPending && (
                  <div
                    className="flex flex-col items-center justify-center h-96 rounded-xl"
                    style={{
                      backgroundColor: "#1f2020",
                      border: "1px dashed rgba(72,72,72,0.5)",
                    }}
                  >
                    <span
                      className="material-symbols-outlined mb-4"
                      style={{ fontSize: "48px", color: "#484848" }}
                    >
                      preview
                    </span>
                    <p style={{ color: "#acabaa" }}>
                      Click "Generate Website" to see a preview
                    </p>
                  </div>
                )}

                {generateMutation.isPending && (
                  <div
                    className="flex flex-col items-center justify-center h-96 rounded-xl"
                    style={{ backgroundColor: "#1f2020" }}
                  >
                    <div className="mb-4">
                      <SpinnerIcon size={32} color="#acabaa" />
                    </div>
                    <p style={{ color: "#acabaa" }}>
                      Generating your website...
                    </p>
                  </div>
                )}

                {generatedHtml && generatedSlug && (
                  <div className="space-y-4">
                    <div
                      className="rounded-xl overflow-hidden"
                      style={{ border: "1px solid rgba(72,72,72,0.3)" }}
                    >
                      <iframe
                        srcDoc={generatedHtml}
                        className="w-full h-96"
                        title="Website Preview"
                        sandbox=""
                      />
                    </div>
                    <p className="text-xs" style={{ color: "#484848" }}>
                      Preview renders the persisted sanitised page (slug: {generatedSlug}) —
                      identical to what visitors see after publish.
                    </p>
                    <div className="flex gap-3 flex-wrap">
                      {!isPublished ? (
                        <button
                          onClick={() => generatedSlug && publishMutation.mutate(generatedSlug)}
                          disabled={publishMutation.isPending}
                          className="flex-1 py-3 rounded-xl font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                          style={{ backgroundColor: "#edfee9", color: "#131313" }}
                        >
                          {publishMutation.isPending ? "Publishing..." : "Publish Website"}
                        </button>
                      ) : (
                        <button
                          onClick={() => generatedSlug && unpublishMutation.mutate(generatedSlug)}
                          disabled={unpublishMutation.isPending}
                          className="flex-1 py-3 rounded-xl font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                          style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.3)" }}
                        >
                          {unpublishMutation.isPending ? "Unpublishing..." : "Unpublish (back to draft)"}
                        </button>
                      )}
                      <button
                        onClick={() => {
                          setGeneratedHtml(null);
                          setGeneratedSlug(null);
                        }}
                        className="px-4 py-3 rounded-xl font-semibold transition-all hover:opacity-90"
                        style={{
                          backgroundColor: "#1f2020",
                          color: "#e7e5e4",
                          border: "1px solid rgba(72,72,72,0.3)",
                        }}
                      >
                        Regenerate
                      </button>
                    </div>
                    {isPublished && (
                      <div
                        className="p-4 rounded-xl flex flex-col sm:flex-row sm:items-center gap-3"
                        style={{ backgroundColor: "#1f2020", border: "1px solid rgba(72,72,72,0.3)" }}
                      >
                        <p className="text-sm flex-1 break-all" style={{ color: "#acabaa" }}>
                          {`${window.location.origin}/website/${generatedSlug}`}
                        </p>
                        <div className="flex gap-2">
                          <button
                            onClick={() => copyShareLink(generatedSlug)}
                            className="px-4 py-2 rounded-xl text-sm font-semibold transition-all hover:opacity-90"
                            style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
                          >
                            {shareCopied ? "Copied!" : "Copy share link"}
                          </button>
                          <a
                            href={publicWebsiteUrl(generatedSlug)}
                            target="_blank"
                            rel="noreferrer"
                            className="px-4 py-2 rounded-xl text-sm font-semibold transition-all hover:opacity-90"
                            style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
                          >
                            Open
                          </a>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </MagicCard>

              {/* Manage list */}
              <MagicCard className="p-6 mt-6">
                <h3 className="text-lg font-bold mb-4" style={{ color: "#e7e5e4" }}>
                  Your Websites
                </h3>
                {websitesLoading && (
                  <p className="text-sm" style={{ color: "#acabaa" }}>Loading websites...</p>
                )}
                {!websitesLoading && websites.length === 0 && (
                  <p className="text-sm" style={{ color: "#acabaa" }}>No websites yet for this project.</p>
                )}
                <div className="flex flex-col gap-3">
                  {websites.map((w) => {
                    const pub = w.is_published || w.status === "published";
                    return (
                      <div
                        key={w.slug}
                        className="p-4 rounded-xl flex flex-col sm:flex-row sm:items-center gap-3"
                        style={{ backgroundColor: "#1f2020", border: "1px solid rgba(72,72,72,0.2)" }}
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-bold truncate" style={{ color: "#e7e5e4" }}>{w.title}</p>
                          <p className="text-xs" style={{ color: "#484848" }}>
                            {w.slug} · {w.content_type}
                          </p>
                        </div>
                        <span
                          className="px-2 py-0.5 rounded text-xs uppercase tracking-widest self-start"
                          style={{
                            fontFamily: "Space Grotesk, sans-serif",
                            backgroundColor: pub ? "rgba(237,254,233,0.1)" : "rgba(140,161,170,0.1)",
                            color: pub ? "#edfee9" : "#8ca1aa",
                          }}
                        >
                          {pub ? "published" : "draft"}
                        </span>
                        <div className="flex gap-2 flex-wrap">
                          <button
                            onClick={() => void handlePreview(w.slug)}
                            disabled={previewLoading === w.slug}
                            className="px-3 py-1.5 rounded-lg text-xs disabled:opacity-50"
                            style={{ color: "#acabaa", border: "1px solid rgba(72,72,72,0.2)" }}
                          >
                            {previewLoading === w.slug ? "Loading..." : "Preview"}
                          </button>
                          {pub && (
                            <a
                              href={publicWebsiteUrl(w.slug)}
                              target="_blank"
                              rel="noreferrer"
                              className="px-3 py-1.5 rounded-lg text-xs"
                              style={{ color: "#acabaa", border: "1px solid rgba(72,72,72,0.2)" }}
                            >
                              Open
                            </a>
                          )}
                          <button
                            onClick={() => visibilityMutation.mutate({ slug: w.slug, vis: pub ? "private" : "public" })}
                            disabled={visibilityMutation.isPending}
                            className="px-3 py-1.5 rounded-lg text-xs disabled:opacity-50"
                            style={{ color: "#acabaa", border: "1px solid rgba(72,72,72,0.2)" }}
                          >
                            Make {pub ? "private" : "public"}
                          </button>
                          <button
                            onClick={() => deleteMutation.mutate(w.slug)}
                            disabled={deleteMutation.isPending}
                            className="px-3 py-1.5 rounded-lg text-xs disabled:opacity-50"
                            style={{ color: "#ee7d77", border: "1px solid rgba(238,125,119,0.3)" }}
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </MagicCard>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
