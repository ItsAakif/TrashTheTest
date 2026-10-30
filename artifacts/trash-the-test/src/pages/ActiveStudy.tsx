import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Redirect, useLocation } from "wouter";
import Layout from "@/components/Layout";
import { SpinnerIcon } from "@/components/icons";
import {
  generateTypedFlashcardsWithFallback,
  generateStudyNotes,
  generateMCQMatrixWithFallback,
  getFlashcards,
  getProjectDocuments,
  listWebsites,
  readAnalysis,
  readMapping,
  readNotes,
  readNotesRecommendation,
  readPyq,
  readQuiz,
  readSyllabus,
  type Document,
} from "@/lib/api-services";
import { useActiveProject } from "@/lib/use-active-project";
import { NotesGenerationConfigDialog } from "@/components/NotesGenerationConfigDialog";
import { NotesArtifactList } from "@/components/NotesArtifactList";
import { DeleteDocumentButton } from "@/components/DeleteDocumentButton";
import type { NotesGenerationConfig } from "@/lib/api-services";

const MUTED = "#acabaa";
const INK = "#e7e5e4";
const PANEL = "#131313";
const TILE = "#1f2020";
const ACCENT = "#edfee9";
const DANGER = "#ee7d77";

function statusLabel(status: Document["processingStatus"]): string {
  if (status === "completed") return "Ready";
  if (status === "processing") return "Indexing";
  if (status === "failed") return "Failed";
  return "Queued";
}

function statusColor(status: Document["processingStatus"]): string {
  if (status === "completed") return ACCENT;
  if (status === "failed") return DANGER;
  return MUTED;
}

function isAuthError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return /401|403|unauthorized|unauthenticated|forbidden/i.test(message);
}

function SectionTitle({ children }: { children: string }) {
  return (
    <h2
      className="text-xl font-bold mb-4"
      style={{ fontFamily: "Inter, sans-serif", color: INK }}
    >
      {children}
    </h2>
  );
}

export default function ActiveStudy() {
  const [, navigate] = useLocation();
  const queryClient = useQueryClient();

  // Single project truth (persists refresh/nav via localStorage pref).
  // The switcher lives ONLY in the navbar: this page never imports it.
  const {
    projects,
    projectsLoading,
    activeProject,
    activeProjectId,
    defaultCreating,
    authRequired,
  } = useActiveProject();

  const projectId = activeProjectId ?? null;
  const [notesConfigOpen, setNotesConfigOpen] = useState(false);
  const [notesConfig, setNotesConfig] = useState<NotesGenerationConfig>({
    mode: "detailed",
    detail: "balanced",
    exam_focus: false,
    memory_aids: false,
    web: false,
    scope: "entire",
  });

  const goLoginOnAuthError = (error: unknown) => {
    if (isAuthError(error)) navigate("/login");
  };

  // Real workspace reads, all scoped to the active project.
  const { data: documents = [], isLoading: docsLoading } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => getProjectDocuments(projectId as string),
    enabled: !!projectId,
  });

  const { data: notesData, isLoading: notesLoading } = useQuery({
    queryKey: ["notes", projectId],
    queryFn: () => readNotes(projectId as string),
    enabled: !!projectId,
  });

  const { data: deck, isLoading: deckLoading } = useQuery({
    queryKey: ["flashcards", projectId],
    queryFn: () => getFlashcards({ projectId: projectId as string }),
    enabled: !!projectId,
  });

  const { data: quizData, isLoading: quizLoading } = useQuery({
    queryKey: ["quiz-content", projectId],
    queryFn: () => readQuiz(projectId as string),
    enabled: !!projectId,
  });

  const { data: syllabusData } = useQuery({
    queryKey: ["academic-syllabus", projectId],
    queryFn: () => readSyllabus(projectId as string),
    enabled: !!projectId,
  });
  const { data: pyqData } = useQuery({
    queryKey: ["academic-pyq", projectId],
    queryFn: () => readPyq(projectId as string),
    enabled: !!projectId,
  });
  const { data: mappingData } = useQuery({
    queryKey: ["academic-mapping", projectId],
    queryFn: () => readMapping(projectId as string),
    enabled: !!projectId,
  });
  const { data: analysisData } = useQuery({
    queryKey: ["academic-analysis", projectId],
    queryFn: () => readAnalysis(projectId as string),
    enabled: !!projectId,
  });
  const { data: notesRecommendation } = useQuery({
    queryKey: ["notes-recommendation", projectId],
    queryFn: () => readNotesRecommendation(projectId as string),
    enabled: !!projectId,
  });

  const { data: websites = [] } = useQuery({
    queryKey: ["websites", projectId],
    queryFn: () => listWebsites(projectId as string),
    enabled: !!projectId,
  });

  const recommendationText = notesRecommendation?.recommendation
    ? `Recommended: ${String(notesRecommendation.recommendation.recommended_mode ?? "").toUpperCase()} / ${String(notesRecommendation.recommendation.recommended_detail ?? "").toLowerCase()}. ${
        notesRecommendation.recommendation.reason ?? "Based on project sources."
      } This suggestion never overrides your selection.`
    : "Recommendation unavailable until project sources are ready. Your selection is always preserved.";

  const refreshGenerated = () => {
    void queryClient.invalidateQueries({ queryKey: ["notes", projectId] });
    void queryClient.invalidateQueries({ queryKey: ["flashcards", projectId] });
    void queryClient.invalidateQueries({ queryKey: ["quiz-content", projectId] });
    void queryClient.invalidateQueries({ queryKey: ["study-notes-pdf", projectId] });
    void queryClient.invalidateQueries({ queryKey: ["academic-insights", projectId] });
  };

  const notesMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project selected.");
      return generateStudyNotes({ projectId, config: notesConfig });
    },
    onSuccess: refreshGenerated,
    onError: goLoginOnAuthError,
  });
  const flashcardsMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project selected.");
      return generateTypedFlashcardsWithFallback({ projectId });
    },
    onSuccess: refreshGenerated,
    onError: goLoginOnAuthError,
  });
  const practiceMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project selected.");
      return generateMCQMatrixWithFallback({ projectId });
    },
    onSuccess: refreshGenerated,
    onError: goLoginOnAuthError,
  });

  // 401 -> login (no valid session; ProtectedRoute also guards the route).
  if (authRequired) {
    return <Redirect to="/login" />;
  }

  // Loading skeleton (project list, or first-run default creation).
  if (projectsLoading || defaultCreating) {
    return (
      <Layout activeNav="study">
        <div
          className="pb-20 md:pb-8 px-6 md:px-12 max-w-7xl mx-auto pt-10"
          aria-busy="true"
          aria-label="Loading your study workspace"
        >
          <div
            className="h-4 w-24 rounded mb-4 animate-pulse"
            style={{ backgroundColor: TILE }}
          />
          <div
            className="h-10 w-2/3 rounded mb-3 animate-pulse"
            style={{ backgroundColor: TILE }}
          />
          <div
            className="h-5 w-1/2 rounded mb-10 animate-pulse"
            style={{ backgroundColor: TILE }}
          />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="p-6 rounded-2xl animate-pulse"
                style={{ backgroundColor: PANEL }}
              >
                <div
                  className="h-5 w-1/2 rounded mb-3"
                  style={{ backgroundColor: TILE }}
                />
                <div
                  className="h-4 w-full rounded mb-2"
                  style={{ backgroundColor: TILE }}
                />
                <div
                  className="h-4 w-3/4 rounded"
                  style={{ backgroundColor: TILE }}
                />
              </div>
            ))}
          </div>
          <p className="mt-8 text-sm" style={{ color: MUTED }}>
            {defaultCreating
              ? "Setting up your personal study workspace..."
              : "Loading your study workspace..."}
          </p>
        </div>
      </Layout>
    );
  }

  // No active project + zero projects (after the create attempt settled):
  // onboarding with a direct upload CTA. No switcher CTA in the body.
  if (!activeProject) {
    return (
      <Layout activeNav="study">
        <div className="pb-20 md:pb-8 px-6 md:px-12 max-w-3xl mx-auto pt-10 text-center">
          <div
            className="px-2 py-1 rounded text-xs uppercase tracking-widest inline-block mb-4"
            style={{
              fontFamily: "Space Grotesk, sans-serif",
              fontSize: "10px",
              color: MUTED,
              backgroundColor: TILE,
            }}
          >
            Study
          </div>
          <h1
            className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4"
            style={{ fontFamily: "Inter, sans-serif", color: INK, lineHeight: "1.1" }}
          >
            Start studying
          </h1>
          <p className="text-lg mb-2" style={{ color: MUTED }}>
            {projects.length === 0
              ? "We could not set up your workspace automatically. Upload your first study material and we will create your project and turn your files into notes, flashcards, and practice questions."
              : "Your study workspace is ready. Upload textbooks, notes, or past papers to begin."}
          </p>
          <p className="text-sm mb-8" style={{ color: MUTED }}>
            Your files stay in your own project — only you can see them.
          </p>
          <button
            onClick={() => navigate("/upload")}
            className="px-8 py-4 rounded-xl font-semibold transition-all hover:opacity-90"
            style={{ backgroundColor: ACCENT, color: "#131313" }}
          >
            Upload your study material
          </button>
        </div>
      </Layout>
    );
  }

  // Active project workspace (post-selection content only).
  const completedDocs = documents.filter((d) => d.processingStatus === "completed");
  // Generation gate (live bug: a document with indexing_state=partial and
  // 196 real chunks still showed "Upload a document first"). A document is
  // usable when it has any indexed state or real pages; only a project with
  // no documents or only failed ones is truly blocked.
  const generationDocs = documents.filter((d) => {
    const state = (d.indexingState ?? "").toLowerCase();
    const indexed = state === "completed" || state === "partial" || state === "indexed";
    return (indexed || (d.pageCount ?? 0) > 0) && d.processingStatus !== "failed";
  });
  const canGenerate = generationDocs.length > 0;
  const hasPartialDocs = generationDocs.some(
    (d) => (d.indexingState ?? "").toLowerCase() === "partial",
  );
  const partialIndexNote = "Partially indexed - generation may skip unreadable pages";
  const pendingDocs = documents.filter(
    (d) => d.processingStatus === "processing" || d.processingStatus === "pending",
  );
  const failedDocs = documents.filter((d) => d.processingStatus === "failed");

  const notesCount = notesData?.state === "ready" ? (notesData.total ?? notesData.notes.length) : 0;
  const notesItems = notesData?.state === "ready" ? notesData.notes : [];
  const flashCount = deck?.cardCount ?? 0;
  const practiceCount =
    quizData?.state === "ready" ? (quizData.total ?? quizData.quizzes.length) : 0;

  const academicModes = [
    { label: "Syllabus", read: syllabusData },
    { label: "Past questions", read: pyqData },
    { label: "Topic mapping", read: mappingData },
    { label: "Trend analysis", read: analysisData },
  ];
  const academicReady = academicModes.filter((m) => m.read?.state === "ready");

  const publishedSites = websites.filter((w) => w.is_published);
  const draftSites = websites.filter((w) => !w.is_published);

  const generating =
    notesMutation.isPending || flashcardsMutation.isPending || practiceMutation.isPending;

  return (
    <Layout activeNav="study">
      <div className="pb-20 md:pb-8 px-6 md:px-12 max-w-7xl mx-auto pt-10">
        {/* Workspace header */}
        <header className="mb-10">
          <div
            className="px-2 py-1 rounded text-xs uppercase tracking-widest inline-block mb-4"
            style={{
              fontFamily: "Space Grotesk, sans-serif",
              fontSize: "10px",
              color: MUTED,
              backgroundColor: TILE,
            }}
          >
            Study
          </div>
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
            <div className="max-w-2xl">
              <h1
                className="text-4xl md:text-5xl font-extrabold tracking-tight mb-3"
                style={{ fontFamily: "Inter, sans-serif", color: INK, lineHeight: "1.1" }}
              >
                {activeProject.name}
              </h1>
              <p className="text-lg" style={{ color: MUTED }}>
                {activeProject.description || "Your personal study workspace."}
              </p>
            </div>
            <span
              className="px-3 py-1.5 rounded text-xs uppercase tracking-widest self-start md:self-auto"
              style={{
                fontFamily: "Space Grotesk, sans-serif",
                color: activeProject.status === "ready" ? ACCENT : MUTED,
                backgroundColor: TILE,
              }}
            >
              {activeProject.status === "ready"
                ? "Ready"
                : activeProject.status === "failed"
                  ? "Failed"
                  : "Processing"}
            </span>
          </div>
        </header>

        {/* Documents */}
        <section className="mb-10">
          <SectionTitle>Your documents</SectionTitle>
          {docsLoading && (
            <p className="text-sm" style={{ color: MUTED }}>
              <SpinnerIcon size={16} color={MUTED} /> Checking your documents...
            </p>
          )}
          {!docsLoading && documents.length === 0 && (
            <div
              className="p-6 rounded-2xl flex flex-col sm:flex-row sm:items-center gap-4"
              style={{ backgroundColor: PANEL }}
            >
              <p className="text-sm flex-1" style={{ color: MUTED }}>
                No documents in this project yet. Upload your textbooks, notes, or past papers to start.
              </p>
              <button
                onClick={() => navigate("/upload")}
                className="px-6 py-3 rounded-xl text-sm font-semibold transition-all hover:opacity-90 whitespace-nowrap"
                style={{ backgroundColor: ACCENT, color: "#131313" }}
              >
                Upload your study material
              </button>
            </div>
          )}
          {!docsLoading && documents.length > 0 && (
            <div className="flex flex-col gap-3">
              {pendingDocs.length > 0 && (
                <p className="text-sm" style={{ color: MUTED }}>
                  Indexing {pendingDocs.length} of {documents.length} document
                  {documents.length === 1 ? "" : "s"} — your materials appear below as soon as each file finishes.
                </p>
              )}
              {failedDocs.length > 0 && (
                <p className="text-sm" style={{ color: DANGER }}>
                  {failedDocs.length} document{failedDocs.length === 1 ? "" : "s"} failed to process. Re-upload{" "}
                  {failedDocs.length === 1 ? "it" : "them"} from the Upload page to try again.
                </p>
              )}
              <ul className="flex flex-col gap-3">
                {documents.map((doc) => (
                  <li
                    key={doc.id}
                    className="p-4 rounded-2xl flex flex-col sm:flex-row sm:items-center gap-3"
                    style={{ backgroundColor: PANEL }}
                  >
                    <div
                      className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                      style={{ backgroundColor: TILE }}
                    >
                      <span className="material-symbols-outlined" style={{ color: MUTED }}>
                        description
                      </span>
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold truncate" style={{ color: INK }}>
                        {doc.fileName}
                      </p>
                      <p
                        className="text-xs"
                        style={{
                          color: statusColor(doc.processingStatus),
                          fontFamily: "Space Grotesk, sans-serif",
                        }}
                      >
                        {statusLabel(doc.processingStatus)}
                        {doc.processingError ? ` — ${doc.processingError}` : ""}
                        {doc.pageCountConfirmed !== undefined && doc.pageCountConfirmed !== null
                          ? ` · ${doc.pageCountConfirmed} pages`
                          : ""}
                      </p>
                    </div>
                    {(doc.fileType ?? "").toLowerCase() === "pdf" && (
                      <button
                        onClick={() => navigate(`/documents/${doc.id}`)}
                        className="px-5 py-2 rounded-xl text-xs font-semibold uppercase tracking-widest transition-all hover:opacity-90 whitespace-nowrap"
                        style={{
                          fontFamily: "Space Grotesk, sans-serif",
                          color: INK,
                          border: "1px solid rgba(72,72,72,0.3)",
                        }}
                      >
                        Open
                      </button>
                    )}
                    {/* Per-document removal: confirm -> DELETE (server checks
                        ownership) -> ["documents"] refetch removes the row. */}
                    <DeleteDocumentButton documentId={doc.id} fileName={doc.fileName} compact />
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>

        {/* Generated resources */}
        <section className="mb-10">
          <SectionTitle>Your study materials</SectionTitle>
          {generating && (
            <p className="text-sm mb-4" style={{ color: MUTED }}>
              <SpinnerIcon size={16} color={MUTED} /> Generating your study materials...
            </p>
          )}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 rounded-2xl" style={{ backgroundColor: PANEL }}>
              <span className="material-symbols-outlined mb-3" style={{ color: ACCENT, display: "block" }}>
                article
              </span>
              <h3 className="text-base font-bold mb-1" style={{ color: INK }}>
                Study notes
              </h3>
              <p className="text-sm mb-4" style={{ color: MUTED }}>
                {notesLoading
                  ? "Checking your notes..."
                  : notesCount > 0
                    ? `${notesCount} note${notesCount === 1 ? "" : "s"} ready`
                    : canGenerate
                      ? hasPartialDocs
                        ? partialIndexNote
                        : "Not generated yet"
                      : "Upload a document first"}
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setNotesConfigOpen(true)}
                  disabled={!canGenerate}
                  className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                  style={{ color: INK, border: "1px solid rgba(72,72,72,0.3)" }}
                >
                  Settings
                </button>
                <button
                  onClick={() => notesMutation.mutate()}
                  disabled={notesMutation.isPending || !canGenerate}
                  className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                  style={{ backgroundColor: ACCENT, color: "#131313" }}
                >
                  {notesMutation.isPending ? "Generating..." : "Generate"}
                </button>
                {notesCount > 0 && (
                  <button
                    onClick={() => navigate(`/study/${projectId}`)}
                    className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:opacity-90"
                    style={{ color: INK, border: "1px solid rgba(72,72,72,0.3)" }}
                  >
                    Read
                  </button>
                )}
              </div>
              {notesMutation.isError && (
                <p className="text-xs mt-3" style={{ color: DANGER }}>
                  {(notesMutation.error as Error)?.message || "Notes generation failed."}
                </p>
              )}
              {/* Every generated note, not just the newest: Open streams that
                  note's PDF into the existing viewer, Delete is ownership
                  checked server-side and refetches the list. */}
              {notesCount > 0 && (
                <div className="mt-4 pt-4" style={{ borderTop: "1px solid rgba(72,72,72,0.15)" }}>
                  <h4
                    className="text-xs font-bold uppercase tracking-widest mb-3"
                    style={{ color: INK }}
                  >
                    Your notes ({notesCount})
                  </h4>
                  <NotesArtifactList
                    projectId={projectId as string}
                    items={notesItems}
                    onChanged={refreshGenerated}
                  />
                </div>
              )}
            </div>

            <div className="p-6 rounded-2xl" style={{ backgroundColor: PANEL }}>
              <span className="material-symbols-outlined mb-3" style={{ color: ACCENT, display: "block" }}>
                style
              </span>
              <h3 className="text-base font-bold mb-1" style={{ color: INK }}>
                Flashcards
              </h3>
              <p className="text-sm mb-4" style={{ color: MUTED }}>
                {deckLoading
                  ? "Checking your flashcards..."
                  : flashCount > 0
                    ? `${flashCount} card${flashCount === 1 ? "" : "s"} ready`
                    : canGenerate
                      ? hasPartialDocs
                        ? partialIndexNote
                        : "Not generated yet"
                      : "Upload a document first"}
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => flashcardsMutation.mutate()}
                  disabled={flashcardsMutation.isPending || !canGenerate}
                  className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                  style={{ backgroundColor: ACCENT, color: "#131313" }}
                >
                  {flashcardsMutation.isPending ? "Generating..." : "Generate"}
                </button>
                {flashCount > 0 && (
                  <button
                    onClick={() => navigate("/flashcards")}
                    className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:opacity-90"
                    style={{ color: INK, border: "1px solid rgba(72,72,72,0.3)" }}
                  >
                    Review
                  </button>
                )}
              </div>
              {flashcardsMutation.isError && (
                <p className="text-xs mt-3" style={{ color: DANGER }}>
                  {(flashcardsMutation.error as Error)?.message || "Flashcard generation failed."}
                </p>
              )}
            </div>

            <div className="p-6 rounded-2xl" style={{ backgroundColor: PANEL }}>
              <span className="material-symbols-outlined mb-3" style={{ color: ACCENT, display: "block" }}>
                quiz
              </span>
              <h3 className="text-base font-bold mb-1" style={{ color: INK }}>
                Practice questions
              </h3>
              <p className="text-sm mb-4" style={{ color: MUTED }}>
                {quizLoading
                  ? "Checking your practice set..."
                  : practiceCount > 0
                    ? `${practiceCount} practice set${practiceCount === 1 ? "" : "s"} ready`
                    : canGenerate
                      ? hasPartialDocs
                        ? partialIndexNote
                        : "Not generated yet"
                      : "Upload a document first"}
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => practiceMutation.mutate()}
                  disabled={practiceMutation.isPending || !canGenerate}
                  className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                  style={{ backgroundColor: ACCENT, color: "#131313" }}
                >
                  {practiceMutation.isPending ? "Generating..." : "Generate"}
                </button>
                {practiceCount > 0 && (
                  <button
                    onClick={() => navigate(`/study/${projectId}`)}
                    className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:opacity-90"
                    style={{ color: INK, border: "1px solid rgba(72,72,72,0.3)" }}
                  >
                    Practise
                  </button>
                )}
              </div>
              {practiceMutation.isError && (
                <p className="text-xs mt-3" style={{ color: DANGER }}>
                  {(practiceMutation.error as Error)?.message || "Practice generation failed."}
                </p>
              )}
            </div>

            <div className="p-6 rounded-2xl" style={{ backgroundColor: PANEL }}>
              <span className="material-symbols-outlined mb-3" style={{ color: ACCENT, display: "block" }}>
                psychology
              </span>
              <h3 className="text-base font-bold mb-1" style={{ color: INK }}>
                AI Tutor
              </h3>
              <p className="text-sm mb-4" style={{ color: MUTED }}>
                {canGenerate
                  ? hasPartialDocs
                    ? partialIndexNote
                    : "Ask questions grounded in your documents"
                  : "Upload a document first"}
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => navigate("/tutor")}
                  disabled={!canGenerate}
                  className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                  style={{ color: INK, border: "1px solid rgba(72,72,72,0.3)" }}
                >
                  Open tutor
                </button>
              </div>
            </div>

            <div className="p-6 rounded-2xl" style={{ backgroundColor: PANEL }}>
              <span className="material-symbols-outlined mb-3" style={{ color: ACCENT, display: "block" }}>
                school
              </span>
              <h3 className="text-base font-bold mb-1" style={{ color: INK }}>
                Academic insights
              </h3>
              <p className="text-sm mb-4" style={{ color: MUTED }}>
                {academicReady.length === 0
                  ? "No insights generated yet"
                  : academicModes
                    .map((m) => `${m.label}: ${m.read?.state === "ready" ? `${m.read.total} saved` : "pending"}`)
                    .join(" · ")}
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => navigate("/academic")}
                  className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:opacity-90"
                  style={{ color: INK, border: "1px solid rgba(72,72,72,0.3)" }}
                >
                  {academicReady.length === 0 ? "Explore insights" : "View insights"}
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* Website / publish state */}
        <section className="mb-4">
          <SectionTitle>Study website</SectionTitle>
          <div className="p-6 rounded-2xl" style={{ backgroundColor: PANEL }}>
            {websites.length === 0 ? (
              <p className="text-sm mb-4" style={{ color: MUTED }}>
                No study website published yet. Turn your notes and flashcards into a shareable revision page when
                you are ready.
              </p>
            ) : (
              <ul className="mb-4 space-y-1">
                {publishedSites.length > 0 && (
                  <li className="text-sm" style={{ color: MUTED }}>
                    Published: {publishedSites.map((w) => w.title || w.slug).join(", ")}
                  </li>
                )}
                {draftSites.length > 0 && (
                  <li className="text-sm" style={{ color: MUTED }}>
                    Drafts: {draftSites.length} unpublished
                    {draftSites.length === 1 ? "" : "d"}
                  </li>
                )}
              </ul>
            )}
            <button
              onClick={() => navigate("/website")}
              className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:opacity-90"
              style={{ color: INK, border: "1px solid rgba(72,72,72,0.3)" }}
            >
              {websites.length === 0 ? "Create your study website" : "Manage website"}
            </button>
          </div>
        </section>
      </div>
       <NotesGenerationConfigDialog
         open={notesConfigOpen}
         onOpenChange={setNotesConfigOpen}
         value={notesConfig}
         onChange={setNotesConfig}
         recommendation={recommendationText}
       />
     </Layout>
  );
}
