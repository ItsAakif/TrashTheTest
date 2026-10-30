import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import Layout from "@/components/Layout";
import MagicCard from "@/components/MagicCard";
import Citations from "@/components/Citations";
import { SparklesIcon, SpinnerIcon } from "@/components/icons";
import { NoteBody, McqTab, AnswerCard } from "@/components/study-content";
import { NotesArtifactList, noteMetaLine } from "@/components/NotesArtifactList";
import { NotesGenerationConfigDialog } from "@/components/NotesGenerationConfigDialog";
import {
  getProject,
  getProjectDocuments,
  generateStudyNotes,
  generatePractice,
  generateExplanation,
  generateAnswer,
  readNotes,
  readQuiz,
  readStudyNotesPdfMeta,
  readNotesRecommendation,
  readHierarchicalNotesRun,
  JobRunError,
  type AnswerResult,
  type ExplanationResult,
  type Job,
  type NotesGenerationConfig,
} from "@/lib/api-services";

interface StudyModeProps {
  params?: { id?: string };
}

// Phase B: exactly one Practice surface (strict MCQs) and no second
// flashcards surface (the standalone /flashcards page owns flashcards):
// the Study tab keeps Notes / Practice / Answers / Explanations only, and
// the "What You'll Get" cards no longer advertise Flashcards either.
// Removed duplicate shells (extra quiz/flashcards tabs, unimplemented map,
// quick-quiz shell, decorative activity feed).
export default function StudyMode({ params }: StudyModeProps) {
  const projectId = params?.id ?? null;
  const [activeTab, setActiveTab] = useState("notes");
  const [notesJob, setNotesJob] = useState<Job | null>(null);
  const [notesDialogOpen, setNotesDialogOpen] = useState(false);
  const [notesConfig, setNotesConfig] = useState<NotesGenerationConfig>({
    mode: "detailed",
    detail: "balanced",
    web: false,
    exam_focus: false,
    memory_aids: false,
    scope: "entire",
  });
  const queryClient = useQueryClient();
  const [, navigate] = useLocation();

  // Fetch project info (no demo id; nothing loads without a real project)
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => (projectId ? getProject(projectId) : Promise.resolve(null)),
    enabled: !!projectId,
  });

  // Study-material readiness: generation is refused without documents
  // (the backend also guards, but the buttons must not offer a dead click).
  const { data: documents = [] } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => (projectId ? getProjectDocuments(projectId) : Promise.resolve([])),
    enabled: !!projectId,
  });
  const hasDocuments = documents.length > 0;

  // Advisory notes recommendation (never overrides the user's selection).
  const { data: notesRecommendation } = useQuery({
    queryKey: ["notes-recommendation", projectId],
    queryFn: () => (projectId ? readNotesRecommendation(projectId) : Promise.resolve(null)),
    enabled: !!projectId,
  });
  const recommendationText = notesRecommendation?.recommendation
    ? `Recommended: ${String(notesRecommendation.recommendation.recommended_mode ?? "").toUpperCase()} / ${String(notesRecommendation.recommendation.recommended_detail ?? "").toLowerCase()}. ${
        notesRecommendation.recommendation.reason ?? "Based on project sources."
      } This suggestion never overrides your selection.`
    : undefined;

  // Persisted generation content (ownership-scoped reads; explicit
  // not_generated empty states, never fabricated).
  const {
    data: notesData,
    isLoading: notesLoading,
    isError: notesError,
    refetch: refetchNotes,
  } = useQuery({
    queryKey: ["notes", projectId],
    queryFn: () => readNotes(projectId as string),
    enabled: !!projectId && activeTab === "notes",
  });

  const {
    data: quizData,
    isLoading: quizLoading,
    isError: quizError,
    refetch: refetchQuiz,
  } = useQuery({
    queryKey: ["quiz-content", projectId],
    queryFn: () => readQuiz(projectId as string),
    enabled: !!projectId && activeTab === "practice",
  });

  // Generated study-notes PDF metadata (Step 8-10): drives the
  // No -> Generate -> Generating -> Ready states and the
  // "Open Study Notes" action into the EXISTING viewer.
  const { data: studyNotesPdf } = useQuery({
    queryKey: ["study-notes-pdf", projectId],
    queryFn: () => readStudyNotesPdfMeta(projectId as string),
    enabled: !!projectId && activeTab === "notes",
  });

  // Partial-success signal: the notes run can be ready with honest source
  // gaps and/or a slow provider.  Read it from the same job output the PDF
  // metadata comes from so the UI can say so without inventing anything.
  const { data: notesRun } = useQuery({
    queryKey: ["notes-run", projectId],
    queryFn: () =>
      projectId ? readHierarchicalNotesRun(projectId) : Promise.resolve(null),
    enabled: !!projectId && activeTab === "notes",
  });

  // Explanation + answer state (query required).
  const [explanationQuery, setExplanationQuery] = useState("");
  const [explanationResult, setExplanationResult] = useState<ExplanationResult | null>(null);
  const [answerQuery, setAnswerQuery] = useState("");
  const [answerMarks, setAnswerMarks] = useState(5);
  const [answerStyle, setAnswerStyle] = useState<"concise" | "detailed" | "exam" | "tutorial">("detailed");
  const [answerResult, setAnswerResult] = useState<AnswerResult | null>(null);

  // MCQ interaction: selected option per question, submitted set.
  const [mcqSelections, setMcqSelections] = useState<Record<string, number>>({});
  const [mcqSubmitted, setMcqSubmitted] = useState<Record<string, boolean>>({});

  // Generate content mutations (ownership-scoped POSTs; reads refresh so
  // generated content persists across refresh).
  const generateNotesMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project selected.");
      return generateStudyNotes({
        projectId,
        config: notesConfig,
        onProgress: setNotesJob,
      });
    },
    onMutate: () => setNotesJob(null),
    onSuccess: () => {
      setNotesJob(null);
      void refetchNotes();
      void queryClient.invalidateQueries({ queryKey: ["notes", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["study-notes-pdf", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["notes-run", projectId] });
    },
    // Keep the last real server stage on timeout/failure. The error surface
    // below exposes the durable job id and offers one explicit retry.
  });

  const generatePracticeMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project selected.");
      return generatePractice({ projectId });
    },
    onSuccess: () => {
      setMcqSelections({});
      setMcqSubmitted({});
      void refetchQuiz();
      void queryClient.invalidateQueries({ queryKey: ["quiz-content", projectId] });
    },
  });

  const generateExplanationMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project selected.");
      return generateExplanation({ projectId, query: explanationQuery });
    },
    onSuccess: (result) => setExplanationResult(result),
  });

  const generateAnswerMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("No project selected.");
      return generateAnswer({
        projectId,
        query: answerQuery,
        marks: answerMarks,
        style: answerStyle,
      });
    },
    onSuccess: (result) => setAnswerResult(result),
  });

  const tabs = [
    { id: "notes", label: "Study Notes" },
    { id: "practice", label: "Practice Questions" },
    { id: "answers", label: "Answers" },
    { id: "explanations", label: "Explanations" },
  ];

  const isGenerating = generateNotesMutation.isPending ||
    generatePracticeMutation.isPending ||
    generateExplanationMutation.isPending || generateAnswerMutation.isPending;

  // Sidebar readiness is derived from real reads, never fabricated.
  const notesReady = notesData?.state === "ready" || studyNotesPdf?.state === "ready";
  const practiceReady = (quizData?.quizzes?.length ?? 0) > 0;
  const notesRunError =
    generateNotesMutation.error instanceof JobRunError
      ? generateNotesMutation.error
      : null;

  return (
    <Layout activeNav="study">
      <div className="pb-20 md:pb-8 px-6 md:px-12 min-h-screen pt-10">
        {/* Project Header */}
        <header className="mb-12 flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div className="max-w-2xl">
            <div className="flex items-center gap-2 mb-4">
              <span
                className="uppercase tracking-widest px-2 py-1 rounded"
                style={{
                  fontFamily: "Space Grotesk, sans-serif",
                  fontSize: "10px",
                  color: "#edfee9",
                  backgroundColor: "rgba(237,254,233,0.1)",
                }}
              >
                {isGenerating ? "Generating..." : "Study Mode"}
              </span>
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: "#edfee9" }} />
            </div>
            <h1
              className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4"
              style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
            >
              {project?.name || "Study Session"}
            </h1>
            <p className="text-lg leading-relaxed" style={{ color: "#acabaa" }}>
              {project?.description || "Generate study materials from your uploaded documents to start learning."}
            </p>
          </div>

          {/* Processing Status Card */}
          <MagicCard
            className="relative"
            style={{ display: "inline-block" }}
            glowColor="198, 198, 199"
            enableStars={false}
          >
            <div className="absolute inset-0 study-glow blur-2xl opacity-50" />
            <div
              className="relative flex items-center gap-4 p-4 rounded-xl"
              style={{
                backgroundColor: "rgba(31,32,32,0.5)",
                backdropFilter: "blur(20px)",
                border: "1px solid rgba(72,72,72,0.1)",
              }}
            >
              <div className="flex flex-col">
                <span
                  className="uppercase tracking-widest"
                  style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
                >
                  {isGenerating ? "Generating..." : "Processing Status"}
                </span>
                <span className="text-sm font-semibold" style={{ color: "#e7e5e4" }}>
                  {isGenerating ? "Generating study materials..." : "Ready"}
                </span>
              </div>
              <div className="w-10 h-10 flex items-center justify-center">
                {isGenerating ? (
                  <SpinnerIcon size={24} color="#c6c6c7" />
                ) : (
                  <span
                    className="material-symbols-outlined"
                    style={{ color: "#c6c6c7" }}
                  >
                    check_circle
                  </span>
                )}
              </div>
            </div>
          </MagicCard>
        </header>

        {/* Tab Navigation */}
        <div
          className="flex items-center gap-8 mb-8 overflow-x-auto no-scrollbar"
          style={{ borderBottom: "1px solid rgba(72,72,72,0.1)" }}
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className="pb-4 text-sm font-semibold uppercase tracking-widest whitespace-nowrap transition-colors"
              style={{
                fontFamily: "Inter, sans-serif",
                color: activeTab === tab.id ? "#e7e5e4" : "#acabaa",
                borderBottom: activeTab === tab.id ? "2px solid #c6c6c7" : "2px solid transparent",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content Grid */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          {/* Main Content */}
          <section className="md:col-span-8 flex flex-col gap-6">
            {activeTab === "notes" && (
              <>
                {/* Persisted notes: sections/headings/content/citations. */}
                {notesLoading && (
                  <div className="p-8 rounded-2xl text-center" style={{ backgroundColor: "#131313" }}>
                    <p style={{ color: "#acabaa" }}>Loading study notes...</p>
                  </div>
                )}
                {!notesLoading && notesError && (
                  <div className="p-8 rounded-2xl text-center" style={{ backgroundColor: "#131313", border: "1px solid rgba(255,80,80,0.25)" }}>
                    <p style={{ color: "#ee7d77" }}>Could not load notes. Check your connection and try again.</p>
                    <button
                      onClick={() => refetchNotes()}
                      className="mt-4 px-6 py-2 rounded-xl text-sm font-semibold"
                      style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
                    >
                      Retry
                    </button>
                  </div>
                )}
                {!notesLoading && !notesError && notesData?.state === "not_generated" && (
                  <div className="p-8 rounded-2xl text-center" style={{ backgroundColor: "#131313" }}>
                    <p style={{ color: "#acabaa" }}>No notes generated yet for this project.</p>
                  </div>
                )}
                {!notesLoading && !notesError && notesData?.state === "ready" && (
                  <div className="flex flex-col gap-6">
                    {/* LIST first: every note for this project (newest first)
                        with Open (this note's PDF in the existing viewer) and
                        Delete (ownership checked server-side, then refetch). */}
                    <div className="p-6 rounded-2xl" style={{ backgroundColor: "#131313" }}>
                      <h3
                        className="text-sm font-bold uppercase tracking-widest mb-4"
                        style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
                      >
                        Your notes ({notesData.total})
                      </h3>
                      <NotesArtifactList
                        projectId={projectId as string}
                        items={notesData.notes}
                        onChanged={() => {
                          void refetchNotes();
                          void queryClient.invalidateQueries({ queryKey: ["study-notes-pdf", projectId] });
                        }}
                      />
                    </div>
                    {notesData.notes.map((note) => (
                      <article key={note.id} className="p-8 rounded-2xl" style={{ backgroundColor: "#131313" }}>
                        <h2 className="text-2xl font-bold mb-1" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>
                          {note.title}
                        </h2>
                        <p
                          className="text-xs mb-4"
                          style={{ color: "#acabaa", fontFamily: "Space Grotesk, sans-serif" }}
                        >
                          {noteMetaLine(note) || "No metadata recorded"}
                        </p>
                        <NoteBody content={note.content} />
                      </article>
                    ))}
                  </div>
                )}
                {/* Study-notes PDF states: No -> Generate -> Generating -> Ready.
                    Ready opens the generated PDF in the EXISTING viewer
                    (/documents/study-notes-{projectId}); metadata shows
                    topics/pages/coverage/date/model, never internal JSON. */}
                {!notesLoading && !notesError && (
                  <div className="p-6 rounded-2xl" style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.15)" }}>
                    <div className="flex items-center justify-between gap-4 mb-2">
                      <h3 className="text-sm font-bold uppercase tracking-widest" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>
                        Study Notes PDF
                      </h3>
                      <button
                        onClick={() => setNotesDialogOpen(true)}
                        className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all hover:opacity-90"
                        style={{ color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.3)" }}
                      >
                        Settings
                      </button>
                    </div>
                    {generateNotesMutation.isPending ? (
                      <div aria-live="polite" className="text-sm" style={{ color: "#acabaa" }}>
                        <p>
                          {notesJob
                            ? `${notesJob.stage || notesJob.status} - ${notesJob.progress}%`
                            : "Queued - waiting for the worker to claim the job..."}
                        </p>
                        <p className="mt-1">
                          {notesJob?.stageMessage ||
                            (notesJob ? "The worker has not reported a new stage yet." : "Waiting for the first job status...")}
                        </p>
                        {notesJob?.id && (
                          <p className="mt-1 font-mono text-xs">Job: {notesJob.id}</p>
                        )}
                      </div>
                    ) : studyNotesPdf?.state === "ready" ? (
                      <div>
                        <p className="text-sm mb-3" style={{ color: "#acabaa" }}>
                          {[
                            studyNotesPdf.version ? `version ${studyNotesPdf.version}` : null,
                            studyNotesPdf.topics !== undefined && studyNotesPdf.topics !== null
                              ? `${studyNotesPdf.topics} topics`
                              : null,
                            studyNotesPdf.pages !== undefined && studyNotesPdf.pages !== null
                              ? `${studyNotesPdf.pages} pages`
                              : null,
                            studyNotesPdf.coverage?.topic_coverage !== undefined &&
                            studyNotesPdf.coverage?.topic_coverage !== null
                              ? `coverage ${Math.round(studyNotesPdf.coverage.topic_coverage * 100)}%`
                              : null,
                            studyNotesPdf.created_at ? studyNotesPdf.created_at.slice(0, 10) : null,
                            studyNotesPdf.model ? `model: ${studyNotesPdf.model}` : null,
                          ]
                            .filter(Boolean)
                            .join(" · ")}
                        </p>
                        {notesRun && notesRun.gaps.length > 0 && (
                          <details className="mb-3 text-sm" style={{ color: "#acabaa" }}>
                            <summary style={{ cursor: "pointer" }}>
                              Notes ready with {notesRun.gaps.length} source gap
                              {notesRun.gaps.length === 1 ? "" : "s"}
                              {notesRun.provider_slow
                                ? " (your AI provider was slow on some topics)"
                                : ""}
                            </summary>
                            <ul className="list-disc ml-5 mt-2 space-y-1">
                              {notesRun.gaps.map((gap, index) => (
                                <li key={`${gap.topic ?? "gap"}-${index}`}>
                                  {gap.topic ?? "Unknown topic"}
                                  {gap.reason ? ` — ${gap.reason}` : ""}
                                </li>
                              ))}
                            </ul>
                          </details>
                        )}
                        {studyNotesPdf.total !== undefined && studyNotesPdf.total > 1 && (
                          <p className="text-sm mb-3" style={{ color: "#acabaa" }}>
                            {studyNotesPdf.total} versions saved. Open any of them from the
                            notes list above.
                          </p>
                        )}
                        <button
                          onClick={() =>
                            navigate(
                              studyNotesPdf.version
                                ? `/documents/study-notes-${projectId}?version=${studyNotesPdf.version}`
                                : `/documents/study-notes-${projectId}`
                            )
                          }
                          className="px-6 py-3 rounded-xl font-semibold transition-all hover:opacity-90"
                          style={{ backgroundColor: "#edfee9", color: "#131313" }}
                        >
                          Open Study Notes
                        </button>
                      </div>
                    ) : (
                      <div className="flex flex-col gap-3">
                        <p className="text-sm" style={{ color: "#acabaa" }}>
                          Notes are ready. The printable PDF is being prepared — check back shortly.
                        </p>
                        <div>
                          <button
                            onClick={() => setNotesDialogOpen(true)}
                            disabled={!hasDocuments}
                            className="px-6 py-3 rounded-xl font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                            style={{ backgroundColor: "#edfee9", color: "#131313" }}
                          >
                            Generate Notes
                          </button>
                          {!hasDocuments && (
                            <p className="mt-2 text-sm" style={{ color: "#acabaa" }}>
                              <a href="/upload" style={{ color: "#edfee9", textDecoration: "underline" }}>
                                Upload study material first
                              </a>
                            </p>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}
                {/* Prominent Generate Call-to-Action */}
                <div
                  className="p-8 md:p-12 rounded-2xl relative overflow-hidden group transition-all duration-300"
                  style={{
                    background: "linear-gradient(135deg, rgba(237,254,233,0.05) 0%, rgba(19,19,19,1) 100%)",
                    border: "2px solid rgba(237,254,233,0.2)",
                  }}
                >
                  <div className="absolute top-0 right-0 p-6 opacity-10 group-hover:opacity-20 transition-opacity">
                    <SparklesIcon size={96} color="#e7e5e4" />
                  </div>
                  <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
                    <div
                      className="w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0"
                      style={{ backgroundColor: "rgba(237,254,233,0.1)" }}
                    >
                      <SparklesIcon size={32} color="#edfee9" label="Generate notes" />
                    </div>
                    <div className="flex-1">
                      <h2 className="text-2xl font-bold mb-2" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>
                        Generate Study Notes
                      </h2>
                      <p className="text-sm mb-4" style={{ color: "#acabaa" }}>
                        Transform your uploaded documents into comprehensive, structured study notes.
                        This is the next step to start learning from your materials.
                      </p>
                      <button
                        onClick={() => setNotesDialogOpen(true)}
                        disabled={generateNotesMutation.isPending || !hasDocuments}
                        className="px-8 py-4 rounded-xl font-semibold transition-all hover:opacity-90 disabled:opacity-50 flex items-center gap-2"
                        style={{ backgroundColor: "#edfee9", color: "#131313" }}
                      >
                        {generateNotesMutation.isPending ? (
                          <>
                            <SpinnerIcon size={18} color="#131313" />
                            Generating Your Notes...
                          </>
                        ) : (
                          <>
                            <SparklesIcon size={18} color="#131313" />
                            Generate Notes Now
                          </>
                        )}
                      </button>
                      {!hasDocuments && (
                        <p className="mt-3 text-sm" style={{ color: "#acabaa" }}>
                          <a href="/upload" style={{ color: "#edfee9", textDecoration: "underline" }}>
                            Upload study material first
                          </a>
                        </p>
                      )}
                      {generateNotesMutation.isError && (
                        <div className="text-sm mt-3" role="alert">
                          <p style={{ color: "#ee7d77" }}>
                            {(generateNotesMutation.error as Error)?.message || "Notes generation failed."}
                          </p>
                          {notesRunError && (
                            <div className="mt-2 flex flex-wrap items-center gap-3">
                              <code className="text-xs break-all">Job: {notesRunError.jobId}</code>
                              <button
                                type="button"
                                onClick={() => generateNotesMutation.mutate()}
                                disabled={generateNotesMutation.isPending}
                                className="px-3 py-1.5 rounded-lg text-xs font-semibold disabled:opacity-50"
                                style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
                              >
                                Retry job
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* What You'll Get: two cards only. Flashcards lives in the
                    standalone /flashcards route (single home for flashcards),
                    so no second flashcards surface is advertised here. */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="p-6 rounded-2xl" style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.05)" }}>
                    <span className="material-symbols-outlined mb-4" style={{ color: "#edfee9", display: "block" }}>article</span>
                    <h4
                      className="text-sm font-bold uppercase tracking-wider mb-2"
                      style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
                    >
                      Structured Notes
                    </h4>
                    <p className="text-sm" style={{ color: "#acabaa" }}>
                      Generated comprehensive notes organized for easy studying.
                    </p>
                  </div>
                  <div className="p-6 rounded-2xl" style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.05)" }}>
                    <span className="material-symbols-outlined mb-4" style={{ color: "#8ca1aa", display: "block" }}>quiz</span>
                    <h4
                      className="text-sm font-bold uppercase tracking-wider mb-2"
                      style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
                    >
                      Practice Questions
                    </h4>
                    <p className="text-sm" style={{ color: "#acabaa" }}>
                      Test your knowledge with generated practice questions.
                    </p>
                  </div>
                </div>
              </>
            )}

            {activeTab === "practice" && (
              <McqTab
                quizData={quizData}
                quizLoading={quizLoading}
                quizError={quizError}
                selections={mcqSelections}
                submitted={mcqSubmitted}
                onSelect={(qid, idx) => setMcqSelections((p) => ({ ...p, [qid]: idx }))}
                onSubmit={(qid) => setMcqSubmitted((p) => ({ ...p, [qid]: true }))}
                onGenerate={() => generatePracticeMutation.mutate()}
                generating={generatePracticeMutation.isPending}
                disabled={!hasDocuments}
                disabledReason="Generate practice questions from your study material."
              />
            )}

            {activeTab === "answers" && (
              <div className="p-8 rounded-2xl" style={{ backgroundColor: "#131313" }}>
                <h2 className="text-2xl font-bold mb-4" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>Marks-aware Answers</h2>
                <div className="flex flex-col md:flex-row gap-3 mb-4">
                  <input
                    type="text"
                    value={answerQuery}
                    onChange={(e) => setAnswerQuery(e.target.value)}
                    placeholder="Enter the exam question..."
                    className="flex-1 px-4 py-3 rounded-xl text-sm"
                    style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.3)" }}
                  />
                  <label className="flex items-center gap-2 text-sm" style={{ color: "#acabaa" }}>
                    Marks
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={answerMarks}
                      onChange={(e) => setAnswerMarks(Math.max(1, Math.min(100, Number(e.target.value) || 1)))}
                      className="w-20 px-3 py-3 rounded-xl text-sm"
                      style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.3)" }}
                    />
                  </label>
                  <select
                    value={answerStyle}
                    onChange={(e) => setAnswerStyle(e.target.value as typeof answerStyle)}
                    className="px-4 py-3 rounded-xl text-sm"
                    style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.3)" }}
                  >
                    {(["concise", "detailed", "exam", "tutorial"] as const).map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => generateAnswerMutation.mutate()}
                    disabled={generateAnswerMutation.isPending || !answerQuery.trim()}
                    className="px-6 py-3 rounded-xl font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                    style={{ backgroundColor: "#edfee9", color: "#131313" }}
                  >
                    {generateAnswerMutation.isPending ? "Answering..." : "Get answer"}
                  </button>
                </div>
                {generateAnswerMutation.isError && (
                  <p className="text-sm mb-4" style={{ color: "#ee7d77" }}>
                    {(generateAnswerMutation.error as Error)?.message || "Answer generation failed."}
                  </p>
                )}
                {!answerResult && !generateAnswerMutation.isPending && (
                  <p className="text-sm" style={{ color: "#acabaa" }}>No answer yet — ask an exam question above.</p>
                )}
                {answerResult && (
                  <AnswerCard result={answerResult} />
                )}
              </div>
            )}

            {activeTab === "explanations" && (
              <div className="p-8 rounded-2xl" style={{ backgroundColor: "#131313" }}>
                <h2 className="text-2xl font-bold mb-4" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>Concept Explanations</h2>
                <div className="flex flex-col md:flex-row gap-3 mb-4">
                  <input
                    type="text"
                    value={explanationQuery}
                    onChange={(e) => setExplanationQuery(e.target.value)}
                    placeholder="Which concept should be explained?"
                    className="flex-1 px-4 py-3 rounded-xl text-sm"
                    style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.3)" }}
                  />
                  <button
                    onClick={() => generateExplanationMutation.mutate()}
                    disabled={generateExplanationMutation.isPending || !explanationQuery.trim()}
                    className="px-6 py-3 rounded-xl font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                    style={{ backgroundColor: "#edfee9", color: "#131313" }}
                  >
                    {generateExplanationMutation.isPending ? "Explaining..." : "Explain"}
                  </button>
                </div>
                {generateExplanationMutation.isError && (
                  <p className="text-sm mb-4" style={{ color: "#ee7d77" }}>
                    {(generateExplanationMutation.error as Error)?.message || "Explanation failed."}
                  </p>
                )}
                {!explanationResult && !generateExplanationMutation.isPending && (
                  <p className="text-sm" style={{ color: "#acabaa" }}>No explanation yet — enter a concept above.</p>
                )}
                {explanationResult && (
                  <div>
                    <p className="text-sm leading-relaxed mb-4" style={{ color: "#e7e5e4" }}>
                      {String(explanationResult.content.explanation ?? "")}
                    </p>
                    {(explanationResult.content.examples ?? []).length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-sm font-bold uppercase tracking-wider mb-2" style={{ color: "#e7e5e4" }}>Examples</h4>
                        <ul className="list-disc ml-5 space-y-1">
                          {(explanationResult.content.examples ?? []).map((ex, i) => (
                            <li key={i} className="text-sm" style={{ color: "#acabaa" }}>{ex}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {(explanationResult.content.related_concepts ?? []).length > 0 && (
                      <div className="mb-4 flex gap-2 flex-wrap">
                        {(explanationResult.content.related_concepts ?? []).map((rc, i) => (
                          <span key={i} className="px-2 py-0.5 rounded text-xs" style={{ backgroundColor: "#1f2020", color: "#8ca1aa" }}>
                            {rc}
                          </span>
                        ))}
                      </div>
                    )}
                    <Citations citations={explanationResult.content.citations} />
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Sidebar: content readiness from real reads only. */}
          <aside className="md:col-span-4 flex flex-col gap-6">
            <div className="p-6 rounded-2xl" style={{ backgroundColor: "#131313" }}>
              <h3
                className="text-sm font-bold uppercase tracking-widest mb-4"
                style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
              >
                Study Material Status
              </h3>
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ backgroundColor: notesReady ? "#edfee9" : "#484848" }}
                  />
                  <p
                    className="uppercase tracking-tighter"
                    style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
                  >
                    {notesLoading ? "Checking notes..." : notesReady ? "Notes ready" : "Notes not generated"}
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <div
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ backgroundColor: practiceReady ? "#edfee9" : "#484848" }}
                  />
                  <p
                    className="uppercase tracking-tighter"
                    style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
                  >
                    {quizLoading ? "Checking practice..." : practiceReady ? "Practice ready" : "Practice not generated"}
                  </p>
                </div>
              </div>
            </div>
          </aside>
        </div>

        <NotesGenerationConfigDialog
          open={notesDialogOpen}
          onOpenChange={setNotesDialogOpen}
          value={notesConfig}
          onChange={setNotesConfig}
          recommendation={recommendationText}
          onConfirm={() => generateNotesMutation.mutate()}
        />
      </div>
    </Layout>
  );
}
