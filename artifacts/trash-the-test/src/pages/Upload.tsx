import { useState, useRef } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import Layout from "@/components/Layout";
import CreateProjectDialog from "@/components/CreateProjectDialog";
import { DeleteDocumentButton } from "@/components/DeleteDocumentButton";
import { SparklesIcon } from "@/components/icons";
import { useLocation } from "wouter";
import {
  uploadDocument,
  streamUploadDocument,
  uploadDocumentsMulti,
  getProjectDocuments,
  getDocumentProgress,
  getJobProgress,
  streamJobProgress,
  cancelJob,
  shouldWarnHeavyDocument,
  heavyDocumentWarning,
  puterProcessDocument,
  PUTER_ENABLED,
  PUTER_HEAVY_PAGE_THRESHOLD,
  type Document,
  type DocumentSourceType,
} from "@/lib/api-services";
import { useActiveProject } from "@/lib/use-active-project";

export default function Upload() {
  const [, navigate] = useLocation();
  const [dragOver, setDragOver] = useState(false);
  const [extractionLevel, setExtractionLevel] = useState("standard");
  const [sourceType, setSourceType] = useState<DocumentSourceType>("study_material");
  const [outputFormats, setOutputFormats] = useState({ flashcards: true, maps: true });
  const [uploadComplete, setUploadComplete] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  // Shared project context (last-selected project, never a blind [0]).
  // No demo fallback: without a project the user must create one first.
  // The inline dropdown + creation dialog feed explicit selections in.
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const {
    projects,
    activeProject,
    activeProjectId: projectId,
    selectProject,
  } = useActiveProject(selectedProjectId);
  // Inline creation entry point: with zero projects the page used to dead-end
  // ("Create one from the Dashboard"). The dialog posts the real API and the
  // created project is selected immediately so upload controls unlock.
  const [createOpen, setCreateOpen] = useState(false);

  // Fetch recent documents for this project (real API state only)
  const { data: documents = [] } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => (projectId ? getProjectDocuments(projectId) : Promise.resolve([])),
    enabled: !!projectId,
  });

  // Indexing state is read fresh from the job/document API in the
  // pollers below; no client-side progress cache is kept.
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  // Document tracked by the progress UI above (single or multi upload).
  // Deleting THAT file has to stop the poller and cancel its job, so a
  // removed document can never keep reporting progress afterwards.
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);
  const stoppedJobsRef = useRef<Set<string>>(new Set());
  const removedDocumentsRef = useRef<Set<string>>(new Set());
  const [cancelling, setCancelling] = useState(false);
  const [multiResults, setMultiResults] = useState<
    Array<{ fileName: string | null; status: string; error: string | null }>
  >([]);

  // Get recent files from API (with provenance: indexing state, page
  // methods, mime, checksum — rendered below, never hidden).
  const recentFiles = documents.slice(0, 3).map((doc: Document) => ({
    id: doc.id,
    name: doc.fileName,
    time: new Date(doc.createdAt).toLocaleDateString(),
    status: doc.processingStatus,
    indexingState: doc.indexingState,
    mimeType: doc.mimeType,
    checksum: doc.checksum,
    pageMethods: doc.pageMethods,
    pageCount: doc.pageCount,
    pageCountConfirmed: doc.pageCountConfirmed,
    fileType: doc.fileType,
    sourceType: doc.sourceType ?? "study_material",
  }));

  // Heavy-document (Puter, default OFF): warning + explicit authorisation.
  // The per-call token lives in local state only and is cleared after the
  // call — never persisted. Deny/fail falls back to local OCR.
  const [pageEstimate, setPageEstimate] = useState("");
  const [puterToken, setPuterToken] = useState("");
  const [puterBusy, setPuterBusy] = useState(false);
  const [puterProgress, setPuterProgress] = useState<number | null>(null);
  const [puterStage, setPuterStage] = useState<string | null>(null);
  const [puterError, setPuterError] = useState<string | null>(null);
  const [puterDone, setPuterDone] = useState<string | null>(null);
  const [puterJobId, setPuterJobId] = useState<string | null>(null);
  const [puterDeclinedLocal, setPuterDeclinedLocal] = useState(false);

  const heavyDocs = recentFiles.filter(
    (f) => f.pageCount !== undefined && f.pageCount !== null &&
      shouldWarnHeavyDocument(f.pageCount)
  );
  const estimateHeavy =
    pageEstimate.trim() !== "" &&
    !Number.isNaN(Number(pageEstimate)) &&
    shouldWarnHeavyDocument(Number(pageEstimate));
  const showPuterPanel = heavyDocs.length > 0 || estimateHeavy;
  const puterTarget = heavyDocs[0] ?? null;

  const handlePuterAuthorise = async () => {
    if (!puterTarget) {
      setPuterError("No heavy document uploaded yet — upload first, then authorise.");
      return;
    }
    if (!puterToken) {
      setPuterError("Enter the per-call Puter token to authorise (it is never stored).");
      return;
    }
    setPuterBusy(true);
    setPuterError(null);
    setPuterDone(null);
    setPuterProgress(0);
    setPuterStage("puter-processing");
    setPuterDeclinedLocal(false);
    try {
      const result = await puterProcessDocument(puterTarget.id, {
        authorised: true,
        userToken: puterToken,
      });
      setPuterJobId(result.jobId);
      setPuterStage(result.stage);
      setPuterProgress(100);
      if (result.jobId) {
        void pollJob(result.jobId);
      }
      setPuterDone(
        `${result.message} Puter pages: ${result.puterPages}, ` +
        `local fallback: ${result.fallbackPages}.`
      );
    } catch (err) {
      setPuterError(
        `${err instanceof Error ? err.message : "Puter processing failed."} ` +
        "Falling back to local OCR — your document still processes normally."
      );
      setPuterDeclinedLocal(true);
    } finally {
      // Never stored: the token is cleared the moment the call settles.
      setPuterToken("");
      setPuterBusy(false);
    }
  };

  const handlePuterCancel = async () => {
    if (puterJobId) {
      try {
        await cancelJob(puterJobId);
      } catch {
        // Cancel is best-effort; the UI still stops tracking below.
      }
      setPuterJobId(null);
    }
    setPuterBusy(false);
    setPuterProgress(null);
    setPuterStage("cancelled");
  };

  const handlePuterDecline = () => {
    setPuterDeclinedLocal(true);
    setPuterError(null);
    setPuterDone("Continuing with local OCR (Puter declined). No Puter call was made.");
  };

  // Upload mutation: real endpoint, real JWT, real job id in response.
  // Files over 10MB go through /upload/stream (no demo fallbacks).
  const uploadMutation = useMutation({
    mutationFn: async ({ file }: { file: File }) => {
      if (!projectId) {
        throw new Error("Create a project first before uploading.");
      }
      const idempotencyKey = `${file.name}-${file.size}-${file.lastModified}`;
      const uploader = file.size > 10 * 1024 * 1024 ? streamUploadDocument : uploadDocument;
      return uploader(file, projectId, { idempotencyKey, sourceType });
    },
    onSuccess: (data) => {
      setUploadedFileName(data.fileName);
      setUploadError(null);
      // Track real indexing progress from the job API, not a timer.
      const jobId = data.jobId;
      const documentId = data.documentId;
      setActiveDocumentId(documentId ?? null);
      if (jobId) {
        pollJob(jobId);
      } else if (documentId) {
        pollDocument(documentId);
      } else {
        setUploadComplete(true);
      }
    },
    onError: (error: Error) => {
      // Surface the real server error instead of a generic toast.
      setUploadError(error.message || "Upload failed. Please try again.");
      setUploading(false);
    },
  });

  // Multi-file upload mutation: POST /api/documents/upload/multi,
  // per-file jobs; one file failing never affects the others.
  const uploadMultiMutation = useMutation({
    mutationFn: async (files: File[]) => {
      if (!projectId) {
        throw new Error("Create a project first before uploading.");
      }
      return uploadDocumentsMulti(files, projectId, sourceType);
    },
    onSuccess: (results) => {
      setMultiResults(
        results.map((r) => ({ fileName: r.fileName, status: r.status, error: r.error }))
      );
      setUploadError(null);
      const firstJob = results.find((r) => r.jobId)?.jobId ?? null;
      const firstDoc = results.find((r) => r.documentId)?.documentId ?? null;
      setActiveDocumentId(firstDoc);
      if (firstJob) {
        pollJob(firstJob);
      } else if (firstDoc) {
        pollDocument(firstDoc);
      } else {
        setUploadComplete(true);
        setUploading(false);
      }
      const failed = results.filter((r) => r.status === "failed");
      if (failed.length > 0) {
        setUploadError(
          `${failed.length} file(s) failed: ${failed.map((f) => `${f.fileName}: ${f.error}`).join("; ")}`
        );
      }
    },
    onError: (error: Error) => {
      setUploadError(error.message || "Multi upload failed. Please try again.");
      setUploading(false);
    },
  });

  // Real progress: SSE stream first (GET /api/jobs/{id}/stream), poll
  // fallback when the stream errors. "not started" renders until the
  // first worker report arrives - no fabricated percentages.
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [jobStage, setJobStage] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const pollJobLoop = async (jobId: string) => {
    for (;;) {
      // The document behind this job may have been deleted while polling:
      // stop silently instead of reporting a terminal state for a removed file.
      if (stoppedJobsRef.current.has(jobId)) {
        setActiveJobId(null);
        setUploading(false);
        return;
      }
      const job = await getJobProgress(jobId);
      if (stoppedJobsRef.current.has(jobId)) {
        setActiveJobId(null);
        setUploading(false);
        return;
      }
      setUploadProgress(job.progress);
      setJobStage(job.stage ?? null);
      if (job.status === "completed") {
        setUploadComplete(true);
        setUploading(false);
        setActiveJobId(null);
        return;
      }
      if (job.status === "failed" || (job.status as string) === "cancelled") {
        setUploadError(job.errorMessage || "Processing failed.");
        setUploading(false);
        setActiveJobId(null);
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
  };

  const pollJob = async (jobId: string) => {
    setUploading(true);
    setUploadProgress(null);
    setJobStage(null);
    setActiveJobId(jobId);
    // Ref-style holder: avoids let-narrowing quirks across the SSE callbacks.
    const streamCtl: { stop: (() => void) | null } = { stop: null };
    let settled = false;
    let streamFailed = false;
    const finish = (ok: boolean, message?: string) => {
      settled = true;
      // A delete of the tracked document already stopped this UI: never
      // surface "Processing failed" for a file the user deliberately removed.
      if (stoppedJobsRef.current.has(jobId)) {
        setUploading(false);
        setActiveJobId(null);
        return;
      }
      if (ok) setUploadComplete(true);
      else setUploadError(message || "Processing failed.");
      setUploading(false);
      setActiveJobId(null);
    };
    try {
      // Prefer the live SSE stream; fall back to polling on error/timeout.
      // A timeout guards streams that end silently (server close with no
      // terminal event): after it fires the poll loop takes over, so the
      // UI can never hang waiting for progress.
      await new Promise<void>((resolve) => {
        const timer = setTimeout(() => resolve(), 30_000);
        streamCtl.stop = streamJobProgress(
          jobId,
          (job) => {
            if (stoppedJobsRef.current.has(jobId)) {
              // Tracked document deleted: close the stream, no error surface.
              clearTimeout(timer);
              finish(true);
              resolve();
              return;
            }
            setUploadProgress(job.progress);
            setJobStage(job.stage ?? null);
            if (job.status === "completed" || job.status === "failed" ||
                (job.status as string) === "cancelled") {
              clearTimeout(timer);
              finish(job.status === "completed", job.errorMessage);
              resolve();
            }
          },
          () => {
            clearTimeout(timer);
            streamFailed = true;
            resolve();
          }
        );
      });
    } catch {
      streamFailed = true;
    } finally {
      streamCtl.stop?.();
    }
    if (!settled) {
      // Stream errored, timed out, or stayed silent: poll to terminal state.
      try {
        await pollJobLoop(jobId);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : "Progress check failed.");
        setUploading(false);
        setActiveJobId(null);
      }
    }
  };

  const handleCancelJob = async () => {
    if (!activeJobId || cancelling) return;
    setCancelling(true);
    try {
      await cancelJob(activeJobId);
      setUploadError("Processing cancelled.");
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Cancel failed.");
    } finally {
      setUploading(false);
      setActiveJobId(null);
      setActiveDocumentId(null);
      setCancelling(false);
    }
  };

  const pollDocument = async (documentId: string) => {
    setUploading(true);
    try {
      for (;;) {
        // Deleted while polling: stop quietly (the row already refetched).
        if (removedDocumentsRef.current.has(documentId)) {
          setUploading(false);
          setUploadProgress(null);
          return;
        }
        const progress = await getDocumentProgress(documentId);
        setUploadProgress(progress.progress);
        setJobStage(progress.stage ?? null);
        if (progress.status === "completed") {
          setUploadComplete(true);
          setUploading(false);
          return;
        }
        if (progress.status === "failed") {
          setUploadError(progress.message || "Processing failed.");
          setUploading(false);
          return;
        }
        if (progress.status === "not_queued") {
          // Honest idle state: queued client-side but no worker report yet.
          setUploadProgress(null);
          setUploading(false);
          setUploadComplete(true);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 1500));
      }
    } catch (err) {
      // 404 after an intentional delete is expected, not a progress failure.
      if (removedDocumentsRef.current.has(documentId)) {
        setUploading(false);
        setUploadProgress(null);
        return;
      }
      setUploadError(err instanceof Error ? err.message : "Progress check failed.");
      setUploading(false);
    }
  };

  const handleUpload = (fileName: string, file?: File) => {
    void fileName;
    setUploadError(null);
    setMultiResults([]);
    setUploading(true);
    setUploadProgress(null);

    if (file) {
      if (!projectId) {
        setUploadError("Create a project first before uploading.");
        setUploading(false);
        return;
      }
      // Real API; progress arrives from the SSE/poll trackers above.
      uploadMutation.mutate({ file });
    } else {
      // No file selected
      setUploading(false);
    }
  };

  const handleFiles = (files: File[]) => {
    if (files.length === 0) return;
    if (!projectId) {
      setUploadError("Create a project first before uploading.");
      return;
    }
    if (files.length === 1) {
      handleUpload(files[0].name, files[0]);
      return;
    }
    // Multi-file: one request, per-file jobs server-side.
    setUploadError(null);
    setUploading(true);
    setUploadProgress(null);
    uploadMultiMutation.mutate(files);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(Array.from(e.dataTransfer.files));
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(Array.from(e.target.files ?? []));
    // Allow re-selecting the same files.
    e.target.value = "";
  };

  const handleStartNew = () => {
    setUploadComplete(false);
    setUploadedFileName("");
    setUploadProgress(null);
    setJobStage(null);
    setUploadError(null);
    setMultiResults([]);
    setActiveJobId(null);
    setActiveDocumentId(null);
  };

  // Deleting a document from the Recent Files list: if it is the one this
  // page is tracking, stop the poller and cancel its job. DELETE
  // /api/documents/{id} also cancels the job server-side, so this is the
  // client half of the same "job cancel" contract (best-effort, no error
  // surface for a file the user intentionally removed).
  const handleDocumentDeleted = (documentId: string) => {
    if (activeDocumentId !== documentId) return;
    const jobId = activeJobId;
    removedDocumentsRef.current.add(documentId);
    setActiveDocumentId(null);
    setActiveJobId(null);
    setUploading(false);
    setUploadProgress(null);
    setJobStage(null);
    if (jobId) {
      stoppedJobsRef.current.add(jobId);
      void cancelJob(jobId).catch(() => undefined);
    }
  };

  return (
    <Layout activeNav="upload">
      <div className="pb-20 md:pb-8 px-6 md:px-12 min-h-screen pt-10">
        {/* Header */}
        <div className="flex flex-col md:flex-row gap-8 mb-12">
          <div className="flex-1">
            <h1
              className="text-5xl md:text-7xl font-black tracking-tighter leading-tight mb-0"
              style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
            >
              Upload Study<br />Material.
            </h1>
          </div>
          <div className="md:max-w-sm">
            <p className="text-lg leading-relaxed" style={{ color: "#acabaa" }}>
              Upload textbooks, notes, or papers. The server extracts the text, chunks it, and indexes it so you can study it.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Upload Zone */}
          <div className="lg:col-span-8">
            <div className="mb-4 grid grid-cols-1 sm:grid-cols-3 gap-2" role="group" aria-label="Source type">
              {([
                ["study_material", "Study Material", "Textbooks, notes, and references"],
                ["syllabus", "Syllabus", "Units, topics, and learning outcomes"],
                ["question_paper", "Question Papers", "Past papers, exams, and question banks"],
              ] as [DocumentSourceType, string, string][]).map(([value, label, description]) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setSourceType(value)}
                  aria-pressed={sourceType === value}
                  className="rounded-xl p-3 text-left transition-opacity hover:opacity-90"
                  style={{
                    backgroundColor: sourceType === value ? "#252626" : "#131313",
                    border: `1px solid ${sourceType === value ? "#c6c6c7" : "rgba(72,72,72,0.3)"}`,
                    color: "#e7e5e4",
                  }}
                >
                  <span className="block text-sm font-semibold">{label}</span>
                  <span className="mt-1 block text-xs" style={{ color: "#acabaa" }}>{description}</span>
                </button>
              ))}
            </div>
            <div
              className="rounded-2xl p-8 transition-all duration-300 relative"
              style={{ backgroundColor: "#131313" }}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              {dragOver && (
                <div
                  className="absolute inset-0 rounded-2xl z-10 flex items-center justify-center"
                  style={{ backgroundColor: "rgba(198,198,199,0.05)", border: "2px dashed rgba(198,198,199,0.3)" }}
                >
                  <span className="text-lg font-bold" style={{ color: "#c6c6c7" }}>Drop here</span>
                </div>
              )}

              {/* Upload Area - Show upload form or completion state */}
              {!uploadComplete ? (
                <div className="flex flex-col items-center py-12">
                  <div
                    className="w-24 h-24 rounded-2xl flex items-center justify-center mb-6"
                    style={{ backgroundColor: "#1f2020" }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "48px", color: "#acabaa" }}>cloud_upload</span>
                  </div>
                  <h2
                    className="text-2xl font-bold mb-2 text-center"
                    style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
                  >
                    Drop Study Material Here
                  </h2>
                  <p
                    className="mb-8 uppercase tracking-widest text-center"
                    style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "12px", color: "#acabaa" }}
                  >
                    PDF, EPUB, TXT, OR IMAGE (MAX 50MB) · MULTIPLE FILES OK
                  </p>

                  <input
                    ref={fileRef}
                    type="file"
                    className="hidden"
                    multiple
                    accept=".pdf,.epub,.txt,.png,.jpg,.jpeg"
                    onChange={handleFileSelect}
                  />
                  <button
                    onClick={() => fileRef.current?.click()}
                    disabled={uploading}
                    className="px-8 py-4 rounded-xl text-sm font-semibold transition-all hover:opacity-90 active:scale-95 disabled:opacity-50"
                    style={{
                      backgroundColor: "#252626",
                      color: "#e7e5e4",
                      fontFamily: "Inter, sans-serif",
                      border: "1px solid rgba(72,72,72,0.2)",
                    }}
                  >
                    Select Files
                  </button>
                </div>
              ) : (
                /* Upload Complete State - Guide to Generation */
                <div className="flex flex-col items-center py-12">
                  <div
                    className="w-24 h-24 rounded-2xl flex items-center justify-center mb-6"
                    style={{ backgroundColor: "rgba(237,254,233,0.1)" }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "48px", color: "#edfee9" }}>check_circle</span>
                  </div>
                  <h2
                    className="text-2xl font-bold mb-2 text-center"
                    style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
                  >
                    Document Ready
                  </h2>
                  <p
                    className="mb-6 text-center max-w-md"
                    style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "12px", color: "#acabaa" }}
                  >
                    {uploadedFileName} has been processed. Now generate study materials to start learning.
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4 w-full max-w-md px-4">
                    <button
                      onClick={() => navigate(activeProject ? `/study/${activeProject.id}` : "/study")}
                      className="flex-1 px-6 py-4 rounded-xl text-sm font-semibold transition-all hover:opacity-90 active:scale-95 flex items-center justify-center gap-2"
                      style={{
                        backgroundColor: "#edfee9",
                        color: "#131313",
                        fontFamily: "Inter, sans-serif",
                      }}
                    >
                      <SparklesIcon size={18} color="#131313" />
                      Generate Study Materials
                    </button>
                    <button
                      onClick={handleStartNew}
                      className="px-6 py-4 rounded-xl text-sm font-semibold transition-all hover:opacity-90 active:scale-95"
                      style={{
                        backgroundColor: "#252626",
                        color: "#e7e5e4",
                        fontFamily: "Inter, sans-serif",
                        border: "1px solid rgba(72,72,72,0.2)",
                      }}
                    >
                      Upload More
                    </button>
                  </div>
                </div>
              )}

              {/* Upload Progress - real job/document state only, never fabricated */}
              {uploading && !uploadComplete && (
                <div className="mt-4 pt-6" style={{ borderTop: "1px solid rgba(72,72,72,0.1)" }}>
                  <div className="flex justify-between items-center mb-3">
                    <div>
                      <span
                        className="uppercase tracking-widest block mb-1"
                        style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
                      >
                        {uploadMutation.isPending || uploadMultiMutation.isPending
                          ? "Uploading..."
                          : uploadProgress === null
                            ? "Queued - waiting for worker"
                            : `Indexing${jobStage ? ` - ${jobStage}` : ""}`}
                      </span>
                      <span className="text-sm font-semibold" style={{ color: "#e7e5e4" }}>{uploadMutation.variables?.file.name || multiResults[0]?.fileName || "Processing..."}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-bold" style={{ fontFamily: "Space Grotesk, sans-serif", color: "#c6c6c7" }}>
                        {uploadProgress === null ? "not started" : `${Math.min(100, Math.round(uploadProgress))}%`}
                      </span>
                      {activeJobId && (
                        <button
                          onClick={handleCancelJob}
                          disabled={cancelling}
                          className="px-3 py-1.5 rounded-lg text-xs uppercase tracking-widest transition-all disabled:opacity-50"
                          style={{
                            fontFamily: "Space Grotesk, sans-serif",
                            color: "#ee7d77",
                            border: "1px solid rgba(238,125,119,0.3)",
                          }}
                        >
                          {cancelling ? "Cancelling..." : "Cancel"}
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="w-full h-1 rounded-full overflow-hidden" style={{ backgroundColor: "#252626" }}>
                    <div
                      className="h-full transition-all duration-500"
                      style={{ width: `${uploadProgress === null ? 0 : Math.min(100, uploadProgress)}%`, backgroundColor: "#c6c6c7" }}
                    />
                  </div>
                  {multiResults.length > 1 && (
                    <div className="mt-3 space-y-1">
                      {multiResults.map((r) => (
                        <p key={r.fileName ?? "unknown"} className="text-xs" style={{ color: r.status === "failed" ? "#ee7d77" : "#acabaa" }}>
                          {r.fileName ?? "unknown"} — {r.status}{r.error ? `: ${r.error}` : ""}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              )}
              {uploadError && (
                <div className="mt-4 p-4 rounded-xl" style={{ backgroundColor: "rgba(255,80,80,0.08)", border: "1px solid rgba(255,80,80,0.25)" }}>
                  <p className="text-sm" style={{ color: "#ff9d9d" }}>{uploadError}</p>
                </div>
              )}
              {/* Heavy-document estimate (pre-upload): page count is known
                  only after upload, so an estimate can surface the warning. */}
              <div className="mt-4 flex items-center gap-3">
                <label className="uppercase tracking-widest" style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}>
                  Estimated pages (heavy threshold ~{PUTER_HEAVY_PAGE_THRESHOLD})
                </label>
                <input
                  value={pageEstimate}
                  onChange={(e) => setPageEstimate(e.target.value)}
                  placeholder="e.g. 120"
                  inputMode="numeric"
                  className="w-28 px-3 py-2 rounded-lg text-sm"
                  style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.2)" }}
                />
              </div>
              {/* Heavy-document warning + explicit Puter authorisation
                  (default OFF). Authorise runs Puter vision then the normal
                  pipeline; declining/failing falls back to local OCR. */}
              {showPuterPanel && (
                <div className="mt-4 p-5 rounded-2xl" style={{ backgroundColor: "#1a1a1a", border: "1px solid rgba(198,198,199,0.25)" }}>
                  <p className="uppercase tracking-widest mb-2" style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#c6c6c7" }}>
                    Heavy document — optional Puter path (OFF by default, threshold ~{PUTER_HEAVY_PAGE_THRESHOLD} pages)
                  </p>
                  <p className="text-sm mb-3" style={{ color: "#acabaa" }}>
                    {puterTarget && puterTarget.pageCount !== undefined && puterTarget.pageCount !== null
                      ? heavyDocumentWarning(puterTarget.pageCount)
                      : `Estimated ~${pageEstimate} pages meets the heavy threshold (~${PUTER_HEAVY_PAGE_THRESHOLD}). Upload the document first, then authorise below — or continue with local OCR.`}
                  </p>
                  {!PUTER_ENABLED && (
                    <p className="text-sm mb-3" style={{ color: "#c6c6c7" }}>
                      Puter is currently disabled on this deployment, so only local OCR runs. No authorisation is requested.
                    </p>
                  )}
                  {PUTER_ENABLED && puterTarget && !puterDone && (
                    <div className="flex flex-col gap-3">
                      <input
                        type="password"
                        value={puterToken}
                        onChange={(e) => setPuterToken(e.target.value)}
                        placeholder="Per-call Puter token (never stored)"
                        autoComplete="off"
                        className="w-full px-4 py-3 rounded-xl text-sm"
                        style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.2)" }}
                      />
                      <div className="flex flex-wrap gap-3">
                        <button
                          onClick={() => void handlePuterAuthorise()}
                          disabled={puterBusy}
                          className="px-6 py-3 rounded-xl text-sm font-semibold transition-all hover:opacity-90 active:scale-95 disabled:opacity-50"
                          style={{ backgroundColor: "#edfee9", color: "#131313", fontFamily: "Inter, sans-serif" }}
                        >
                          {puterBusy ? "Authorising…" : "Authorise Puter processing"}
                        </button>
                        <button
                          onClick={handlePuterDecline}
                          disabled={puterBusy}
                          className="px-6 py-3 rounded-xl text-sm font-semibold transition-all hover:opacity-90 active:scale-95 disabled:opacity-50"
                          style={{ backgroundColor: "#252626", color: "#e7e5e4", fontFamily: "Inter, sans-serif", border: "1px solid rgba(72,72,72,0.2)" }}
                        >
                          Continue with local OCR
                        </button>
                        {(puterBusy || puterJobId) && (
                          <button
                            onClick={() => void handlePuterCancel()}
                            className="px-6 py-3 rounded-xl text-sm font-semibold transition-all hover:opacity-90 active:scale-95"
                            style={{ color: "#ee7d77", border: "1px solid rgba(238,125,119,0.3)" }}
                          >
                            Cancel
                          </button>
                        )}
                      </div>
                      {(puterProgress !== null || puterStage) && (
                        <div>
                          <p className="text-xs mb-1" style={{ color: "#acabaa", fontFamily: "Space Grotesk, sans-serif" }}>
                            {puterStage ? `stage: ${puterStage}` : "authorising…"}{puterProgress !== null ? ` · ${Math.min(100, Math.round(puterProgress))}%` : ""}
                          </p>
                          <div className="w-full h-1 rounded-full overflow-hidden" style={{ backgroundColor: "#252626" }}>
                            <div
                              className="h-full transition-all duration-500"
                              style={{ width: `${puterProgress === null ? 0 : Math.min(100, puterProgress)}%`, backgroundColor: "#c6c6c7" }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                  {puterError && (
                    <p className="text-sm mt-3" style={{ color: "#ff9d9d" }}>{puterError}</p>
                  )}
                  {puterDone && (
                    <p className="text-sm mt-3" style={{ color: "#edfee9" }}>{puterDone}</p>
                  )}
                  {puterDeclinedLocal && !puterError && !puterDone && (
                    <p className="text-sm mt-3" style={{ color: "#acabaa" }}>Local OCR continues — Puter was not called.</p>
                  )}
                </div>
              )}
              {projects.length > 1 && (
                <div className="mt-4">
                  <label className="uppercase tracking-widest block mb-2" style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}>
                    Target project
                  </label>
                  <select
                    value={projectId ?? ""}
                    onChange={(e) => {
                      const id = e.target.value || null;
                      setSelectedProjectId(id);
                      if (id) selectProject(id);
                    }}
                    className="w-full px-4 py-3 rounded-xl text-sm"
                    style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.2)" }}
                  >
                    {projects.map((p) => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </select>
                </div>
              )}
              {projects.length === 0 && (
                <div className="mt-4 p-4 rounded-xl" style={{ backgroundColor: "#1f2020", border: "1px solid rgba(72,72,72,0.15)" }}>
                  <p className="text-sm mb-3" style={{ color: "#acabaa" }}>
                    No projects yet. Create one to unlock uploading.
                  </p>
                  <button
                    onClick={() => setCreateOpen(true)}
                    className="px-6 py-3 rounded-xl text-sm font-semibold transition-all hover:opacity-90 active:scale-95"
                    style={{ backgroundColor: "#edfee9", color: "#131313", fontFamily: "Inter, sans-serif" }}
                  >
                    Create a project
                  </button>
                </div>
              )}
              <CreateProjectDialog
                open={createOpen}
                onOpenChange={setCreateOpen}
                onCreated={(project) => {
                  setSelectedProjectId(project.id);
                  selectProject(project.id);
                }}
              />
            </div>

            {/* Settings — local-only display preferences (not sent to the
                backend; uploads always use server-side validation). */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
              <div className="p-6 rounded-2xl" style={{ backgroundColor: "#131313" }}>
                <span
                  className="uppercase tracking-widest block mb-4"
                  style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
                >
                  Extraction Level (local display preference)
                </span>
                <div className="space-y-3">
                  {[
                    { id: "standard", label: "Standard view" },
                    { id: "deep", label: "Detailed view" },
                  ].map((option) => (
                    <label key={option.id} className="flex items-center gap-3 cursor-pointer">
                      <div
                        className="w-4 h-4 rounded-full flex items-center justify-center transition-all"
                        style={{
                          border: `2px solid ${extractionLevel === option.id ? "#c6c6c7" : "#484848"}`,
                          backgroundColor: extractionLevel === option.id ? "#c6c6c7" : "transparent",
                        }}
                        onClick={() => setExtractionLevel(option.id)}
                      >
                        {extractionLevel === option.id && (
                          <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: "#3f4041" }} />
                        )}
                      </div>
                      <span className="text-sm" style={{ color: extractionLevel === option.id ? "#e7e5e4" : "#acabaa" }}>{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="p-6 rounded-2xl" style={{ backgroundColor: "#131313" }}>
                <span
                  className="uppercase tracking-widest block mb-4"
                  style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
                >
                  Output Format (local display preference)
                </span>
                <div className="space-y-3">
                  {[
                    { id: "flashcards", label: "Flashcards" },
                    { id: "maps", label: "Study Notes" },
                  ].map((option) => (
                    <label key={option.id} className="flex items-center gap-3 cursor-pointer">
                      <div
                        className="w-4 h-4 rounded flex items-center justify-center transition-all cursor-pointer"
                        style={{
                          backgroundColor: outputFormats[option.id as keyof typeof outputFormats] ? "#c6c6c7" : "transparent",
                          border: `2px solid ${outputFormats[option.id as keyof typeof outputFormats] ? "#c6c6c7" : "#484848"}`,
                        }}
                        onClick={() => setOutputFormats(prev => ({ ...prev, [option.id]: !prev[option.id as keyof typeof outputFormats] }))}
                      >
                        {outputFormats[option.id as keyof typeof outputFormats] && (
                          <span className="material-symbols-outlined" style={{ fontSize: "12px", color: "#3f4041", fontVariationSettings: "'FILL' 1" }}>check</span>
                        )}
                      </div>
                      <span className="text-sm" style={{ color: outputFormats[option.id as keyof typeof outputFormats] ? "#e7e5e4" : "#acabaa" }}>{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel */}
          <div className="lg:col-span-4 flex flex-col gap-6">
            {/* Context Window Status */}
            <div
              className="p-6 rounded-2xl"
              style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.1)" }}
            >
              <div className="flex items-center gap-2 mb-4">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: "#edfee9" }} />
                <span
                  className="uppercase tracking-widest"
                  style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
                >
                  System Ready
                </span>
              </div>
              <h3
                className="text-xl font-bold mb-3"
                style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
              >
                Processing Pipeline
              </h3>
              <p className="text-sm leading-relaxed mb-6" style={{ color: "#acabaa" }}>
                Documents are processed on the server (text extraction, chunking, indexing).
                Scanned pages use local OCR; progress above reflects the real job state.
              </p>
              <div className="flex gap-2 flex-wrap">
                {["OCR Enabled", "Latex Support"].map((tag) => (
                  <span
                    key={tag}
                    className="px-3 py-1 rounded text-xs"
                    style={{
                      backgroundColor: "#1f2020",
                      color: "#acabaa",
                      fontFamily: "Space Grotesk, sans-serif",
                      border: "1px solid rgba(72,72,72,0.15)",
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            {/* Recent Files */}
            <div
              className="p-6 rounded-2xl"
              style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.1)" }}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-bold" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>Recent Files</h3>
              </div>
              <div className="space-y-4">
                {recentFiles.length > 0 ? (
                  recentFiles.map((file) => (
                    <div key={file.id} className="flex items-start gap-3 group">
                      <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                        style={{ backgroundColor: "#1f2020" }}
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: "16px", color: "#acabaa" }}>description</span>
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium transition-opacity" style={{ color: "#e7e5e4" }}>{file.name}</p>
                        <p className="text-xs" style={{ color: "#acabaa" }}>
                          {file.time} · {file.indexingState === "partial" ? "partially indexed" : file.status} · {file.sourceType.replaceAll("_", " ")}
                        </p>
                        {/* Provenance: indexing state, page methods, mime, checksum. */}
                        <p className="text-xs mt-1" style={{ color: "#484848", fontFamily: "Space Grotesk, sans-serif" }}>
                          {[file.indexingState ? `indexing: ${file.indexingState}` : null,
                            file.mimeType ? `mime: ${file.mimeType}` : null,
                            file.checksum ? `sha256: ${file.checksum.slice(0, 12)}...` : null,
                            file.pageCountConfirmed !== undefined && file.pageCountConfirmed !== null
                              ? `pages: ${file.pageCountConfirmed}` : null,
                          ].filter(Boolean).join(" · ") || "provenance pending"}
                        </p>
                        {file.pageMethods && Object.keys(file.pageMethods).length > 0 && (
                          <p className="text-xs" style={{ color: "#484848", fontFamily: "Space Grotesk, sans-serif" }}>
                            methods: {Object.entries(file.pageMethods).map(([k, v]) =>
                              typeof v === "string" ? `${k}: ${v}` : `${k}: ${v.method ?? "?"}${v.count !== undefined ? ` x${v.count}` : ""}`
                            ).join(", ")}
                          </p>
                        )}
                        {file.pageCount !== undefined && file.pageCount !== null &&
                          shouldWarnHeavyDocument(file.pageCount) && (
                          <p className="text-xs mt-1" style={{ color: "#c6c6c7" }}>
                            {heavyDocumentWarning(file.pageCount)}
                          </p>
                        )}
                        {/* PDF viewer (Phase B) + per-document delete. */}
                        <div className="mt-2 flex flex-wrap items-center gap-2">
                          {(file.fileType ?? "").toLowerCase() === "pdf" && (
                            <button
                              onClick={() => navigate(`/documents/${file.id}`)}
                              className="px-4 py-1.5 rounded-lg text-xs uppercase tracking-widest transition-all hover:opacity-90"
                              style={{
                                fontFamily: "Space Grotesk, sans-serif",
                                color: "#e7e5e4",
                                border: "1px solid rgba(72,72,72,0.3)",
                              }}
                            >
                              Open
                            </button>
                          )}
                          <DeleteDocumentButton
                            documentId={file.id}
                            fileName={file.name}
                            compact
                            onDeleted={handleDocumentDeleted}
                          />
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-sm" style={{ color: "#acabaa" }}>No files uploaded yet</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
