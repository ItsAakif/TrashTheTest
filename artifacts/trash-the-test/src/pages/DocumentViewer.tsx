import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
// Framework-agnostic PDF rendering straight on pdf.js v6. The old viewer
// library stack declared a pdfjs-dist <= 3 peer range and is not React 19
// compatible, so it was removed entirely.
import * as pdfjsLib from "pdfjs-dist";
import type { PDFDocumentProxy, RenderTask } from "pdfjs-dist";
// Worker bundled by Vite (?url) so the viewer works offline and in Docker
// without a CDN.
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import "@/lib/pdf-text-layer.css";
import Layout from "@/components/Layout";
import { SpinnerIcon } from "@/components/icons";
import {
  fetchDocumentBytes,
  fetchStudyNotesPdf,
  getDocument,
  readStudyNotesPdfMeta,
} from "@/lib/api-services";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 4;
const ZOOM_STEP = 0.25;
// Tailwind `p-4` on the page frame (16px each side); subtract it when fitting.
const PAGE_FRAME_GUTTER = 32;

const round2 = (n: number): number => Math.round(n * 100) / 100;

const controlStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minWidth: "2rem",
  height: "2rem",
  padding: "0 0.5rem",
  borderRadius: "0.5rem",
  backgroundColor: "#252626",
  color: "#e7e5e4",
  border: "1px solid #333333",
  fontSize: "0.875rem",
  lineHeight: 1,
};

interface DocumentViewerProps {
  params?: { id?: string };
}

export default function DocumentViewer({ params }: DocumentViewerProps) {
  const documentId = params?.id ?? null;
  const [location, navigate] = useLocation();

  // Study-notes artifact mode: /documents/study-notes-{projectId} opens
  // the GENERATED study-notes PDF in this SAME viewer (no second viewer).
  // Bytes come from the academic PDF endpoint; ?page= citation jumps keep
  // working exactly like source documents.
  const STUDY_NOTES_PREFIX = "study-notes-";
  const isStudyNotes = (documentId ?? "").startsWith(STUDY_NOTES_PREFIX);
  const studyNotesProjectId = isStudyNotes
    ? (documentId as string).slice(STUDY_NOTES_PREFIX.length)
    : null;

  // Citation jumps land here as /documents/:id?page=N (1-based, from the
  // chunk's page provenance). Derive the target from the reactive location
  // rather than reading window.location once during mount; a citation link
  // can update the query while this viewer remains mounted.
  const citationPage = (() => {
    const query = location.includes("?") ? location.slice(location.indexOf("?") + 1) : "";
    const raw = new URLSearchParams(query).get("page");
    const parsed = raw !== null ? Number.parseInt(raw, 10) : Number.NaN;
    return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : 1;
  })();

  // Study-notes notes are versioned: /documents/study-notes-{projectId}?version=N
  // opens THAT saved note instead of only the newest one. No version means the
  // latest READY version (unchanged behaviour).
  const studyNotesVersion = (() => {
    const query = location.includes("?") ? location.slice(location.indexOf("?") + 1) : "";
    const raw = new URLSearchParams(query).get("version");
    const parsed = raw !== null ? Number.parseInt(raw, 10) : Number.NaN;
    return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : null;
  })();

  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [fetching, setFetching] = useState(true);
  const [reloadKey, setReloadKey] = useState(0);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [pageNumber, setPageNumber] = useState(citationPage);
  const [pageInput, setPageInput] = useState(String(citationPage));
  const [zoom, setZoom] = useState(1);
  const [fitWidth, setFitWidth] = useState(false);
  const [renderedScale, setRenderedScale] = useState(1);
  const [containerWidth, setContainerWidth] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const textLayerContainerRef = useRef<HTMLDivElement | null>(null);
  const pageFrameRef = useRef<HTMLDivElement | null>(null);
  const pdfDocRef = useRef<PDFDocumentProxy | null>(null);
  const renderTokenRef = useRef(0);
  const renderTaskRef = useRef<RenderTask | null>(null);
  const textLayerRef = useRef<{ cancel(): void } | null>(null);
  const appliedCitationPageRef = useRef(citationPage);

  // Fetch the JWT-scoped bytes (never in a URL) ONCE per (documentId,
  // reloadKey), parse the PDF ONCE, and store the PDFDocumentProxy. Page
  // navigation and zoom never refetch bytes and never re-parse the document.
  useEffect(() => {
    if (!documentId) {
      setFetching(false);
      setPdfDoc(null);
      pdfDocRef.current = null;
      return;
    }
    let cancelled = false;
    setFetching(true);
    setFetchError(null);
    setRenderError(null);
    setPdfDoc(null);
    pdfDocRef.current = null;

    const load =
      isStudyNotes && studyNotesProjectId
        ? fetchStudyNotesPdf(studyNotesProjectId, studyNotesVersion)
        : fetchDocumentBytes(documentId);

    load
      .then((blob) => blob.arrayBuffer())
      .then((buffer) => {
        if (cancelled) return null;
        // `pdf.js` takes ownership of the typed array.
        return pdfjsLib.getDocument({ data: new Uint8Array(buffer) }).promise;
      })
      .then((doc) => {
        if (!doc) return;
        if (cancelled) {
          doc.loadingTask.destroy().catch(() => {});
          return;
        }
        pdfDocRef.current = doc;
        setPdfDoc(doc);
        setFetching(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setFetchError(err instanceof Error ? err.message : "Document unavailable.");
        setFetching(false);
      });

    return () => {
      cancelled = true;
      renderTokenRef.current += 1;
      renderTaskRef.current?.cancel();
      renderTaskRef.current = null;
      textLayerRef.current?.cancel();
      textLayerRef.current = null;
      const doc = pdfDocRef.current;
      if (doc) {
        doc.loadingTask.destroy().catch(() => {});
        pdfDocRef.current = null;
      }
    };
  }, [documentId, reloadKey, isStudyNotes, studyNotesProjectId, studyNotesVersion]);

  // Per-document UI state reset when the document identity changes.
  useEffect(() => {
    setPageNumber(citationPage);
    setZoom(1);
    setFitWidth(false);
    appliedCitationPageRef.current = citationPage;
    // Intentionally keyed on document identity only: a later ?page change
    // jumps through the citation effect below without resetting zoom.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentId, reloadKey]);

  // A later ?page= change (a citation link clicked while this viewer stays
  // mounted) updates the current page without a reload.
  useEffect(() => {
    if (!pdfDoc) return;
    if (citationPage === appliedCitationPageRef.current) return;
    appliedCitationPageRef.current = citationPage;
    setPageNumber(Math.min(Math.max(1, citationPage), pdfDoc.numPages));
  }, [citationPage, pdfDoc]);

  // Keep the page in range once the real page count is known.
  useEffect(() => {
    if (!pdfDoc) return;
    setPageNumber((p) => Math.min(Math.max(1, p), pdfDoc.numPages));
  }, [pdfDoc]);

  useEffect(() => {
    setPageInput(String(pageNumber));
  }, [pageNumber]);

  // Measure the frame while fit-width is active so resize re-fits.
  useEffect(() => {
    if (!fitWidth) return;
    const measure = () => setContainerWidth(pageFrameRef.current?.clientWidth ?? 0);
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [fitWidth, pdfDoc]);

  // Render the current page into the canvas + a selectable pdf.js text layer.
  // Re-runs on [pdfDoc, pageNumber, zoom, fitWidth, containerWidth] only.
  useEffect(() => {
    if (!pdfDoc) return;
    const canvas = canvasRef.current;
    const textContainer = textLayerContainerRef.current;
    if (!canvas || !textContainer) return;

    const token = ++renderTokenRef.current;
    let cancelled = false;

    renderTaskRef.current?.cancel();
    renderTaskRef.current = null;
    textLayerRef.current?.cancel();
    textLayerRef.current = null;

    const run = async () => {
      try {
        const page = await pdfDoc.getPage(pageNumber);
        if (cancelled || token !== renderTokenRef.current) return;

        const baseViewport = page.getViewport({ scale: 1 });
        const available = (pageFrameRef.current?.clientWidth ?? 0) - PAGE_FRAME_GUTTER;
        const scale =
          fitWidth && available > 0
            ? Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, available / baseViewport.width))
            : zoom;
        const viewport = page.getViewport({ scale });
        setRenderedScale(scale);

        const outputScale = window.devicePixelRatio || 1;
        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);
        canvas.style.width = `${Math.floor(viewport.width)}px`;
        canvas.style.height = `${Math.floor(viewport.height)}px`;

        const renderTask = page.render({
          canvas,
          viewport,
          transform:
            outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : undefined,
        });
        renderTaskRef.current = renderTask;
        await renderTask.promise;
        if (cancelled || token !== renderTokenRef.current) return;
        renderTaskRef.current = null;

        // pdf.js v6 sizes the text layer from custom properties the viewer
        // normally sets on the page element; set them before constructing.
        textContainer.replaceChildren();
        textContainer.style.setProperty("--total-scale-factor", String(viewport.scale));
        textContainer.style.setProperty("--scale-round-x", "1px");
        textContainer.style.setProperty("--scale-round-y", "1px");

        const textContent = await page.getTextContent();
        if (cancelled || token !== renderTokenRef.current) return;

        const textLayer = new pdfjsLib.TextLayer({
          textContentSource: textContent,
          container: textContainer,
          viewport,
        });
        textLayerRef.current = textLayer;
        await textLayer.render();
        if (cancelled || token !== renderTokenRef.current) return;
        textLayerRef.current = null;

        setRenderError(null);
      } catch (err) {
        if (cancelled || token !== renderTokenRef.current) return;
        // Cancellation is an expected outcome of rapid page changes.
        if ((err as { name?: string } | null)?.name === "RenderingCancelledException") return;
        setRenderError(err instanceof Error ? err.message : "Could not render this PDF.");
      }
    };

    void run();

    return () => {
      cancelled = true;
      renderTaskRef.current?.cancel();
      renderTaskRef.current = null;
      textLayerRef.current?.cancel();
      textLayerRef.current = null;
    };
  }, [pdfDoc, pageNumber, zoom, fitWidth, containerWidth]);

  // Prefetch neighbouring pages so prev/next swap without a parse stall.
  useEffect(() => {
    if (!pdfDoc) return;
    for (const n of [pageNumber - 1, pageNumber + 1]) {
      if (n >= 1 && n <= pdfDoc.numPages) {
        pdfDoc.getPage(n).catch(() => {});
      }
    }
  }, [pdfDoc, pageNumber]);

  // Wrap on the first/last page instead of erroring.
  const goToPrev = useCallback(() => {
    setPageNumber((p) => {
      const total = pdfDocRef.current?.numPages ?? 1;
      return p <= 1 ? total : p - 1;
    });
  }, []);
  const goToNext = useCallback(() => {
    setPageNumber((p) => {
      const total = pdfDocRef.current?.numPages ?? 1;
      return p >= total ? 1 : p + 1;
    });
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        goToPrev();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        goToNext();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [goToPrev, goToNext]);

  const commitPageInput = useCallback(() => {
    const parsed = Number.parseInt(pageInput, 10);
    const total = pdfDocRef.current?.numPages ?? 1;
    if (Number.isFinite(parsed)) {
      setPageNumber(Math.min(Math.max(1, parsed), total));
    } else {
      setPageInput(String(pageNumber));
    }
  }, [pageInput, pageNumber]);

  const onPageInputKeyDown = (e: ReactKeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitPageInput();
    }
  };

  const {
    data: document,
    isLoading: metaLoading,
    isError: metaError,
  } = useQuery({
    queryKey: ["document", documentId],
    queryFn: () => getDocument(documentId as string),
    enabled: !!documentId && !isStudyNotes,
  });

  const {
    data: studyNotesMeta,
    isLoading: studyNotesMetaLoading,
    isError: studyNotesMetaError,
  } = useQuery({
    queryKey: ["study-notes-pdf", studyNotesProjectId, studyNotesVersion ?? "latest"],
    queryFn: () =>
      readStudyNotesPdfMeta(studyNotesProjectId as string, studyNotesVersion),
    enabled: isStudyNotes && !!studyNotesProjectId,
  });

  const retry = () => {
    // Re-runs the fetch/parse effect in place (no full-page reload).
    setFetchError(null);
    setRenderError(null);
    setReloadKey((k) => k + 1);
  };

  const isPdf = isStudyNotes || (document?.fileType ?? "pdf").toLowerCase() === "pdf";

  // Header copy: generated artifact metadata (topics/pages/coverage/date/
  // model) instead of document provenance — same title bar, no second UI.
  const headerTitle = isStudyNotes
    ? (studyNotesMeta?.state === "ready" ? (studyNotesMeta.title ?? "Study Notes") : "Study Notes")
    : (metaLoading ? "Loading document..." : document?.fileName ?? "Document");
  const coverage = studyNotesMeta?.state === "ready" ? studyNotesMeta.coverage : undefined;
  const headerSubtitle = isStudyNotes
    ? [
        studyNotesMeta?.topics !== undefined && studyNotesMeta?.topics !== null
          ? `${studyNotesMeta.topics} topics`
          : null,
        studyNotesMeta?.pages !== undefined && studyNotesMeta?.pages !== null
          ? `${studyNotesMeta.pages} pages`
          : null,
        coverage?.topic_coverage !== undefined && coverage?.topic_coverage !== null
          ? `coverage ${Math.round(coverage.topic_coverage * 100)}%`
          : null,
        studyNotesMeta?.model ? `model: ${studyNotesMeta.model}` : null,
        studyNotesMeta?.created_at ? studyNotesMeta.created_at.slice(0, 10) : null,
        studyNotesMeta?.version !== undefined ? `v${studyNotesMeta.version}` : null,
      ]
        .filter(Boolean)
        .join(" · ") || "Generated study notes"
    : `${document?.pageCount !== undefined && document?.pageCount !== null
      ? `${document.pageCount} pages · `
      : ""}${document?.processingStatus ? `status: ${document.processingStatus}` : ""}${
      document?.indexingState ? ` · indexing: ${document.indexingState}` : ""
    }`;
  const headerLoading = isStudyNotes ? studyNotesMetaLoading : metaLoading;
  const headerError = isStudyNotes ? studyNotesMetaError : metaError;
  const backTarget = isStudyNotes && studyNotesProjectId ? `/study/${studyNotesProjectId}` : "/upload";
  const backLabel = isStudyNotes ? "Back to study" : "Back to study material";

  const displayZoom = Math.round((fitWidth ? renderedScale : zoom) * 100);

  return (
    <Layout activeNav="upload">
      <div className="pb-20 md:pb-8 px-6 md:px-12 min-h-screen pt-10">
        <button
          onClick={() => navigate(backTarget)}
          className="mb-6 text-sm flex items-center gap-2"
          style={{ color: "#acabaa" }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
            arrow_back
          </span>
          {backLabel}
        </button>

        {/* Title bar: real document name + state, never a placeholder. */}
        <header className="mb-6">
          <h1
            className="text-2xl md:text-3xl font-extrabold tracking-tight mb-2"
            style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
          >
            {headerLoading ? "Loading..." : headerTitle}
          </h1>
          <p className="text-sm" style={{ color: "#acabaa" }}>
            {headerSubtitle}
          </p>
          {headerError && (
            <p className="text-sm mt-2" style={{ color: "#ee7d77" }}>
              Could not load details. Check your connection and try again.
            </p>
          )}
        </header>

        {!documentId && (
          <div className="p-8 rounded-2xl text-center" style={{ backgroundColor: "#131313" }}>
            <p style={{ color: "#acabaa" }}>No document selected.</p>
          </div>
        )}

        {documentId && !isPdf && !metaLoading && document && (
          <div className="p-8 rounded-2xl text-center" style={{ backgroundColor: "#131313" }}>
            <p style={{ color: "#acabaa" }}>
              Preview is available for PDF files. This file is{" "}
              {document.fileType.toUpperCase()} and cannot be rendered here.
            </p>
          </div>
        )}

        {documentId && (isPdf || metaLoading || !document) && (
          <div className="rounded-2xl overflow-hidden" style={{ backgroundColor: "#131313" }}>
            {fetching || fetchError || !pdfDoc ? (
              <div className="p-12 text-center" style={{ minHeight: "60vh" }}>
                {fetching ? (
                  <>
                    <span className="block mb-4" style={{ display: "flex", justifyContent: "center" }}>
                      <SpinnerIcon size={32} color="#acabaa" />
                    </span>
                    <p style={{ color: "#acabaa" }}>Loading document...</p>
                  </>
                ) : fetchError ? (
                  <>
                    <p className="mb-4" style={{ color: "#ee7d77" }}>
                      {fetchError}
                    </p>
                    <button
                      onClick={retry}
                      className="px-6 py-3 rounded-xl text-sm font-semibold"
                      style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
                    >
                      Retry
                    </button>
                  </>
                ) : (
                  <SpinnerIcon size={32} color="#acabaa" />
                )}
              </div>
            ) : (
              <div>
                {/* Toolbar: page navigation, zoom and fit-width. Keyboard
                    arrows (left/right) wrap on the first/last page. */}
                <div
                  className="flex flex-wrap items-center gap-2 px-3 py-2"
                  style={{ borderBottom: "1px solid #252626" }}
                >
                  <button
                    onClick={goToPrev}
                    aria-label="Previous page"
                    style={controlStyle}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                      chevron_left
                    </span>
                  </button>
                  <div className="flex items-center gap-2 text-sm" style={{ color: "#acabaa" }}>
                    <input
                      value={pageInput}
                      onChange={(e) => setPageInput(e.target.value)}
                      onBlur={commitPageInput}
                      onKeyDown={onPageInputKeyDown}
                      inputMode="numeric"
                      aria-label="Page number"
                      style={{
                        width: "3.5rem",
                        height: "2rem",
                        textAlign: "center",
                        borderRadius: "0.5rem",
                        backgroundColor: "#0b0b0b",
                        border: "1px solid #333333",
                        color: "#e7e5e4",
                      }}
                    />
                    <span aria-live="polite">of {pdfDoc.numPages}</span>
                  </div>
                  <button
                    onClick={goToNext}
                    aria-label="Next page"
                    style={controlStyle}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                      chevron_right
                    </span>
                  </button>

                  <div className="flex items-center gap-2 ml-auto">
                    <button
                      onClick={() => {
                        setFitWidth(false);
                        setZoom((z) => Math.max(MIN_ZOOM, round2(z - ZOOM_STEP)));
                      }}
                      aria-label="Zoom out"
                      style={controlStyle}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                        zoom_out
                      </span>
                    </button>
                    <span
                      aria-live="polite"
                      style={{ color: "#acabaa", minWidth: "3rem", textAlign: "center", fontSize: "0.875rem" }}
                    >
                      {displayZoom}%
                    </span>
                    <button
                      onClick={() => {
                        setFitWidth(false);
                        setZoom((z) => Math.min(MAX_ZOOM, round2(z + ZOOM_STEP)));
                      }}
                      aria-label="Zoom in"
                      style={controlStyle}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                        zoom_in
                      </span>
                    </button>
                    <button
                      onClick={() => setFitWidth((f) => !f)}
                      aria-pressed={fitWidth}
                      aria-label="Fit width"
                      style={{
                        ...controlStyle,
                        backgroundColor: fitWidth ? "#3a3a3a" : "#252626",
                        borderColor: fitWidth ? "#5a5a5a" : "#333333",
                      }}
                    >
                      Fit width
                    </button>
                  </div>
                </div>

                <div
                  ref={pageFrameRef}
                  className="overflow-auto p-4"
                  style={{ height: "80vh", backgroundColor: "#0b0b0b" }}
                >
                  <div className="mx-auto" style={{ position: "relative", width: "fit-content" }}>
                    <canvas ref={canvasRef} style={{ display: "block" }} />
                    <div ref={textLayerContainerRef} className="pdf-text-layer" />
                  </div>
                  {renderError && (
                    <div className="py-10 text-center max-w-md mx-auto">
                      <p className="mb-4" style={{ color: "#ee7d77" }}>
                        {renderError}
                      </p>
                      <button
                        onClick={retry}
                        className="px-6 py-3 rounded-xl text-sm font-semibold"
                        style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
                      >
                        Retry
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </Layout>
  );
}
