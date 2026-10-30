import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Layout from "@/components/Layout";
import {
  generateSyllabus,
  generatePyq,
  generateMapping,
  generateAnalysis,
  generateMetrics,
  readSyllabus,
  readPyq,
  readMapping,
  readAnalysis,
  readInsights,
  type SyllabusContent,
  type PYQContent,
  type MappingContent,
  type AnalysisContent,
} from "@/lib/api-services";
import { useActiveProject, ProjectSelector } from "@/lib/use-active-project";

type TabId = "syllabus" | "pyq" | "mapping" | "analysis" | "insights";

const TABS: Array<{ id: TabId; label: string }> = [
  { id: "syllabus", label: "Syllabus" },
  { id: "pyq", label: "PYQs" },
  { id: "mapping", label: "Mapping" },
  { id: "analysis", label: "Analysis" },
  { id: "insights", label: "Insights" },
];

function EmptyState({ message, action }: { message: string; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center rounded-xl border border-dashed border-neutral-700">
      <span className="material-symbols-outlined text-5xl mb-4 text-neutral-500">
        inbox
      </span>
      <p className="text-neutral-400 max-w-md">{message}</p>
      {action && <div className="mt-6">{action}</div>}
    </div>
  );
}

function GenerateButton({
  label,
  pending,
  onClick,
}: {
  label: string;
  pending: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={pending}
      className="px-5 py-2.5 rounded-lg font-medium bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white"
    >
      {pending ? "Working..." : label}
    </button>
  );
}

export default function Academic() {
  const [tab, setTab] = useState<TabId>("syllabus");
  const queryClient = useQueryClient();

  const { projects, activeProject, activeProjectId: projectId, selectProject } =
    useActiveProject();

  const syllabusQuery = useQuery({
    queryKey: ["academic-syllabus", projectId],
    queryFn: () => readSyllabus(projectId as string),
    enabled: !!projectId,
  });
  const pyqQuery = useQuery({
    queryKey: ["academic-pyq", projectId],
    queryFn: () => readPyq(projectId as string),
    enabled: !!projectId,
  });
  const mappingQuery = useQuery({
    queryKey: ["academic-mapping", projectId],
    queryFn: () => readMapping(projectId as string),
    enabled: !!projectId,
  });
  const analysisQuery = useQuery({
    queryKey: ["academic-analysis", projectId],
    queryFn: () => readAnalysis(projectId as string),
    enabled: !!projectId,
  });
  const insightsQuery = useQuery({
    queryKey: ["academic-insights", projectId],
    queryFn: () => readInsights(projectId as string),
    enabled: !!projectId && tab === "insights",
  });

  const refreshAll = () => {
    queryClient.invalidateQueries({ queryKey: ["academic-syllabus"] });
    queryClient.invalidateQueries({ queryKey: ["academic-pyq"] });
    queryClient.invalidateQueries({ queryKey: ["academic-mapping"] });
    queryClient.invalidateQueries({ queryKey: ["academic-analysis"] });
    queryClient.invalidateQueries({ queryKey: ["academic-insights"] });
  };

  const syllabusMutation = useMutation({
    mutationFn: () => generateSyllabus({ projectId: projectId as string }),
    onSuccess: refreshAll,
  });
  const pyqMutation = useMutation({
    mutationFn: () => generatePyq({ projectId: projectId as string }),
    onSuccess: refreshAll,
  });
  const mappingMutation = useMutation({
    mutationFn: () => generateMapping({ projectId: projectId as string }),
    onSuccess: refreshAll,
  });
  const analysisMutation = useMutation({
    mutationFn: () => generateAnalysis({ projectId: projectId as string }),
    onSuccess: refreshAll,
  });
  const metricsMutation = useMutation({
    mutationFn: () => generateMetrics({ projectId: projectId as string }),
    onSuccess: refreshAll,
  });

  const latest = <T,>(read: { state: string; total: number } & Record<string, unknown>, key: string): T | null => {
    if (!read || read.state !== "ready" || read.total === 0) return null;
    const items = (read[key] ?? []) as Array<{ content: T }>;
    return items.length > 0 ? items[0].content : null;
  };

  const syllabus = latest<SyllabusContent>(syllabusQuery.data as never, "syllabi");
  const pyqs = latest<PYQContent>(pyqQuery.data as never, "pyq_sets");
  const mapping = latest<MappingContent>(mappingQuery.data as never, "mappings");
  const analysis = latest<AnalysisContent>(analysisQuery.data as never, "analyses");

  const renderSyllabus = () => {
    if (syllabusQuery.isLoading) return <p className="text-neutral-400">Loading syllabus...</p>;
    if (!syllabus) {
      return (
        <EmptyState
          message="No syllabus extracted yet for this project. Upload syllabus or course material, then extract units and topics."
          action={
            <GenerateButton
              label="Extract syllabus"
              pending={syllabusMutation.isPending}
              onClick={() => syllabusMutation.mutate()}
            />
          }
        />
      );
    }
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">{syllabus.title}</h2>
          <GenerateButton
            label="Re-extract"
            pending={syllabusMutation.isPending}
            onClick={() => syllabusMutation.mutate()}
          />
        </div>
        {syllabus.units.map((unit) => (
          <div key={unit.unit_id} className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4">
            <h3 className="font-medium">
              {unit.title}{" "}
              <span className="text-xs text-neutral-500">({unit.unit_id})</span>
            </h3>
            {unit.topics.length > 0 && (
              <ul className="mt-2 list-disc list-inside text-sm text-neutral-300">
                {unit.topics.map((topic, i) => (
                  <li key={i}>{topic}</li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderPyq = () => {
    if (pyqQuery.isLoading) return <p className="text-neutral-400">Loading PYQs...</p>;
    if (!pyqs || pyqs.questions.length === 0) {
      return (
        <EmptyState
          message="No previous-year questions extracted yet. Upload question papers, then extract questions with year and marks."
          action={
            <GenerateButton
              label="Extract PYQs"
              pending={pyqMutation.isPending}
              onClick={() => pyqMutation.mutate()}
            />
          }
        />
      );
    }
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">
            {pyqs.title} ({pyqs.questions.length} questions)
          </h2>
          <GenerateButton
            label="Re-extract"
            pending={pyqMutation.isPending}
            onClick={() => pyqMutation.mutate()}
          />
        </div>
        {pyqs.questions.map((q, i) => (
          <div key={i} className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4">
            <p className="text-sm">{q.question}</p>
            <div className="mt-2 flex flex-wrap gap-2 text-xs">
              <span className="px-2 py-0.5 rounded bg-neutral-800 text-neutral-300">{q.year}</span>
              <span className="px-2 py-0.5 rounded bg-neutral-800 text-neutral-300">{q.marks} marks</span>
              {q.unit_id && (
                <span className="px-2 py-0.5 rounded bg-indigo-900/60 text-indigo-200">{q.unit_id}</span>
              )}
              {q.topic && (
                <span className="px-2 py-0.5 rounded bg-neutral-800 text-neutral-400">{q.topic}</span>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderMapping = () => {
    if (mappingQuery.isLoading) return <p className="text-neutral-400">Loading mapping...</p>;
    if (!mapping || mapping.mappings.length === 0) {
      return (
        <EmptyState
          message={
            !syllabus
              ? "Mapping needs an extracted syllabus first."
              : !pyqs
                ? "Mapping needs extracted PYQs first."
                : "No mapping computed yet. Mapping assigns each PYQ to its best syllabus unit."
          }
          action={
            syllabus && pyqs ? (
              <GenerateButton
                label="Compute mapping"
                pending={mappingMutation.isPending}
                onClick={() => mappingMutation.mutate()}
              />
            ) : undefined
          }
        />
      );
    }
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">
            {mapping.syllabus_title} - {mapping.coverage_pct}% mapped, {mapping.unmapped_count} unmapped
          </h2>
          <GenerateButton
            label="Recompute"
            pending={mappingMutation.isPending}
            onClick={() => mappingMutation.mutate()}
          />
        </div>
        {mapping.mappings.map((m) => (
          <div key={m.question_index} className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4">
            <p className="text-sm">{m.question}</p>
            <div className="mt-2 flex flex-wrap gap-2 text-xs">
              {m.unit_id ? (
                <span className="px-2 py-0.5 rounded bg-emerald-900/60 text-emerald-200">
                  {m.unit_title} ({m.unit_id})
                </span>
              ) : (
                <span className="px-2 py-0.5 rounded bg-amber-900/60 text-amber-200">Unmapped</span>
              )}
              <span className="px-2 py-0.5 rounded bg-neutral-800 text-neutral-400">
                {(m.confidence * 100).toFixed(0)}% - {m.method.replace("_", " ")}
              </span>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderInsights = () => {
    const insights = insightsQuery.data;
    if (insightsQuery.isLoading) return <p className="text-neutral-400">Loading insights...</p>;
    if (!insights || insights.state !== "ready") {
      return (
        <EmptyState
          message={
            insights?.message ??
            "No study insights computed yet. Insights summarise per-topic frequency, marks share, hotspots and gaps from your syllabus, documents and past questions."
          }
          action={
            <GenerateButton
              label="Generate insights"
              pending={metricsMutation.isPending}
              onClick={() => metricsMutation.mutate()}
            />
          }
        />
      );
    }
    return (
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Study insights</h2>
        {insights.cards.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {insights.cards.map((card, i) => (
              <div key={i} className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4">
                <h3 className="text-sm font-bold uppercase tracking-wider mb-2 text-neutral-200">
                  {String(card.card).replace("_", " ")}
                </h3>
                {Object.entries(card)
                  .filter(([key]) => key !== "card")
                  .map(([key, value]) => (
                    <p key={key} className="text-sm text-neutral-400">
                      {key.replace(/_/g, " ")}:{" "}
                      {Array.isArray(value) ? value.join(", ") || "none" : String(value ?? "—")}
                    </p>
                  ))}
              </div>
            ))}
          </div>
        )}
        {insights.table.length > 0 && (
          <div className="rounded-xl border border-neutral-800 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-neutral-400 border-b border-neutral-800">
                  <th className="px-4 py-2 font-medium">Topic</th>
                  <th className="px-4 py-2 font-medium">Unit</th>
                  <th className="px-4 py-2 font-medium">Frequency</th>
                  <th className="px-4 py-2 font-medium">Marks</th>
                  <th className="px-4 py-2 font-medium">Share</th>
                  <th className="px-4 py-2 font-medium">Hotspot</th>
                </tr>
              </thead>
              <tbody>
                {insights.table.map((row, i) => (
                  <tr key={i} className="border-b border-neutral-800/50 text-neutral-300">
                    <td className="px-4 py-2">{row.topic}</td>
                    <td className="px-4 py-2 text-neutral-400">{row.unit}</td>
                    <td className="px-4 py-2">{row.frequency}</td>
                    <td className="px-4 py-2">{row.marks_total}</td>
                    <td className="px-4 py-2">{row.marks_share}%</td>
                    <td className="px-4 py-2">{row.hotspot_rank > 0 ? `#${row.hotspot_rank}` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {insights.never_asked.length > 0 && (
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4">
            <h3 className="text-sm font-bold uppercase tracking-wider mb-2 text-neutral-200">
              Never asked
            </h3>
            <p className="text-sm text-neutral-400">{insights.never_asked.join(", ")}</p>
          </div>
        )}
        {insights.narrative && (
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4">
            <p className="text-sm text-neutral-300">{insights.narrative}</p>
          </div>
        )}
        {insights.provenance_note && (
          <p className="text-xs text-neutral-500">grounding: {insights.provenance_note}</p>
        )}
      </div>
    );
  };

  const renderAnalysis = () => {
    if (analysisQuery.isLoading) return <p className="text-neutral-400">Loading analysis...</p>;
    if (!analysis || analysis.frequency.length === 0) {
      return (
        <EmptyState
          message="No analysis yet. Analysis shows per-unit question frequency and marks alignment once PYQs are mapped."
          action={
            mapping ? (
              <GenerateButton
                label="Compute analysis"
                pending={analysisMutation.isPending}
                onClick={() => analysisMutation.mutate()}
              />
            ) : undefined
          }
        />
      );
    }
    const maxCount = Math.max(1, ...analysis.frequency.map((f) => f.count));
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">
            {analysis.syllabus_title} - {analysis.total_questions} questions, {analysis.total_marks} marks
          </h2>
          <GenerateButton
            label="Recompute"
            pending={analysisMutation.isPending}
            onClick={() => analysisMutation.mutate()}
          />
        </div>
        <p className="text-sm text-neutral-400">
          {analysis.mapped_questions} mapped, {analysis.unmapped_questions} unmapped (
          {analysis.coverage_pct}% coverage)
        </p>
        <div className="space-y-3">
          {analysis.frequency.map((f) => (
            <div key={f.unit_id} className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">
                  {f.unit_title} <span className="text-xs text-neutral-500">({f.unit_id})</span>
                </span>
                <span className="text-neutral-400">
                  {f.count} questions - {f.marks_total} marks - {f.share_pct}%
                </span>
              </div>
              <div className="mt-2 h-2 rounded bg-neutral-800">
                <div
                  className="h-2 rounded bg-indigo-500"
                  style={{ width: `${(f.count / maxCount) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <Layout activeNav="academic">
      <div className="pb-20 md:pb-8 px-6 md:px-12 min-h-screen pt-10 max-w-5xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">Academic Insights</h1>
        <p className="text-neutral-400 mb-4">
          Syllabus extraction, previous-year questions, unit mapping, and marks
          alignment - all grounded in your uploaded documents.
        </p>
        <div className="mb-6">
          <ProjectSelector
            projects={projects}
            activeProjectId={projectId}
            onSelect={selectProject}
          />
        </div>

        {!projectId ? (
          <EmptyState message="No project yet. Upload study material to create a project first." />
        ) : (
          <>
            <div className="flex gap-2 mb-6 border-b border-neutral-800 pb-2">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`px-4 py-2 rounded-t-lg font-medium ${
                    tab === t.id
                      ? "bg-neutral-800 text-white"
                      : "text-neutral-400 hover:text-white"
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {tab === "syllabus" && renderSyllabus()}
            {tab === "pyq" && renderPyq()}
            {tab === "mapping" && renderMapping()}
            {tab === "analysis" && renderAnalysis()}
            {tab === "insights" && renderInsights()}
          </>
        )}
      </div>
    </Layout>
  );
}
