import { useState } from "react";
import { useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import Layout from "@/components/Layout";
import MagicCard from "@/components/MagicCard";
import { requestOpenSwitcher } from "@/components/ProjectSwitcher";
import { DeleteDocumentButton } from "@/components/DeleteDocumentButton";
import { getProjectDocuments, type Project, type Document } from "@/lib/api-services";
import { useActiveProject } from "@/lib/use-active-project";

interface ActivityItem {
  id: string;
  icon: string;
  title: string;
  subtitle: string;
  badge: string;
  badgeColor: string;
  badgeBg: string;
}

export default function Dashboard() {
  const [, navigate] = useLocation();
  const [searchValue, setSearchValue] = useState("");
  // Tag filter wired to real document statuses (not decorative).
  const [activeTag, setActiveTag] = useState("All");
  // Creation lives ONLY in the navbar project switcher (single path):
  // this page fires the open-switcher event instead of owning a dialog.

  // Shared project context (last-selected project, never a blind [0]).
  // The ["projects"] query inside the hook is the single cached source.
  const { activeProject, activeProjectId, projectsLoading: isLoading, projectsError: isError } =
    useActiveProject();

  // Fetch documents for the active project
  const { data: documents = [] } = useQuery({
    queryKey: ["documents", activeProjectId],
    queryFn: () => activeProjectId ? getProjectDocuments(activeProjectId) : Promise.resolve([]),
    enabled: !!activeProjectId,
    staleTime: 30000,
  });

  // Build recent activity from real documents, filtered by the live
  // search box + status tag filter (both wired, never decorative).
  const recentActivity: ActivityItem[] = documents
    .filter((doc: Document) => {
      const q = searchValue.trim().toLowerCase();
      if (q && !doc.fileName.toLowerCase().includes(q)) return false;
      if (activeTag === "All") return true;
      return doc.processingStatus === activeTag.toLowerCase();
    })
    .slice(0, 5).map((doc: Document) => {
    const statusMap: Record<string, { badge: string; color: string; bg: string }> = {
      completed: { badge: "PROCESSED", color: "#8ca1aa", bg: "rgba(42,62,70,0.3)" },
      processing: { badge: "PROCESSING", color: "#edfee9", bg: "rgba(237,254,233,0.1)" },
      pending: { badge: "PENDING", color: "#c6c6c7", bg: "rgba(198,198,199,0.1)" },
      failed: { badge: "FAILED", color: "#ee7d77", bg: "rgba(238,125,119,0.2)" },
    };
    const status = statusMap[doc.processingStatus] || statusMap.pending;
    
    return {
      id: doc.id,
      icon: "description",
      title: doc.fileName,
      subtitle: `Uploaded ${new Date(doc.createdAt).toLocaleDateString()}`,
      badge: status.badge,
      badgeColor: status.color,
      badgeBg: status.bg,
    };
  });

  // Format project for display
  const formatProjectStatus = (project: Project) => {
    switch (project.status) {
      case "processing":
        return { badge: "Active Processing", color: "#edfee9" };
      case "ready":
        return { badge: "Ready", color: "#8ca1aa" };
      case "failed":
        return { badge: "Failed", color: "#ff6b6b" };
      default:
        return { badge: "Unknown", color: "#acabaa" };
    }
  };

  return (
    <Layout activeNav="overview">
      <div className="pb-20 md:pb-8 px-6 md:px-12 max-w-7xl mx-auto pt-10">
        {/* Hero Search */}
        <section className="mb-16">
          <div className="max-w-3xl">
            <h2
              className="text-4xl md:text-5xl font-extrabold tracking-tighter mb-6 leading-none"
              style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
            >
              Good morning, <span style={{ color: "#b8b9b9" }}>Researcher</span>.
            </h2>
            <div className="relative group">
              <div
                className="absolute -inset-1 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000"
                style={{ background: "linear-gradient(to right, rgba(198,198,199,0.2), rgba(140,161,170,0.2))" }}
              />
              <div
                className="relative flex items-center p-4 rounded-xl transition-all duration-300"
                style={{
                  backgroundColor: "#131313",
                  borderBottom: "2px solid rgba(72,72,72,0.2)",
                }}
              >
                <span className="material-symbols-outlined mr-4" style={{ color: "#c6c6c7", fontSize: "28px" }}>search</span>
                <input
                  value={searchValue}
                  onChange={(e) => setSearchValue(e.target.value)}
                  className="bg-transparent border-none w-full focus:outline-none text-lg"
                  placeholder="Search projects, transcripts, or ask the AI tutor..."
                  style={{ fontFamily: "Manrope, sans-serif", color: "#e7e5e4" }}
                />
                <div className="flex gap-2">
                  <span
                    className="px-2 py-1 rounded text-xs"
                    style={{
                      backgroundColor: "#1f2020",
                      color: "#acabaa",
                      fontFamily: "Space Grotesk, sans-serif",
                      border: "1px solid rgba(72,72,72,0.1)",
                      fontSize: "10px",
                    }}
                  >ESC</span>
                </div>
              </div>
            </div>
            <div className="flex gap-3 mt-6 overflow-x-auto no-scrollbar pb-2">
              {["All", "Completed", "Processing", "Pending", "Failed"].map((tag) => (
                <button
                  key={tag}
                  onClick={() => setActiveTag(tag)}
                  className="whitespace-nowrap px-4 py-1.5 rounded-full text-xs transition-colors"
                  style={{
                    backgroundColor: activeTag === tag ? "#2a3e46" : "#1f2020",
                    color: activeTag === tag ? "#e7e5e4" : "#acabaa",
                    fontFamily: "Space Grotesk, sans-serif",
                    border: `1px solid ${activeTag === tag ? "#454747" : "rgba(72,72,72,0.05)"}`,
                  }}
                >
                  {tag}
                </button>
              ))}
            </div>
            <div className="mt-6">
              <button
                onClick={() => requestOpenSwitcher(true)}
                className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold transition-all hover:opacity-90 active:scale-95"
                style={{ backgroundColor: "#edfee9", color: "#131313", fontFamily: "Inter, sans-serif" }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>add</span>
                New Project
              </button>
            </div>
          </div>
        </section>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          {/* Primary Active Project Card */}
          <MagicCard
            className="md:col-span-8 md:row-span-2 glass-panel rounded-3xl p-8 flex flex-col justify-between relative group"
            style={{ minHeight: "360px" }}
            enableTilt={true}
            glowColor="140, 161, 170"
          >
            <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
              <span className="material-symbols-outlined" style={{ fontSize: "120px", fontVariationSettings: "'FILL' 1" }}>terminal</span>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-4">
                {activeProject?.status === "processing" && (
                  <span className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: "#edfee9" }} />
                )}
                <span
                  className="uppercase tracking-widest"
                  style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: formatProjectStatus(activeProject || {} as Project).color }}
                >
                  {isLoading ? "Loading..." : (activeProject ? formatProjectStatus(activeProject).badge : "No Active Project")}
                </span>
              </div>
              <h3
                className="text-3xl font-bold mb-2"
                style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
              >
                {isLoading ? "Loading..." : (activeProject?.name || "No Projects Yet")}
              </h3>
              <p className="text-sm leading-relaxed max-w-md" style={{ color: "#acabaa" }}>
                {isError 
                    ? "Unable to load project. Please check your connection." 
                    : (activeProject?.description || "Upload a document to get started with your first project.")}
              </p>
            </div>
            {activeProject ? (
              <>
                <div className="mt-8">
                  <div
                    className="w-full h-1 rounded-full mb-4 overflow-hidden"
                    style={{ backgroundColor: "#252626" }}
                  >
                    <div
                      className="h-full transition-all duration-1000"
                      style={{ width: `${activeProject.progress}%`, backgroundColor: "#c6c6c7" }}
                    />
                  </div>
                  <div className="flex justify-between items-center">
                    <span
                      className="text-xs"
                      style={{ fontFamily: "Space Grotesk, sans-serif", color: "#acabaa" }}
                    >
                      {activeProject.progress}% COMPILED
                    </span>
                    <button
                      className="flex items-center gap-2 text-sm font-semibold transition-colors cursor-pointer"
                      style={{ color: "#c6c6c7", fontFamily: "Inter, sans-serif" }}
                      onClick={() => navigate(`/study/${activeProject.id}`)}
                    >
                      Open Workspace
                      <span className="material-symbols-outlined">arrow_forward</span>
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="mt-8">
                <button
                  className="flex items-center gap-2 text-sm font-semibold transition-colors cursor-pointer"
                  style={{ color: "#c6c6c7", fontFamily: "Inter, sans-serif" }}
                  onClick={() => requestOpenSwitcher(true)}
                >
                  Create a project to get started
                  <span className="material-symbols-outlined">arrow_forward</span>
                </button>
              </div>
            )}
          </MagicCard>

          {/* Stats Cards - Real Data */}
          <MagicCard
            className="md:col-span-4 rounded-3xl p-6 flex flex-col justify-between"
            style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.05)" }}
            glowColor="140, 161, 170"
          >
            <div className="flex justify-between items-start">
              <span className="material-symbols-outlined" style={{ color: "#8ca1aa" }}>folder</span>
              <span
                className="uppercase"
                style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
              >
                Documents
              </span>
            </div>
            <div>
              <span className="text-4xl font-black" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>
                {documents.length}
              </span>
              <p className="text-xs mt-1" style={{ color: "#acabaa" }}>
                {documents.length === 1 ? "document uploaded" : "documents uploaded"}
              </p>
            </div>
          </MagicCard>

          <MagicCard
            className="md:col-span-4 rounded-3xl p-6 flex flex-col justify-between"
            style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.05)" }}
            glowColor="237, 254, 233"
          >
            <div className="flex justify-between items-start">
              <span className="material-symbols-outlined" style={{ color: "#edfee9" }}>check_circle</span>
              <span
                className="uppercase"
                style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "10px", color: "#acabaa" }}
              >
                Status
              </span>
            </div>
            <div>
              <span className="text-4xl font-black" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>
                {documents.filter((d: Document) => d.processingStatus === "completed").length}
              </span>
              <p className="text-xs mt-1" style={{ color: "#acabaa" }}>documents ready to study</p>
            </div>
          </MagicCard>

          {/* Recent Documents List */}
          <MagicCard
            className="md:col-span-12 lg:col-span-7 rounded-3xl p-8"
            style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.05)" }}
            glowColor="140, 161, 170"
            enableStars={false}
          >
            <div className="flex justify-between items-center mb-12">
              <h4 className="text-xl font-bold" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>Recent Documents</h4>
              <button
                className="text-xs hover:underline"
                style={{ fontFamily: "Space Grotesk, sans-serif", color: "#c6c6c7" }}
                onClick={() => navigate("/upload")}
              >
                View All
              </button>
            </div>
            <div className="space-y-12">
              {recentActivity.map((item) => (
                <MagicCard
                  key={item.id}
                  className="flex items-center cursor-pointer rounded-xl px-4 py-3"
                  style={{ backgroundColor: "transparent" }}
                  glowColor="140, 161, 170"
                  enableStars={false}
                  onClick={() => navigate(`/documents/${item.id}`)}
                >
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center mr-6 transition-colors"
                    style={{ backgroundColor: "#1f2020" }}
                  >
                    <span className="material-symbols-outlined" style={{ color: "#acabaa" }}>{item.icon}</span>
                  </div>
                  <div className="flex-1">
                    <h5 className="font-semibold" style={{ color: "#e7e5e4" }}>{item.title}</h5>
                    <p className="text-xs mt-1" style={{ color: "#acabaa" }}>{item.subtitle}</p>
                  </div>
                  <div className="text-right">
                    <span
                      className="px-3 py-1 text-xs rounded-md"
                      style={{
                        fontFamily: "Space Grotesk, sans-serif",
                        fontSize: "10px",
                        color: item.badgeColor,
                        backgroundColor: item.badgeBg,
                      }}
                    >
                      {item.badge}
                    </span>
                  </div>
                  <div className="ml-4" onClick={(event) => event.stopPropagation()}>
                    {/* Per-document removal, same confirm + DELETE + refetch
                        flow as the Upload and Study document lists. */}
                    <DeleteDocumentButton documentId={item.id} fileName={item.title} compact />
                  </div>
                </MagicCard>
              ))}
            </div>
          </MagicCard>

          {/* Quick Action Cards */}
          <div className="md:col-span-12 lg:col-span-5 grid grid-cols-1 gap-6">
            <MagicCard
              className="rounded-3xl p-6 cursor-pointer"
              style={{
                background: "linear-gradient(to bottom right, #1f2020, #131313)",
                border: "1px solid rgba(72,72,72,0.1)",
              }}
              glowColor="198, 198, 199"
              onClick={() => navigate("/upload")}
            >
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
                style={{ backgroundColor: "rgba(198,198,199,0.1)" }}
              >
                <span className="material-symbols-outlined" style={{ color: "#c6c6c7" }}>cloud_upload</span>
              </div>
              <h5 className="font-bold mb-2" style={{ color: "#e7e5e4" }}>Upload Documents</h5>
              <p className="text-xs mb-4" style={{ color: "#acabaa" }}>Add new study materials to your project. We will process and analyze them.</p>
              <button className="text-xs flex items-center gap-1 transition-all" style={{ fontFamily: "Space Grotesk, sans-serif", color: "#c6c6c7" }}>
                Upload Now <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>chevron_right</span>
              </button>
            </MagicCard>

            <MagicCard
              className="rounded-3xl p-6 cursor-pointer"
              style={{
                background: "linear-gradient(to bottom right, #1f2020, #131313)",
                border: "1px solid rgba(72,72,72,0.1)",
              }}
              glowColor="140, 161, 170"
            >
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
                style={{ backgroundColor: "rgba(140,161,170,0.1)" }}
              >
                <span className="material-symbols-outlined" style={{ color: "#8ca1aa" }}>forum</span>
              </div>
              <h5 className="font-bold mb-2" style={{ color: "#e7e5e4" }}>AI Tutor</h5>
              <p className="text-xs mb-4" style={{ color: "#acabaa" }}>Ask questions about your uploaded documents and get instant answers.</p>
              <button className="text-xs flex items-center gap-1 transition-all" style={{ fontFamily: "Space Grotesk, sans-serif", color: "#8ca1aa" }} onClick={() => navigate("/tutor")}>
                Initiate Chat <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>chevron_right</span>
              </button>
            </MagicCard>
          </div>
        </div>
      </div>

      {/* FAB */}
      <div className="fixed bottom-24 right-8 md:bottom-12 md:right-12 z-50">
        <button
          className="w-14 h-14 rounded-2xl flex items-center justify-center hover:scale-110 active:scale-90 transition-all duration-300 shadow-xl"
          style={{ backgroundColor: "#c6c6c7", color: "#3f4041" }}
          onClick={() => navigate("/upload")}
        >
          <span className="material-symbols-outlined" style={{ fontSize: "30px", fontVariationSettings: "'FILL' 1" }}>add</span>
        </button>
      </div>
    </Layout>
  );
}
