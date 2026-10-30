import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { useQueryClient } from "@tanstack/react-query";
import { logout } from "@/lib/api-services";
import { useActiveProject } from "@/lib/use-active-project";
import ProjectSwitcher, {
  OPEN_SWITCHER_EVENT,
} from "@/components/ProjectSwitcher";

interface LayoutProps {
  children: React.ReactNode;
  activeNav?: string;
}

// Single navigation source for the whole shell: the sidebar renders all
// entries. The topbar no longer lists features separately — it hosts the
// GCP-style project selector (single source of project truth). Every
// control below is wired to a real handler; nothing decorative ships.
const NAV = [
  { id: "overview", label: "Overview", icon: "grid_view", href: "/dashboard", top: false },
  { id: "upload", label: "Upload Study Material", icon: "cloud_upload", href: "/upload", top: false },
  { id: "study", label: "Study", icon: "menu_book", href: "/study", top: false },
  { id: "flashcards", label: "Flashcards", icon: "style", href: "/flashcards", top: false },
  { id: "tutor", label: "AI Tutor", icon: "psychology", href: "/tutor", top: false },
  { id: "academic", label: "Academic Insights", icon: "school", href: "/academic", top: false },
  { id: "publish", label: "Publish Study Website", icon: "public", href: "/website", top: false },
];

const MOBILE_NAV = [
  { icon: "grid_view", label: "Overview", href: "/dashboard" },
  { icon: "cloud_upload", label: "Upload", href: "/upload" },
  { icon: "menu_book", label: "Study", href: "/study" },
  { icon: "person_outline", label: "Profile", href: "/profile" },
];

export default function Layout({
  children,
  activeNav = "overview",
}: LayoutProps) {
  const [location, navigate] = useLocation();
  const queryClient = useQueryClient();

  // Active project is the single source of truth (localStorage-persisted
  // via useActiveProject): the topbar selector reflects it on every page.
  const {
    activeProject,
    projectsLoading: projectsLoadingNav,
    projectsError: projectsErrorNav,
  } = useActiveProject();

  // Project switcher modal (single creation path lives inside it).
  // Pages without their own dialog fire OPEN_SWITCHER_EVENT to open it.
  const [switcherOpen, setSwitcherOpen] = useState(false);
  const [createSignal, setCreateSignal] = useState(0);
  useEffect(() => {
    const handler = (e: Event) => {
      setSwitcherOpen(true);
      if ((e as CustomEvent<boolean>).detail === true) {
        setCreateSignal((n) => n + 1);
      }
    };
    window.addEventListener(OPEN_SWITCHER_EVENT, handler);
    return () => window.removeEventListener(OPEN_SWITCHER_EVENT, handler);
  }, []);

  const selectorLabel = projectsLoadingNav
    ? "Loading..."
    : projectsErrorNav
      ? "Projects unavailable"
      : (activeProject?.name ?? "Select project");

  // Sign Out: server logout (best-effort revoke) + local session clear +
  // query cache reset, then land on /login. ProtectedRoute's
  // getCurrentUser guard keeps every protected route closed afterwards.
  const handleSignOut = async () => {
    try {
      await logout();
    } finally {
      queryClient.clear();
      navigate("/login");
    }
  };

  const isActive = (href: string) =>
    location === href || (href !== "/dashboard" && location.startsWith(href));

  return (
    <div
      className="min-h-screen"
      style={{ backgroundColor: "#0e0e0e", color: "#e7e5e4" }}
    >
      {/* Top Navigation */}
      <header
        className="fixed top-0 w-full z-50 flex justify-between items-center px-8 h-16"
        style={{
          backgroundColor: "rgba(14,14,14,0.85)",
          backdropFilter: "blur(20px)",
          borderBottom: "1px solid rgba(72,72,72,0.15)",
          boxShadow: "0 20px 50px rgba(0,0,0,0.5)",
        }}
      >
        <div className="flex items-center gap-8">
          <span
            className="text-xl font-black tracking-tighter cursor-pointer"
            style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
            onClick={() => navigate("/")}
          >
            TRASHTHETEST
          </span>
          <nav
            className="flex gap-6 items-center"
            style={{ fontFamily: "Inter, sans-serif" }}
            aria-label="Project"
          >
            <button
              onClick={() => setSwitcherOpen(true)}
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all hover:opacity-90"
              style={{
                backgroundColor: "#1f2020",
                color: "#e7e5e4",
                border: "1px solid rgba(72,72,72,0.2)",
              }}
              aria-haspopup="dialog"
              title="Select a project"
            >
              <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                folder
              </span>
              <span className="max-w-56 truncate">
                Project: {selectorLabel}
              </span>
              <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                expand_more
              </span>
            </button>
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <button
            className="p-2 rounded-lg transition-all hover:opacity-80"
            style={{ color: "#acabaa" }}
            onClick={() => navigate("/settings")}
            aria-label="Settings"
            title="Settings"
          >
            <span className="material-symbols-outlined">settings</span>
          </button>
          <button
            className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all hover:opacity-80"
            style={{ backgroundColor: "#454747", color: "#c6c6c7" }}
            onClick={() => navigate("/profile")}
            aria-label="Profile"
            title="Profile"
          >
            R
          </button>
        </div>
      </header>

      {/* Sidebar */}
      <aside
        className="hidden md:flex flex-col py-6 px-4 gap-4 h-full fixed left-0 top-0 z-40"
        style={{
          width: "256px",
          backgroundColor: "#131313",
          borderRight: "1px solid rgba(72,72,72,0.15)",
        }}
      >
        <div className="flex items-center gap-3 px-2 mt-16 mb-4">
          <div
            className="icon-tile w-8 h-8 rounded-lg flex items-center justify-center"
            style={{
              background: "linear-gradient(to top right, #c6c6c7, #454747)",
            }}
          >
            <span
              className="material-symbols-outlined"
              style={{
                color: "#3f4041",
                fontSize: "16px",
                fontVariationSettings: "'FILL' 1",
              }}
            >
              terminal
            </span>
          </div>
          <div>
            <h1
              className="text-base font-black leading-tight"
              style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
            >
              TRASHTHETEST
            </h1>
            <p
              className="uppercase tracking-widest"
              style={{
                fontFamily: "Space Grotesk, sans-serif",
                color: "#acabaa",
                fontSize: "10px",
              }}
            >
              Study Workspace
            </p>
          </div>
        </div>

        <nav className="flex flex-col gap-1 flex-1">
          {NAV.map((item) => {
            const active = isActive(item.href);
            return (
              <div
                key={item.id}
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-300 ease-out cursor-pointer text-sm font-medium"
                style={{
                  fontFamily: "Inter, sans-serif",
                  backgroundColor: active ? "#1f2020" : "transparent",
                  color: active ? "#e7e5e4" : "#acabaa",
                }}
                onClick={() => navigate(item.href)}
              >
                <span className="material-symbols-outlined">{item.icon}</span>
                <span>{item.label}</span>
              </div>
            );
          })}
        </nav>

        <div
          className="flex flex-col gap-1"
          style={{
            borderTop: "1px solid rgba(72,72,72,0.15)",
            paddingTop: "16px",
          }}
        >
          <div
            className="flex items-center gap-3 px-3 py-2 text-sm transition-all cursor-pointer hover:opacity-80"
            style={{ color: "#acabaa" }}
            onClick={() => navigate("/settings")}
          >
            <span className="material-symbols-outlined">settings</span>
            <span>Settings</span>
          </div>
          <div
            className="flex items-center gap-3 px-3 py-2 text-sm transition-all cursor-pointer hover:opacity-80"
            style={{ color: "#acabaa" }}
            onClick={() => void handleSignOut()}
          >
            <span className="material-symbols-outlined">logout</span>
            <span>Sign Out</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="min-h-screen pt-16 md:pl-64">{children}</main>

      {/* Mobile Bottom Nav (every link resolves to a real route) */}
      <nav
        className="md:hidden fixed bottom-0 left-0 w-full flex justify-around items-center px-6 pb-6 pt-3 z-50 rounded-t-2xl"
        style={{
          backgroundColor: "rgba(14,14,14,0.9)",
          backdropFilter: "blur(24px)",
          borderTop: "1px solid rgba(72,72,72,0.2)",
          boxShadow: "0 -10px 40px rgba(0,0,0,0.6)",
        }}
      >
        {MOBILE_NAV.map((item) => (
          <div
            key={item.label}
            className="flex flex-col items-center justify-center p-2 transition-all active:scale-95 cursor-pointer"
            style={{ color: location === item.href ? "#e7e5e4" : "#acabaa" }}
            onClick={() => navigate(item.href)}
          >
            <span className="material-symbols-outlined">{item.icon}</span>
            <span
              className="uppercase tracking-widest mt-1"
              style={{
                fontFamily: "Space Grotesk, sans-serif",
                fontSize: "10px",
              }}
            >
              {item.label}
            </span>
          </div>
        ))}
      </nav>

      {/* GCP-style project switcher: the single creation + deletion path. */}
      <ProjectSwitcher
        open={switcherOpen}
        onOpenChange={setSwitcherOpen}
        createSignal={createSignal}
      />
    </div>
  );
}
