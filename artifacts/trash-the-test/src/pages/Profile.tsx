import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import Layout from "@/components/Layout";
import { getCurrentUser, getProjects } from "@/lib/api-services";
import { readLastProjectId } from "@/lib/use-active-project";

/** Minimal profile: email/username/project info (no dead controls). */
export default function Profile() {
  const [, navigate] = useLocation();
  const { data: user, isLoading } = useQuery({ queryKey: ["me"], queryFn: getCurrentUser });
  const { data: projects = [] } = useQuery({ queryKey: ["projects"], queryFn: getProjects });
  const lastProjectId = readLastProjectId();
  const lastProject = projects.find((p) => p.id === lastProjectId);

  return (
    <Layout activeNav="overview">
      <div className="pb-20 md:pb-8 px-6 md:px-12 max-w-4xl mx-auto pt-10">
        <header className="mb-10">
          <h1
            className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4"
            style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4", lineHeight: "1.1" }}
          >
            Profile
          </h1>
        </header>
        <div className="p-8 rounded-2xl" style={{ backgroundColor: "#131313" }}>
          {isLoading ? (
            <p className="text-sm" style={{ color: "#acabaa" }}>Loading profile...</p>
          ) : !user ? (
            <p className="text-sm" style={{ color: "#ee7d77" }}>
              Could not load your profile. Check your connection and try again.
            </p>
          ) : (
            <div className="flex flex-col gap-3 text-sm" style={{ color: "#acabaa" }}>
              <p>
                Name: <span style={{ color: "#e7e5e4" }}>{user.name || "-"}</span>
              </p>
              <p>
                Email: <span style={{ color: "#e7e5e4" }}>{user.email}</span>
              </p>
              <p>
                Projects: <span style={{ color: "#e7e5e4" }}>{projects.length}</span>
              </p>
              {lastProject && (
                <p>
                  Last opened project:{" "}
                  <span style={{ color: "#e7e5e4" }}>{lastProject.name}</span>
                </p>
              )}
              <div className="flex gap-2 mt-4 flex-wrap">
                <button
                  onClick={() => navigate("/settings")}
                  className="px-4 py-2 rounded-xl text-sm font-semibold"
                  style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
                >
                  Open settings
                </button>
                <button
                  onClick={() => navigate("/dashboard")}
                  className="px-4 py-2 rounded-xl text-sm font-semibold"
                  style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
                >
                  Back to overview
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
