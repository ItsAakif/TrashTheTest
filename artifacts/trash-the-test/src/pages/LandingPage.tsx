import { useLocation } from "wouter";
import Silk from "../components/Silk";
import MagicCard from "../components/MagicCard";
import CardSwap, { Card } from "../components/CardSwap";
import CardNav from "../components/CardNav";

export default function LandingPage() {
  const [, navigate] = useLocation();
  const features = [
    "Reads PDFs, lecture notes, and scanned pages",
    "Generates flashcards and practice questions",
    "AI tutor that knows your study materials",
    "Study notes grounded in your documents",
  ];

  return (
    <div
      className="min-h-screen"
      style={{ backgroundColor: "#0e0e0e", color: "#e7e5e4" }}
    >
      {/* Top Nav */}
      <CardNav
        logoText="TRASHTHETEST"
        items={[
          {
            label: "Study",
            bgColor: "#131313",
            textColor: "#e7e5e4",
            links: [
              {
                label: "Dashboard",
                ariaLabel: "Go to Dashboard",
                onClick: () => navigate("/dashboard"),
              },
              {
                label: "Upload Materials",
                ariaLabel: "Upload study materials",
                onClick: () => navigate("/dashboard"),
              },
            ],
          },
          {
            label: "Artifacts",
            bgColor: "#1a1a1a",
            textColor: "#e7e5e4",
            links: [
              {
                label: "Active Study",
                ariaLabel: "Active Study sessions",
                onClick: () => navigate("/study"),
              },
              {
                label: "Flashcards",
                ariaLabel: "View Flashcards",
                onClick: () => navigate("/study"),
              },
            ],
          },
          {
            label: "Science",
            bgColor: "#1f1f1f",
            textColor: "#e7e5e4",
            links: [
              {
                label: "AI Tutor",
                ariaLabel: "Open AI Tutor",
                onClick: () => navigate("/study"),
              },
              {
                label: "Knowledge Maps",
                ariaLabel: "View Knowledge Maps",
                onClick: () => navigate("/study"),
              },
            ],
          },
        ]}
        baseColor="rgba(14, 14, 14, 0.55)"
        menuColor="#e7e5e4"
        buttonBgColor="#c6c6c7"
        buttonTextColor="#3f4041"
        buttonLabel="Access System"
        onButtonClick={() => navigate("/dashboard")}
        ease="power3.out"
      />

      {/* Hero Section */}
      <section
        className="min-h-screen flex flex-col justify-center px-8 md:px-16 pt-20 pb-16"
        style={{ position: "relative", overflow: "hidden" }}
      >
        <div
          style={{
            position: "absolute",
            inset: "-20%",
            zIndex: 0,
            maskImage:
              "radial-gradient(ellipse 60% 55% at 50% 45%, black 0%, transparent 70%)",
            WebkitMaskImage:
              "radial-gradient(ellipse 60% 55% at 50% 45%, black 0%, transparent 70%)",
          }}
        >
          <Silk
            speed={5}
            scale={1}
            color="#7B7481"
            noiseIntensity={1.5}
            rotation={0}
          />
        </div>
        <div
          className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-center"
          style={{ position: "relative", zIndex: 1 }}
        >
          <div>
            <h1
              className="font-black tracking-tighter leading-none mb-8"
              style={{
                fontFamily: "Inter, sans-serif",
                fontSize: "clamp(3rem, 8vw, 7rem)",
                lineHeight: "0.9",
              }}
            >
              <span style={{ color: "#acabaa", fontSize: "0.7em" }}>
                Discard the
              </span>
              <br />
              <em>Static.</em>
              <br />
              <span style={{ color: "#acabaa", fontSize: "0.7em" }}>
                Embrace the
              </span>
              <br />
              <em>Dynamic.</em>
            </h1>
            <p
              className="text-lg leading-relaxed mb-12 max-w-lg font-light"
              style={{ color: "#acabaa" }}
            >
              Information is an artifact. We treat your data as a living,
              evolving landscape—moving beyond the static dissolved tiles of
              textbooks to give your material a productive life.
            </p>
            <div className="flex gap-4 flex-wrap">
              <button
                className="px-8 py-4 rounded-xl text-sm font-bold uppercase tracking-wider transition-all active:scale-95"
                style={{
                  background: "linear-gradient(to top right, #c6c6c7, #454747)",
                  color: "#3f4041",
                  fontFamily: "Inter, sans-serif",
                }}
                onClick={() => navigate("/dashboard")}
              >
                Start Studying
              </button>
              <button
                className="px-8 py-4 rounded-xl text-sm font-bold uppercase tracking-wider transition-all"
                style={{
                  backgroundColor: "transparent",
                  color: "#acabaa",
                  fontFamily: "Inter, sans-serif",
                  border: "1px solid rgba(72,72,72,0.3)",
                }}
                onClick={() => navigate("/study")}
              >
                View Artifacts
              </button>
            </div>
          </div>

          {/* Visual Element */}
          <div className="hidden lg:flex justify-center">
            <div className="relative">
              <CardSwap
                cardDistance={30}
                verticalDistance={35}
                delay={3500}
                width={480}
                height={270}
              >
                <Card>
                  <img src="/screenshot-dashboard.png" alt="Dashboard" />
                </Card>
                <Card>
                  <img src="/screenshot-active-study.png" alt="Active Study" />
                </Card>
                <Card>
                  <img src="/screenshot-flashcards.png" alt="Flashcards" />
                </Card>
                <Card>
                  <img src="/screenshot-ai-tutor.png" alt="AI Tutor" />
                </Card>
              </CardSwap>
            </div>
          </div>
        </div>
      </section>

      {/* "The Molecular Shift" Section */}
      <section className="py-32 px-8 md:px-16">
        <div className="max-w-7xl mx-auto">
          <div
            className="text-xs uppercase tracking-widest mb-12"
            style={{
              fontFamily: "Space Grotesk, sans-serif",
              color: "#acabaa",
            }}
          >
            Provided By
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2
                className="text-4xl md:text-5xl font-black tracking-tight mb-6"
                style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
              >
                The Molecular
                <br />
                Shift
              </h2>
              <p
                className="text-lg font-light leading-relaxed mb-8"
                style={{ color: "#acabaa" }}
              >
                TrashTheTest study is a three-path. We fracture that box into a
                semantic web of interconnected insights. Your static PDF becomes
                a living, connected web of insights.
              </p>
              <div className="space-y-4">
                {features.map((f) => (
                  <div key={f} className="flex items-start gap-4">
                    <span
                      className="material-symbols-outlined mt-0.5"
                      style={{ color: "#c6c6c7", fontSize: "18px" }}
                    >
                      check_circle
                    </span>
                    <span className="text-sm" style={{ color: "#acabaa" }}>
                      {f}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {[
                "Study Notes",
                "Flashcards",
                "Practice Questions",
                "AI Tutor",
              ].map((item, i) => (
                <MagicCard
                  key={item}
                  className="p-6 rounded-2xl"
                  style={{
                    backgroundColor: i % 2 === 0 ? "#131313" : "#1f2020",
                    border: "1px solid rgba(72,72,72,0.1)",
                    borderRadius: "16px",
                  }}
                  enableStars
                  enableBorderGlow
                  enableTilt={false}
                  enableMagnetism={false}
                  clickEffect
                  particleCount={12}
                  glowColor="140, 161, 170"
                  spotlightRadius={400}
                >
                  <span
                    className="material-symbols-outlined mb-3"
                    style={{
                      color: "#454747",
                      display: "block",
                      fontSize: "32px",
                      fontVariationSettings: "'FILL' 1",
                      position: "relative",
                      zIndex: 2,
                    }}
                  >
                    {["hub", "memory", "network_node", "schema"][i]}
                  </span>
                  <p
                    className="text-sm font-semibold"
                    style={{
                      color: "#acabaa",
                      position: "relative",
                      zIndex: 2,
                    }}
                  >
                    {item}
                  </p>
                </MagicCard>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* "Interactions as Artifacts" Section */}
      <section
        className="py-32 px-8 md:px-16"
        style={{ backgroundColor: "#0a0a0a" }}
      >
        <div className="max-w-7xl mx-auto text-center">
          <span
            className="text-xs uppercase tracking-widest block mb-6"
            style={{
              fontFamily: "Space Grotesk, sans-serif",
              color: "#acabaa",
            }}
          >
            The Artifacts Interface
          </span>
          <h2
            className="text-5xl md:text-7xl font-black tracking-tighter mb-16"
            style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
          >
            Interactions as
            <br />
            Artifacts.
          </h2>

          {/* Demo Card */}
          <div className="max-w-md mx-auto">
            <MagicCard
              className="p-8 rounded-2xl text-left mb-8"
              style={{
                backgroundColor: "#131313",
                border: "1px solid rgba(72,72,72,0.15)",
                borderRadius: "16px",
              }}
              enableStars
              enableBorderGlow
              enableTilt={false}
              enableMagnetism={false}
              clickEffect
              particleCount={12}
              glowColor="140, 161, 170"
            >
              <div
                className="text-xs mb-4"
                style={{
                  fontFamily: "Space Grotesk, sans-serif",
                  color: "#acabaa",
                  position: "relative",
                  zIndex: 2,
                }}
              >
                FLASHCARD 01/28
              </div>
              <h3
                className="text-xl font-bold mb-4"
                style={{
                  fontFamily: "Inter, sans-serif",
                  color: "#e7e5e4",
                  position: "relative",
                  zIndex: 2,
                }}
              >
                How does the{" "}
                <span style={{ color: "#8ca1aa" }}>Primacy Effect</span>{" "}
                influence long-term retention in high-stakes environments?
              </h3>
              <button
                className="text-xs uppercase tracking-widest"
                style={{
                  fontFamily: "Space Grotesk, sans-serif",
                  color: "#c6c6c7",
                  border: "1px solid rgba(198,198,199,0.2)",
                  padding: "8px 16px",
                  borderRadius: "4px",
                  position: "relative",
                  zIndex: 2,
                }}
              >
                Detail Card
              </button>
            </MagicCard>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-32 px-8 md:px-16">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <MagicCard
            className="p-8 rounded-2xl"
            style={{
              backgroundColor: "#131313",
              border: "1px solid rgba(72,72,72,0.1)",
              borderRadius: "16px",
            }}
            enableStars
            enableBorderGlow
            enableTilt={false}
            enableMagnetism={false}
            clickEffect
            particleCount={12}
            glowColor="140, 161, 170"
          >
            <div
              className="text-xs uppercase tracking-widest mb-4"
              style={{
                fontFamily: "Space Grotesk, sans-serif",
                color: "#acabaa",
                position: "relative",
                zIndex: 2,
              }}
            >
              AI Reasoning Stream
            </div>
            <div
              className="space-y-3"
              style={{ position: "relative", zIndex: 2 }}
            >
              {[
                "Reading your documents...",
                "Structuring notes and flashcards...",
                "Writing practice questions...",
              ].map((item, i) => (
                <div key={item} className="flex items-center gap-3">
                  <span
                    className="w-1.5 h-1.5 rounded-full animate-pulse"
                    style={{
                      backgroundColor: i === 1 ? "#c6c6c7" : "#454747",
                      animationDelay: `${i * 0.4}s`,
                    }}
                  />
                  <span
                    className="text-xs"
                    style={{
                      fontFamily: "Space Grotesk, sans-serif",
                      color: "#acabaa",
                    }}
                  >
                    {item}
                  </span>
                </div>
              ))}
            </div>
          </MagicCard>

          <div>
            <h2
              className="text-5xl md:text-7xl font-black tracking-tighter mb-6"
              style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
            >
              Study
              <br />
              Smarter.
            </h2>
            <p
              className="text-lg font-light leading-relaxed"
              style={{ color: "#acabaa" }}
            >
              We don't just summarize. Upload your material, generate notes,
              flashcards, and practice questions, and study them with an AI
              tutor. One workspace for learning.
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section
        className="py-32 px-8 md:px-16 text-center"
        style={{ backgroundColor: "#0a0a0a" }}
      >
        <h2
          className="font-black tracking-tighter mb-12"
          style={{
            fontFamily: "Inter, sans-serif",
            fontSize: "clamp(4rem, 12vw, 12rem)",
            color: "#e7e5e4",
            lineHeight: "0.9",
          }}
        >
          BEYOND THE PAGE.
        </h2>
        <div className="flex gap-6 justify-center">
          <button
            className="px-8 py-4 rounded-xl text-sm font-bold uppercase tracking-wider transition-all"
            style={{
              backgroundColor: "#e7e5e4",
              color: "#0e0e0e",
              fontFamily: "Inter, sans-serif",
            }}
            onClick={() => navigate("/dashboard")}
          >
            Start Studying
          </button>
          <button
            className="px-8 py-4 rounded-xl text-sm font-bold uppercase tracking-wider transition-all"
            style={{
              backgroundColor: "transparent",
              color: "#acabaa",
              fontFamily: "Inter, sans-serif",
              border: "1px solid rgba(72,72,72,0.3)",
            }}
            onClick={() => navigate("/study")}
          >
            View Artifacts
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer
        className="py-12 px-8 md:px-16"
        style={{ borderTop: "1px solid rgba(72,72,72,0.1)" }}
      >
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start gap-8">
          <div>
            <span
              className="text-lg font-black tracking-tighter block mb-2"
              style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
            >
              TRASHTHETEST
            </span>
            <p className="text-xs" style={{ color: "#484848" }}>
              Redefining the future of academic learning solutions
            </p>
          </div>
          <div
            className="text-xs"
            style={{
              fontFamily: "Space Grotesk, sans-serif",
              color: "#484848",
            }}
          >
            © 2026 TrashTheTest. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
