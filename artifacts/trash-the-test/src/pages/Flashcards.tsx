import { useState, useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import Layout from "@/components/Layout";
import MagicCard from "@/components/MagicCard";
import {
  generateTypedFlashcardsWithFallback,
  getFlashcards,
  getFlashcardDecks,
} from "@/lib/api-services";
import { useActiveProject } from "@/lib/use-active-project";

// Flashcard type for real API data (empty deck when nothing generated).
type Flashcard = {
  id: string | number;
  front: string;
  back: string;
  category?: string;
  difficulty?: "easy" | "medium" | "hard";
};

// Honest generation outcome: the typed endpoint answers 200 with
// state "not_generated" when no card could be built from the project's
// sources (no artifact, no deck). Surface that message instead of
// appearing to do nothing on a click.
function noticeFromResult(result: unknown): string | null {
  if (!result || typeof result !== "object") return null;
  const state = (result as { state?: string }).state;
  if (!state || state === "ready") return null;
  const message = (result as { message?: string }).message;
  return (
    message ||
    "No flashcards could be generated from this project's sources yet."
  );
}

export default function Flashcards() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [filter, setFilter] = useState("all");

  // Shared project context (last-selected project, never a blind [0]).
  const { activeProject, activeProjectId: projectId } = useActiveProject();

  // Fetch flashcards from API
  const {
    data: flashcardsData,
    isLoading: cardsLoading,
    isError: cardsError,
    refetch: refetchCards,
  } = useQuery({
    queryKey: ["flashcards", projectId],
    queryFn: () => getFlashcards({ projectId: projectId as string }),
    enabled: !!projectId,
  });

  // Generate flashcards from the project's material (persisted server-side;
  // the read above refreshes so the deck survives a page refresh).
  // Same path as Study Generate: the typed academic endpoint, with the
  // legacy endpoint only as a deployment fallback.
  const queryClient = useQueryClient();
  const [generationNotice, setGenerationNotice] = useState<string | null>(null);
  const generateMutation = useMutation({
    mutationFn: () => {
      if (!projectId) throw new Error("Upload study material first.");
      return generateTypedFlashcardsWithFallback({ projectId });
    },
    onSuccess: (result) => {
      setGenerationNotice(noticeFromResult(result));
      void refetchCards();
      void queryClient.invalidateQueries({ queryKey: ["flashcards", projectId] });
    },
    onError: (error: Error) => {
      setGenerationNotice(error.message || "Generation failed.");
    },
  });

  // Fetch decks from API
  const { data: decksData, isLoading: decksLoading } = useQuery({
    queryKey: ["flashcard-decks", projectId],
    queryFn: () => getFlashcardDecks({ projectId: projectId as string }),
    enabled: !!projectId,
  });

  // Use API data - show empty state if no cards
  const allCards: Flashcard[] = flashcardsData?.cards || [];

  // Get unique categories from cards
  const categories = useMemo(() => {
    if (!allCards.length) return ["all"];
    const cats = new Set(allCards.map((c) => c.category || "General"));
    return ["all", ...Array.from(cats)];
  }, [allCards]);

  const filteredCards = useMemo(() => {
    if (!allCards.length) return [];
    return filter === "all"
      ? allCards
      : allCards.filter((c) => (c.category || "General") === filter);
  }, [allCards, filter]);

  // Get current card safely
  const card: Flashcard | null =
    filteredCards.length > 0
      ? filteredCards[currentIndex % filteredCards.length]
      : null;

  // Reset index when filter changes
  if (currentIndex >= filteredCards.length && filteredCards.length > 0) {
    setCurrentIndex(0);
  }

  const next = () => {
    setRevealed(false);
    setCurrentIndex((prev) => (prev + 1) % filteredCards.length);
  };

  const prev = () => {
    setRevealed(false);
    setCurrentIndex(
      (prev) => (prev - 1 + filteredCards.length) % filteredCards.length,
    );
  };

  const difficultyColor = {
    easy: "#edfee9",
    medium: "#c6c6c7",
    hard: "#ee7d77",
  };

  // Get navigate from useLocation hook
  const [, navigate] = useLocation();

  // Show loading state while fetching
  if (cardsLoading) {
    return (
      <Layout activeNav="flashcards">
        <div className="pb-20 md:pb-8 px-6 md:px-12 min-h-screen pt-10">
          <div className="flex flex-col items-center justify-center h-96">
            <span
              className="material-symbols-outlined text-6xl mb-4 animate-pulse"
              style={{ color: "#454747" }}
            >
              style
            </span>
            <h1
              className="text-2xl font-bold mb-2"
              style={{ color: "#e7e5e4" }}
            >
              Loading Flashcards...
            </h1>
          </div>
        </div>
      </Layout>
    );
  }

  // Show empty state if no cards
  if (allCards.length === 0) {
    return (
      <Layout activeNav="flashcards">
        <div className="pb-20 md:pb-8 px-6 md:px-12 min-h-screen pt-10">
          <div className="flex flex-col items-center justify-center h-96">
            <span
              className="material-symbols-outlined text-6xl mb-4"
              style={{ color: "#454747" }}
            >
              style
            </span>
            <h1
              className="text-2xl font-bold mb-2"
              style={{ color: "#e7e5e4" }}
            >
              No Flashcards Yet
            </h1>
            <p
              className="text-center max-w-md mb-6"
              style={{ color: "#acabaa" }}
            >
              Upload study material and generate flashcards to start studying. Your
              flashcards will appear here.
            </p>
            {cardsError && (
              <p className="text-sm text-center max-w-md mb-6" style={{ color: "#ee7d77" }}>
                Could not load flashcards. Check your connection and try again.
              </p>
            )}
            <div className="flex gap-4">
              <button
                onClick={() => generateMutation.mutate()}
                disabled={generateMutation.isPending || !projectId}
                className="px-6 py-3 rounded-xl text-sm font-semibold transition-all disabled:opacity-50"
                style={{ backgroundColor: "#edfee9", color: "#131313" }}
              >
                {generateMutation.isPending ? "Generating..." : "Generate flashcards"}
              </button>
              <button
                onClick={() => navigate("/upload")}
                className="px-6 py-3 rounded-xl text-sm font-semibold transition-all"
                style={{ backgroundColor: "#c6c6c7", color: "#3f4041" }}
              >
                Upload Study Material
              </button>
            </div>
            {generationNotice && (
              <p className="text-sm text-center max-w-md mt-4" style={{ color: "#ee7d77" }}>
                {generationNotice}
              </p>
            )}
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout activeNav="flashcards">
      <div className="pb-20 md:pb-8 px-6 md:px-12 min-h-screen pt-10">
        {/* Error banner (honest fetch error, no placeholder content) */}
        {cardsError && (
          <div className="mb-6 px-4 py-3 rounded-xl text-sm" style={{ backgroundColor: "rgba(255,80,80,0.08)", border: "1px solid rgba(255,80,80,0.25)", color: "#ff9d9d" }}>
            Could not refresh flashcards. Showing the last loaded cards.
          </div>
        )}

        {/* Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-12">
          <div>
            <h1
              className="text-4xl font-black tracking-tight mb-2"
              style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
            >
              Flashcards
            </h1>
            <p style={{ color: "#acabaa" }}>
              {allCards.length} cards ·{" "}
              {activeProject?.name || "AI-generated from your materials"}
            </p>
          </div>
          {categories.length > 1 && (
            <div className="flex gap-2 flex-wrap">
              {categories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => {
                    setFilter(cat);
                    setCurrentIndex(0);
                    setRevealed(false);
                  }}
                  className="px-4 py-1.5 rounded-full text-xs transition-all capitalize"
                  style={{
                    fontFamily: "Space Grotesk, sans-serif",
                    backgroundColor: filter === cat ? "#c6c6c7" : "#1f2020",
                    color: filter === cat ? "#3f4041" : "#acabaa",
                    border: "1px solid rgba(72,72,72,0.1)",
                  }}
                >
                  {cat === "all" ? "All Cards" : cat}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Main Flashcard */}
        <div className="max-w-2xl mx-auto mb-12">
          <div className="relative">
            <div
              className="absolute -inset-1 rounded-3xl blur opacity-25"
              style={{
                background:
                  "linear-gradient(to right, rgba(198,198,199,0.2), rgba(140,161,170,0.2))",
              }}
            />
            {card ? (
              <MagicCard
                className="relative p-10 rounded-2xl cursor-pointer"
                style={{
                  backgroundColor: "#131313",
                  minHeight: "320px",
                  border: "1px solid rgba(72,72,72,0.1)",
                }}
                glowColor="140, 161, 170"
                enableTilt={true}
              >
                <div
                  onClick={() => setRevealed(!revealed)}
                  style={{
                    minHeight: "100%",
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  {/* Card Meta */}
                  <div className="flex justify-between items-center mb-8">
                    <span
                      className="px-2 py-1 rounded text-xs"
                      style={{
                        fontFamily: "Space Grotesk, sans-serif",
                        fontSize: "10px",
                        color: "#8ca1aa",
                        backgroundColor: "rgba(42,62,70,0.3)",
                      }}
                    >
                      {card.category?.toUpperCase() || "GENERAL"}
                    </span>
                    <div className="flex items-center gap-3">
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{
                          backgroundColor:
                            difficultyColor[
                              card.difficulty as keyof typeof difficultyColor
                            ] || difficultyColor.medium,
                        }}
                      />
                      <span
                        className="capitalize text-xs"
                        style={{
                          fontFamily: "Space Grotesk, sans-serif",
                          color: "#acabaa",
                        }}
                      >
                        {card.difficulty}
                      </span>
                      <span
                        className="text-xs"
                        style={{
                          fontFamily: "Space Grotesk, sans-serif",
                          color: "#acabaa",
                        }}
                      >
                        {currentIndex + 1}/{filteredCards.length}
                      </span>
                    </div>
                  </div>

                  {/* Question */}
                  <div
                    className="flex flex-col justify-center"
                    style={{ minHeight: "160px" }}
                  >
                    {!revealed ? (
                      <div>
                        <p
                          className="text-xl font-bold leading-relaxed mb-6"
                          style={{
                            fontFamily: "Inter, sans-serif",
                            color: "#e7e5e4",
                          }}
                        >
                          {card.front}
                        </p>
                        <div className="flex justify-center">
                          <div className="text-center">
                            <span
                              className="text-xs uppercase tracking-widest"
                              style={{
                                fontFamily: "Space Grotesk, sans-serif",
                                color: "#484848",
                              }}
                            >
                              Tap to reveal
                            </span>
                            <div className="flex justify-center gap-1 mt-2">
                              {[0, 1, 2].map((i) => (
                                <span
                                  key={i}
                                  className="w-1.5 h-1.5 rounded-full animate-pulse"
                                  style={{
                                    backgroundColor: "#454747",
                                    animationDelay: `${i * 0.3}s`,
                                  }}
                                />
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <p
                          className="text-sm font-bold mb-4"
                          style={{
                            fontFamily: "Space Grotesk, sans-serif",
                            color: "#8ca1aa",
                            textTransform: "uppercase",
                            letterSpacing: "0.1em",
                          }}
                        >
                          Answer
                        </p>
                        <p
                          className="leading-relaxed"
                          style={{ color: "#acabaa" }}
                        >
                          {card.back}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </MagicCard>
            ) : (
              <div
                className="p-10 rounded-2xl"
                style={{
                  backgroundColor: "#131313",
                  minHeight: "320px",
                  border: "1px solid rgba(72,72,72,0.1)",
                }}
              >
                <div className="flex flex-col items-center justify-center h-full">
                  <p style={{ color: "#acabaa" }}>No flashcards available</p>
                </div>
              </div>
            )}
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between mt-8">
            <button
              onClick={prev}
              className="px-6 py-3 rounded-xl text-sm font-semibold transition-all active:scale-95"
              style={{
                backgroundColor: "#1f2020",
                color: "#e7e5e4",
                fontFamily: "Inter, sans-serif",
                border: "1px solid rgba(72,72,72,0.15)",
              }}
            >
              ← Previous
            </button>

            {revealed && (
              <div className="flex gap-3">
                <button
                  onClick={next}
                  className="px-4 py-2 rounded-lg text-xs transition-all"
                  style={{
                    backgroundColor: "rgba(127,41,39,0.2)",
                    color: "#ee7d77",
                    border: "1px solid rgba(238,125,119,0.2)",
                  }}
                >
                  Hard
                </button>
                <button
                  onClick={next}
                  className="px-4 py-2 rounded-lg text-xs transition-all"
                  style={{
                    backgroundColor: "rgba(37,38,38,0.5)",
                    color: "#acabaa",
                    border: "1px solid rgba(72,72,72,0.2)",
                  }}
                >
                  Okay
                </button>
                <button
                  onClick={next}
                  className="px-4 py-2 rounded-lg text-xs transition-all"
                  style={{
                    backgroundColor: "rgba(237,254,233,0.1)",
                    color: "#edfee9",
                    border: "1px solid rgba(237,254,233,0.2)",
                  }}
                >
                  Easy
                </button>
              </div>
            )}

            <button
              onClick={next}
              className="px-6 py-3 rounded-xl text-sm font-semibold transition-all active:scale-95"
              style={{
                background: "linear-gradient(to top right, #c6c6c7, #454747)",
                color: "#3f4041",
                fontFamily: "Inter, sans-serif",
              }}
            >
              Next →
            </button>
          </div>
        </div>

        {/* All Cards Grid */}
        <div>
          <h2
            className="text-xl font-bold mb-8"
            style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
          >
            All Cards
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredCards.map((fc, idx) => (
              <MagicCard
                key={fc.id}
                className="p-6 rounded-2xl cursor-pointer"
                style={{
                  backgroundColor:
                    idx === currentIndex % filteredCards.length
                      ? "#1f2020"
                      : "#131313",
                  border:
                    idx === currentIndex % filteredCards.length
                      ? "1px solid rgba(198,198,199,0.2)"
                      : "1px solid rgba(72,72,72,0.05)",
                }}
                glowColor="140, 161, 170"
              >
                <div
                  onClick={() => {
                    setCurrentIndex(idx);
                    setRevealed(false);
                  }}
                >
                  <div className="flex justify-between items-start mb-3">
                    <span
                      className="text-xs"
                      style={{
                        fontFamily: "Space Grotesk, sans-serif",
                        color: "#8ca1aa",
                      }}
                    >
                      {fc.category}
                    </span>
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor:
                          difficultyColor[
                            fc.difficulty as keyof typeof difficultyColor
                          ] || difficultyColor.medium,
                      }}
                    />
                  </div>
                  <p
                    className="text-sm font-semibold leading-relaxed"
                    style={{ color: "#e7e5e4" }}
                  >
                    {fc.front}
                  </p>
                </div>
              </MagicCard>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
}
