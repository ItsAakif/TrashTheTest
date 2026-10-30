import { useState } from "react";
import Citations from "@/components/Citations";
import type { Citation, AnswerResult } from "@/lib/api-services";

interface SectionShape {
  heading?: string;
  content?: unknown;
  subsections?: SectionShape[];
  citations?: Citation[];
}

/** Render persisted note content: sections/headings/content/citations. */
export function NoteBody({ content }: { content: unknown }) {
  const sections = extractSections(content);
  if (sections.length === 0) {
    // No internal JSON on screen: a string renders as prose, anything
    // else gets an honest placeholder (never a raw object dump).
    if (typeof content === "string" && content.trim().length > 0) {
      return (
        <p className="text-sm leading-relaxed whitespace-pre-wrap" style={{ color: "#acabaa" }}>
          {content}
        </p>
      );
    }
    return (
      <p className="text-sm leading-relaxed" style={{ color: "#acabaa" }}>
        Notes content is being prepared in a readable format. Generate again to refresh this note.
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-6">
      {sections.map((s, i) => (
        <NoteSectionView key={i} section={s} depth={0} />
      ))}
    </div>
  );
}

function extractSections(content: unknown): SectionShape[] {
  if (!content || typeof content !== "object") return [];
  const obj = content as Record<string, unknown>;
  const raw = obj.sections ?? obj.notes ?? [];
  if (!Array.isArray(raw)) return [];
  return raw.filter((s): s is SectionShape => !!s && typeof s === "object");
}

function NoteSectionView({ section, depth }: { section: SectionShape; depth: number }) {
  const HeadingTag = depth === 0 ? "h3" : "h4";
  return (
    <section>
      {section.heading && (
        <HeadingTag
          className={depth === 0 ? "text-lg font-bold mb-2" : "text-base font-bold mb-2"}
          style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}
        >
          {section.heading}
        </HeadingTag>
      )}
      {typeof section.content === "string" && (
        <p className="text-sm leading-relaxed whitespace-pre-wrap" style={{ color: "#acabaa" }}>
          {section.content}
        </p>
      )}
      <Citations citations={section.citations} />
      {(section.subsections ?? []).length > 0 && (
        <div className="ml-4 mt-3 flex flex-col gap-4" style={{ borderLeft: "1px solid rgba(72,72,72,0.3)", paddingLeft: "1rem" }}>
          {(section.subsections ?? []).map((sub, i) => (
            <NoteSectionView key={i} section={sub} depth={depth + 1} />
          ))}
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------

interface McqQuestion {
  question?: string;
  options?: string[];
  correct_answer?: number;
  correctAnswer?: number;
  explanation?: string;
  citations?: Citation[];
}

interface QuizData {
  state?: string;
  message?: string;
  quizzes?: Array<{ id: string; title: string; questions: unknown[] }>;
}

interface McqTabProps {
  quizData?: QuizData;
  quizLoading: boolean;
  quizError: boolean;
  selections: Record<string, number>;
  submitted: Record<string, boolean>;
  onSelect: (qid: string, idx: number) => void;
  onSubmit: (qid: string) => void;
  onGenerate: () => void;
  generating: boolean;
  /** Disable generation (e.g. no study material uploaded yet). */
  disabled?: boolean;
  /** Reason shown when disabled (e.g. "Upload study material first"). */
  disabledReason?: string;
}

/** MCQs: answer hidden until submit, then correct/incorrect + explanation + citation. */
export function McqTab(props: McqTabProps) {
  const { quizData, quizLoading, quizError } = props;
  if (quizLoading) {
    return (
      <div className="p-8 rounded-2xl text-center" style={{ backgroundColor: "#131313" }}>
        <p style={{ color: "#acabaa" }}>Loading practice questions...</p>
      </div>
    );
  }
  if (quizError) {
    return (
      <div className="p-8 rounded-2xl text-center" style={{ backgroundColor: "#131313", border: "1px solid rgba(255,80,80,0.25)" }}>
        <p style={{ color: "#ee7d77" }}>Could not load practice questions. Check your connection and try again.</p>
      </div>
    );
  }
  const quizzes = quizData?.quizzes ?? [];
  if (quizData?.state === "not_generated" || quizzes.length === 0) {
    return (
      <div className="p-8 rounded-2xl text-center" style={{ backgroundColor: "#131313" }}>
        <p className="mb-4" style={{ color: "#acabaa" }}>No practice questions generated yet for this project.</p>
        <button
          onClick={props.onGenerate}
          disabled={props.generating || props.disabled}
          className="px-6 py-3 rounded-xl font-semibold transition-all hover:opacity-90 disabled:opacity-50"
          style={{ backgroundColor: "#edfee9", color: "#131313" }}
        >
          {props.generating ? "Generating..." : "Generate practice questions"}
        </button>
        {props.disabled && props.disabledReason && (
          <p className="mt-3 text-sm" style={{ color: "#acabaa" }}>
            {props.disabledReason}{" "}
            <a href="/upload" style={{ color: "#edfee9", textDecoration: "underline" }}>
              Upload study material first
            </a>
          </p>
        )}
      </div>
    );
  }
  // Progress + score from real interaction state (submitted answers only).
  // qid scheme matches the render loop below: `${quiz.id}-${questionIndex}`.
  const allQuestions = quizzes.flatMap((quiz) =>
    (quiz.questions as McqQuestion[]).map((q, i) => ({ qid: `${quiz.id}-${i}`, q }))
  );
  const total = allQuestions.length;
  const answered = allQuestions.filter(({ qid }) => !!props.submitted[qid]).length;
  const correct = allQuestions.filter(({ qid, q }) => {
    if (!props.submitted[qid]) return false;
    const right = q.correct_answer ?? q.correctAnswer ?? -1;
    return props.selections[qid] === right;
  }).length;
  return (
    <div className="flex flex-col gap-6">
      <div
        className="px-6 py-4 rounded-2xl flex flex-wrap items-center gap-4"
        style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.15)" }}
        aria-live="polite"
      >
        <span className="text-sm font-semibold" style={{ color: "#e7e5e4" }}>
          Progress: {answered} of {total} answered
        </span>
        <span className="text-sm" style={{ color: "#acabaa" }}>
          Score: {correct} correct{answered > 0 ? ` (${Math.round((correct / answered) * 100)}%)` : ""}
        </span>
      </div>
      {quizzes.map((quiz) => (
        <div key={quiz.id} className="p-8 rounded-2xl" style={{ backgroundColor: "#131313" }}>
          <h3 className="text-xl font-bold mb-4" style={{ color: "#e7e5e4" }}>{quiz.title}</h3>
          <div className="flex flex-col gap-6">
            {(quiz.questions as McqQuestion[]).map((q, qi) => {
              const qid = `${quiz.id}-${qi}`;
              const correct = q.correct_answer ?? q.correctAnswer ?? -1;
              const isSubmitted = !!props.submitted[qid];
              const selected = props.selections[qid];
              return (
                <div key={qid} className="p-4 rounded-xl" style={{ border: "1px solid rgba(72,72,72,0.2)" }}>
                  <p className="text-sm font-medium mb-3" style={{ color: "#e7e5e4" }}>{q.question}</p>
                  <div className="flex flex-col gap-2">
                    {(q.options ?? []).map((opt, oi) => {
                      const isPicked = selected === oi;
                      const isRight = isSubmitted && oi === correct;
                      const isWrong = isSubmitted && isPicked && oi !== correct;
                      return (
                        <button
                          key={oi}
                          disabled={isSubmitted}
                          onClick={() => props.onSelect(qid, oi)}
                          className="text-left p-3 rounded-lg transition-all text-sm disabled:cursor-default"
                          style={{
                            border: isRight
                              ? "1px solid rgba(237,254,233,0.5)"
                              : isWrong
                                ? "1px solid rgba(238,125,119,0.5)"
                                : isPicked
                                  ? "1px solid rgba(198,198,199,0.3)"
                                  : "1px solid rgba(72,72,72,0.2)",
                            backgroundColor: isRight
                              ? "rgba(237,254,233,0.08)"
                              : isWrong
                                ? "rgba(238,125,119,0.08)"
                                : isPicked
                                  ? "rgba(198,198,199,0.1)"
                                  : "transparent",
                            color: "#e7e5e4",
                          }}
                        >
                          {opt}
                          {isRight && " — correct"}
                          {isWrong && " — your answer"}
                        </button>
                      );
                    })}
                  </div>
                  {!isSubmitted ? (
                    <button
                      onClick={() => props.onSubmit(qid)}
                      disabled={selected === undefined}
                      className="mt-3 px-5 py-2 rounded-xl text-sm font-semibold transition-all hover:opacity-90 disabled:opacity-50"
                      style={{ backgroundColor: "#252626", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.2)" }}
                    >
                      Submit answer
                    </button>
                  ) : (
                    <div className="mt-3">
                      <p className="text-sm font-semibold" style={{ color: selected === correct ? "#edfee9" : "#ee7d77" }}>
                        {selected === correct ? "Correct!" : "Incorrect."}
                      </p>
                      {q.explanation && (
                        <p className="text-sm mt-1" style={{ color: "#acabaa" }}>{q.explanation}</p>
                      )}
                      <Citations citations={q.citations} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------

/** Answers: question/marks/depth/key points/citations + word-band UI. */
export function AnswerCard({ result }: { result: AnswerResult }) {
  const content = result.content ?? {};
  const text = String(content.answer_text ?? "");
  const words = text.split(/\s+/).filter(Boolean).length;
  const band = result.word_band;
  const inBand = words >= band.min_words && words <= band.max_words;
  const keyPoints = Array.isArray(content.key_points) ? (content.key_points as string[]) : [];
  return (
    <div>
      {content.question && (
        <p className="text-sm font-medium mb-2" style={{ color: "#e7e5e4" }}>{String(content.question)}</p>
      )}
      <div className="flex gap-2 flex-wrap mb-3">
        <span className="px-2 py-0.5 rounded text-xs" style={{ backgroundColor: "#1f2020", color: "#8ca1aa" }}>
          {result.marks} marks
        </span>
        <span className="px-2 py-0.5 rounded text-xs" style={{ backgroundColor: "#1f2020", color: "#8ca1aa" }}>
          target {band.min_words}-{band.max_words} words
        </span>
        <span
          className="px-2 py-0.5 rounded text-xs"
          style={{ backgroundColor: inBand ? "rgba(237,254,233,0.1)" : "rgba(238,125,119,0.1)", color: inBand ? "#edfee9" : "#ee7d77" }}
        >
          {words} words{inBand ? " · in band" : " · out of band"}
        </span>
      </div>
      <p className="text-sm leading-relaxed whitespace-pre-wrap mb-3" style={{ color: "#e7e5e4" }}>{text}</p>
      {keyPoints.length > 0 && (
        <ul className="list-disc ml-5 space-y-1 mb-3">
          {keyPoints.map((kp, i) => (
            <li key={i} className="text-sm" style={{ color: "#acabaa" }}>{kp}</li>
          ))}
        </ul>
      )}
      <Citations citations={content.citations} />
      {content.provenance && typeof content.provenance === "object" && (
        <p className="text-xs mt-2" style={{ color: "#484848" }}>
          grounding: {String((content.provenance as { support?: string }).support ?? "unknown")}
          {(content.provenance as { note?: string }).note ? ` — ${String((content.provenance as { note?: string }).note)}` : ""}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

interface DeckCard {
  id: string;
  front: string;
  back: string;
  difficulty?: "easy" | "medium" | "hard";
}

/** Persisted flashcard deck with flip/next interaction. */
export function PersistedDeck({ cards }: { cards: DeckCard[] }) {
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const card = cards.length > 0 ? cards[index % cards.length] : null;
  if (!card) return null;
  return (
    <div className="p-6 rounded-2xl" style={{ backgroundColor: "#1f2020" }}>
      <p className="text-xs uppercase tracking-widest mb-3" style={{ color: "#acabaa" }}>
        Saved deck · {index + 1} / {cards.length}
      </p>
      <div className="cursor-pointer" onClick={() => setRevealed((r) => !r)}>
        <p className="text-base font-bold" style={{ color: "#e7e5e4" }}>{card.front}</p>
        {revealed && (
          <p className="text-sm mt-2" style={{ color: "#acabaa" }}>{card.back}</p>
        )}
        {!revealed && (
          <p className="text-xs mt-2" style={{ color: "#484848" }}>Click to reveal</p>
        )}
      </div>
      <div className="flex gap-2 mt-4">
        <button
          onClick={() => { setRevealed(false); setIndex((i) => (i + 1) % cards.length); }}
          className="px-4 py-2 rounded-xl text-xs uppercase tracking-widest"
          style={{ color: "#acabaa", border: "1px solid rgba(72,72,72,0.2)" }}
        >
          Next card
        </button>
      </div>
    </div>
  );
}
