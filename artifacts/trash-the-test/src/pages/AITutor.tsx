import { useState, useRef, useEffect } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import Layout from "@/components/Layout";
import Citations from "@/components/Citations";
import {
  sendTutorMessage,
  startStudySession,
  type Citation,
} from "@/lib/api-services";
import { useActiveProject } from "@/lib/use-active-project";
import { TutorMarkdown } from "@/lib/tutor-markdown";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: string[];
  citations?: Citation[];
  context?: string;
  isError?: boolean;
}

export default function AITutor() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "0",
      role: "assistant",
      content: "Hi! I'm your AI tutor. I can answer questions based on your uploaded documents. Upload materials and ask me anything about them.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  // Real tutor session id from POST /api/study/session/start (never faked).
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const { activeProject, activeProjectId, projectsLoading, projectsError } =
    useActiveProject();
  const activeProjectName = activeProject?.name;

  const ensureSession = async (): Promise<string> => {
    if (sessionId) return sessionId;
    if (!activeProjectId) {
      throw new Error("Upload a document and create a project first.");
    }
    try {
      const session = await startStudySession({
        projectId: activeProjectId,
        sessionType: "tutor",
      });
      setSessionId(session.sessionId);
      setSessionError(null);
      return session.sessionId;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not start a tutor session.";
      setSessionError(message);
      throw err;
    }
  };

  const sendMessageMutation = useMutation({
    mutationFn: async (message: string) => {
      const sid = await ensureSession();
      // The backend classifies the intent, retrieves its own evidence, and
      // answers with citations plus the exact context it prompted with.
      // No second client-side search is issued: two retrievals can disagree,
      // and a display-only result must never overwrite the grounded citation
      // list the server actually used.
      return await sendTutorMessage({ sessionId: sid, message });
    },
  });

  const sendMessage = async () => {
    if (!input.trim() || isTyping) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await sendMessageMutation.mutateAsync(input);

      // T408: coerce to displayable text; the API guarantees a string but
      // a stale cache or proxy must never render [object Object].
      const displayText =
        typeof response.response === "string"
          ? response.response
          : String(response.response ?? "I could not produce an answer.");
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: displayText,
        timestamp: new Date(),
        sources: response.sources,
        citations: response.citations ?? undefined,
        context: response.context,
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      // Readable provider errors (B4): surface the real failure reason
      // (typed 404/not_generated/5xx detail) instead of a generic line,
      // so the student knows what to do next. Never render [object Object].
      const detail = err instanceof Error ? err.message : "Unknown error.";
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `I could not answer that: ${detail}`,
        timestamp: new Date(),
        isError: true,
      };
      setMessages((prev) => [...prev, aiMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const suggestedPrompts = [
    "What can you help me learn?",
    "How do I generate study materials?",
    "What documents have I uploaded?",
    "How do I publish a study website?",
  ];

  return (
    <Layout activeNav="tutor">
      <div className="h-screen flex flex-col pt-16 md:pl-0" style={{ maxHeight: "100vh" }}>
        <div className="flex flex-1 overflow-hidden">
          {/* Chat Area */}
          <div className="flex-1 flex flex-col">
            {/* Chat Header */}
            <div
              className="px-6 md:px-12 py-6 flex-shrink-0"
              style={{ borderBottom: "1px solid rgba(72,72,72,0.1)" }}
            >
              <div className="flex items-center gap-4">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center relative"
                  style={{ background: "linear-gradient(to top right, #2a3e46, #1f2020)" }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: "20px", color: "#8ca1aa" }}>psychology</span>
                  <span
                    className="absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 animate-pulse"
                    style={{ backgroundColor: "#edfee9", borderColor: "#0e0e0e" }}
                  />
                </div>
                <div>
                  <h1 className="font-bold" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>AI Tutor</h1>
                  <p className="text-xs" style={{ color: "#8ca1aa" }}>
                    {sendMessageMutation.isPending ? "Processing..." : activeProjectName ? `Active on: ${activeProjectName}` : "No project selected"}
                  </p>
                </div>
              </div>
              {projectsLoading && (
                <p className="text-xs mt-2" style={{ color: "#484848" }}>Loading projects...</p>
              )}
              {projectsError && (
                <p className="text-xs mt-2" style={{ color: "#ee7d77" }}>Could not load projects. Check your connection and log in again.</p>
              )}
              {sessionError && (
                <p className="text-xs mt-2" style={{ color: "#ee7d77" }}>{sessionError}</p>
              )}
              {!projectsLoading && !activeProjectId && !projectsError && (
                <p className="text-xs mt-2" style={{ color: "#acabaa" }}>Upload a document first to start a tutor session.</p>
              )}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-6 md:px-12 py-8 space-y-8 no-scrollbar">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                  {msg.role === "assistant" && (
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-1"
                      style={{ background: "linear-gradient(to top right, #2a3e46, #1f2020)" }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "16px", color: "#8ca1aa" }}>psychology</span>
                    </div>
                  )}
                  <div className={`max-w-2xl ${msg.role === "user" ? "text-right" : ""}`}>
                    <div
                      className="inline-block px-6 py-4 rounded-2xl text-sm leading-relaxed"
                      style={{
                        backgroundColor: msg.role === "user" ? "#252626" : "#131313",
                        color: msg.role === "user" ? "#e7e5e4" : msg.isError ? "#ee7d77" : "#acabaa",
                        border: msg.role === "assistant" ? "1px solid rgba(72,72,72,0.1)" : "none",
                        maxWidth: "100%",
                        textAlign: "left",
                      }}
                    >
                      <TutorMarkdown text={msg.content} />
                    </div>
                    {/* Retrieval context used for this answer */}
                    {msg.context && (
                      <details className="mt-2 text-left">
                        <summary className="text-xs cursor-pointer" style={{ color: "#8ca1aa" }}>
                          Retrieval context
                        </summary>
                        <p className="mt-1 text-xs leading-relaxed whitespace-pre-wrap" style={{ color: "#484848" }}>
                          {msg.context.slice(0, 800)}{msg.context.length > 800 ? "..." : ""}
                        </p>
                      </details>
                    )}
                    <Citations citations={msg.citations} sources={msg.sources} />
                    <p
                      className="mt-2 text-xs"
                      style={{ color: "#484848", fontFamily: "Space Grotesk, sans-serif" }}
                    >
                      {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </p>
                  </div>
                  {msg.role === "user" && (
                    <div
                      className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 text-xs font-bold"
                      style={{ backgroundColor: "#454747", color: "#c6c6c7" }}
                    >
                      R
                    </div>
                  )}
                </div>
              ))}

              {isTyping && (
                <div className="flex gap-4 justify-start">
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ background: "linear-gradient(to top right, #2a3e46, #1f2020)" }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "16px", color: "#8ca1aa" }}>psychology</span>
                  </div>
                  <div
                    className="px-6 py-4 rounded-2xl flex items-center gap-2"
                    style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.1)" }}
                  >
                    {[0, 1, 2].map((i) => (
                      <span
                        key={i}
                        className="w-2 h-2 rounded-full animate-pulse"
                        style={{ backgroundColor: "#acabaa", animationDelay: `${i * 0.2}s` }}
                      />
                    ))}
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Suggested Prompts */}
            <div className="px-6 md:px-12 pb-4 flex-shrink-0">
              <div className="flex gap-2 overflow-x-auto no-scrollbar">
                {suggestedPrompts.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => setInput(prompt)}
                    disabled={isTyping}
                    className="whitespace-nowrap px-3 py-1.5 rounded-full text-xs transition-all flex-shrink-0 disabled:opacity-50"
                    style={{
                      backgroundColor: "#1f2020",
                      color: "#acabaa",
                      fontFamily: "Space Grotesk, sans-serif",
                      border: "1px solid rgba(72,72,72,0.1)",
                    }}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Area */}
            <div
              className="px-6 md:px-12 py-6 flex-shrink-0"
              style={{ borderTop: "1px solid rgba(72,72,72,0.1)" }}
            >
              <div
                className="flex items-end gap-4 p-4 rounded-2xl"
                style={{
                  backgroundColor: "#131313",
                  border: "1px solid rgba(72,72,72,0.2)",
                }}
              >
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  className="flex-1 bg-transparent border-none focus:outline-none resize-none text-sm"
                  placeholder="Ask the monolith anything..."
                  rows={1}
                  disabled={isTyping}
                  style={{ color: "#e7e5e4", fontFamily: "Manrope, sans-serif", minHeight: "24px", maxHeight: "120px" }}
                />
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || isTyping}
                  className="w-10 h-10 rounded-xl flex items-center justify-center transition-all disabled:opacity-40 active:scale-90"
                  style={{
                    background: "linear-gradient(to top right, #c6c6c7, #454747)",
                    color: "#3f4041",
                  }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>send</span>
                </button>
              </div>
              <p
                className="text-center mt-3 text-xs"
                style={{ fontFamily: "Space Grotesk, sans-serif", color: "#484848" }}
              >
                Answers come from your study material
              </p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
