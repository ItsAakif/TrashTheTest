import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import Layout from "@/components/Layout";
import {
  getCurrentUser,
  getProjects,
  getProviderStatus,
  listProviderModels,
  removeProviderSettings,
  saveProviderSettings,
  testProviderConnection,
  logout,
  type AIProviderName,
  type AIProviderStatus,
  type AIModelEntry,
} from "@/lib/api-services";

const inputStyle: React.CSSProperties = {
  backgroundColor: "#1f2020",
  color: "#e7e5e4",
  border: "1px solid rgba(72,72,72,0.2)",
  fontFamily: "Inter, sans-serif",
};

function formatContext(contextLength?: number): string | null {
  if (!contextLength || contextLength <= 0) return null;
  if (contextLength >= 1000) return `${Math.round(contextLength / 1000)}k ctx`;
  return `${contextLength} ctx`;
}

function ModelCombobox({
  provider,
  value,
  onChange,
  preserveId,
}: {
  provider: AIProviderName;
  value: string;
  onChange: (id: string) => void;
  /** Currently effective/saved id: always kept visible so it never orphans. */
  preserveId?: string;
}) {
  const [input, setInput] = useState(value);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const [debounced, setDebounced] = useState("");
  const boxRef = useRef<HTMLDivElement>(null);

  // Keep the textbox in sync when a selection is committed elsewhere
  // (e.g. after Save clears it), but never clobber in-progress typing
  // while the dropdown is open.
  useEffect(() => {
    if (!open) setInput(value);
  }, [value, open]);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(input.trim()), 250);
    return () => clearTimeout(timer);
  }, [input]);

  const {
    data: catalog,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["provider-models", provider, debounced],
    queryFn: async ({ signal }) => {
      const base = {
        provider,
        ...(debounced ? { search: debounced } : {}),
        limit: 100,
      } as const;
      // Live first (server 1h cache + static fallback). Any transport /
      // HTTP failure here must NOT kill the dropdown: fall back to the
      // curated static list client-side and surface a subtle notice.
      // 401/403 is different — the JWT is dead, so prompt sign-in
      // instead of blaming the catalog.
      try {
        return await listProviderModels({ ...base, live: true, signal });
      } catch (err) {
        if (signal?.aborted) throw err;
        const status = (err as { status?: number })?.status;
        if (status === 401 || status === 403) {
          throw new Error("Session expired — please sign in again.");
        }
        try {
          const fallback = await listProviderModels({ ...base, live: false, signal });
          return { ...fallback, source: fallback.source ?? "static" };
        } catch {
          throw err;
        }
      }
    },
    staleTime: 60_000,
    // One retry for transient blips; never retry auth failures or aborts
    // (each debounced keystroke already refetches — 3x retry per
    // keystroke was hammering the API with 401s).
    retry: (count, err) => {
      const status = (err as { status?: number })?.status;
      if (status === 401 || status === 403) return false;
      if ((err as Error)?.name === "AbortError") return false;
      if ((err as Error)?.message === "Session expired — please sign in again.") return false;
      return count < 1;
    },
  });
  const source = catalog?.source;
  // Static fallback (client or server side) still renders the full list
  // with a subtle notice + Retry — never a dead error.
  const isFallback = source === "static" || source === "mixed" || source === "cache";
  const isAuthError =
    (error as Error | null)?.message === "Session expired — please sign in again.";
  const entries: AIModelEntry[] = catalog?.models?.[provider] ?? [];

  // Client-side filter for instant feedback between debounced fetches.
  const filtered = useMemo(() => {
    const needle = input.trim().toLowerCase();
    const list = needle
      ? entries.filter(
          (e) =>
            e.id.toLowerCase().includes(needle) ||
            e.label.toLowerCase().includes(needle) ||
            (e.name ?? "").toLowerCase().includes(needle)
        )
      : entries;
    // Preserve the current/saved value even when it is absent from the
    // live list (e.g. openrouter/auto or a retired id).
    const ids = new Set(list.map((e) => e.id));
    const out = [...list];
    for (const pin of [value, preserveId]) {
      if (pin && !ids.has(pin)) {
        out.unshift({ id: pin, label: `${pin} (current)` });
        ids.add(pin);
      }
    }
    return out.slice(0, 100);
  }, [entries, input, value, preserveId]);

  useEffect(() => {
    setHighlight(0);
  }, [filtered.length, debounced]);

  const commit = (id: string) => {
    onChange(id.trim());
    setInput(id.trim());
    setOpen(false);
  };

  return (
    <div ref={boxRef} className="relative">
      <input
        value={input}
        onChange={(e) => {
          setInput(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          // Delay so click-to-select still fires before close.
          setTimeout(() => setOpen(false), 120);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setOpen(true);
            setHighlight((h) => Math.min(h + 1, Math.max(filtered.length - 1, 0)));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setHighlight((h) => Math.max(h - 1, 0));
          } else if (e.key === "Enter") {
            e.preventDefault();
            if (open && filtered[highlight]) commit(filtered[highlight].id);
            else if (input.trim()) commit(input);
          } else if (e.key === "Escape") {
            setOpen(false);
          }
        }}
        placeholder="Search models or type a custom id..."
        role="combobox"
        aria-expanded={open}
        aria-label="Model"
        autoComplete="off"
        className="px-3 py-2 rounded-xl text-sm focus:outline-none w-full"
        style={inputStyle}
      />
      {open && (
        <div
          className="absolute z-10 mt-1 w-full rounded-xl overflow-hidden"
          style={{ backgroundColor: "#1f2020", border: "1px solid rgba(72,72,72,0.3)" }}
        >
          {isLoading && (
            <p className="px-3 py-2 text-sm" style={{ color: "#acabaa" }}>
              Loading models...
            </p>
          )}
          {isError && (
            <div className="px-3 py-2 text-sm flex items-center justify-between gap-2">
              <span style={{ color: "#ee7d77" }}>
                {isAuthError
                  ? "Session expired — please sign in again."
                  : "Could not load models. Type a custom id or retry."}
              </span>
              <button
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => void refetch()}
                className="px-2 py-1 rounded-lg text-xs font-semibold"
                style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
              >
                Retry
              </button>
            </div>
          )}
          {!isLoading && !isError && filtered.length === 0 && (
            <p className="px-3 py-2 text-sm" style={{ color: "#acabaa" }}>
              {input.trim()
                ? `No matches — press Enter to use "${input.trim()}" as a custom id (validated on Test).`
                : "No models found."}
            </p>
          )}
          {!isLoading && !isError && filtered.length > 0 && (
            <ul role="listbox" className="max-h-64 overflow-auto py-1">
              {filtered.map((m, i) => {
                const hint = [
                  m.name && m.name !== m.id ? m.name : null,
                  formatContext(m.context_length),
                  m.pricing,
                ]
                  .filter(Boolean)
                  .join(" · ");
                return (
                  <li key={m.id}>
                    <button
                      type="button"
                      role="option"
                      aria-selected={value === m.id}
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => commit(m.id)}
                      onMouseEnter={() => setHighlight(i)}
                      className="w-full text-left px-3 py-2 text-sm"
                      style={{
                        backgroundColor: i === highlight ? "#2a3e46" : "transparent",
                        color: "#e7e5e4",
                      }}
                    >
                      <span style={{ fontFamily: "Space Grotesk, sans-serif" }}>{m.id}</span>
                      {hint && (
                        <span className="block text-xs" style={{ color: "#acabaa" }}>
                          {m.label !== m.id ? `${m.label}` : ""}
                          {m.label !== m.id && hint ? " · " : ""}
                          {m.label === m.id ? hint : hint.replace(`${m.label} · `, "")}
                        </span>
                      )}
                      {!hint && m.label !== m.id && (
                        <span className="block text-xs" style={{ color: "#acabaa" }}>
                          {m.label}
                        </span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
          {isFallback && !isLoading && !isError && (
            <div
              className="px-3 py-1 text-xs flex items-center justify-between gap-2"
              style={{ color: "#484848" }}
            >
              <span>Showing curated list (live catalog unreachable).</span>
              <button
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => void refetch()}
                className="px-2 py-0.5 rounded-lg text-xs font-semibold"
                style={{ backgroundColor: "#252626", color: "#acabaa" }}
              >
                Retry live
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AIProviderCard({
  provider,
  status,
  onChanged,
}: {
  provider: AIProviderName;
  status: AIProviderStatus;
  onChanged: () => void;
}) {
  const entry = status.providers.find((p) => p.provider === provider);
  const configured = entry?.configured ?? false;
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [apiKey, setApiKey] = useState("");
  const [replacing, setReplacing] = useState(false);
  const [notice, setNotice] = useState<{ kind: "ok" | "err"; text: string } | null>(null);

  // Live catalog + searchable combobox. Any well-formed custom id typed
  // into the combobox is also accepted and validated live on Test.
  const effectiveModel = useMemo(
    () => selectedModel || entry?.effective_model || "",
    [selectedModel, entry?.effective_model]
  );

  const saveMutation = useMutation({
    mutationFn: () => {
      const model = selectedModel.trim();
      if (!model) throw new Error("Choose a model or type a custom model id.");
      if (!apiKey.trim()) throw new Error("Enter your API key.");
      return saveProviderSettings({ provider, apiKey: apiKey.trim(), model });
    },
    onSuccess: (saved) => {
      setNotice({ kind: "ok", text: `Saved ${saved.provider} key (${saved.key_hint_masked}).` });
      setApiKey("");
      setReplacing(false);
      onChanged();
    },
    onError: (err) =>
      setNotice({ kind: "err", text: err instanceof Error ? err.message : "Save failed." }),
  });

  const testMutation = useMutation({
    mutationFn: () => {
      if (configured && !replacing && !apiKey.trim()) {
        return testProviderConnection({ provider, useSaved: true });
      }
      const model = selectedModel.trim();
      if (!model) throw new Error("Choose a model or type a custom model id.");
      if (!apiKey.trim()) throw new Error("Enter your API key to test.");
      return testProviderConnection({ provider, apiKey: apiKey.trim(), model });
    },
    onSuccess: (res) =>
      setNotice({
        kind: "ok",
        text: res.warning ?? `Connection ok (${res.latency_ms}ms).`,
      }),
    onError: (err) => {
      const raw = err instanceof Error ? err.message : "Test failed.";
      let text = raw;
      try {
        const parsed = JSON.parse(raw) as { code?: string; message?: string };
        if (parsed?.message) text = `${parsed.message}`;
      } catch {
        // Plain message already.
      }
      setNotice({ kind: "err", text });
    },
  });

  const removeMutation = useMutation({
    mutationFn: () => removeProviderSettings(provider),
    onSuccess: () => {
      setNotice({ kind: "ok", text: `Removed ${provider} key. Using app-provided key.` });
      setApiKey("");
      setSelectedModel("");
      setReplacing(false);
      onChanged();
    },
    onError: (err) =>
      setNotice({ kind: "err", text: err instanceof Error ? err.message : "Remove failed." }),
  });

  const title = provider === "groq" ? "Groq" : "OpenRouter";

  return (
    <div className="p-6 rounded-2xl" style={{ backgroundColor: "#131313" }}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-bold" style={{ color: "#e7e5e4" }}>
          {title}
        </h3>
        <span
          className="px-2 py-0.5 rounded text-xs uppercase tracking-widest"
          style={{
            fontFamily: "Space Grotesk, sans-serif",
            backgroundColor: configured ? "rgba(237,254,233,0.1)" : "rgba(140,161,170,0.1)",
            color: configured ? "#edfee9" : "#8ca1aa",
          }}
        >
          {configured ? "Using your key" : "app-provided"}
        </span>
      </div>

      {configured && !replacing ? (
        <div className="flex flex-col gap-3 mt-4">
          <p className="text-sm" style={{ color: "#acabaa" }}>
            Key <span style={{ fontFamily: "Space Grotesk, sans-serif" }}>{entry?.key_hint_masked}</span>
            {" · "}Model <span style={{ color: "#e7e5e4" }}>{entry?.model}</span>
          </p>
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => testMutation.mutate()}
              disabled={testMutation.isPending}
              className="px-4 py-2 rounded-xl text-sm font-semibold disabled:opacity-50"
              style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
            >
              {testMutation.isPending ? "Testing..." : "Test connection"}
            </button>
            <button
              onClick={() => { setReplacing(true); setNotice(null); }}
              className="px-4 py-2 rounded-xl text-sm font-semibold"
              style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
            >
              Replace
            </button>
            <button
              onClick={() => removeMutation.mutate()}
              disabled={removeMutation.isPending}
              className="px-4 py-2 rounded-xl text-sm font-semibold disabled:opacity-50"
              style={{ color: "#ee7d77", border: "1px solid rgba(238,125,119,0.3)" }}
            >
              {removeMutation.isPending ? "Removing..." : "Remove"}
            </button>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-3 mt-4">
          <label className="text-sm flex flex-col gap-1" style={{ color: "#acabaa" }}>
            Model (search live catalog or type a custom id)
            <ModelCombobox
              provider={provider}
              value={selectedModel}
              onChange={setSelectedModel}
              preserveId={entry?.effective_model || undefined}
            />
          </label>
          <label className="text-sm flex flex-col gap-1" style={{ color: "#acabaa" }}>
            API key (stored encrypted, never shown again)
            <input
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              type="password"
              autoComplete="off"
              placeholder={configured ? "Enter replacement key..." : "Enter your key..."}
              className="px-3 py-2 rounded-xl text-sm focus:outline-none"
              style={inputStyle}
            />
          </label>
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => testMutation.mutate()}
              disabled={testMutation.isPending}
              className="px-4 py-2 rounded-xl text-sm font-semibold disabled:opacity-50"
              style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
            >
              {testMutation.isPending ? "Testing..." : "Test"}
            </button>
            <button
              onClick={() => saveMutation.mutate()}
              disabled={saveMutation.isPending}
              className="px-4 py-2 rounded-xl text-sm font-semibold disabled:opacity-50"
              style={{ backgroundColor: "#edfee9", color: "#131313" }}
            >
              {saveMutation.isPending ? "Saving..." : "Save"}
            </button>
            {configured && (
              <button
                onClick={() => { setReplacing(false); setApiKey(""); setNotice(null); }}
                className="px-4 py-2 rounded-xl text-sm font-semibold"
                style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
              >
                Cancel
              </button>
            )}
          </div>
          {effectiveModel && (
            <p className="text-xs" style={{ color: "#484848" }}>
              Will use model: {effectiveModel}
            </p>
          )}
        </div>
      )}

      {notice && (
        <p className="text-sm mt-3" style={{ color: notice.kind === "ok" ? "#edfee9" : "#ee7d77" }}>
          {notice.text}
        </p>
      )}
    </div>
  );
}

export default function Settings() {
  const [, navigate] = useLocation();
  const queryClient = useQueryClient();
  const [activeProvider, setActiveProvider] = useState<AIProviderName>("groq");

  const { data: user } = useQuery({ queryKey: ["me"], queryFn: getCurrentUser });
  const { data: projects = [] } = useQuery({ queryKey: ["projects"], queryFn: getProjects });
  const {
    data: providerStatus,
    isLoading: statusLoading,
    isError: statusError,
    refetch: refetchStatus,
  } = useQuery({ queryKey: ["provider-status"], queryFn: getProviderStatus });

  const handleSignOut = async () => {
    try {
      await logout();
    } finally {
      queryClient.clear();
      navigate("/login");
    }
  };

  return (
    <Layout activeNav="overview">
      <div className="pb-20 md:pb-8 px-6 md:px-12 max-w-4xl mx-auto pt-10">
        <header className="mb-10">
          <h1
            className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4"
            style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4", lineHeight: "1.1" }}
          >
            Settings
          </h1>
          <p className="text-lg" style={{ color: "#acabaa" }}>
            Account, preferences, and your AI provider keys.
          </p>
        </header>

        {/* Profile */}
        <section className="p-6 rounded-2xl mb-6" style={{ backgroundColor: "#131313" }}>
          <h2 className="text-lg font-bold mb-4" style={{ color: "#e7e5e4" }}>
            Profile
          </h2>
          <div className="flex flex-col gap-2 text-sm" style={{ color: "#acabaa" }}>
            <p>
              Name: <span style={{ color: "#e7e5e4" }}>{user?.name || "-"}</span>
            </p>
            <p>
              Email: <span style={{ color: "#e7e5e4" }}>{user?.email || "-"}</span>
            </p>
            <p>
              Projects: <span style={{ color: "#e7e5e4" }}>{projects.length}</span>
            </p>
          </div>
          <div className="flex gap-2 mt-4 flex-wrap">
            <button
              onClick={() => navigate("/profile")}
              className="px-4 py-2 rounded-xl text-sm font-semibold"
              style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
            >
              View profile
            </button>
            <button
              onClick={() => void handleSignOut()}
              className="px-4 py-2 rounded-xl text-sm font-semibold"
              style={{ color: "#ee7d77", border: "1px solid rgba(238,125,119,0.3)" }}
            >
              Sign Out
            </button>
          </div>
        </section>

        {/* AI Provider */}
        <section className="mb-6">
          <h2 className="text-lg font-bold mb-2" style={{ color: "#e7e5e4" }}>
            AI Provider
          </h2>
          <p className="text-sm mb-4" style={{ color: "#acabaa" }}>
            Your saved key is used for Study Notes, Flashcards, Practice Questions,
            explanations, AI Tutor answers, and study websites. Without a saved key,
            the app-provided key is used.
          </p>
          {statusLoading && (
            <p className="text-sm" style={{ color: "#acabaa" }}>Loading provider status...</p>
          )}
          {statusError && (
            <div className="p-6 rounded-2xl" style={{ backgroundColor: "#131313" }}>
              <p className="text-sm mb-4" style={{ color: "#ee7d77" }}>
                Could not load provider status. Check your connection and try again.
              </p>
              <button
                onClick={() => refetchStatus()}
                className="px-4 py-2 rounded-xl text-sm font-semibold"
                style={{ backgroundColor: "#252626", color: "#e7e5e4" }}
              >
                Retry
              </button>
            </div>
          )}
          {providerStatus && (
            <>
              <div
                className="p-6 rounded-2xl mb-4"
                style={{ backgroundColor: "#131313", border: "1px solid rgba(72,72,72,0.1)" }}
              >
                <p className="text-sm" style={{ color: "#acabaa" }}>
                  Currently using:{" "}
                  <span style={{ color: "#e7e5e4" }}>
                    {providerStatus.effective.using_user_key ? "Using your key" : "app-provided"}
                    {" · "}
                    {providerStatus.effective.provider}
                    {" · "}
                    {providerStatus.effective.model || "default model"}
                  </span>
                </p>
              </div>
              <div className="flex gap-2 mb-4">
                {(["groq", "openrouter"] as AIProviderName[]).map((p) => (
                  <button
                    key={p}
                    onClick={() => setActiveProvider(p)}
                    className="px-4 py-2 rounded-xl text-sm font-semibold"
                    style={{
                      backgroundColor: activeProvider === p ? "#2a3e46" : "#1f2020",
                      color: activeProvider === p ? "#e7e5e4" : "#acabaa",
                      border: "1px solid rgba(72,72,72,0.2)",
                    }}
                  >
                    {p === "groq" ? "Groq" : "OpenRouter"}
                  </button>
                ))}
              </div>
              <AIProviderCard
                provider={activeProvider}
                status={providerStatus}
                onChanged={() => {
                  void refetchStatus();
                  void queryClient.invalidateQueries({ queryKey: ["provider-status"] });
                }}
              />
            </>
          )}
        </section>
      </div>
    </Layout>
  );
}
