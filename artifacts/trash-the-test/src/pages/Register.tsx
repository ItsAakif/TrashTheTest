import { useState } from "react";
import { useLocation } from "wouter";
import { useQueryClient } from "@tanstack/react-query";
import Layout from "@/components/Layout";
import { register } from "@/lib/api-services";

export default function Register() {
  const [, navigate] = useLocation();
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setPending(true);
    try {
      // POST /api/auth/register; JWT persisted to localStorage by the helper.
      const { user } = await register(email.trim(), password, name.trim());
      // Same session-seeding as Login: a fresh-cached ["me"] = null from an
      // earlier logged-out guard check must not shadow the new session.
      queryClient.setQueryData(["me"], user);
      navigate("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed.");
    } finally {
      setPending(false);
    }
  };

  return (
    <Layout activeNav="dashboard">
      <div className="pb-20 md:pb-8 px-6 md:px-12 min-h-screen pt-10">
        <div className="max-w-md mx-auto">
          <h1 className="text-4xl font-extrabold tracking-tight mb-2" style={{ fontFamily: "Inter, sans-serif", color: "#e7e5e4" }}>
            Register
          </h1>
          <p className="mb-8" style={{ color: "#acabaa" }}>
            Create an account to upload documents and generate study materials.
          </p>
          <form onSubmit={submit} className="p-6 rounded-2xl flex flex-col gap-4" style={{ backgroundColor: "#131313" }}>
            <label className="flex flex-col gap-2 text-sm" style={{ color: "#acabaa" }}>
              Name
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="px-4 py-3 rounded-xl text-sm"
                style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.3)" }}
              />
            </label>
            <label className="flex flex-col gap-2 text-sm" style={{ color: "#acabaa" }}>
              Email
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="px-4 py-3 rounded-xl text-sm"
                style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.3)" }}
              />
            </label>
            <label className="flex flex-col gap-2 text-sm" style={{ color: "#acabaa" }}>
              Password (min 8 characters)
              <input
                type="password"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="px-4 py-3 rounded-xl text-sm"
                style={{ backgroundColor: "#1f2020", color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.3)" }}
              />
            </label>
            {error && (
              <p className="text-sm" style={{ color: "#ee7d77" }}>{error}</p>
            )}
            <button
              type="submit"
              disabled={pending}
              className="py-3 rounded-xl font-semibold transition-all hover:opacity-90 disabled:opacity-50"
              style={{ backgroundColor: "#edfee9", color: "#131313" }}
            >
              {pending ? "Registering..." : "Register"}
            </button>
            <button
              type="button"
              onClick={() => navigate("/login")}
              className="text-sm"
              style={{ color: "#8ca1aa" }}
            >
              Have an account? Log in
            </button>
          </form>
        </div>
      </div>
    </Layout>
  );
}
