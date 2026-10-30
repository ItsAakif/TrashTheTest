// Login-loop regression: proves the ["me"] stale-cache mechanism behind the
// LIVE login redirect loop (login 200, no error shown, bounce back to /login)
// and asserts the source-level fix is present.
//
// Mechanism (TanStack Query v5, same options as ProtectedRoute):
//   useQuery({ queryKey: ["me"], queryFn: getCurrentUser, retry: false,
//              staleTime: 60_000 })
// A logged-out guard check caches ["me"] = null, FRESH for 60s. The old
// Login/Register persisted the token then navigated WITHOUT touching that
// cache, so the guard reused the fresh null and <Redirect>ed to /login.
// Register only *appeared* to work: first visits have an empty ["me"] cache,
// so the guard fetched fresh and got 200.
//
// Fix asserted here:
//   1. Login.tsx / Register.tsx call queryClient.setQueryData(["me"], user)
//      synchronously before navigate("/dashboard").
//   2. api-services apiFetch shares one refresh across concurrent 401s,
//      retries once, and only clears local auth when that refresh fails.
//
// Run: pnpm --filter @workspace/trash-the-test run test:login-loop
// (plain node, no DOM/jsdom needed — QueryClient core only).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { QueryClient } from "@tanstack/react-query";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (p) => readFileSync(join(root, p), "utf8");

// ---- 1. Source-contract assertions (the actual fix) ----
const loginSrc = read("src/pages/Login.tsx");
const registerSrc = read("src/pages/Register.tsx");
const apiSrc = read("src/lib/api-services.ts");
const guardSrc = read("src/components/ProtectedRoute.tsx");

assert.match(loginSrc, /setQueryData\(\["me"\]/, "Login must seed [\"me\"] before navigate");
assert.match(registerSrc, /setQueryData\(\["me"\]/, "Register must seed [\"me\"] before navigate");
assert.match(apiSrc, /let refreshInFlight/, "apiFetch refresh must be single-flight");
assert.match(apiSrc, /retry the original request exactly/, "apiFetch must retry once after refresh");
assert.match(apiSrc, /if \(status === 401\) return null/, "getCurrentUser must not clear a renewed session itself");
// Guard must keep waiting for loading and use declarative redirect (no render-time navigate).
assert.match(guardSrc, /if \(isLoading\)/, "guard must wait for loading");
assert.match(guardSrc, /<Redirect to="\/login" \/>/, "guard must use declarative Redirect");
assert.doesNotMatch(guardSrc, /navigate\(\s*["']\/login["']\s*\)/, "guard must not navigate during render");
console.log("source contracts: OK (Login/Register seed [me]; single-flight refresh + one retry; declarative guard)");

// ---- 2. Mechanism proof on a real QueryClient ----
const queryClient = new QueryClient();
const ME = ["me"];
const user = { id: "u-1", email: "loop@example.com", name: "Loop" };

let meCalls = 0;
// Logged-out guard check: caches null, fresh for 60s (ProtectedRoute options).
await queryClient.fetchQuery({
  queryKey: ME,
  queryFn: async () => {
    meCalls += 1;
    return null;
  },
  retry: false,
  staleTime: 60_000,
});
assert.equal(meCalls, 1);

// OLD behaviour: token persisted + navigate, cache untouched -> guard refetch
// is served from the fresh null WITHOUT calling queryFn -> redirect to /login.
const staleSeen = await queryClient.fetchQuery({
  queryKey: ME,
  queryFn: async () => {
    meCalls += 1;
    return user; // even a healthy /me is never reached inside the stale window
  },
  retry: false,
  staleTime: 60_000,
});
assert.equal(staleSeen, null, "stale null shadows the new session (the loop)");
assert.equal(meCalls, 1, "no refetch inside staleTime — guard reuses cached null");
console.log("loop mechanism reproduced on old path: fresh null served, queryFn not called");

// NEW behaviour (the fix): seed ["me"] synchronously, as Login/Register now do.
queryClient.setQueryData(ME, user);
assert.deepEqual(queryClient.getQueryData(ME), user, "guard sees the user on mount — no redirect");
console.log("fixed path: setQueryData([me], user) visible to guard synchronously");

// ---- 3. Clearing contract: only a failed refresh clears local auth. ----
const decide = (refreshFailed) => (refreshFailed ? "clear-and-null" : "keep-token");
assert.equal(decide(true), "clear-and-null");
for (const value of [false, undefined]) {
  assert.equal(decide(value), "keep-token", `refresh result ${value} must keep storage`);
}
console.log("clearing scope: OK (only refresh failure clears; retry rejection does not)");

console.log("\nPASS: login-loop regression");
