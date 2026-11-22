"""One-command semantic-cache smoke test (live stack required).

    docker compose exec api python -m scripts.smoke_semantic_cache

Invalidates the project cache, then checks:
1. first answer  -> miss
2. exact repeat  -> hit
3. paraphrase    -> hit or miss (reported, threshold-dependent)
4. near-miss     -> must NOT return the dot-product answer
5. stats endpoint
"""

from __future__ import annotations

import asyncio
import os

import httpx

BASE = "http://localhost:8000"
EMAIL = os.getenv("DEMO_EMAIL", "demo@trashthetest.dev")
PASSWORD = os.getenv("DEMO_PASSWORD", "DemoPass!123")
PROJECT = os.getenv("DEMO_PROJECT", "Python Programming")


async def ask(c, headers, pid, message):
    r = await c.post("/api/study/tutor/message", headers=headers,
                     json={"project_id": pid, "message": message})
    return r.status_code, (r.json() if r.status_code == 200 else {"raw": r.text[:200]})


async def main() -> int:
    async with httpx.AsyncClient(base_url=BASE, timeout=300) as c:
        login = await c.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
        login.raise_for_status()
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        projects = (await c.get("/api/projects", headers=headers)).json()["projects"]
        project = next((p for p in projects if p.get("title") == PROJECT), None)
        if project is None:
            print(f"project {PROJECT!r} not found")
            return 2
        pid = project["id"]
        await c.post("/api/semantic-cache/invalidate", json={"project_id": pid})

        results = {}
        base_q = "What do the light reactions produce?"
        status, first = await ask(c, headers, pid, base_q)
        results["first_miss"] = not first.get("cache_hit", False)
        status, second = await ask(c, headers, pid, base_q)
        results["exact_hit"] = bool(second.get("cache_hit"))

        status, para = await ask(c, headers, pid,
                                 "Which products come out of the light-dependent reactions?")
        results["paraphrase_hit"] = bool(para.get("cache_hit"))

        status, dot = await ask(c, headers, pid, "What is the dot product of two vectors?")
        status, near = await ask(c, headers, pid,
                                 "What is the cross product of two vectors?")
        results["near_miss_rejected"] = not near.get("cache_hit", False)
        # The cross-product question may legitimately have no source in this
        # project. What must never happen is reusing a *substantive*
        # dot-product answer; two identical no-evidence replies are fine.
        dot_resp = (dot.get("response") or "").lower()
        dot_missing = ("could not find" in dot_resp) or ("does not contain" in dot_resp)
        results["near_miss_not_dot_answer"] = dot_missing or (
            (near.get("response") or "") != (dot.get("response") or "")
        )

        stats = (await c.get("/api/semantic-cache/stats")).json()
        print("first_miss:", results["first_miss"])
        print("exact_hit:", results["exact_hit"])
        print("paraphrase_hit:", results["paraphrase_hit"])
        print("near_miss_rejected:", results["near_miss_rejected"])
        print("near_miss_not_dot_answer:", results["near_miss_not_dot_answer"])
        print("cache_stats:", stats)
        print("PASS" if all(results.values()) else "CHECK THE REPORTED VALUES")
        return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
