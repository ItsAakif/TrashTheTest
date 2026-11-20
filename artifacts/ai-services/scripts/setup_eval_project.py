"""Create/seed the RAG Eval project from tests/eval/fixtures.

Run inside the api container:
    python -m scripts.setup_eval_project
Uses the demo account (local seed) and waits for each ingestion job.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import httpx

BASE = "http://localhost:8000"
EMAIL = os.getenv("DEMO_EMAIL", "demo@trashthetest.dev")
PASSWORD = os.getenv("DEMO_PASSWORD", "DemoPass!123")


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="RAG Eval")
    parser.add_argument("--dir", default="tests/eval/fixtures")
    args = parser.parse_args(argv)
    fixtures = Path(args.dir)
    title = args.title

    async with httpx.AsyncClient(base_url=BASE, timeout=300) as c:
        r = await c.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
        r.raise_for_status()
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

        projects = (await c.get("/api/projects", headers=headers)).json()["projects"]
        project = next((p for p in projects if p.get("title") == title), None)
        if project is None:
            r = await c.post(
                "/api/projects",
                json={"title": title, "description": "Retrieval evaluation fixtures"},
                headers=headers,
            )
            r.raise_for_status()
            project = r.json()
        pid = project["id"]
        print("project", pid)

        for path in sorted(fixtures.glob("*.txt")):
            with path.open("rb") as fh:
                r = await c.post(
                    "/api/documents/upload",
                    files={"file": (path.name, fh, "text/plain")},
                    data={"project_id": pid, "source_type": "study_material"},
                    headers=headers,
                )
            if r.status_code >= 300:
                print("upload failed", path.name, r.status_code, r.text[:160])
                continue
            body = r.json()
            jid = body.get("jobId")
            print("uploaded", path.name, "job", jid)
            for _ in range(120):
                prog = await c.get(f"/api/jobs/{jid}/progress", headers=headers)
                if prog.status_code == 200:
                    j = prog.json()
                    if j.get("status") in ("completed", "failed", "cancelled"):
                        print("  ", j.get("status"), j.get("stage"))
                        break
                await asyncio.sleep(3)

        print("PROJECT_ID", pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
