"""T101: fail-fast settings, config-sourced model ids, .env.example coverage."""

import os
import subprocess
import sys

import pytest

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_without_secrets(code, cwd):
    env = {k: v for k, v in os.environ.items() if k not in ("JWT_SECRET", "DATABASE_URL")}
    env["PYTHONPATH"] = SERVICE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=60,
    )


def test_import_without_jwt_secret_names_the_variable(tmp_path):
    """Importing config.settings with no secrets fails naming the variable."""
    proc = _run_without_secrets("import config.settings", str(tmp_path))
    assert proc.returncode != 0
    assert "JWT_SECRET" in (proc.stderr or "")


def test_import_without_database_url_names_the_variable(tmp_path, monkeypatch):
    """JWT present but DATABASE_URL missing still fail-fasts by name."""
    monkeypatch.setenv("JWT_SECRET", "test-secret-at-least-32-chars-long-0123456789")
    env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    env["PYTHONPATH"] = SERVICE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", "import config.settings"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "DATABASE_URL" in (proc.stderr or "")


def test_no_hardcoded_llama_model_ids():
    """Model ids come only from config: no llama-3 literal in code."""
    offenders = []
    needle = "llama" + "-3"  # built, not literal
    for root, _, files in os.walk(SERVICE_DIR):
        if ".venv" in root or "__pycache__" in root or os.path.basename(root) == "tests":
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for lineno, line in enumerate(fh, 1):
                    if needle in line:
                        offenders.append(f"{path}:{lineno}")
    assert offenders == [], f"hardcoded model ids: {offenders}"


def test_env_example_documents_every_variable():
    """Every _Env field is documented in .env.example (placeholders only)."""
    from config.settings import _Env

    example_path = os.path.join(SERVICE_DIR, ".env.example")
    with open(example_path, encoding="utf-8") as fh:
        example = fh.read()
    missing = [f for f in _Env.model_fields if f not in example]
    assert missing == [], f"undocumented variables: {missing}"


def test_wildcard_cors_refused_at_boot(monkeypatch):
    """CORS_ORIGINS='*' raises instead of booting with credentials+wildcard."""
    from config.settings import _Env

    monkeypatch.setenv("JWT_SECRET", "test-secret-at-least-32-chars-long-0123456789")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(Exception, match=r"\*"):
        _Env()
