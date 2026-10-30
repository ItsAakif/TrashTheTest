# Model Strategy Layer for TrashTheTest

## Overview

This module implements the Model Strategy Layer - a foundational abstraction that allows AI/ML components to be interchangeable without affecting the rest of the system.

## ⚡️ Local Mode (NO API KEYS REQUIRED!)

The system is designed to run **entirely locally** with **NO external API keys needed**!

| Component | Provider | Local? | API Key Required? |
|-----------|----------|--------|-------------------|
| OCR | PP-OCRv6 tiny via onnxruntime (CPU) | ✅ Yes | ❌ NO |
| Embeddings | Qwen3-Embedding-0.6B 1024d (qwen3-embedding) | ✅ Yes | ❌ NO |
| LLM | Groq API | ❌ No | ✅ Yes (free tier) |
| TTS | Kokoro | ✅ Yes | ❌ NO |
| Search | Tavily + Jina | ❌ No | ✅ Yes |

## Architecture

```
services/strategies/
├── __init__.py
├── base.py          # Abstract base classes (Protocols)
├── ocr/
│   ├── __init__.py
│   └── ppocr_onnx.py  # PP-OCRv6 provider (CPU onnxruntime, tiny det/rec)
├── embedding/
│   ├── __init__.py
│   └── bge_small.py # Local embedding provider
├── llm/
│   ├── __init__.py
│   └── groq.py     # Groq API provider
├── tts/
│   ├── __init__.py
│   └── kokoro.py   # Kokoro TTS provider
└── search/
    ├── __init__.py
    └── tavily.py   # Tavily + Jina provider

config/
├── __init__.py
└── settings.py     # Central configuration

factory.py           # Strategy factory/registry
main.py             # FastAPI application entry point
requirements.txt    # Python dependencies
```

## 🚀 Quick Start (Local OCR)

### Step 1: Install Dependencies

```bash
cd artifacts/ai-services

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: PP-OCR models (CPU ONNX, tiny det/rec)

```bash
# PP-OCRv6 tiny models download on first use into the paddleocr cache.
# CPU-only: onnxruntime executes det/rec; no GPU wheel, no server needed.
pip install "paddleocr>=3.7" onnxruntime
```

### Step 3: Run the FastAPI Service

```bash
# From artifacts/ai-services (the app entry point is main:app at the
# package root; services.main does not exist).
uvicorn main:app --host 0.0.0.0 --port 8000

# Or via the __main__ entry point (same host/port from settings):
python main.py
```

### Step 4: Database migrations + background worker

```bash
# Apply migrations (needs DATABASE_URL + JWT_SECRET in .env):
alembic upgrade head

# Run the job worker in a second terminal. Postgres SKIP LOCKED is the
# single source of truth; --once claims one batch then exits (handy for
# smoke tests), --concurrency bounds in-flight jobs (default 2,
# WORKER_CONCURRENCY env wins when set):
python -m services.jobs.worker --once
python -m services.jobs.worker --concurrency 4
```

### Step 5: Vector index / reindex scripts (live Postgres required)

```bash
# After a bulk load, build the HNSW index (idempotent):
python -m scripts.build_vector_index --project <project-id>
python -m scripts.build_vector_index --project <project-id> --prod  # CONCURRENTLY

# Backfill Qwen3-1024 embeddings (resumable; refuses without --all):
python -m scripts.reindex_embeddings --all --batch-size 64
```

## Environment Variables

For full local operation, you only need:

```bash
# Optional: Groq API for LLM (free tier)
GROQ_API_KEY=your_groq_api_key

# Optional: Tavily for web search
TAVILY_API_KEY=your_tavily_api_key

# Server
HOST=0.0.0.0
PORT=8000
```

## API Endpoints

Once running, access the FastAPI docs at `http://localhost:8000/docs`

### OCR
- `POST /api/ocr/extract` - Extract text from file (PDF, image)

### Embeddings
- `POST /api/embeddings/generate` - Generate embeddings for texts

### LLM
- `POST /api/llm/generate` - Generate text with LLM

### TTS
- `POST /api/tts/synthesize` - Synthesize text to speech

### Search
- `POST /api/search/query` - Search the web
- `POST /api/search/extract` - Extract content from URL

## Troubleshooting

### OCR not working?
PP-OCR runs in-process on CPU via onnxruntime (no server):
```bash
# Check the provider health from Python
python -c "import asyncio;from services.strategies.ocr.ppocr_onnx import PPOCROnxProvider;print(asyncio.run(PPOCROnxProvider().health()))"
```

### Page-level routing
Normal pages take PP-OCR locally. Heavy documents (default ~100 pages,
`PUTER_HEAVY_PAGE_THRESHOLD`) may offer the optional user-authorised Puter
path instead; Puter is OFF by default and any failure degrades explicitly
(never silent, provenance preserved).