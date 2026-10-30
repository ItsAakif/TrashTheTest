# TrashTheTest ai-services (CPU-only default; GPU not required for tier 0/1).
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
# Wheels-only friendly install: modern pip/setuptools/wheel avoids legacy
# setup.py source builds (e.g. old grpcio sdist -> missing pkg_resources).
# CPU-only: never install GPU wheels (no paddlepaddle-gpu / CUDA builds).
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
