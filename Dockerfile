# ──────────────────────────────────────────────────────────────────────────────
# PyTorch NGC base — CUDA 12.6, Python 3.10, PyTorch pre-installed
# Swap the tag for whichever NGC release matches your driver:
#   24.12-py3  →  CUDA 12.6  (driver ≥ 525)
#   24.09-py3  →  CUDA 12.6  (driver ≥ 525)
#   24.03-py3  →  CUDA 12.4  (driver ≥ 520)
# ──────────────────────────────────────────────────────────────────────────────
ARG NGC_TAG=24.12-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (skip torch — already in the NGC image) ──────────────────────
COPY requirements.txt .
RUN grep -vE "^(torch|#)" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# ── App source ────────────────────────────────────────────────────────────────
COPY . .

# ── Streamlit port ────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Default: launch dashboard (override with `docker run ... python train_and_backtest.py ...`) ──
ENTRYPOINT ["streamlit", "run", "dashboard.py", \
            "--server.address=0.0.0.0", \
            "--server.port=8501", \
            "--server.headless=true"]
