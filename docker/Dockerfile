# syntax=docker/dockerfile:1
# One Dockerfile, one build target per service (see docker/docker-compose.yml):
#   stt     faster-whisper, GPU          tts     Supertonic + ffmpeg, GPU
#   ml      embed + gender (CPU torch)   app     bot (calls) + web (pages)
# Build context is the repo root.

ARG PYTHON_IMAGE=python:3.11-slim

# ---------------------------------------------------------------- base
FROM ${PYTHON_IMAGE} AS base
# pip: long timeout + retries (the NVIDIA wheels are hundreds of MB); downloads are kept in a
# BuildKit cache mount (not in the image), so a failed build resumes instead of starting over.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 \
    PIP_CACHE_DIR=/root/.cache/pip \
    HOME=/home/app PYTHONPATH=/app
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -u 1000 -m -d /home/app app
WORKDIR /app

# ---------------------------------------------------------------- CUDA 12 runtime libs
# onnxruntime-gpu (TTS) and CTranslate2 (Whisper) both need cuBLAS 12, cuDNN 9,
# cuFFT 11, cuRAND 10 and cudart 12. The GPU driver comes from the host (nvidia runtime).
FROM base AS cuda12
RUN --mount=type=cache,target=/root/.cache/pip pip install \
      "nvidia-cuda-runtime-cu12==12.9.*" "nvidia-cublas-cu12==12.9.*" "nvidia-cudnn-cu12==9.*" \
      "nvidia-curand-cu12==10.3.*" "nvidia-cufft-cu12==11.4.*" "nvidia-cuda-nvrtc-cu12==12.9.*"
ENV NV=/usr/local/lib/python3.11/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NV}/cuda_runtime/lib:${NV}/cublas/lib:${NV}/cudnn/lib:${NV}/curand/lib:${NV}/cufft/lib:${NV}/cuda_nvrtc/lib

# ---------------------------------------------------------------- stt
FROM cuda12 AS stt
COPY docker/requirements/stt.txt /tmp/req.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/req.txt
COPY ari_app /app/ari_app
COPY services /app/services
USER app
WORKDIR /app/services
EXPOSE 8101
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s \
  CMD curl -fs http://127.0.0.1:8101/health | grep -q '"ready":true'
CMD ["uvicorn", "stt_service:app", "--host", "0.0.0.0", "--port", "8101"]

# ---------------------------------------------------------------- tts
FROM cuda12 AS tts
COPY docker/requirements/tts.txt /tmp/req.txt
# supertonic pulls the CPU onnxruntime; replace it with the GPU build
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/req.txt \
 && pip uninstall -y onnxruntime || true \
 && pip install "onnxruntime-gpu==1.26.0"
COPY ari_app /app/ari_app
COPY services /app/services
USER app
WORKDIR /app/services
EXPOSE 8102
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s \
  CMD curl -fs http://127.0.0.1:8102/health | grep -q '"ready":true'
CMD ["uvicorn", "tts_service:app", "--host", "0.0.0.0", "--port", "8102"]

# ---------------------------------------------------------------- embed + gender (CPU torch)
FROM base AS ml
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
COPY docker/requirements/ml.txt /tmp/req.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/req.txt
# librosa's numba JIT caches next to its install (read-only for the app user) -> use /tmp
ENV NUMBA_CACHE_DIR=/tmp/numba
COPY ari_app /app/ari_app
COPY services /app/services
USER app
WORKDIR /app/services
# command (embed_service / gender_service) and port are set in docker-compose.yml

# ---------------------------------------------------------------- bot + web
FROM base AS app
COPY docker/requirements/app.txt /tmp/req.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/req.txt
ENV NUMBA_CACHE_DIR=/tmp/numba
COPY ari_app /app/ari_app
COPY registration_questions.json complaint_questions.json /app/
USER app
WORKDIR /app
CMD ["python", "-m", "ari_app.server"]
