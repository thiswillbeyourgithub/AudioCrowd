# Lightweight Python image with uv for dependency management.
# The app uses PEP 723 inline script metadata, so uv run handles deps automatically.
FROM python:3.12-slim

# Install uv (fast Python package manager) and libsndfile (needed by soundfile).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY AudioCrowd.py .

# Pre-install dependencies so container startup is fast.
# Uses uv run which reads PEP 723 inline script metadata from AudioCrowd.py.
RUN uv run --no-project AudioCrowd.py --help || true

# Drop privileges: create a non-root user and remove setuid/setgid binaries.
RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid 1000 -m appuser \
    && find / -perm /6000 -type f -exec chmod a-s {} + 2>/dev/null || true
USER appuser

EXPOSE 7860

# uv run handles dependency resolution via PEP 723 inline metadata in AudioCrowd.py.
CMD ["uv", "run", "--no-project", "AudioCrowd.py"]
