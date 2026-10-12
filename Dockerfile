# vocnu-engine with ffmpeg and English OCR for image subtitles. Speech recognition runs on the CPU here;
# for an NVIDIA GPU, install with `uv sync --extra cuda` on the host instead.
FROM python:3.12-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr tesseract-ocr-eng fonts-noto-core \
  && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

WORKDIR /engine
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY vocnu_engine ./vocnu_engine
RUN uv sync --frozen --no-dev
ENV PATH="/engine/.venv/bin:$PATH" XDG_CACHE_HOME=/cache

WORKDIR /media
ENTRYPOINT ["vocnu-engine"]
