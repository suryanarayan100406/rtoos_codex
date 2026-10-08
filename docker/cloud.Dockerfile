FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/drishti
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY server ./server
RUN python3 -m pip install --no-cache-dir -e ".[ingest,poses,neural,geo,server]"
COPY . .
ENTRYPOINT ["drishti"]
