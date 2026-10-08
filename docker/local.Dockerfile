FROM python:3.12-slim

WORKDIR /opt/drishti
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY server ./server
RUN python -m pip install --no-cache-dir -e ".[ingest,poses,geo,server]"
COPY . .
ENTRYPOINT ["drishti"]
