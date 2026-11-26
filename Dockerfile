# =============================================================================
# Stage 1: Build Frontend
# =============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies
COPY frontend/package*.json ./
RUN npm ci

# Build production bundle
COPY frontend/ ./
RUN npm run build

# =============================================================================
# Stage 2: Python Runtime
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch first, so the requirements install does not pull the CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Export the SigLIP 2 text tower (all the app needs at query time), then drop
# the full checkpoint so the vision tower never ships in the image.
# The model must match the one the corpus embeddings were built with.
ARG SIGLIP_MODEL_ID=google/siglip2-so400m-patch14-384
ENV SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID}
# Only the small config module and the export script go in before the export, so
# editing scorer.py does not invalidate this layer (and its multi-gigabyte download)
COPY model/siglip_config.py model/
COPY scripts/export_text_tower.py scripts/
RUN python scripts/export_text_tower.py && rm -rf /root/.cache/huggingface
ENV HF_HUB_OFFLINE=1

# Copy application code and the curated diagram corpus
COPY model/*.py model/
COPY main.py auth.py ./
COPY agent/ agent/
COPY mcp_server/ mcp_server/
COPY corpus/metadata.json corpus/embeddings.npz corpus/
COPY corpus/images/ corpus/images/

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Railway uses PORT env var (default 8080 for local testing)
ENV PORT=8080
EXPOSE ${PORT}

# Start server (Railway sets PORT automatically)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
