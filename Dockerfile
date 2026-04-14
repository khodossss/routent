FROM python:3.12-slim

WORKDIR /app

# System deps for sentence-transformers / transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY routent/requirements.txt ./requirements.txt

# CPU-only PyTorch (much smaller image)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir python-dotenv datasets tiktoken sentence-transformers \
        langchain-core langchain-openai langchain-google-genai langchain-huggingface

COPY routent/ ./routent/

ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

ENTRYPOINT ["python", "-u", "routent/scripts/train.py"]
