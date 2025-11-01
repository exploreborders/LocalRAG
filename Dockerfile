FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models
RUN python -m spacy download de_core_news_sm && \
    python -m spacy download fr_core_news_sm && \
    python -m spacy download es_core_news_sm && \
    python -m spacy download it_core_news_sm && \
    python -m spacy download pt_core_news_sm && \
    python -m spacy download nl_core_news_sm && \
    python -m spacy download sv_core_news_sm && \
    python -m spacy download pl_core_news_sm && \
    python -m spacy download zh_core_web_sm && \
    python -m spacy download ja_core_news_sm && \
    python -m spacy download ko_core_news_sm

# Copy the application
COPY . .

# Create data directory
RUN mkdir -p data models

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Run the application
CMD ["streamlit", "run", "web_interface/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]