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

# Copy the application
COPY . .

# Create data directory
RUN mkdir -p data models

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "web_interface/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]