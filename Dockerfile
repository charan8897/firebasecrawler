# Firebase Chain Chatbot - Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY config.py .
COPY firebase_chain_chatbot.py .
COPY chatbot.sh .

# Make chatbot.sh executable
RUN chmod +x chatbot.sh

# Create .env file location for mounting
RUN touch .env && chmod 644 .env

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import config; config.get_firestore_client()" || exit 1

# Default to interactive mode
ENTRYPOINT ["python3", "firebase_chain_chatbot.py"]
CMD []
