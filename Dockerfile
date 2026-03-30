FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable defaults (can be overridden at runtime)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4-turbo"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('OK')"

# Run the inference script
CMD ["python", "inference.py"]