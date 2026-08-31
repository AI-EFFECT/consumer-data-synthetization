FROM python:3.11-slim

LABEL maintainer="carlos.t.santos@inesctec.pt"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get -y install curl && rm -rf /var/lib/apt/lists/*

# Define build argument for requirements file (defaults to CPU version)
# This allows docker-compose to pass "requirementsGPU.txt" if needed
ARG REQUIREMENTS_FILE=requirementsCPU.txt

# Copy the specific requirements file to a standard name
# Note: The source path includes the folder because build context is root
COPY synthetic_data_generation/${REQUIREMENTS_FILE} ./requirements.txt

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY synthetic_data_generation/ .
COPY auth.py ./auth.py

# Expose the port
EXPOSE 600

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:600/docs || exit 1

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "600"]