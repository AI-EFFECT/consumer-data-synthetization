FROM python:3.11-slim

LABEL maintainer="carlos.t.santos@inesctec.pt"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get -y install curl && rm -rf /var/lib/apt/lists/*

ARG REQUIREMENTS_FILE=requirementsCPU.txt

COPY synthetic_data_generation/${REQUIREMENTS_FILE} ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY synthetic_data_generation/dgan_wrapper.py .
COPY synthetic_data_generation/doppelganger.py .
COPY synthetic_model_training/main.py .
COPY synthetic_model_training/common ./common
COPY auth.py ./auth.py

EXPOSE 601

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:601/docs || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "601"]
