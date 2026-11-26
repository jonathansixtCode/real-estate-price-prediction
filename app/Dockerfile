FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY app/requirements_deploy.txt ./requirements_deploy.txt
RUN pip install --no-cache-dir -r requirements_deploy.txt

COPY src/ ./src/
COPY models/ ./models/
COPY app/ ./app/

EXPOSE 8000 8501

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
