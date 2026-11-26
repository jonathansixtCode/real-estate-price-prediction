FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY c:\Users\Jonathan\Documents\Development\real-estate-price-prediction\app/requirements_deploy.txt ./requirements_deploy.txt
RUN pip install --no-cache-dir -r requirements_deploy.txt

COPY src/ ./src/
COPY models/ ./models/
COPY c:\Users\Jonathan\Documents\Development\real-estate-price-prediction\app/ ./c:\Users\Jonathan\Documents\Development\real-estate-price-prediction\app/

EXPOSE 8000 8501

CMD ["uvicorn", "c:\Users\Jonathan\Documents\Development\real-estate-price-prediction\app.api:app", "--host", "0.0.0.0", "--port", "8000"]
