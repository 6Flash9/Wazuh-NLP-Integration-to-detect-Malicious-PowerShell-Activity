FROM python:3.9-slim

   # Update package list and install dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       python3-dev \
       && rm -rf /var/lib/apt/lists/*

   WORKDIR /app

   COPY api.py .
   COPY model.joblib .
   COPY vectorizer.joblib .

   RUN pip install --no-cache-dir flask joblib requests scikit-learn waitress

   CMD ["python", "api.py"]
