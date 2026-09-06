FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 7860
ENV PORT=7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "web_mobile_app:app", "--timeout", "120", "--workers", "1"]
