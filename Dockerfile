FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY server.py .
ENV SETER_MODEL=shoresh-labs/seter-pii-he-ar-fast
# download the model at build time so the container runs offline
RUN python -c "from transformers import pipeline; pipeline('token-classification', model='${SETER_MODEL}')"
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
