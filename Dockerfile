FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY server.py .
ENV SETER_MODEL=shoresh-labs/seter-pii-he-ar-fast
# download the model at build time so the container runs offline
# optional Hugging Face token (private/gated models, higher rate limits), passed as a build
# secret so it is never stored in an image layer: docker build --secret id=hf_token,env=HF_TOKEN .
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python -c "from transformers import pipeline; pipeline('token-classification', model='${SETER_MODEL}')"
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
