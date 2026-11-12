FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY server.py .
# repo@version: the image is built with, and keeps running, exactly this version
ENV SETER_MODEL=shoresh-labs/seter-pii-he-ar-fast@v1.1
# download the model at build time so the container runs offline
# optional Hugging Face token (private/gated models, higher rate limits), passed as a build
# secret so it is never stored in an image layer: docker build --secret id=hf_token,env=HF_TOKEN .
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python -c "import os; from transformers import pipeline; m, _, r = os.environ['SETER_MODEL'].partition('@'); pipeline('token-classification', model=m, revision=r or None)"
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
