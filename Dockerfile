FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY server.py .
# repo@version. Another model: docker build --build-arg SETER_MODEL=shoresh-labs/seter-pii-he-ar@v1.2 .
ARG SETER_MODEL=shoresh-labs/seter-pii-he-ar-fast@v1.1
ENV SETER_MODEL=$SETER_MODEL SETER_BAKED_MODEL=$SETER_MODEL
# download the model at build time so the container runs offline
# optional Hugging Face token (private/gated models, higher rate limits), passed as a build
# secret so it is never stored in an image layer: docker build --secret id=hf_token,env=HF_TOKEN .
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python -c "import server; print('baked in:', server.MODEL, server.REVISION, server.COMMIT)"
# The model is in the image; never ask the Hub again. A tag can be moved, and without this line a container
# with network access would re-resolve it at every start. Only the exact SETER_MODEL above can be loaded
# (the same repo without its @version cannot: that means "newest", which needs the Hub). For anything else,
# rebuild with --build-arg, or start the container with -e HF_HUB_OFFLINE=0.
ENV HF_HUB_OFFLINE=1
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
