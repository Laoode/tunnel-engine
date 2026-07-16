# Developer Guide: Streaming LLMs from S3-Compatible Object Storage to vLLM

A practical guide for serving Large Language Models directly from S3-compatible object storage using the Run:ai Model Streamer.

**Test Environment**
- **Model:** Qwen3.5-4B (9.3 GB)
- **vLLM:** v0.21+
- **Date:** 16 July 2026

---

# 1. Choosing an Object Storage

The Run:ai Model Streamer works with any **S3-compatible object storage**. The backend can be replaced without changing the vLLM serving workflow.

During this research the following providers were evaluated:

| Provider | Free Tier | S3 Compatible | Egress | Notes |
|-----------|-----------|---------------|---------|------|
| Backblaze B2 | 10 GB | ✓ | Limited | No credit card required. |
| Cloudflare R2 | 10 GB | ✓ | Free | Requires credit card. |
| Wasabi | 30-day Trial / 1 TB | ✓ | Free | Tested successfully. |
| MinIO | Self-hosted | ✓ | Local | Suitable for on-prem deployments. |

---

# 2. Bucket Best Practices

- Private bucket
- DNS-compatible bucket name
- Object Lock disabled
- Default encryption is sufficient
- Restrict credentials to the target bucket only

---

# 3. Create an S3 Application Key

Do **not** use provider master credentials.

Create an application key with:

- Read
- Write
- Bucket-scoped permission

Export credentials:

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

---

# 4. Upload Model

Download from Hugging Face:

```bash
pip install huggingface_hub awscli

hf download Qwen/Qwen3.5-4B \
    --local-dir Qwen3.5-4B
```

Upload to your bucket:

```bash
aws s3 sync \
    Qwen3.5-4B \
    s3://llm-qwen3.5-4b-bucket/Qwen3.5-4B \
    --endpoint-url https://your-s3-endpoint
```

---

# 5. Serve with vLLM

Install:

```bash
pip install "vllm[runai]"
```

Serve directly from object storage:

```bash
AWS_ACCESS_KEY_ID="..." \
AWS_SECRET_ACCESS_KEY="..." \
AWS_ENDPOINT_URL="https://your-s3-endpoint" \
RUNAI_STREAMER_S3_ENDPOINT="https://your-s3-endpoint" \
AWS_EC2_METADATA_DISABLED=true \
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
vllm serve s3://llm-qwen3.5-4b-bucket/Qwen3.5-4B \
    --load-format runai_streamer \
    --served-model-name Qwen3.5-4B \
    --attention-backend FLASHINFER \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000
```

Example request:

```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model":"Qwen3.5-4B",
  "messages":[
    {
      "role":"user",
      "content":"Hello"
    }
  ]
}'
```

---

# 6. Architecture

## Traditional Deployment

```
 Hugging Face Hub
         │
         ▼
 Local SSD
         │
         ▼
     CPU RAM
         │
         ▼
     GPU VRAM
         │
         ▼
       vLLM
```

Requires the full checkpoint on local storage.

---

## Object Storage Streaming

```
 S3-Compatible Bucket
         │
         ▼
 Run:ai Streamer
         │
         ▼
     CPU RAM
         │
         ▼
     GPU VRAM
         │
         ▼
       vLLM
```

No full checkpoint is permanently stored on local disk.

---

# 7. Storage Verification

A fresh Lightning AI instance was used for validation.

Observed after serving a **9.3 GB** model:

- Local storage usage remained unchanged.
- No `.safetensors` files were created.
- Only lightweight metadata (~23 MB) was cached.

Cached assets include:

```
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
README.md
metadata
```

No model weight files were found:

```bash
find ~/.cache -name "*.safetensors"
```

Result:

```
(no output)
```

This indicates that Run:ai Streamer does **not** create a persistent local copy of the checkpoint.

---

# 8. Portability

The serving workflow is provider-independent.

```
                vLLM
                 │
                 ▼
        Run:ai Streamer
                 │
                 ▼
         S3-Compatible API
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
 Backblaze B2   R2       MinIO
      ▼          ▼          ▼
          Model Weights
```

Migrating providers typically only requires changing:

```text
AWS_ENDPOINT_URL
RUNAI_STREAMER_S3_ENDPOINT
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

No changes to the application code or `vllm serve` command.

---

# 9. Provider Evaluation

## Backblaze B2

**Pros**

- 10 GB free
- No credit card
- Easy setup
- Native S3 API

**Cons**

During testing, serving a single 9.3 GB model immediately exhausted the monthly free download allowance. The documented free bandwidth policy did not match the observed behavior for repeated model streaming. or maybe have to regist cc to get 3x.

---

## Cloudflare R2

**Pros**

- Zero egress
- Native S3 API
- Excellent for production

**Cons**

- Credit card required
- Account verification may fail depending on region/payment method

---

## Wasabi

**Pros**

- Native S3 API
- No egress charges
- No API request charges
- No cc required
- 30-day trial (1 TB)

During testing, repeated `vllm serve` executions streamed successfully from Wasabi without bandwidth restrictions or additional charges. + point there's Singapore ap-southeast-1 region.

> Note: Potential used in production

---

# Conclusion

For development and research:

- **Backblaze B2** is the easiest free starting point.
- **Cloudflare R2** is the best long-term option if account verification succeeds.
- **Wasabi** provided the smoothest experience during testing and was selected as the storage backend for this project.

---

# References

- https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer/ 
- https://secure.backblaze.com 
- https://checkthat.ai/brands/wasabi/pricing 
- https://www.reddit.com/r/AnaloguePocket/comments/1e3y4nt/anyone_else_getting_there_was_an_error_processing/