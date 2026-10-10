# Financial Intelligence Agent

Financial Intelligence Agent is a React and TypeScript workspace backed by a FastAPI service. Users can upload a tabular dataset, inspect paged rows, ask text questions, and submit voice questions through the guarded financial-analysis graph.

The frontend does not generate sample answers, fake financial values, charts, market quotes, or document results. It only presents values returned by the API and labels capabilities that the API does not expose.

## Architecture

```text
frontend (React, TypeScript, Vite)
  └─ FastAPI API client
       ├─ dataset upload and preview
       ├─ text and voice query services
       └─ LangGraph guardrail → understanding → analysis → evidence → reasoning → output guardrail
```

Dataset analysis uses the backend financial, trend, and anomaly engines. Text and voice turns can share a session ID; session history is bounded, in-memory, and cleared when the backend process restarts. Dataset storage is also in-memory.

## Start the backend

Create a private root `.env` with the backend's required credentials and configuration. Never put backend credentials in frontend variables.

```bash
python -m pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

The API reads `OPENAI_API_KEY` for query guardrails and, by default, reasoning. To route financial reasoning through local Qwen, set `FINANCIAL_REASONING_PROVIDER=qwen`; guardrail classifiers still use OpenAI. Qwen model loading is lazy and never triggers training. See the QLoRA section below. LangSmith tracing is enabled only when `LANGSMITH_TRACING=true`; configure its key and project in the backend environment if remote traces are intended.

## Start the frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

`VITE_API_BASE_URL` is public browser configuration and defaults to `http://127.0.0.1:8000`. The backend allows the default Vite localhost origins and local Vite fallback ports. Set `FRONTEND_ORIGINS` for another frontend origin.

See [frontend/README.md](frontend/README.md) for page behavior, scripts, and API mapping.

## API currently exposed

| Method and path | Purpose |
|---|---|
| `GET /health` | API process health only |
| `GET /model/status` | Configured reasoning provider, model, adapter availability/load state, quantization, and device |
| `POST /datasets/upload` | Upload CSV, XLSX, or XLS; maximum 50 MB |
| `GET /datasets/{dataset_id}?offset=0&limit=25` | Dataset metadata, column types, and paged preview |
| `POST /query` | JSON dataset ID, question, optional session ID; returns answer, result, evidence, intent, and route |
| `POST /voice/query` | Multipart dataset ID, optional session ID, and audio; returns transcript, answer, route metadata, and generated audio |

Analytics prompts in the UI call `POST /query` and use its financial, trend, or anomaly route. There are no standalone chart or KPI endpoints.

## Current limitations

- The API has no dataset listing or deletion operation, and dataset upload timestamps are not returned.
- Conversation history is not available through a backend history endpoint or persistent storage.
- The API has no document-ingestion endpoint. Internal RAG code is not exposed as a usable document-upload flow.
- The query request has no market-symbol field. The MCP graph node therefore cannot receive a symbol through the current API contract.
- `GET /health` does not report LangGraph, LLM, RAG, or MCP component status.
- Voice processing requires browser microphone permission and configured backend speech services.

## QLoRA training and local inference

The behavioral SFT dataset is in `fine_tuning/datasets/{train,validation,test}.jsonl`. `fine_tuning/train.py` validates all three splits, uses 4-bit NF4 QLoRA with the pinned Transformers/PEFT/TRL stack, and writes a portable PEFT adapter to `fine_tuning/output/final_adapter/`. Training requires a CUDA GPU and the model weights available locally or from Hugging Face. For the pinned RTX 3050 CUDA environment, install `requirements-qwen-cu121.txt` after `requirements.txt`. The adapter is not merged into the base model.

```bash
python fine_tuning/train.py
python fine_tuning/inference.py --mode base
python fine_tuning/inference.py --mode finetuned
python fine_tuning/evaluate.py
```

The evaluator compares both model modes on the same held-out behavioral examples and records reference-term recall, unsupported numeric values, latency, generated tokens, and GPU memory when available. For the existing RAGAS judge metrics, set `RUN_RAGAS_LLM_EVAL=1` and configure `OPENAI_API_KEY`; that option makes external OpenAI calls. To deploy the trained adapter, set `FINANCIAL_REASONING_PROVIDER=qwen` and optionally configure `QWEN_BASE_MODEL`, `QWEN_ADAPTER_PATH`, `QWEN_LOAD_ADAPTER`, `QWEN_4BIT`, `QWEN_DEVICE`, and `QWEN_REQUIRE_ADAPTER`. With lazy loading, `/model/status` distinguishes an available adapter from one actually loaded into memory. A required but missing adapter is a configuration error.

## Verification

Backend tests run without external model downloads when the local embedding model is cached:

```bash
HF_HUB_OFFLINE=1 python -m pytest -q
python -m compileall -q api core tests
```

Frontend checks:

```bash
cd frontend
npm run typecheck
npm run lint
npm test -- --run
npm run build
```

Paid OpenAI and remote LangSmith integration checks are isolated in `tests/integration/` and require explicit opt-in environment flags. Their generated test dataset and prompt are synthetic.
