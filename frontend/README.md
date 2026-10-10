# Financial Intelligence Agent frontend

React, TypeScript, and Vite application for the FastAPI service in this repository. The dark analyst workspace uses React Router, TanStack Query for paged dataset-detail requests, a typed API client, and the existing CSS design tokens. Routes that lack backend support state that limitation instead of generating local demo results.

## Run locally

```bash
cp .env.example .env
npm install
npm run dev
```

Start the API from the repository root:

```bash
uvicorn api.main:app --reload --port 8000
```

Set `VITE_API_BASE_URL` to the public FastAPI base URL. The backend defaults to allowing Vite at `http://localhost:5173` and `http://127.0.0.1:5173`; configure `FRONTEND_ORIGINS` when using another origin.

Only public browser configuration belongs in this app's `.env`. Never put OpenAI, LangSmith, database, or cloud credentials in a `VITE_` variable.

## Pages

- `/` and `/dashboard` — active dataset metadata, current-tab questions, quick actions, and the actual API health status. KPI and component health cards say when the API does not provide those values.
- `/chat` — text questions through `POST /query`, including the active dataset and session ID. Successful replies show only response metadata and evidence returned by the API.
- `/voice` — browser `MediaRecorder` and `POST /voice/query`; microphone permission and provider errors are shown. Text and voice share a session ID.
- `/datasets` — CSV/XLS/XLSX upload and a page of active-dataset preview.
- `/datasets/:datasetId` — schema and paginated preview from `GET /datasets/{dataset_id}`. It shows an error if the in-memory backend no longer has that ID.
- `/analytics` — submits overview, trend, anomaly, and comparison prompts through the real query route. It does not fabricate charts or metrics.
- `/documents` — explains that document upload/indexing is not exposed by the backend.
- `/history` — current page-session history only. It is cleared by refresh; there is no persistent history endpoint.
- `/settings` — real health status and public API base URL, plus browser-local voice autoplay and compact-mode preferences.

## Backend contract used

| Frontend feature | Endpoint | Request and response |
|---|---|---|
| Health badge/settings | `GET /health` | Returns `status` and `service` |
| Dataset upload | `POST /datasets/upload` | Multipart `file`; returns ID, filename, row/column counts, and names |
| Dataset preview/details | `GET /datasets/{dataset_id}?offset=…&limit=…` | Returns dataset metadata, types, and paged preview rows |
| Chat and analytics prompts | `POST /query` | JSON `dataset_id`, `question`, optional `session_id`; response contains answer, result, evidence, intent, and route |
| Voice | `POST /voice/query` | Multipart dataset ID, session ID, audio; returns transcript, answer, route metadata, and generated audio |

The API accepts CSV, XLSX, and XLS files up to 50 MB. Datasets and backend session turns are held in process memory and are lost when the backend restarts. The UI retains the active dataset ID in browser session storage and message text only in page memory.

## Known backend limitations

- There is no dataset list or delete endpoint; the UI can only show the active dataset it knows about.
- There is no document upload/index/list endpoint. Internal RAG code is not presented as an available document feature.
- There is no market symbol field in the query request and no market-data status endpoint, so the frontend does not claim MCP market querying works.
- There is no persistent history endpoint or upload timestamp in the dataset response.
- `GET /health` does not report LangGraph, model, RAG, or MCP component status.
- Analytics requests go through the existing guarded query graph; separate chart/KPI data endpoints are not available.

## Development checks

```bash
npm run typecheck
npm run lint
npm test -- --run
npm run build
```

The tests cover API request payloads, upload progress, backend and guardrail errors, and server-rendered dashboard states. Browser/provider round trips require a running API and, for voice, working microphone permission and configured speech services.
