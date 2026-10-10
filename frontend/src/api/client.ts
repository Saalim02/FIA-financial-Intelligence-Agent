import type { Dataset, DatasetPreview, ModelStatus, QueryResult, VoiceResponse } from '../types'

const configuredBaseUrl = import.meta.env?.VITE_API_BASE_URL?.trim()
const baseUrl = (configuredBaseUrl || 'http://127.0.0.1:8000').replace(/\/$/, '')

type ApiFailureKind = 'unavailable' | 'timeout' | 'endpoint'

export class ApiError extends Error {
  readonly kind: ApiFailureKind
  readonly status?: number
  readonly datasetMissing: boolean

  constructor(message: string, kind: ApiFailureKind, status?: number, datasetMissing = false) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
    this.datasetMissing = datasetMissing
  }
}

function reportUnavailable() {
  if (typeof window !== 'undefined') window.dispatchEvent(new Event('fia-api-unavailable'))
}

function datasetIdForRequest(path: string, init?: RequestInit): string | null {
  if (path.startsWith('/datasets/')) {
    const id = path.split('/')[2]?.split('?')[0]
    return id ? decodeURIComponent(id) : null
  }
  if (path === '/query' && typeof init?.body === 'string') {
    try {
      const body: unknown = JSON.parse(init.body)
      if (typeof body === 'object' && body !== null && 'dataset_id' in body && typeof body.dataset_id === 'string') return body.dataset_id
    } catch { /* Invalid request data is reported by the endpoint. */ }
  }
  if (path === '/voice/query' && init?.body instanceof FormData) {
    const id = init.body.get('dataset_id')
    return typeof id === 'string' ? id : null
  }
  return null
}

async function probeHealth(): Promise<boolean> {
  const controller = new AbortController()
  const timeout = globalThis.setTimeout(() => controller.abort(), 4_000)
  try {
    const response = await fetch(`${baseUrl}/health`, { signal: controller.signal })
    return response.ok
  } catch {
    return false
  } finally {
    globalThis.clearTimeout(timeout)
  }
}

export function apiErrorMessage(error: unknown, action: 'query' | 'upload' | 'voice' | 'dataset' | 'model' = 'query'): string {
  if (error instanceof ApiError) {
    if (error.kind === 'unavailable') return 'Cannot connect to the FIA API.'
    if (error.kind === 'timeout') return 'The FIA API took too long to respond. Please try again.'
    if (error.datasetMissing) return 'This dataset is no longer available. Upload it again to continue.'
    if (error.status === 400 && /guardrail|safety requirements|prompt injection|unsafe request/i.test(error.message)) {
      return "Your request couldn't be processed because it didn't meet FIA's safety requirements."
    }
  }
  if (action === 'upload') return 'Dataset upload failed. Please try again.'
  if (action === 'voice') return 'FIA voice request failed. Please try again.'
  if (action === 'dataset') return 'The dataset request failed. Please check the active dataset and try again.'
  if (action === 'model') return 'Model status unavailable.'
  return 'FIA could not complete this analysis. Please try again.'
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timeout = globalThis.setTimeout(() => controller.abort(), 120_000)
  let response: Response
  try {
    response = await fetch(`${baseUrl}${path}`, { ...init, signal: init?.signal ?? controller.signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(`Request timed out: ${path}`, 'timeout')
    }
    const backendReachable = await probeHealth()
    if (backendReachable) throw new ApiError(`Endpoint request failed: ${path}`, 'endpoint')
    reportUnavailable()
    throw new ApiError(`FIA API is unavailable: ${path}`, 'unavailable')
  } finally {
    globalThis.clearTimeout(timeout)
  }
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body: unknown = await response.json()
      if (typeof body === 'object' && body !== null && 'detail' in body && typeof body.detail === 'string') message = body.detail
    } catch { /* Retain the endpoint status when the response is not JSON. */ }
    const missingDatasetId = response.status === 404 ? datasetIdForRequest(path, init) : null
    const datasetMissing = Boolean(missingDatasetId && (
      path.startsWith('/datasets/') || /\bdataset\s+(?:was\s+)?not found\b/i.test(message)
    ))
    if (missingDatasetId && datasetMissing) {
      if (typeof window !== 'undefined') window.dispatchEvent(new CustomEvent('fia-dataset-unavailable', { detail: { datasetId: missingDatasetId } }))
    }
    throw new ApiError(message, 'endpoint', response.status, datasetMissing)
  }
  try {
    return await response.json() as T
  } catch {
    throw new ApiError(`Invalid response from ${path}`, 'endpoint', response.status)
  }
}

export const api = {
  baseUrl,
  health: () => request<{ status: string; service: string }>('/health'),
  modelStatus: () => request<ModelStatus>('/model/status'),
  preview: (id: string, offset = 0, limit = 25) =>
    request<DatasetPreview>(`/datasets/${encodeURIComponent(id)}?offset=${offset}&limit=${limit}`),
  upload(file: File, onProgress: (percent: number) => void) {
    return new Promise<Dataset>((resolve, reject) => {
      const form = new FormData()
      form.append('file', file)
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${baseUrl}/datasets/upload`)
      xhr.timeout = 120_000
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100))
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText) as Dataset) }
          catch { reject(new ApiError('Invalid dataset upload response.', 'endpoint', xhr.status)) }
        } else {
          let message = `Upload failed (${xhr.status})`
          try {
            const body: unknown = JSON.parse(xhr.responseText)
            if (typeof body === 'object' && body !== null && 'detail' in body && typeof body.detail === 'string') message = body.detail
          } catch { /* Keep the HTTP status. */ }
          reject(new ApiError(message, 'endpoint', xhr.status))
        }
      }
      xhr.onerror = () => {
        void probeHealth().then((backendReachable) => {
          if (!backendReachable) reportUnavailable()
          reject(new ApiError(
            backendReachable ? 'Dataset upload endpoint failed.' : 'FIA API is unavailable.',
            backendReachable ? 'endpoint' : 'unavailable',
          ))
        })
      }
      xhr.ontimeout = () => reject(new ApiError('Dataset upload timed out.', 'timeout'))
      xhr.send(form)
    })
  },
  query(datasetId: string, question: string, sessionId: string) {
    return request<{ dataset_id: string; question: string; result: QueryResult }>('/query', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id: datasetId, question, session_id: sessionId }),
    })
  },
  voice(datasetId: string | null, sessionId: string, audio: Blob) {
    const form = new FormData()
    if (datasetId) form.append('dataset_id', datasetId)
    form.append('session_id', sessionId)
    const audioMime = audio.type.split(';')[0]
    const type = audioMime === 'audio/mp4' ? 'mp4'
      : audioMime === 'audio/ogg' ? 'ogg'
        : audioMime === 'audio/mpeg' || audioMime === 'audio/mp3' ? 'mp3'
          : audioMime === 'audio/wav' || audioMime === 'audio/x-wav' ? 'wav'
            : 'webm'
    form.append('audio', audio, `recording.${type}`)
    return request<VoiceResponse>('/voice/query', { method: 'POST', body: form })
  },
}
