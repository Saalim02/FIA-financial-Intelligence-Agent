import { after, afterEach, describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { readFile, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import ts from 'typescript'

const source = await readFile(new URL('./client.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText
const modulePath = join(tmpdir(), `fia-api-client-${process.pid}.mjs`)
await writeFile(modulePath, compiled)
const { api, apiErrorMessage, ApiError } = await import(pathToFileURL(modulePath))
after(() => rm(modulePath, { force: true }))
afterEach(() => {
  delete globalThis.fetch
  delete globalThis.XMLHttpRequest
  delete globalThis.window
})

describe('API client', () => {
  it('sends text questions with the active dataset and backend session', async () => {
    let requestBody
      globalThis.fetch = async (_url, init) => {
        requestBody = JSON.parse(init.body)
        return new Response(JSON.stringify({
          dataset_id: 'data-1', question: 'What were sales?',
        result: { answer: 'Backend answer', result: [], evidence: [], intent: 'financial_analysis', route: 'financial',
          visualization: { chart_type: 'line', title: 'Units over time', x_label: 'Year', y_label: 'Units', x: ['2023'], y: [8], metric: 'Units', value_format: 'number' } },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }

    const result = await api.query('data-1', 'What were sales?', 'session-1')

    assert.equal(result.result.answer, 'Backend answer')
    assert.equal(result.result.visualization.metric, 'Units')
    assert.deepEqual(requestBody, { dataset_id: 'data-1', question: 'What were sales?', session_id: 'session-1' })
  })

  it('surfaces backend validation errors', async () => {
    globalThis.fetch = async () => new Response(
      JSON.stringify({ detail: 'Dataset file exceeds the 50 MB limit.' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    )
    await assert.rejects(api.preview('bad-id'), /Dataset file exceeds the 50 MB limit/)
  })

  it('reads real model and adapter status from the backend', async () => {
    let requestedUrl
    globalThis.fetch = async (url) => {
      requestedUrl = url
      return new Response(JSON.stringify({ provider: 'qwen', base_model: 'Qwen/Qwen2.5-1.5B-Instruct',
        adapter_loaded: false, adapter_available: true, quantized: true, device: 'auto', mode: 'finetuned_pending' }),
      { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    const status = await api.modelStatus()
    assert.match(requestedUrl, /\/model\/status$/)
    assert.equal(status.adapter_available, true)
    assert.equal(status.adapter_loaded, false)
  })

  it('hides internal guardrail details behind a clear safety message', async () => {
    globalThis.fetch = async () => new Response(
      JSON.stringify({ detail: 'Prompt injection detected by classifier.' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    )
    await assert.rejects(api.query('data-1', 'unsafe request', 'session-1'), (error) => {
      assert.match(apiErrorMessage(error, 'query'), /didn't meet FIA's safety requirements/)
      return true
    })
  })

  it('posts recorded audio and session identity to the voice endpoint', async () => {
    let requestBody
    globalThis.fetch = async (_url, init) => {
      requestBody = init.body
      return new Response(JSON.stringify({
        dataset_id: 'data-1', transcript: 'Sales?', answer: 'Answer', intent: null,
        route: null, audio_base64: 'YWJj', audio_format: 'mp3',
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    const audio = new Blob(['recording'], { type: 'audio/webm' })

    const result = await api.voice('data-1', 'session-1', audio)

    assert.equal(result.transcript, 'Sales?')
    assert.equal(requestBody.get('dataset_id'), 'data-1')
    assert.equal(requestBody.get('session_id'), 'session-1')
    assert.equal(requestBody.get('audio').name, 'recording.webm')
  })

  it('uploads the chosen file and reports backend progress', async () => {
    const progress = []
    class FakeXHR {
      upload = { onprogress: null }
      status = 200
      responseText = JSON.stringify({ dataset_id: 'data-2', filename: 'ledger.csv', rows: 3, columns: 1, column_names: ['Sales'] })
      open(method, url) { this.method = method; this.url = url }
      send(body) {
        this.body = body
        this.upload.onprogress({ lengthComputable: true, loaded: 50, total: 100 })
        this.onload()
      }
    }
    globalThis.XMLHttpRequest = FakeXHR
    const file = new File(['Sales\n10'], 'ledger.csv', { type: 'text/csv' })

    const result = await api.upload(file, (value) => progress.push(value))

    assert.equal(result.dataset_id, 'data-2')
    assert.deepEqual(progress, [50])
  })

  it('distinguishes an unreachable API from a failed query endpoint', async () => {
    globalThis.fetch = async () => { throw new TypeError('Failed to fetch') }
    await assert.rejects(api.query('data-1', 'Show trend', 'session-1'), (error) => {
      assert.ok(error instanceof ApiError)
      assert.equal(error.kind, 'unavailable')
      assert.equal(apiErrorMessage(error, 'query'), 'Cannot connect to the FIA API.')
      return true
    })

    globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'Internal server error.' }), {
      status: 500, headers: { 'Content-Type': 'application/json' },
    })
    await assert.rejects(api.query('data-1', 'Show trend', 'session-1'), (error) => {
      assert.ok(error instanceof ApiError)
      assert.equal(error.kind, 'endpoint')
      assert.equal(apiErrorMessage(error, 'query'), 'FIA could not complete this analysis. Please try again.')
      return true
    })
  })

  it('keeps a healthy API distinct from a failed query route', async () => {
    globalThis.fetch = async (url) => url.endsWith('/health')
      ? new Response(JSON.stringify({ status: 'ok', service: 'fia' }), { status: 200 })
      : new Response(JSON.stringify({ detail: 'Internal server error.' }), { status: 500 })

    assert.equal((await api.health()).status, 'ok')
    await assert.rejects(api.query('data-1', 'Show trend', 'session-1'), (error) => {
      assert.equal(error.kind, 'endpoint')
      assert.equal(apiErrorMessage(error, 'query'), 'FIA could not complete this analysis. Please try again.')
      return true
    })
  })

  it('probes health after a fetch-level query failure before declaring the API unavailable', async () => {
    globalThis.fetch = async (url) => {
      if (url.endsWith('/health')) return new Response(JSON.stringify({ status: 'ok' }), { status: 200 })
      throw new TypeError('Request was blocked at the query endpoint')
    }

    await assert.rejects(api.query('data-1', 'Show trend', 'session-1'), (error) => {
      assert.equal(error.kind, 'endpoint')
      assert.equal(apiErrorMessage(error, 'query'), 'FIA could not complete this analysis. Please try again.')
      return true
    })
  })

  it('reports the exact stale dataset id when a dataset request returns 404', async () => {
    const events = []
    const eventTarget = new EventTarget()
    eventTarget.addEventListener('fia-dataset-unavailable', (event) => events.push(event))
    globalThis.window = eventTarget
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'Dataset not found' }), {
      status: 404, headers: { 'Content-Type': 'application/json' },
    })

    await assert.rejects(api.preview('stale-dataset'))

    assert.equal(events.length, 1)
    assert.equal(events[0].detail.datasetId, 'stale-dataset')
  })

  it('does not clear the active dataset for an unrelated query 404', async () => {
    const events = []
    const eventTarget = new EventTarget()
    eventTarget.addEventListener('fia-dataset-unavailable', (event) => events.push(event))
    globalThis.window = eventTarget
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'Not Found' }), {
      status: 404, headers: { 'Content-Type': 'application/json' },
    })

    await assert.rejects(api.query('active-id', 'Show trend', 'session-1'), (error) => {
      assert.equal(error.datasetMissing, false)
      return true
    })

    assert.equal(events.length, 0)
  })

  it('uses operation-specific upload and voice errors', () => {
    const endpointError = new ApiError('Internal details', 'endpoint', 500)
    assert.equal(apiErrorMessage(endpointError, 'upload'), 'Dataset upload failed. Please try again.')
    assert.equal(apiErrorMessage(endpointError, 'voice'), 'FIA voice request failed. Please try again.')
  })
})
