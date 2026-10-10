import { after, describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { readFile, writeFile, rm } from 'node:fs/promises'
import { fileURLToPath, pathToFileURL } from 'node:url'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'

const componentUrl = new URL('./DashboardPage.tsx', import.meta.url)
const source = await readFile(componentUrl, 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    jsx: ts.JsxEmit.ReactJSX,
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText
const modulePath = fileURLToPath(new URL(`./.DashboardPage.${process.pid}.mjs`, import.meta.url))
await writeFile(modulePath, compiled)
const { default: DashboardPage } = await import(pathToFileURL(modulePath))
after(() => rm(modulePath, { force: true }))

describe('DashboardPage', () => {
  it('shows the real empty state and does not invent metrics', () => {
    const html = renderToStaticMarkup(React.createElement(DashboardPage, {
      dataset: null,
      messages: [],
      apiStatus: 'connected',
      onNavigate: () => {},
    }))

    assert.match(html, /No dataset selected/)
    assert.match(html, /Standalone KPI summaries are not exposed by the API yet/)
    assert.match(html, /API Connected/)
    assert.doesNotMatch(html, /\$[0-9]|\bRevenue\s+[0-9]/)
  })

  it('renders dataset values and recent question from supplied application state', () => {
    const html = renderToStaticMarkup(React.createElement(DashboardPage, {
      dataset: { dataset_id: 'generated-id', filename: 'local.csv', rows: 2, columns: 2, column_names: ['Revenue', 'Profit'] },
      messages: [{ role: 'user', content: 'What was revenue?', createdAt: '2026-01-01T10:00:00.000Z' }],
      apiStatus: 'disconnected',
      onNavigate: () => {},
    }))

    assert.match(html, /local\.csv/)
    assert.match(html, /generated-id/)
    assert.match(html, /What was revenue\?/)
    assert.match(html, /API Unavailable/)
  })
})
