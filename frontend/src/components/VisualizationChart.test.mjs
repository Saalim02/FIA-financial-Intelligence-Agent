import { after, describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { readFile, writeFile, rm } from 'node:fs/promises'
import { fileURLToPath, pathToFileURL } from 'node:url'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'

const componentUrl = new URL('./VisualizationChart.tsx', import.meta.url)
const source = await readFile(componentUrl, 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText
const modulePath = fileURLToPath(new URL(`./.VisualizationChart.${process.pid}.mjs`, import.meta.url))
await writeFile(modulePath, compiled)
const { default: VisualizationChart } = await import(pathToFileURL(modulePath))
after(() => rm(modulePath, { force: true }))

const chart = {
  chart_type: 'line', title: 'Margin over time', x_label: 'Month', y_label: 'Margin',
  x: ['2024-01', '2024-02'], y: [12.3, 15.1], metric: 'Margin', granularity: 'month', value_format: 'number',
  summary: { direction: 'increasing', start: 12.3, end: 15.1, change_percent: 22.76 },
}

describe('VisualizationChart', () => {
  it('renders a supplied visualization with the backend summary and points', () => {
    const html = renderToStaticMarkup(React.createElement(VisualizationChart, { visualization: chart }))
    assert.match(html, /Margin over time/)
    assert.match(html, /22\.76%/)
    assert.match(html, /aria-label="Margin by Month"/)
    assert.match(html, /chart-line/)
  })

  it('renders nothing for a null visualization', () => {
    assert.equal(renderToStaticMarkup(React.createElement(VisualizationChart, { visualization: null })), '')
  })

  it('shows a safe empty state for empty chart data', () => {
    const html = renderToStaticMarkup(React.createElement(VisualizationChart, { visualization: { ...chart, x: [], y: [] } }))
    assert.match(html, /There isn’t enough chart data/)
    assert.doesNotMatch(html, /<svg/)
  })

  it('keeps a dataset-backed trend response renderable when visualization is malformed', () => {
    const result = { answer: 'Sales decreased over time.', evidence: [], visualization: { type: 'line', labels: ['2024'], data: [12] } }
    const html = renderToStaticMarkup(React.createElement(VisualizationChart, { visualization: result.visualization }))
    assert.match(result.answer, /Sales decreased/)
    assert.match(html, /Visualization unavailable/)
    assert.doesNotMatch(html, /<svg/)
  })

  it('does not throw for scalar, null, or unexpected visualization payloads', () => {
    for (const visualization of [undefined, null, 14, 'not a chart', { chart_type: 'bar', x: null, y: [1] }]) {
      assert.doesNotThrow(() => renderToStaticMarkup(React.createElement(VisualizationChart, { visualization })))
    }
  })

  it('renders a category result using the returned backend values', () => {
    const html = renderToStaticMarkup(React.createElement(VisualizationChart, {
      visualization: { ...chart, chart_type: 'bar', title: 'Sales by Region', x: ['East', 'West'], y: [24, 31], summary: undefined },
    }))
    assert.match(html, /Sales by Region/)
    assert.match(html, /chart-bar/)
  })

  it('shows a gap for null trend points without plotting them as zero', () => {
    const html = renderToStaticMarkup(React.createElement(VisualizationChart, {
      visualization: { ...chart, x: ['2009', '2010', '2011'], y: [42, null, 34], summary: undefined },
    }))
    assert.match(html, /chart-line/)
    assert.equal((html.match(/class="chart-point"/g) ?? []).length, 2)
    assert.doesNotMatch(html, /2010: \$0/)
  })

  it('shows an empty state when every visualization point is missing', () => {
    const html = renderToStaticMarkup(React.createElement(VisualizationChart, {
      visualization: { ...chart, x: ['2009', '2010'], y: [null, null], summary: undefined },
    }))
    assert.match(html, /There isn’t enough chart data/)
    assert.doesNotMatch(html, /<svg/)
  })
})
