import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { readFile, writeFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import ts from 'typescript'

const source = await readFile(new URL('./datasetSession.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText
const modulePath = join(process.env.TMPDIR ?? '/tmp', `fia-dataset-session-${process.pid}.mjs`)
await writeFile(modulePath, compiled)
const { confirmsDatasetMissing, restoreStillMatchesActiveDataset } = await import(pathToFileURL(modulePath))
await rm(modulePath, { force: true })

describe('active dataset restoration', () => {
  it('accepts a restore only while it still matches the active, persisted selection', () => {
    assert.equal(restoreStillMatchesActiveDataset('dataset-1', 4, 'dataset-1', 4, 'dataset-1'), true)
  })

  it('ignores an old restore after a new upload activates another dataset', () => {
    assert.equal(restoreStillMatchesActiveDataset('old-id', 4, 'new-id', 5, 'new-id'), false)
  })

  it('does not clear active selection for temporary preview failures', () => {
    assert.equal(confirmsDatasetMissing(503, 'Backend unavailable'), false)
    assert.equal(confirmsDatasetMissing(undefined, 'Dataset not found'), false)
  })

  it('clears only after the backend confirms the dataset is missing', () => {
    assert.equal(confirmsDatasetMissing(404, 'Dataset not found: missing-id'), true)
    assert.equal(confirmsDatasetMissing(404, 'Route not found'), false)
  })
})
