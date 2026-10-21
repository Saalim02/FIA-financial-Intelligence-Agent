import { after, describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { readFile, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import ts from 'typescript'

const source = await readFile(new URL('./uuid.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText
const modulePath = join(tmpdir(), `fia-uuid-${process.pid}.mjs`)
await writeFile(modulePath, compiled)
const { generateUUID } = await import(pathToFileURL(modulePath))
after(() => rm(modulePath, { force: true }))

describe('generateUUID', () => {
  const originalCrypto = globalThis.crypto
  after(() => Object.defineProperty(globalThis, 'crypto', { configurable: true, value: originalCrypto }))

  it('uses crypto.randomUUID when available', () => {
    Object.defineProperty(globalThis, 'crypto', {
      configurable: true,
      value: { randomUUID: () => 'preferred-uuid' },
    })
    assert.equal(generateUUID(), 'preferred-uuid')
  })

  it('creates an RFC 4122 version 4 UUID using getRandomValues as fallback', () => {
    Object.defineProperty(globalThis, 'crypto', {
      configurable: true,
      value: { getRandomValues: (bytes) => bytes.fill(0) },
    })
    assert.equal(generateUUID(), '00000000-0000-4000-8000-000000000000')
  })

  it('fails clearly when no secure random source is available', () => {
    Object.defineProperty(globalThis, 'crypto', { configurable: true, value: undefined })
    assert.throws(generateUUID, /Secure UUID generation is unavailable/)
  })
})
