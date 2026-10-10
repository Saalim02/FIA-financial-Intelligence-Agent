import { spawnSync } from 'node:child_process'

const result = spawnSync(process.execPath, [
  '--test',
  'src/api/client.test.mjs',
  'src/pages/DashboardPage.test.mjs',
  'src/components/VisualizationChart.test.mjs',
  'src/state/datasetSession.test.mjs',
], { cwd: process.cwd(), stdio: 'inherit' })

process.exitCode = result.status ?? 1
