export type Dataset = {
  dataset_id: string
  filename: string
  rows: number
  columns: number
  column_names: string[]
}

export type DatasetPreview = Dataset & {
  column_types: Record<string, string>
  offset: number
  limit: number
  preview: Record<string, unknown>[]
}

export type Evidence = { source: string; findings: unknown }
export type TrendSummary = {
  direction: 'increasing' | 'decreasing' | 'stable'
  start: number | null
  end: number | null
  change_percent: number | null
}
export type Visualization = {
  chart_type: 'line' | 'bar'
  title: string
  x_label: string
  y_label: string
  x: string[]
  y: (number | null)[]
  metric: string
  time_column?: string
  granularity?: string
  dimension?: string
  aggregation?: string
  value_format: 'currency' | 'number'
  summary?: TrendSummary
}
export type QueryResult = {
  answer: string | null
  result: unknown
  evidence: Evidence[]
  intent: string | null
  route: string | null
  visualization?: Visualization | null
}
export type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
  createdAt?: string
  result?: QueryResult
  audioUrl?: string
}
export type VoiceResponse = {
  dataset_id: string | null
  transcript: string
  answer: string
  intent: string | null
  route: string | null
  audio_base64: string
  audio_format: string
  visualization?: Visualization | null
}
export type ModelStatus = {
  provider: string
  model?: string
  base_model?: string
  adapter_loaded: boolean
  adapter_available?: boolean
  quantized: boolean
  device: string
  mode?: string
}
