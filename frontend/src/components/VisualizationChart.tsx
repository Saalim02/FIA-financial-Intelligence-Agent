import { Component, useState, type ErrorInfo, type ReactNode } from 'react'
import type { Visualization } from '../types'

type Props = { visualization: unknown }

const W = 760
const H = 300
const LEFT = 76
const RIGHT = 22
const TOP = 24
const BOTTOM = 62

function formatValue(value: number, format: Visualization['value_format']) {
  if (format === 'currency') {
    return new Intl.NumberFormat('en-US', {
      style: 'currency', currency: 'USD', notation: Math.abs(value) >= 10_000 ? 'compact' : 'standard',
      maximumFractionDigits: Math.abs(value) >= 10_000 ? 2 : 2,
    }).format(value)
  }
  return new Intl.NumberFormat('en-US', { notation: Math.abs(value) >= 10_000 ? 'compact' : 'standard', maximumFractionDigits: 2 }).format(value)
}

function formatLabel(value: string) {
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const date = new Date(`${value}T00:00:00Z`)
    if (!Number.isNaN(date.getTime())) return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' }).format(date)
  }
  return value
}

function axisLabel(value: string) {
  const label = formatLabel(value)
  return label.length > 14 ? `${label.slice(0, 12)}…` : label
}

function isVisualization(value: unknown): value is Visualization {
  if (!value || typeof value !== 'object') return false
  const item = value as Record<string, unknown>
  if (item.chart_type !== 'line' && item.chart_type !== 'bar') return false
  if (![item.title, item.x_label, item.y_label, item.metric].every((field) => typeof field === 'string')) return false
  if (!Array.isArray(item.x) || !item.x.every((label) => typeof label === 'string')) return false
  if (!Array.isArray(item.y) || !item.y.every((point) => point === null || (typeof point === 'number' && Number.isFinite(point)))) return false
  if (item.value_format !== 'currency' && item.value_format !== 'number') return false
  if (item.summary !== undefined) {
    if (!item.summary || typeof item.summary !== 'object') return false
    const summary = item.summary as Record<string, unknown>
    if (!['increasing', 'decreasing', 'stable'].includes(String(summary.direction))) return false
    if (summary.start !== null && (typeof summary.start !== 'number' || !Number.isFinite(summary.start))) return false
    if (summary.end !== null && (typeof summary.end !== 'number' || !Number.isFinite(summary.end))) return false
    if (summary.change_percent !== null && (typeof summary.change_percent !== 'number' || !Number.isFinite(summary.change_percent))) return false
  }
  return true
}

function ChartContent({ visualization }: { visualization: Visualization }) {
  const [hovered, setHovered] = useState<number | null>(null)
  const { x, y } = visualization
  const validValues = y.filter((value): value is number => value !== null)
  if (!x.length || !y.length || x.length !== y.length || !validValues.length) {
    return <section className="visualization-card" aria-label="Chart unavailable"><div className="visualization-heading"><div><span className="eyebrow">DATA VISUALIZATION</span><h3>{visualization.title}</h3></div></div><p className="visualization-empty">There isn’t enough chart data to display this result.</p></section>
  }

  const plotWidth = W - LEFT - RIGHT
  const plotHeight = H - TOP - BOTTOM
  const min = Math.min(0, ...validValues)
  const max = Math.max(0, ...validValues)
  const span = max - min || 1
  const yPosition = (value: number) => TOP + (max - value) / span * plotHeight
  const xPosition = (index: number) => LEFT + (x.length === 1 ? plotWidth / 2 : index / (x.length - 1) * plotWidth)
  const tickValues = Array.from({ length: 5 }, (_, index) => max - span * index / 4)
  const paths: string[] = []
  let activePath: string[] = []
  y.forEach((value, index) => {
    if (value === null) {
      if (activePath.length) paths.push(activePath.join(' '))
      activePath = []
      return
    }
    activePath.push(`${activePath.length ? 'L' : 'M'} ${xPosition(index)} ${yPosition(value)}`)
  })
  if (activePath.length) paths.push(activePath.join(' '))
  const barWidth = Math.min(34, plotWidth / Math.max(1, x.length) * 0.62)
  const summary = visualization.summary
  const firstLabel = formatLabel(x[0])
  const lastLabel = formatLabel(x[x.length - 1])

  return <section className="visualization-card" aria-label={`${visualization.chart_type} chart: ${visualization.title}`}>
    <div className="visualization-heading"><div><span className="eyebrow">{visualization.chart_type === 'line' ? 'TREND ANALYSIS' : 'CATEGORY COMPARISON'}</span><h3>{visualization.title}</h3><p>{visualization.y_label} · {visualization.x_label}{visualization.granularity ? ` · ${visualization.granularity}` : ''}</p></div><span className="chart-type-chip">{visualization.chart_type === 'line' ? 'LINE' : 'BAR'}</span></div>
    {summary && summary.start !== null && summary.end !== null && <div className="trend-summary"><div><span>Direction</span><strong className={summary.direction}>{summary.direction}</strong></div><div><span>Start · {firstLabel}</span><strong>{formatValue(summary.start, visualization.value_format)}</strong></div><div><span>End · {lastLabel}</span><strong>{formatValue(summary.end, visualization.value_format)}</strong></div><div><span>Change</span><strong className={summary.direction}>{summary.change_percent === null ? '—' : `${summary.change_percent > 0 ? '+' : ''}${summary.change_percent.toFixed(2)}%`}</strong></div></div>}
    <div className="chart-frame">
      {hovered !== null && y[hovered] !== null && <div className="chart-tooltip" role="status"><strong>{formatLabel(x[hovered])}</strong><span>{formatValue(y[hovered], visualization.value_format)}</span></div>}
      <svg className="financial-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${visualization.y_label} by ${visualization.x_label}`}>
        {tickValues.map((value, index) => { const py = TOP + index / 4 * plotHeight; return <g key={index}><line x1={LEFT} y1={py} x2={W - RIGHT} y2={py} className="chart-gridline"/><text x={LEFT - 10} y={py + 4} textAnchor="end" className="chart-tick">{formatValue(value, visualization.value_format)}</text></g> })}
        <text x={18} y={TOP + plotHeight / 2} textAnchor="middle" className="chart-axis-label" transform={`rotate(-90 18 ${TOP + plotHeight / 2})`}>{visualization.y_label}</text>
        {visualization.chart_type === 'line' ? <>
          {paths.map((d, index) => <path key={index} d={d} className="chart-line" />)}
          {y.map((value, index) => value === null ? null : <circle key={`${x[index]}-${index}`} cx={xPosition(index)} cy={yPosition(value)} r={hovered === index ? 6 : 4} className="chart-point" onMouseEnter={() => setHovered(index)} onMouseLeave={() => setHovered(null)} onFocus={() => setHovered(index)} onBlur={() => setHovered(null)} tabIndex={0} aria-label={`${formatLabel(x[index])}: ${formatValue(value, visualization.value_format)}`} />)}
        </> : y.map((value, index) => { if (value === null) return null; const zero = yPosition(0); const py = yPosition(value); return <rect key={`${x[index]}-${index}`} x={xPosition(index) - barWidth / 2} y={Math.min(zero, py)} width={barWidth} height={Math.max(1, Math.abs(zero - py))} rx={3} className="chart-bar" onMouseEnter={() => setHovered(index)} onMouseLeave={() => setHovered(null)} onFocus={() => setHovered(index)} onBlur={() => setHovered(null)} tabIndex={0} aria-label={`${x[index]}: ${formatValue(value, visualization.value_format)}`} /> })}
        {x.map((label, index) => {
          const step = Math.max(1, Math.ceil(x.length / 7))
          if (index !== 0 && index !== x.length - 1 && index % step !== 0) return null
          return <text key={`${label}-${index}`} x={xPosition(index)} y={H - 25} textAnchor="middle" className="chart-tick x-tick"><title>{formatLabel(label)}</title>{axisLabel(label)}</text>
        })}
        <text x={LEFT + plotWidth / 2} y={H - 5} textAnchor="middle" className="chart-axis-label">{visualization.x_label}</text>
      </svg>
    </div>
  </section>
}

class ChartErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() { return { failed: true } }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Chat visualization could not be rendered.', error, info.componentStack)
  }

  render() {
    return this.state.failed
      ? <section className="visualization-card" aria-label="Chart unavailable"><p className="visualization-empty">Visualization unavailable</p></section>
      : this.props.children
  }
}

export default function VisualizationChart({ visualization }: Props) {
  if (visualization === null || visualization === undefined) return null
  if (!isVisualization(visualization)) {
    return <section className="visualization-card" aria-label="Chart unavailable"><p className="visualization-empty">Visualization unavailable</p></section>
  }
  return <ChartErrorBoundary><ChartContent visualization={visualization} /></ChartErrorBoundary>
}
