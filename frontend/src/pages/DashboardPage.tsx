import { Activity, ArrowUpRight, BarChart3, Clock3, Database, FileSpreadsheet, Headphones, MessageSquareText, ShieldCheck } from 'lucide-react'
import type { ChatMessage, Dataset, ModelStatus } from '../types'

type Props = {
  dataset: Dataset | null
  messages: ChatMessage[]
  apiStatus: 'connected' | 'disconnected' | 'checking'
  modelStatus?: ModelStatus | null
  onNavigate: (path: string) => void
}

export default function DashboardPage({ dataset, messages, apiStatus, modelStatus, onNavigate }: Props) {
  const recentQuestions = messages.flatMap((message, index) => message.role === 'user' ? [{
    question: message,
    response: messages[index + 1],
  }] : []).slice(-4).reverse()
  const statusLabel = apiStatus === 'connected' ? 'Connected' : apiStatus === 'checking' ? 'Checking' : 'Unavailable'

  return <div className="page-scroll dashboard-page">
    <div className="dashboard-welcome">
      <div><div className="eyebrow">FINANCIAL INTELLIGENCE AGENT</div><h1>Financial Intelligence</h1><p>Understand your financial data with natural language.</p></div>
      <div className={`dashboard-api-status ${apiStatus}`}><span className="status-indicator" /> API {statusLabel}</div>
    </div>

    <section className="dashboard-grid">
      <article className="surface dashboard-dataset-card">
        <div className="dashboard-card-heading"><div className="upload-glyph"><Database size={18} /></div><div><span className="eyebrow">CURRENT DATASET</span><h2>{dataset?.filename ?? 'No dataset selected'}</h2></div></div>
        {dataset ? <>
          <div className="dashboard-dataset-stats"><div><span>Rows</span><strong>{dataset.rows.toLocaleString()}</strong></div><div><span>Columns</span><strong>{dataset.columns}</strong></div><div><span>Status</span><strong className="dataset-live"><i /> In memory</strong></div></div>
          <div className="dataset-id-line">Dataset ID <code title={dataset.dataset_id}>{dataset.dataset_id}</code></div>
          <button className="dashboard-link" onClick={() => onNavigate(`/datasets/${encodeURIComponent(dataset.dataset_id)}`)}>Explore dataset <ArrowUpRight size={14} /></button>
        </> : <div className="dashboard-empty-copy"><p>Upload a CSV or Excel file to explore its schema and ask questions.</p><button className="primary-button" onClick={() => onNavigate('/datasets')}>Upload dataset <ArrowUpRight size={14} /></button></div>}
      </article>

      <article className="surface dashboard-metrics-card">
        <div className="dashboard-card-heading"><div className="upload-glyph"><BarChart3 size={18} /></div><div><span className="eyebrow">KEY METRICS</span><h2>Backend results only</h2></div></div>
        <div className="dashboard-unavailable"><Activity size={18} /><p>Standalone KPI summaries are not exposed by the API yet.</p><span>Run an analysis in Chat or Analytics to see values returned by the agent.</span></div>
      </article>

      <article className="surface dashboard-actions-card">
        <div className="dashboard-card-heading"><div className="upload-glyph"><ArrowUpRight size={18} /></div><div><span className="eyebrow">QUICK ACTIONS</span><h2>Continue your analysis</h2></div></div>
        <div className="dashboard-actions">
          <button onClick={() => onNavigate('/datasets')}><FileSpreadsheet size={16} /><span>Upload dataset</span><ArrowUpRight size={13} /></button>
          <button onClick={() => onNavigate('/chat')}><MessageSquareText size={16} /><span>Ask a question</span><ArrowUpRight size={13} /></button>
          <button onClick={() => onNavigate('/analytics')}><BarChart3 size={16} /><span>Analyze trends</span><ArrowUpRight size={13} /></button>
          <button onClick={() => onNavigate('/voice')}><Headphones size={16} /><span>Voice assistant</span><ArrowUpRight size={13} /></button>
        </div>
      </article>

      <article className="surface dashboard-history-card">
        <div className="dashboard-card-heading"><div className="upload-glyph"><Clock3 size={18} /></div><div><span className="eyebrow">RECENT QUESTIONS</span><h2>Current session</h2></div><button className="text-button" onClick={() => onNavigate('/history')}>View history</button></div>
        {recentQuestions.length ? <div className="dashboard-recent-list">{recentQuestions.map(({ question, response }, index) => <div className="dashboard-recent-row" key={`${question.createdAt ?? question.content}-${index}`}><span>{question.content}</span><span>{response?.result?.route ?? 'No route data'}</span></div>)}</div> : <div className="dashboard-unavailable"><MessageSquareText size={18} /><p>No questions in this session yet.</p><span>Questions appear here after you use Chat or Voice.</span></div>}
      </article>

      <article className="surface dashboard-system-card">
        <div className="dashboard-card-heading"><div className="upload-glyph"><ShieldCheck size={18} /></div><div><span className="eyebrow">SYSTEM STATUS</span><h2>Available checks</h2></div></div>
        <div className="dashboard-system-row"><span>FastAPI health</span><strong className={apiStatus}><i />{statusLabel}</strong></div>
        <div className="dashboard-system-row"><span>Reasoning provider</span><strong>{modelStatus?.provider ?? 'Unavailable'}</strong></div>
        <div className="dashboard-system-row"><span>QLoRA adapter</span><strong>{modelStatus?.adapter_loaded ? 'Loaded' : modelStatus?.adapter_available ? 'Available · lazy load' : modelStatus?.provider === 'qwen' ? 'Unavailable' : 'Not applicable'}</strong></div>
        <p className="dashboard-system-note">Model details come from the backend status endpoint.</p>
      </article>
    </section>
  </div>
}
