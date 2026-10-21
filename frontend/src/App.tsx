import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import {
  Activity, ArrowUpRight, AudioLines, BarChart3, Check, ChevronLeft, Copy,
  ChevronRight, CircleHelp, Clock3, Database, FileSpreadsheet, FileUp, Headphones,
  FileText, LayoutDashboard, MessageSquareText, Mic, MicOff, MoreHorizontal, Play, Plus, RotateCcw,
  Send, Settings2, ShieldCheck, Sparkles, Square, Volume2, X,
} from 'lucide-react'
import { ApiError, api, apiErrorMessage } from './api/client'
import { generateUUID } from './utils/uuid'
import { confirmsDatasetMissing, restoreStillMatchesActiveDataset } from './state/datasetSession'
import type { ChatMessage, Dataset, DatasetPreview, Evidence, ModelStatus, QueryResult, VoiceResponse } from './types'
import VisualizationChart from './components/VisualizationChart'

const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const DatasetDetailsPage = lazy(() => import('./pages/DatasetDetailsPage'))
const DocumentsPage = lazy(() => import('./pages/DocumentsPage'))

type Status = 'connected' | 'disconnected' | 'checking'
const routes = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/chat', label: 'Chat analyst', icon: MessageSquareText },
  { to: '/voice', label: 'Voice mode', icon: Headphones },
  { to: '/datasets', label: 'Datasets', icon: Database },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
  { to: '/documents', label: 'Documents', icon: FileText },
  { to: '/history', label: 'History', icon: Clock3 },
  { to: '/settings', label: 'Settings', icon: Settings2 },
]

function newSession() {
  return generateUUID()
}

function savedDatasetId(): string | null {
  try { return typeof sessionStorage === 'undefined' ? null : sessionStorage.getItem('fia.datasetId') }
  catch { return null }
}

function restoringDataset(id: string): Dataset {
  return { dataset_id: id, filename: 'Restoring active dataset…', rows: 0, columns: 0, column_names: [] }
}

export default function App() {
  const navigate = useNavigate()
  const location = useLocation()
  const [dataset, setDataset] = useState<Dataset | null>(() => {
    const id = savedDatasetId()
    return id ? restoringDataset(id) : null
  })
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sessionId, setSessionId] = useState(newSession)
  const [health, setHealth] = useState<Status>('checking')
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [modelStatusError, setModelStatusError] = useState('')
  const [datasetLoading, setDatasetLoading] = useState(() => Boolean(savedDatasetId()))
  const [datasetRestoreError, setDatasetRestoreError] = useState('')
  const [voiceAutoPlay, setVoiceAutoPlay] = useState(() => localStorage.getItem('fia.voiceAutoPlay') !== 'false')
  const [compactMode, setCompactMode] = useState(() => localStorage.getItem('fia.compactMode') === 'true')
  const audioUrls = useRef(new Set<string>())
  const healthEpoch = useRef(0)
  const datasetRestoreEpoch = useRef(0)
  const activeDatasetId = useRef(dataset?.dataset_id ?? null)
  useEffect(() => () => audioUrls.current.forEach((url) => URL.revokeObjectURL(url)), [])

  const refreshHealth = useCallback(async () => {
    const epoch = healthEpoch.current
    try {
      await api.health()
      if (epoch === healthEpoch.current) setHealth('connected')
    } catch {
      if (epoch === healthEpoch.current) setHealth('disconnected')
    }
    try {
      const status = await api.modelStatus()
      setModelStatus(status); setModelStatusError('')
    } catch (error) {
      setModelStatus(null)
      setModelStatusError(apiErrorMessage(error, 'model'))
    }
  }, [])
  useEffect(() => {
    void refreshHealth()
    const timer = window.setInterval(() => void refreshHealth(), 10_000)
    return () => window.clearInterval(timer)
  }, [refreshHealth])
  useEffect(() => {
    const markUnavailable = () => {
      healthEpoch.current += 1
      setHealth('disconnected')
    }
    window.addEventListener('fia-api-unavailable', markUnavailable)
    return () => window.removeEventListener('fia-api-unavailable', markUnavailable)
  }, [])
  useEffect(() => {
    const clearUnavailableDataset = (event: Event) => {
      const { datasetId } = (event as CustomEvent<{ datasetId?: string }>).detail ?? {}
      if (!datasetId || activeDatasetId.current !== datasetId) return
      datasetRestoreEpoch.current += 1
      sessionStorage.removeItem('fia.datasetId')
      activeDatasetId.current = null
      setDataset(null)
      setPreview(null)
      setDatasetLoading(false)
      setDatasetRestoreError('The active dataset is no longer available. Upload it again to continue.')
    }
    window.addEventListener('fia-dataset-unavailable', clearUnavailableDataset)
    return () => window.removeEventListener('fia-dataset-unavailable', clearUnavailableDataset)
  }, [])
  useEffect(() => {
    const id = savedDatasetId()
    if (!id) return
    activeDatasetId.current = id
    const requestEpoch = datasetRestoreEpoch.current
    const isCurrentRestore = () => restoreStillMatchesActiveDataset(
      id, requestEpoch, activeDatasetId.current, datasetRestoreEpoch.current, savedDatasetId(),
    )
    setDatasetRestoreError('')
    setDatasetLoading(true)
    api.preview(id, 0, 25).then((response) => {
      if (!isCurrentRestore()) return
      setPreview(response)
      setDataset({ dataset_id: response.dataset_id, filename: response.filename ?? 'Dataset', rows: response.rows,
        columns: response.columns, column_names: response.column_names })
    }).catch((error: unknown) => {
      if (!isCurrentRestore()) return
      if (error instanceof ApiError && confirmsDatasetMissing(error.status, error.message)) {
        sessionStorage.removeItem('fia.datasetId')
        activeDatasetId.current = null
        setDataset(null)
        setPreview(null)
        setDatasetRestoreError(apiErrorMessage(error, 'dataset'))
      } else {
        setDatasetRestoreError('Could not refresh dataset details. The active dataset selection is preserved.')
      }
    }).finally(() => {
      if (requestEpoch === datasetRestoreEpoch.current && activeDatasetId.current === id) setDatasetLoading(false)
    })
  }, [])

  const activateDataset = useCallback((data: Dataset, destination = '/chat') => {
    datasetRestoreEpoch.current += 1
    activeDatasetId.current = data.dataset_id
    setDatasetRestoreError('')
    setDatasetLoading(false)
    setDataset(data)
    setPreview(null)
    setMessages([])
    setSessionId(newSession())
    sessionStorage.setItem('fia.datasetId', data.dataset_id)
    navigate(destination)
  }, [navigate])

  const startNewConversation = useCallback(() => {
    setMessages([])
    setSessionId(newSession())
  }, [])

  const fetchPreview = useCallback(async (offset = 0) => {
    if (!dataset) return
    const result = await api.preview(dataset.dataset_id, offset, 25)
    setPreview(result)
  }, [dataset])

  const ask = useCallback(async (question: string): Promise<QueryResult> => {
    if (!dataset) throw new Error('Upload a dataset before asking a question.')
    const createdAt = new Date().toISOString()
    setMessages((items) => [...items, { role: 'user', content: question, createdAt }])
    try {
      const response = await api.query(dataset.dataset_id, question, sessionId)
      setMessages((items) => [...items, { role: 'assistant', content: response.result.answer ?? 'The API returned no answer.', createdAt: new Date().toISOString(), result: response.result }])
      return response.result
    } catch (error) {
      setMessages((items) => [...items, { role: 'assistant', content: apiErrorMessage(error, 'query'), createdAt: new Date().toISOString() }])
      throw error
    }
  }, [dataset, sessionId])

  const onVoiceResponse = useCallback(async (response: VoiceResponse) => {
    const audioUrl = URL.createObjectURL(new Blob(
      [Uint8Array.from(atob(response.audio_base64), (char) => char.charCodeAt(0))],
      { type: response.audio_format.toLowerCase() === 'mp3' ? 'audio/mpeg' : `audio/${response.audio_format}` },
    ))
    audioUrls.current.add(audioUrl)
    const createdAt = new Date().toISOString()
    setMessages((items) => [...items,
      { role: 'user', content: response.transcript, createdAt },
      { role: 'assistant', content: response.answer, createdAt: new Date().toISOString(), result: { answer: response.answer, evidence: [], intent: response.intent, route: response.route, result: null, visualization: response.visualization ?? null }, audioUrl },
    ])
  }, [])

  const title = useMemo(() => {
    if (location.pathname === '/') return 'Dashboard'
    const route = routes.find((item) => location.pathname.startsWith(item.to))
    return route?.label ?? (location.pathname === '/settings' ? 'Settings' : 'Workspace')
  }, [location.pathname])

  return <div className={`app-shell${compactMode ? ' compact-mode' : ''}`}>
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Activity size={18} strokeWidth={2.4} /></div><div><strong>FIA</strong><span>FINANCIAL INTELLIGENCE AGENT</span></div></div>
      <div className="workspace-switch"><div className="workspace-avatar">F</div><div className="workspace-copy"><strong>Financial workspace</strong><span>Workspace</span></div><MoreHorizontal size={17} className="muted" /></div>
      <div className="nav-label">WORKSPACE</div>
      <nav className="main-nav" aria-label="Primary navigation">{routes.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} aria-label={label} title={label} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}><Icon size={17} strokeWidth={1.8} /><span>{label}</span>{to === '/history' && messages.length > 0 && <span className="nav-count">{messages.filter((m) => m.role === 'user').length}</span>}</NavLink>)}</nav>
      <div className="sidebar-dataset">
        <div className="sidebar-dataset-heading"><span className="nav-label">ACTIVE DATASET</span><button className="icon-button tiny" aria-label="Upload dataset" onClick={() => navigate('/datasets')}><Plus size={15} /></button></div>
        {dataset ? <button className="active-file" onClick={() => navigate('/datasets')}><FileSpreadsheet size={17} /><span><strong>{dataset.filename}</strong><small>{dataset.rows.toLocaleString()} rows · {dataset.columns} columns</small></span><span className="file-dot" /></button> : <button className="no-file" onClick={() => navigate('/datasets')}><FileUp size={17} /><span>Add a dataset</span><ArrowUpRight size={14} /></button>}
      </div>
      <div className="sidebar-bottom"><div className="profile"><div className="profile-avatar">FI</div><div><strong>Local workspace</strong><span>Account service unavailable</span></div><MoreHorizontal size={17} className="muted" /></div></div>
    </aside>

    <main className="main-shell">
      <header className="topbar"><div className="crumb"><span>Workspace</span><ChevronRight size={14} /><strong>{title}</strong></div><div className="topbar-right"><button className={`backend-status ${health}`} onClick={() => void refreshHealth()} title="Check FIA API connection"><span className="status-indicator" />{health === 'connected' ? 'API connected' : health === 'checking' ? 'Checking API' : 'FIA API unavailable'}</button><div className="topbar-divider" />{dataset ? <div className="top-dataset"><Database size={15} /><span>{dataset.filename}</span><span className="top-dataset-dot" /></div> : <span className="top-dataset empty">No dataset selected</span>}</div></header>
      {datasetRestoreError && <div className="dataset-restore-error" role="alert"><span>{datasetRestoreError}</span><button onClick={() => { setDatasetRestoreError(''); navigate('/datasets') }}>Upload dataset <ArrowUpRight size={13} /></button></div>}
      <Suspense fallback={<div className="route-loading" role="status"><span className="spinner" /> Loading workspace…</div>}><Routes>
        <Route path="/" element={<DashboardPage dataset={dataset} messages={messages} apiStatus={health} modelStatus={modelStatus} onNavigate={navigate} />} />
        <Route path="/dashboard" element={<DashboardPage dataset={dataset} messages={messages} apiStatus={health} modelStatus={modelStatus} onNavigate={navigate} />} />
        <Route path="/chat" element={<ChatPage dataset={dataset} messages={messages} ask={ask} onNavigate={navigate} onNewConversation={startNewConversation} />} />
        <Route path="/voice" element={<VoicePage dataset={dataset} sessionId={sessionId} messages={messages} ask={ask} onVoiceResponse={onVoiceResponse} onNavigate={navigate} autoPlay={voiceAutoPlay} />} />
        <Route path="/datasets/:datasetId" element={<DatasetDetailsPage onActivate={activateDataset} />} />
        <Route path="/datasets" element={<DatasetsPage dataset={dataset} preview={preview} onActivate={activateDataset} onPage={fetchPreview} loading={datasetLoading} onNavigate={navigate} />} />
        <Route path="/analytics" element={<AnalyticsPage dataset={dataset} ask={ask} onNavigate={navigate} />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="/history" element={<HistoryPage messages={messages} dataset={dataset} onNavigate={navigate} />} />
        <Route path="/settings" element={<SettingsPage health={health} modelStatus={modelStatus} modelStatusError={modelStatusError} apiBase={api.baseUrl} refreshHealth={refreshHealth} voiceAutoPlay={voiceAutoPlay} compactMode={compactMode} onVoiceAutoPlay={(value) => { setVoiceAutoPlay(value); localStorage.setItem('fia.voiceAutoPlay', String(value)) }} onCompactMode={(value) => { setCompactMode(value); localStorage.setItem('fia.compactMode', String(value)) }} />} />
        <Route path="*" element={<ChatPage dataset={dataset} messages={messages} ask={ask} onNavigate={navigate} onNewConversation={startNewConversation} />} />
      </Routes></Suspense>
    </main>
  </div>
}

function PageHeading({ eyebrow, title, subtitle, action }: { eyebrow: string; title: string; subtitle: string; action?: React.ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>{action}</div>
}

function ContextPanel({ dataset, onQuickAsk, onNavigate }: { dataset: Dataset | null; onQuickAsk: (q: string) => void; onNavigate: (to: string) => void }) {
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  useEffect(() => {
    if (dataset) void api.preview(dataset.dataset_id, 0, 1).then(setPreview).catch(() => setPreview(null))
    else setPreview(null)
  }, [dataset])
  const actions = ['Summarize this dataset', 'Show revenue trend', 'Find unusual values']
  return <aside className="context-panel">
    <div className="context-heading"><h3>Dataset context</h3><CircleHelp size={15} className="muted" /></div>
    {dataset ? <>
      <div className="context-file"><div className="file-type-icon"><FileSpreadsheet size={17} /></div><div><strong title={dataset.filename}>{dataset.filename}</strong><span>Active dataset</span></div><Check size={15} className="success-icon" /></div>
      <div className="dataset-stats"><div><span>Rows</span><strong>{dataset.rows.toLocaleString()}</strong></div><div><span>Columns</span><strong>{dataset.columns}</strong></div></div>
      <div className="context-section"><div className="section-title">COLUMNS <button className="text-button" onClick={() => onNavigate('/datasets')}>View all</button></div><div className="column-list">{(preview?.column_names ?? dataset.column_names).slice(0, 6).map((column) => <span key={column} className="column-chip"><i />{column}</span>)}{dataset.columns > 6 && <span className="more-columns">+{dataset.columns - 6} more</span>}</div></div>
      <div className="context-section quick-questions"><div className="section-title">QUICK QUESTIONS</div>{actions.map((question) => <button key={question} onClick={() => onQuickAsk(question)}><Sparkles size={13} />{question}<ArrowUpRight size={13} /></button>)}</div>
      <div className="context-note"><ShieldCheck size={15} /><span>Answers are grounded in this dataset and reviewed by backend guardrails.</span></div>
    </> : <div className="context-empty"><div className="empty-icon"><Database size={19} /></div><strong>No dataset selected</strong><span>Upload a financial file to see its structure and ask questions.</span></div>}
  </aside>
}

function ChatPage({ dataset, messages, ask, onNavigate, onNewConversation }: { dataset: Dataset | null; messages: ChatMessage[]; ask: (q: string) => Promise<QueryResult>; onNavigate: (to: string) => void; onNewConversation: () => void }) {
  const [question, setQuestion] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const endRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, pending])
  async function submit(text = question) {
    const value = text.trim()
    if (!value || pending) return
    setQuestion(''); setError(''); setPending(true)
    try { await ask(value) } catch (e) { setError(e instanceof Error ? e.message : 'Request failed') }
    finally { setPending(false) }
  }
  return <div className="workspace-grid">
    <section className="chat-workspace">
      <PageHeading eyebrow="FINANCIAL ANALYST · WORKSPACE" title="Chat with your financial data" subtitle={dataset ? `Ask a question about ${dataset.filename}. Answers are generated from your active dataset.` : 'Upload a dataset to begin an analysis grounded in your numbers.'} action={<div className="chat-heading-actions">{messages.length > 0 && <button className="outline-button" onClick={onNewConversation}><RotateCcw size={14} /> New conversation</button>}<button className="outline-button voice-shortcut" onClick={() => onNavigate('/voice')}><Mic size={15} /> Voice mode</button></div>} />
      {!dataset ? <div className="chat-empty"><div className="empty-orbit"><div className="orbit-core"><Sparkles size={22} /></div><i /><b /></div><h2>Start with your data</h2><p>Upload a CSV or Excel workbook. Your analyst will use the backend to inspect and reason over its contents.</p><button className="primary-button" onClick={() => onNavigate('/datasets')}><FileUp size={16} /> Upload dataset <ArrowUpRight size={15} /></button></div> : messages.length === 0 ? <div className="welcome-block"><div className="welcome-icon"><Sparkles size={19} /></div><div className="welcome-copy"><h2>What would you like to understand?</h2><p>Questions are sent to your financial analysis backend. Start with a prompt below or write your own.</p></div><div className="prompt-grid">{['What are the key financial metrics?', 'Show the trend over time', 'Which categories perform best?', 'Find unusual values in the data'].map((q, i) => <button className="prompt-card" key={q} onClick={() => void submit(q)}><span className={`prompt-index p${i + 1}`}>0{i + 1}</span><span>{q}</span><ArrowUpRight size={14} /></button>)}</div></div> : <div className="message-list">{messages.map((message, index) => <MessageBubble key={`${index}-${message.role}`} message={message} onRetry={message.role === 'assistant' && !message.result && messages[index - 1]?.role === 'user' ? () => void submit(messages[index - 1].content) : undefined} />)}{pending && <div className="assistant-message"><div className="assistant-avatar"><Sparkles size={14} /></div><div className="thinking"><span /><span /><span /> Analyzing your question</div></div>}<div ref={endRef} /></div>}
      {error && <div className="inline-error"><X size={15} />{error}<button onClick={() => setError('')} aria-label="Dismiss"><X size={14} /></button></div>}
      {dataset && <div className="composer-wrap"><form className="composer" onSubmit={(e) => { e.preventDefault(); void submit() }}><button type="button" className="composer-plus" aria-label="Upload another dataset" onClick={() => onNavigate('/datasets')}><Plus size={17} /></button><textarea aria-label="Ask about your dataset" placeholder="Ask a question about your data..." value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void submit() } }} rows={1} /><button type="button" className="composer-mic" onClick={() => onNavigate('/voice')} aria-label="Open voice mode"><Mic size={17} /></button><button type="submit" className="send-button" disabled={!question.trim() || pending} aria-label="Send question"><Send size={15} /></button></form><div className="composer-caption"><span><ShieldCheck size={12} /> Backend grounded analysis</span><span>Enter to send · Shift + Enter for a new line</span></div></div>}
    </section>
    <ContextPanel dataset={dataset} onQuickAsk={(q) => void submit(q)} onNavigate={onNavigate} />
  </div>
}

function MessageBubble({ message, onRetry }: { message: ChatMessage; onRetry?: () => void }) {
  const [playing, setPlaying] = useState(false)
  const [audioError, setAudioError] = useState('')
  const audioRef = useRef<HTMLAudioElement>(null)
  const audio = message.audioUrl
  async function toggleAudio() {
    if (!audioRef.current) return
    setAudioError('')
    if (playing) {
      audioRef.current.pause()
      setPlaying(false)
      return
    }
    try {
      await audioRef.current.play()
      setPlaying(true)
    } catch {
      setPlaying(false)
      setAudioError('The response audio could not be played in this browser.')
    }
  }
  const timestamp = message.createdAt ? new Date(message.createdAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : ''
  const content = typeof message.content === 'string' ? message.content : 'The API returned an invalid answer.'
  const result = message.result && typeof message.result === 'object' ? message.result : undefined
  const evidence = Array.isArray(result?.evidence) ? result.evidence.filter((item): item is Evidence => Boolean(item) && typeof item === 'object') : []
  return message.role === 'user' ? <div className="user-message"><div><span className="message-time">{timestamp}</span>{content}</div><span className="user-avatar">SA</span></div> : <div className="assistant-message"><div className="assistant-avatar"><Sparkles size={14} /></div><div className="assistant-body"><div className="assistant-meta"><strong>FIA</strong><span>·</span><span>{timestamp}</span></div><div className="answer-text">{renderAssistantAnswer(content)}</div><VisualizationChart visualization={result?.visualization} />{result && <div className="answer-tags">{result.route && <span className="route-tag"><i />{result.intent?.replaceAll('_', ' ') ?? result.route}</span>}{evidence.length > 0 && <span className="evidence-tag"><ShieldCheck size={12} /> {evidence.length} evidence source{evidence.length === 1 ? '' : 's'}</span>}<span className="evidence-tag"><ShieldCheck size={12} /> Guardrails passed</span></div>}{result && evidence.length > 0 && <EvidenceDetails evidence={evidence} />}{audio && <div className="audio-replay"><audio ref={audioRef} src={audio} onEnded={() => setPlaying(false)} /><button onClick={() => void toggleAudio()}><Volume2 size={14} />{playing ? 'Pause response' : 'Play voice response'}{playing ? <Square size={12} /> : <Play size={12} />}</button></div>}{audioError && <div className="upload-error voice-error" role="alert">{audioError}</div>}<div className="message-actions"><button onClick={() => void navigator.clipboard?.writeText(content)} aria-label="Copy response"><Copy size={13} />Copy</button>{onRetry && <button onClick={onRetry} aria-label="Retry failed question"><RotateCcw size={13} />Retry</button>}</div></div></div>
}

function renderAssistantAnswer(text: string) {
  const lines = text.split(/\r?\n/)
  const nodes: React.ReactNode[] = []
  let paragraph: string[] = []
  const flushParagraph = () => {
    if (paragraph.length) nodes.push(<p key={`p-${nodes.length}`}>{paragraph.join(' ')}</p>)
    paragraph = []
  }
  const cells = (line: string) => line.trim().replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim())

  for (let index = 0; index < lines.length;) {
    const line = lines[index]
    const heading = line.match(/^#{1,4}\s+(.+)$/)
    if (!line.trim()) { flushParagraph(); index += 1; continue }
    if (heading) {
      flushParagraph()
      nodes.push(<h4 key={`h-${nodes.length}`}>{heading[1]}</h4>)
      index += 1
      continue
    }
    if (line.includes('|') && index + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[index + 1])) {
      flushParagraph()
      const headers = cells(line)
      index += 2
      const rows: string[][] = []
      while (index < lines.length && lines[index].includes('|')) rows.push(cells(lines[index++]))
      nodes.push(<div className="answer-table-scroll" key={`table-${nodes.length}`}><table><thead><tr>{headers.map((header, cellIndex) => <th key={cellIndex}>{header}</th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{headers.map((_, cellIndex) => <td key={cellIndex}>{row[cellIndex] ?? ''}</td>)}</tr>)}</tbody></table></div>)
      continue
    }
    const listItem = line.match(/^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$/)
    if (listItem) {
      flushParagraph()
      const ordered = /^\s*\d+[.)]/.test(line)
      const items: string[] = []
      while (index < lines.length) {
        const match = lines[index].match(/^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$/)
        if (!match || ordered !== /^\s*\d+[.)]/.test(lines[index])) break
        items.push(match[1])
        index += 1
      }
      const List = ordered ? 'ol' : 'ul'
      nodes.push(<List key={`list-${nodes.length}`}>{items.map((item, itemIndex) => <li key={itemIndex}>{item}</li>)}</List>)
      continue
    }
    paragraph.push(line.trim())
    index += 1
  }
  flushParagraph()
  return nodes
}

function EvidenceDetails({ evidence }: { evidence: Evidence[] }) {
  return <details className="message-evidence"><summary>View evidence returned by the backend</summary><div>{evidence.map((source, index) => <section key={`${source.source}-${index}`}><strong>{source.source}</strong><ul>{evidenceLines(source).map((line, lineIndex) => <li key={`${index}-${lineIndex}`}>{line}</li>)}</ul></section>)}</div></details>
}

function evidenceLines(source: Evidence): string[] {
  const findings = source.findings
  if (Array.isArray(findings)) return findings.map((finding) => {
    if (finding && typeof finding === 'object') {
      const record = finding as Record<string, unknown>
      const body = [record.text, record.finding, record.description].find((value): value is string => typeof value === 'string')
      const metadata = record.metadata && typeof record.metadata === 'object' ? record.metadata as Record<string, unknown> : {}
      const location = [metadata.source, metadata.page].filter((value) => typeof value === 'string' || typeof value === 'number').join(' · ')
      if (body) return location ? `${body} (${location})` : body
    }
    return typeof finding === 'string' ? finding : JSON.stringify(finding) ?? String(finding)
  })
  if (typeof findings === 'string') return [findings]
  return [JSON.stringify(findings) ?? String(findings)]
}

function DatasetsPage({ dataset, preview, onActivate, onPage, loading, onNavigate }: {
  dataset: Dataset | null; preview: DatasetPreview | null
  onActivate: (value: Dataset) => void; onPage: (offset?: number) => Promise<void>; loading: boolean
  onNavigate: (path: string) => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [pageError, setPageError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const allowed = (candidate: File) => {
    const extension = candidate.name.split('.').pop()?.toLowerCase()
    if (!['csv', 'xlsx', 'xls'].includes(extension ?? '')) {
      setError('Choose a CSV, XLSX, or XLS file. These are the formats accepted by the backend.')
      return false
    }
    if (candidate.size > 50 * 1024 * 1024) {
      setError('This file exceeds the backend 50 MB limit.')
      return false
    }
    setError(''); setFile(candidate); return true
  }
  async function upload() {
    if (!file || busy) return
    setBusy(true); setProgress(0); setError('')
    try {
      const data = await api.upload(file, setProgress)
      onActivate(data)
    } catch (e) { setError(apiErrorMessage(e, 'upload')) }
    finally { setBusy(false); setProgress(null) }
  }
  async function page(offset: number) {
    setPageError('')
    try { await onPage(offset) } catch (e) { setPageError(apiErrorMessage(e, 'dataset')) }
  }
  const offset = preview?.offset ?? 0
  const limit = preview?.limit ?? 25
  const rows = preview?.preview ?? []
  return <div className="page-scroll datasets-page">
    <PageHeading eyebrow="DATA MANAGEMENT" title="Dataset manager" subtitle="Upload a financial file and inspect the active dataset. The API does not list all stored datasets." />
    <div className="dataset-layout">
      <section className="surface upload-surface"><div className="section-heading"><div className="upload-glyph"><FileUp size={18} /></div><div><h2>Upload a dataset</h2><p>CSV and Excel files are parsed by your connected backend.</p></div></div>
        <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); const incoming = e.dataTransfer.files[0]; if (incoming) allowed(incoming) }}>
          <input ref={inputRef} type="file" accept=".csv,.xlsx,.xls" hidden onChange={(e) => { const incoming = e.target.files?.[0]; if (incoming) allowed(incoming) }} />
          <div className="upload-cloud"><FileUp size={21} /></div><strong>{file?.name ?? 'Drop your file here'}</strong><span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ready to upload` : 'or browse from your computer'}</span><button className="browse-button" onClick={() => inputRef.current?.click()} disabled={busy}>Browse files</button><small>Supported by backend: .csv · .xlsx · .xls</small>
        </div>
        {progress !== null && <div className="progress-wrap"><div><span>{progress === 100 ? 'Processing on backend…' : 'Uploading file…'}</span><span>{progress}%</span></div><div className="progress-track"><i style={{ width: `${progress}%` }} /></div></div>}
        {error && <div className="upload-error"><X size={15} />{error}</div>}
        <button className="primary-button upload-submit" onClick={() => void upload()} disabled={!file || busy}>{busy ? <><span className="spinner" /> Uploading…</> : <><FileUp size={15} /> Upload dataset <ArrowUpRight size={15} /></>}</button>
        <div className="upload-footnote"><ShieldCheck size={14} /> Files are processed by the backend. The API accepts files up to 50 MB.</div>
      </section>
      <section className="surface active-dataset-surface"><div className="section-heading"><div className="upload-glyph"><Database size={18} /></div><div><h2>Active dataset</h2><p>Dataset metadata returned by the API.</p></div></div>
        {dataset ? <><div className="dataset-current"><div className="file-type-icon large"><FileSpreadsheet size={20} /></div><div className="current-file-copy"><strong>{dataset.filename}</strong><span>In memory on this backend process</span></div><span className="active-badge"><i />Active</span></div><div className="dataset-metadata"><div><span>Rows</span><strong>{dataset.rows.toLocaleString()}</strong></div><div><span>Columns</span><strong>{dataset.columns}</strong></div><div><span>Format</span><strong>{dataset.filename.split('.').pop()?.toUpperCase() ?? '—'}</strong></div><div><span>Dataset ID</span><strong className="id-value" title={dataset.dataset_id}>{dataset.dataset_id.slice(0, 8)}…</strong></div></div><div className="schema-preview-title"><span>SCHEMA</span><span>{dataset.columns} columns</span></div><div className="schema-chips">{(preview?.column_names ?? dataset.column_names).map((column) => <span key={column}>{column}</span>)}</div><button className="outline-button dataset-details-link" onClick={() => onNavigate(`/datasets/${encodeURIComponent(dataset.dataset_id)}`)}>Open paginated dataset view <ArrowUpRight size={14} /></button></> : <div className="dataset-placeholder"><Database size={20} /><span>Upload a file to activate a dataset.</span></div>}
      </section>
    </div>
    <section className="surface table-surface"><div className="table-header"><div><div className="eyebrow">DATA EXPLORER</div><h2>Preview rows</h2><p>{dataset ? `A paginated sample from ${dataset.filename}.` : 'Select a dataset to load a preview.'}</p></div><span className="sample-badge">{rows.length} rows shown</span></div>
      {pageError && <div className="upload-error"><X size={15} />{pageError}</div>}
      {dataset && (preview || loading) ? <><div className="table-scroll">{loading ? <div className="table-loading"><span className="spinner" /> Loading dataset metadata…</div> : <table><thead><tr>{preview?.column_names.map((column) => <th key={column}>{column}<small>{preview.column_types[column]}</small></th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={`${offset}-${i}`}>{preview?.column_names.map((column) => <td key={column}>{formatCell(row[column])}</td>)}</tr>)}</tbody></table>}</div><div className="pagination"><span>Showing {rows.length ? offset + 1 : 0}–{Math.min(offset + limit, dataset.rows)} of {dataset.rows.toLocaleString()}</span><div><button aria-label="Previous page" disabled={offset === 0} onClick={() => void page(Math.max(0, offset - limit))}><ChevronLeft size={15} /></button><button aria-label="Next page" disabled={offset + limit >= dataset.rows} onClick={() => void page(offset + limit)}><ChevronRight size={15} /></button></div></div></> : <div className="preview-empty"><FileSpreadsheet size={20} /><span>Dataset preview appears here after upload.</span></div>}
    </section>
  </div>
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) return <span className="null-cell">—</span>
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

type VoiceStatus = 'idle' | 'listening' | 'processing' | 'speaking' | 'error'
const FIA_GREETING = 'Hello, this is FIA. How can I help you?'
function VoicePage({ dataset, sessionId, messages, ask, onVoiceResponse, onNavigate, autoPlay }: {
  dataset: Dataset | null; sessionId: string; messages: ChatMessage[]; ask: (q: string) => Promise<QueryResult>
  onVoiceResponse: (response: VoiceResponse) => Promise<void>; onNavigate: (to: string) => void; autoPlay: boolean
}) {
  const [status, setStatus] = useState<VoiceStatus>('idle')
  const [error, setError] = useState('')
  const [manualQuestion, setManualQuestion] = useState('')
  const [greeted, setGreeted] = useState(false)
  const greetingStarted = useRef(false)
  const requestPending = useRef(false)
  const mounted = useRef(true)
  const recorder = useRef<MediaRecorder | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const chunks = useRef<Blob[]>([])
  const audioRef = useRef<HTMLAudioElement>(null)
  const latestResponse = [...messages].reverse().find((m) => m.role === 'assistant' && m.audioUrl)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      window.speechSynthesis?.cancel()
      if (recorder.current?.state === 'recording') recorder.current.stop()
      stream.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])
  function activateFia() {
    if (greetingStarted.current) return
    greetingStarted.current = true
    setGreeted(true)
    if (!('speechSynthesis' in window) || typeof SpeechSynthesisUtterance === 'undefined') {
      setStatus('error')
      setError('Spoken greeting is not supported by this browser. Voice responses may still be played from FIA audio.')
      return
    }
    const greeting = new SpeechSynthesisUtterance(FIA_GREETING)
    setStatus('speaking')
    greeting.onstart = () => setStatus('speaking')
    greeting.onend = () => setStatus('idle')
    greeting.onerror = () => setStatus('idle')
    try { window.speechSynthesis.speak(greeting) }
    catch {
      setStatus('error')
      setError('FIA could not play the greeting. Check your browser audio settings and try again.')
    }
  }
  useEffect(() => {
    if (latestResponse?.audioUrl) {
      if (audioRef.current) audioRef.current.src = latestResponse.audioUrl
      if (autoPlay && audioRef.current) {
        void audioRef.current.play().then(() => setStatus('speaking')).catch(() => {
          setStatus('idle')
          setError('The response audio could not be played. Use Replay or check your browser audio settings.')
        })
      } else {
        setStatus('idle')
      }
    }
  }, [latestResponse?.audioUrl, autoPlay])
  function stopRecording() { if (recorder.current?.state === 'recording') recorder.current.stop() }
  async function startRecording() {
    setError('')
    if (requestPending.current || status === 'processing' || status === 'listening' || status === 'speaking') return
    if (!greeted) { activateFia(); return }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { setStatus('error'); setError('Audio recording is not supported in this browser.'); return }
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (!mounted.current) { media.getTracks().forEach((track) => track.stop()); return }
      stream.current = media
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4'].find((type) => MediaRecorder.isTypeSupported(type))
      let instance: MediaRecorder
      try { instance = new MediaRecorder(media, mimeType ? { mimeType } : undefined) }
      catch { media.getTracks().forEach((track) => track.stop()); stream.current = null; throw new Error('This browser does not support a recording format accepted by FIA.') }
      recorder.current = instance; chunks.current = []
      instance.ondataavailable = (event) => { if (event.data.size) chunks.current.push(event.data) }
      instance.onerror = () => {
        media.getTracks().forEach((track) => track.stop())
        stream.current = null
        setStatus('error')
        setError('The browser could not record audio.')
      }
      instance.onstop = async () => {
        media.getTracks().forEach((track) => track.stop()); stream.current = null
        if (!mounted.current) return
        const audioType = (instance.mimeType || mimeType || 'audio/webm').split(';')[0]
        const audio = new Blob(chunks.current, { type: audioType })
        chunks.current = []
        if (!audio.size) { setStatus('error'); setError('No audio was recorded. Please try again.'); return }
        if (audio.size > 10 * 1024 * 1024) { setStatus('error'); setError('That recording exceeds the 10 MB limit. Please record a shorter question.'); return }
        requestPending.current = true
        setStatus('processing')
        try { const response = await api.voice(dataset?.dataset_id ?? null, sessionId, audio); await onVoiceResponse(response); setStatus('idle') }
        catch (e) { setStatus('error'); setError(apiErrorMessage(e, 'voice')) }
        finally { requestPending.current = false }
      }
      instance.start(); setStatus('listening')
    } catch (e) {
      setStatus('error')
      setError(e instanceof Error && e.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow microphone access in your browser settings and retry.' : e instanceof Error ? e.message : 'Could not open the microphone.')
    }
  }
  async function askTyped(e: React.FormEvent) {
    e.preventDefault()
    if (!manualQuestion.trim() || status === 'processing' || status === 'listening' || requestPending.current) return
    const question = manualQuestion.trim(); setManualQuestion(''); setStatus('processing')
    try { await ask(question) } catch { /* the chat page stores the request error */ }
    finally { setStatus('idle') }
  }
  const label = status === 'listening' ? 'Listening…' : status === 'processing' ? 'Thinking…' : status === 'speaking' ? 'FIA is speaking…' : status === 'error' ? 'Something went wrong. Try again.' : greeted ? 'Click to speak' : 'Activate FIA to begin'
  return <div className="page-scroll voice-page"><PageHeading eyebrow="FIA VOICE EXPERIENCE" title="Talk with FIA" subtitle="Ask about financial data, documents, market data, or FIA's capabilities. Financial questions use the guarded backend workflow." action={<button className="outline-button" onClick={() => onNavigate('/chat')}><MessageSquareText size={15} /> Text chat</button>} />
    <div className="voice-layout"><section className="surface voice-console"><div className="voice-session-top"><span className="live-pill"><i /> VOICE SESSION</span><span className="voice-dataset">{dataset ? <><Database size={13} />{dataset.filename}</> : 'No dataset selected'}</span></div>
      <div className={`voice-visual ${status}`}><div className="voice-ring ring-one"/><div className="voice-ring ring-two"/><div className="voice-ring ring-three"/><div className="voice-disc">{status === 'processing' ? <span className="spinner light" /> : status === 'speaking' ? <AudioLines size={28} /> : status === 'error' ? <MicOff size={26} /> : <Mic size={27} />}</div></div>
      <div className="voice-state" role="status" aria-live="polite"><span className={`state-dot ${status}`} />{label}</div><p className="voice-hint">{status === 'listening' ? 'FIA is listening. Select stop when you have finished speaking.' : status === 'processing' ? 'Your question is being transcribed and processed.' : status === 'speaking' ? 'FIA is speaking. You can replay or stop the audio.' : !greeted ? 'FIA will greet you once when you activate the voice experience.' : dataset ? 'Ask about your data or ask FIA about its capabilities.' : 'Identity and capability questions work without a dataset. Upload one for financial analysis.'}</p>
      <div className="voice-buttons">{status === 'listening' ? <button className="record-button stop" onClick={stopRecording}><Square size={15} fill="currentColor" /> Stop recording</button> : <button className={`record-button ${status === 'processing' || status === 'speaking' ? 'disabled' : ''}`} disabled={status === 'processing' || status === 'speaking'} onClick={() => void startRecording()} aria-label={greeted ? 'Start recording a question for FIA' : 'Activate FIA and hear greeting'}>{greeted ? <Mic size={17} /> : <AudioLines size={17} />}{greeted ? 'Start speaking' : 'Activate FIA'}</button>}{status === 'speaking' && <button className="outline-button" onClick={() => { audioRef.current?.pause(); window.speechSynthesis?.cancel(); setStatus('idle') }}><Square size={13} /> Stop audio</button>}{latestResponse?.audioUrl && <button className="outline-button" onClick={() => { const player = audioRef.current; if (!player) return; player.currentTime = 0; void player.play().then(() => { setError(''); setStatus('speaking') }).catch(() => { setStatus('idle'); setError('The response audio could not be played. Check your browser audio settings and try again.') }) }} aria-label="Replay FIA response"><Play size={14} /> Replay answer</button>}</div>
      <audio ref={audioRef} onEnded={() => setStatus('idle')} onError={() => { setStatus('idle'); setError('The response audio could not be played. Use Replay or check your browser audio settings.') }} hidden />{error && <div className="upload-error voice-error" role="alert"><X size={15} />{error}</div>}
      {!dataset && <button className="text-link" onClick={() => onNavigate('/datasets')}>Upload a dataset for financial analysis <ArrowUpRight size={13} /></button>}
    </section><section className="surface voice-transcript"><div className="section-heading"><div className="upload-glyph"><AudioLines size={18} /></div><div><h2>Conversation with FIA</h2><p>Voice turns share the backend session when a dataset is active.</p></div></div>{messages.length || greeted ? <div className="voice-turns">{greeted && <div className="voice-turn assistant"><span>FIA</span><p>{FIA_GREETING}</p></div>}{messages.slice(-8).map((message, i) => <div className={`voice-turn ${message.role}`} key={`${i}-${message.role}`}><span>{message.role === 'user' ? 'YOU' : 'FIA'}</span><p>{message.content}</p>{message.role === 'assistant' && <VisualizationChart visualization={message.result?.visualization} />}</div>)}</div> : <div className="voice-empty"><AudioLines size={20} /><span>Your conversation with FIA will appear here.</span></div>}{!dataset && <p className="voice-hint">Typed financial questions need an active dataset. Use the microphone for FIA identity and capability questions.</p>}<form className="voice-text-form" onSubmit={(e) => void askTyped(e)}><input value={manualQuestion} onChange={(e) => setManualQuestion(e.target.value)} placeholder="Type a question about your dataset…" disabled={!dataset || status === 'processing' || status === 'listening'} /><button aria-label="Send question to FIA" disabled={!manualQuestion.trim() || !dataset || status === 'processing'}><Send size={15} /></button></form></section></div>
    <div className="voice-disclosure"><ShieldCheck size={14} /><span>Browser audio is sent to the configured backend for transcription. No recordings are saved by this interface.</span></div>
  </div>
}

function AnalyticsPage({ dataset, ask, onNavigate }: { dataset: Dataset | null; ask: (q: string) => Promise<QueryResult>; onNavigate: (to: string) => void }) {
  const [running, setRunning] = useState('')
  const [error, setError] = useState('')
  const [answer, setAnswer] = useState('')
  const questions = [
    { title: 'Financial overview', description: 'Ask the backend to analyze available numeric metrics.', prompt: 'Summarize the key financial metrics in this dataset.', icon: LayoutDashboard, accent: 'lavender' },
    { title: 'Trend analysis', description: 'Route a question to the backend trend analyzer.', prompt: 'Show the trend over time for the main metric.', icon: Activity, accent: 'mint' },
    { title: 'Anomaly detection', description: 'Use the backend anomaly route to find unusual values.', prompt: 'Find unusual values or anomalies in this dataset.', icon: Sparkles, accent: 'amber' },
    { title: 'Category comparison', description: 'Compare numeric results across detected dimensions.', prompt: 'Which categories have the highest values?', icon: BarChart3, accent: 'blue' },
  ]
  async function run(prompt: string) {
    if (!dataset || running) return
    setRunning(prompt); setError(''); setAnswer('')
    try { const result = await ask(prompt); setAnswer(result.answer ?? 'The backend returned no answer.') }
    catch (e) { setError(e instanceof Error ? e.message : 'Analysis request failed.') }
    finally { setRunning('') }
  }
  return <div className="page-scroll analytics-page"><PageHeading eyebrow="ANALYSIS TOOLS" title="Explore your numbers" subtitle="Each analysis runs as a real question through the backend agent and its active guardrails." action={<button className="outline-button" onClick={() => onNavigate('/chat')}><MessageSquareText size={15} /> Open chat</button>} />
    {!dataset && <div className="notice-banner"><Database size={16} /><span>Upload a dataset to run an analysis.</span><button onClick={() => onNavigate('/datasets')}>Go to datasets <ArrowUpRight size={13} /></button></div>}
    <div className="analytics-summary surface"><div className="summary-icon"><Activity size={18} /></div><div><span>BACKEND ANALYSIS</span><strong>{dataset ? dataset.filename : 'No active dataset'}</strong><small>{dataset ? `${dataset.rows.toLocaleString()} records · ${dataset.columns} columns` : 'Choose a dataset to start.'}</small></div><div className="summary-route"><span>AVAILABLE ROUTES</span><strong><i />Financial <i />Trend <i />Anomaly</strong></div></div>
    <div className="analysis-cards">{questions.map(({ title, description, prompt, icon: Icon, accent }) => <article className="analysis-card surface" key={title}><div className={`analysis-icon ${accent}`}><Icon size={18} /></div><span className="analysis-route-tag">BACKEND QUERY</span><h2>{title}</h2><p>{description}</p><div className="analysis-prompt">“{prompt}”</div><button className="analysis-run" disabled={!dataset || !!running} onClick={() => void run(prompt)}>{running === prompt ? <><span className="spinner" /> Running analysis…</> : <>Run analysis <ArrowUpRight size={14} /></>}</button></article>)}</div>
    {(answer || error) && <div className={error ? 'inline-error' : 'analysis-result'}>{error ? <><X size={15} />{error}</> : <><Check size={15} />{answer}<button className="text-button" onClick={() => onNavigate('/history')}>View session history <ArrowUpRight size={13} /></button></>}</div>}
    <div className="analytics-note"><ShieldCheck size={14} /><span>Trends and anomalies use backend analysis. Charts and metrics are shown only when the API provides them.</span></div>
  </div>
}

function HistoryPage({ messages, dataset, onNavigate }: { messages: ChatMessage[]; dataset: Dataset | null; onNavigate: (to: string) => void }) {
  const userTurns = messages.filter((message) => message.role === 'user')
  return <div className="page-scroll history-page"><PageHeading eyebrow="CURRENT SESSION" title="Conversation history" subtitle="Turns from the active in-memory session are kept for this browser tab." action={<button className="outline-button" onClick={() => onNavigate('/chat')}><MessageSquareText size={15} /> Back to chat</button>} />
    <div className="surface history-surface">{userTurns.length ? <>{userTurns.map((message, index) => { const response = messages[messages.indexOf(message) + 1]; const timestamp = message.createdAt ? new Date(message.createdAt).toLocaleString() : 'Time unavailable'; return <div className="history-item" key={`${index}-${message.content}`}><div className="history-index">{String(index + 1).padStart(2, '0')}</div><div className="history-copy"><span>YOU ASKED · {timestamp}</span><strong>{message.content}</strong>{response && <p>{response.content}</p>}</div><div className="history-meta">{response?.result?.route && <span className="route-tag"><i />{response.result.route}</span>}{dataset && <span>{dataset.filename}</span>}<span>Current session</span></div></div> })}</> : <div className="history-empty"><div className="empty-icon"><Clock3 size={19} /></div><h2>No turns yet</h2><p>Ask your first question in chat or voice mode. Conversation memory is kept by the backend for this session.</p><button className="primary-button" onClick={() => onNavigate('/chat')}><MessageSquareText size={15} /> Start chatting <ArrowUpRight size={14} /></button></div>}</div>
    <div className="history-retention"><ShieldCheck size={14} /><span>Message content stays in app memory and is cleared when the page is refreshed. The backend keeps up to 10 turns per session while its process is running.</span></div>
  </div>
}

function SettingsPage({ health, modelStatus, modelStatusError, apiBase, refreshHealth, voiceAutoPlay, compactMode, onVoiceAutoPlay, onCompactMode }: {
  health: Status; modelStatus: ModelStatus | null; modelStatusError: string; apiBase: string; refreshHealth: () => Promise<void>; voiceAutoPlay: boolean; compactMode: boolean
  onVoiceAutoPlay: (value: boolean) => void; onCompactMode: (value: boolean) => void
}) {
  return <div className="page-scroll settings-page"><PageHeading eyebrow="PREFERENCES & CONNECTIONS" title="Settings" subtitle="Local preferences and backend connection details." />
    <section className="surface settings-card"><div className="settings-card-heading"><div className="upload-glyph"><Activity size={18} /></div><div><h2>Backend connection</h2><p>Connection is checked against the actual health endpoint.</p></div></div><div className="setting-row"><div><strong>API status</strong><span>GET /health</span></div><div className={`setting-value ${health}`}><i />{health === 'connected' ? 'Connected' : health === 'checking' ? 'Checking…' : 'Disconnected'}</div></div><div className="setting-row"><div><strong>API address</strong><span>Configured from VITE_API_BASE_URL or the default</span></div><code>{apiBase}</code></div><div className="setting-actions"><button className="outline-button" onClick={() => void refreshHealth()}><Activity size={14} /> Check connection</button></div></section>
    <section className="surface settings-card"><div className="settings-card-heading"><div className="upload-glyph"><Sparkles size={18} /></div><div><h2>Reasoning model</h2><p>Live configuration reported by GET /model/status.</p></div></div>{modelStatus ? <><div className="setting-row"><div><strong>Provider</strong><span>Active reasoning provider</span></div><code>{modelStatus.provider}</code></div><div className="setting-row"><div><strong>Model</strong><span>Base model identifier</span></div><code>{modelStatus.base_model ?? modelStatus.model ?? 'Not reported'}</code></div><div className="setting-row"><div><strong>QLoRA adapter</strong><span>{modelStatus.adapter_available ? 'Adapter artifact found' : modelStatus.provider === 'qwen' ? 'Adapter artifact unavailable' : 'Not applicable'}</span></div><strong>{modelStatus.adapter_loaded ? 'Loaded' : 'Not loaded'}</strong></div><div className="setting-row"><div><strong>Runtime</strong><span>Quantization · device · mode</span></div><code>{modelStatus.quantized ? '4-bit' : 'full precision'} · {modelStatus.device} · {modelStatus.mode ?? modelStatus.provider}</code></div></> : <div className="setting-row"><div><strong>Model status</strong><span>{modelStatusError || 'Waiting for the backend status endpoint.'}</span></div><span className="setting-description">Unavailable</span></div>}</section>
    <section className="surface settings-card"><div className="settings-card-heading"><div className="upload-glyph"><ShieldCheck size={18} /></div><div><h2>Data and privacy</h2><p>How this interface handles your dataset and conversation.</p></div></div><div className="setting-row"><div><strong>Dataset storage</strong><span>In-memory backend storage</span></div><span className="setting-description">A backend restart clears datasets and sessions.</span></div><div className="setting-row"><div><strong>Conversation context</strong><span>Server-side session memory</span></div><span className="setting-description">Up to 10 turns per session, scoped to its active dataset.</span></div><div className="setting-row"><div><strong>Voice</strong><span>Browser recording → backend transcription and speech</span></div><span className="setting-description">Audio is not stored by the interface.</span></div></section>
    <section className="surface settings-card"><div className="settings-card-heading"><div className="upload-glyph"><Activity size={18} /></div><div><h2>Preferences</h2><p>Saved in this browser only; these settings are not backend capabilities.</p></div></div><label className="setting-row setting-toggle"><span><strong>Play voice answers automatically</strong><small>When disabled, use Play response on the assistant message.</small></span><input type="checkbox" checked={voiceAutoPlay} onChange={(event) => onVoiceAutoPlay(event.target.checked)} /></label><label className="setting-row setting-toggle"><span><strong>Compact workspace</strong><small>Reduce workspace spacing for denser views.</small></span><input type="checkbox" checked={compactMode} onChange={(event) => onCompactMode(event.target.checked)} /></label><div className="setting-row"><div><strong>Appearance</strong><span>Dark theme</span></div><span className="setting-description">The current interface provides a dark theme.</span></div></section>
  </div>
}
