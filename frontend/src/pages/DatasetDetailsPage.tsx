import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, ArrowRight, Database, FileSpreadsheet, MessageSquareText, RefreshCw } from 'lucide-react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Dataset, DatasetPreview } from '../types'

type Props = { onActivate: (dataset: Dataset, destination?: string) => void }

export default function DatasetDetailsPage({ onActivate }: Props) {
  const { datasetId = '' } = useParams()
  const navigate = useNavigate()
  const [offset, setOffset] = useState(0)
  const limit = 25
  const query = useQuery<DatasetPreview, Error>({
    queryKey: ['dataset-preview', datasetId, offset],
    queryFn: () => api.preview(datasetId, offset, limit),
    enabled: Boolean(datasetId),
    staleTime: 30_000,
    retry: false,
  })
  const preview = query.data ?? null

  const activeDataset: Dataset | null = preview ? {
    dataset_id: preview.dataset_id,
    filename: preview.filename,
    rows: preview.rows,
    columns: preview.columns,
    column_names: preview.column_names,
  } : null

  return <div className="page-scroll dataset-details-page">
    <div className="page-heading"><div><div className="eyebrow">DATASET EXPLORER</div><h1>{preview?.filename ?? 'Dataset details'}</h1><p>Paginated metadata and rows returned by the backend.</p></div><button className="outline-button" onClick={() => navigate('/datasets')}><ArrowLeft size={14} /> All datasets</button></div>
    {query.isPending && <div className="surface dataset-details-state"><span className="spinner" /> Loading dataset metadata…</div>}
    {query.error && <div className="surface dataset-details-state error-state"><p>{query.error.message}</p><button className="outline-button" onClick={() => void query.refetch()}><RefreshCw size={14} /> Retry</button></div>}
    {preview && <>
      <section className="surface dataset-details-summary">
        <div className="dataset-details-file"><div className="file-type-icon large"><FileSpreadsheet size={20} /></div><div><span className="eyebrow">ACTIVE IN BACKEND MEMORY</span><h2>{preview.filename}</h2><code>{preview.dataset_id}</code></div></div>
        <div className="dashboard-dataset-stats"><div><span>Rows</span><strong>{preview.rows.toLocaleString()}</strong></div><div><span>Columns</span><strong>{preview.columns}</strong></div><div><span>Preview page</span><strong>{preview.preview.length} rows</strong></div></div>
        <div className="dataset-detail-actions"><button className="primary-button" disabled={!activeDataset} onClick={() => activeDataset && onActivate(activeDataset)}><MessageSquareText size={14} /> Use in chat</button><button className="outline-button" disabled={!activeDataset} onClick={() => activeDataset && onActivate(activeDataset, '/analytics')}><Database size={14} /> Analyze</button></div>
      </section>
      <section className="surface table-surface dataset-details-table">
        <div className="table-header"><div><div className="eyebrow">SCHEMA & ROWS</div><h2>Dataset preview</h2><p>Column types and at most 25 rows are loaded per page.</p></div><span className="sample-badge">{preview.rows.toLocaleString()} total rows</span></div>
        <div className="schema-chips dataset-detail-schema">{preview.column_names.map((column) => <span key={column}>{column}<small>{preview.column_types[column]}</small></span>)}</div>
        <div className="table-scroll"><table><thead><tr>{preview.column_names.map((column) => <th key={column}>{column}<small>{preview.column_types[column]}</small></th>)}</tr></thead><tbody>{query.isFetching ? <tr><td colSpan={preview.column_names.length} className="table-loading"><span className="spinner" /> Loading page…</td></tr> : preview.preview.map((row, rowIndex) => <tr key={`${offset}-${rowIndex}`}>{preview.column_names.map((column) => <td key={column}>{formatCell(row[column])}</td>)}</tr>)}</tbody></table></div>
        <div className="pagination"><span>Showing {preview.preview.length ? offset + 1 : 0}–{Math.min(offset + preview.preview.length, preview.rows)} of {preview.rows.toLocaleString()}</span><div><button aria-label="Previous page" disabled={offset === 0 || query.isFetching} onClick={() => setOffset(Math.max(0, offset - limit))}><ArrowLeft size={14} /></button><button aria-label="Next page" disabled={offset + limit >= preview.rows || query.isFetching} onClick={() => setOffset(offset + limit)}><ArrowRight size={14} /></button></div></div>
      </section>
    </>}
  </div>
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) return <span className="null-cell">—</span>
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
