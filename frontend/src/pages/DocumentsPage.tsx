import { ArrowUpRight, FileText, Search, ShieldAlert } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

export default function DocumentsPage() {
  const navigate = useNavigate()
  return <div className="page-scroll documents-page">
    <div className="page-heading"><div><div className="eyebrow">DOCUMENT INTELLIGENCE</div><h1>Documents</h1><p>Search indexed financial reports when document ingestion is available.</p></div></div>
    <section className="surface documents-unavailable">
      <div className="documents-icon"><FileText size={21} /></div>
      <span className="eyebrow">RAG DOCUMENTS</span>
      <h2>Document ingestion is not available through the API</h2>
      <p>The backend has internal PDF retrieval code, but it does not expose document upload, indexing, listing, or deletion endpoints. This page will not accept or imply an upload until those operations exist.</p>
      <div className="documents-capabilities"><div><Search size={15} /><span>Retrieval status</span><strong>Not available</strong></div><div><ShieldAlert size={15} /><span>Indexed documents</span><strong>Not reported by API</strong></div></div>
      <button className="outline-button" onClick={() => navigate('/chat')}>Return to dataset analysis <ArrowUpRight size={14} /></button>
    </section>
  </div>
}
