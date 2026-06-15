import { ExternalLink, FileCheck } from 'lucide-react';

interface Props {
  documentUrl: string;
  rawArtifactPath: string;
  parserVersion: string;
  sourceHash: string;
}

export default function FilingAuditMetadata({
  documentUrl,
  rawArtifactPath,
  parserVersion,
  sourceHash,
}: Props) {
  return (
    <section className="glass-card p-4" aria-label="Filing source audit metadata">
      <div className="flex items-start gap-3">
        <div
          className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{ background: 'var(--violet-8)' }}
        >
          <FileCheck className="w-4 h-4" style={{ color: 'var(--accent-violet)' }} />
        </div>
        <div className="min-w-0 text-[12px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          <a
            href={documentUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 font-medium"
            style={{ color: 'var(--text-luminous)' }}
          >
            Official source document
            <ExternalLink className="w-3 h-3" />
          </a>
          <div className="mt-2 grid gap-1">
            <span>Parser: {parserVersion}</span>
            <span className="break-all">Source hash: {sourceHash}</span>
            <span className="break-all">Raw artifact: {rawArtifactPath}</span>
          </div>
        </div>
      </div>
    </section>
  );
}
