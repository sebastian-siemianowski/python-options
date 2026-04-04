/**
 * Multi-Format Data Export (Story 12.3)
 *
 * Exports data as CSV, JSON, or clipboard. Respects current filter/sort state.
 */

export type ExportFormat = 'csv' | 'json' | 'clipboard';

interface ExportOptions {
  /** Filename without extension */
  filename: string;
  /** Column headers for CSV - keys map to data object properties */
  columns: { key: string; label: string }[];
  /** Data rows */
  data: Record<string, unknown>[];
  /** Format to export */
  format: ExportFormat;
}

function formatValue(val: unknown): string {
  if (val == null) return '';
  if (typeof val === 'number') return String(val);
  if (typeof val === 'boolean') return val ? 'Yes' : 'No';
  return String(val);
}

function toCSV(columns: ExportOptions['columns'], data: ExportOptions['data']): string {
  const header = columns.map((c) => `"${c.label.replace(/"/g, '""')}"`).join(',');
  const rows = data.map((row) =>
    columns.map((c) => {
      const val = formatValue(row[c.key]);
      // Escape CSV values
      if (val.includes(',') || val.includes('"') || val.includes('\n')) {
        return `"${val.replace(/"/g, '""')}"`;
      }
      return val;
    }).join(','),
  );
  return [header, ...rows].join('\n');
}

function toJSON(columns: ExportOptions['columns'], data: ExportOptions['data']): string {
  const mapped = data.map((row) => {
    const obj: Record<string, unknown> = {};
    for (const c of columns) {
      obj[c.label] = row[c.key];
    }
    return obj;
  });
  return JSON.stringify(mapped, null, 2);
}

function toClipboard(columns: ExportOptions['columns'], data: ExportOptions['data']): string {
  const header = columns.map((c) => c.label).join('\t');
  const rows = data.map((row) =>
    columns.map((c) => formatValue(row[c.key])).join('\t'),
  );
  return [header, ...rows].join('\n');
}

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function exportData(opts: ExportOptions): Promise<'success' | 'error'> {
  try {
    const { columns, data, filename, format } = opts;

    if (format === 'csv') {
      const content = toCSV(columns, data);
      downloadFile(content, `${filename}.csv`, 'text/csv;charset=utf-8');
      return 'success';
    }

    if (format === 'json') {
      const content = toJSON(columns, data);
      downloadFile(content, `${filename}.json`, 'application/json');
      return 'success';
    }

    if (format === 'clipboard') {
      const content = toClipboard(columns, data);
      await navigator.clipboard.writeText(content);
      return 'success';
    }

    return 'error';
  } catch {
    return 'error';
  }
}
