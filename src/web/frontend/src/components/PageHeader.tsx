interface Props {
  title: string;
  children?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
}

export default function PageHeader({ title, children, subtitle, action }: Props) {
  return (
    <div className="mb-10 fade-up">
      <div className="flex items-end justify-between">
        <div>
          <h1
            className="text-[32px] font-bold tracking-tight leading-none page-title-glow"
            style={{
              color: 'var(--text-luminous)',
              letterSpacing: '-0.03em',
            }}
          >
            {title}
          </h1>
          {(children || subtitle) && (
            <div className="text-sm mt-3 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{children ?? subtitle}</div>
          )}
        </div>
        {action && <div>{action}</div>}
      </div>
      <div className="divider-fade mt-8" />
    </div>
  );
}
