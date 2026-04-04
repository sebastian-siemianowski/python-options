interface Props {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}

export default function PageHeader({ title, children, action }: Props) {
  return (
    <div className="mb-10 fade-up">
      <div className="flex items-end justify-between">
        <div>
          <h1
            className="text-[32px] font-bold tracking-tight leading-none"
            style={{
              color: 'var(--text-luminous)',
              letterSpacing: '-0.03em',
              textShadow: '0 0 80px var(--violet-15)',
            }}
          >
            {title}
          </h1>
          <p className="text-sm mt-3 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{children}</p>
        </div>
        {action && <div>{action}</div>}
      </div>
      <div className="divider-fade mt-8" />
    </div>
  );
}
