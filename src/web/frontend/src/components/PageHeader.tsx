interface Props {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}

export default function PageHeader({ title, children, action }: Props) {
  return (
    <div className="mb-8 fade-up">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-[28px] font-bold tracking-tight leading-none" style={{ color: 'var(--text-luminous)', letterSpacing: '-0.025em' }}>{title}</h1>
          <p className="text-[13px] mt-2.5 leading-relaxed" style={{ color: 'var(--text-muted)' }}>{children}</p>
        </div>
        {action && <div>{action}</div>}
      </div>
      <div className="divider-fade mt-7" />
    </div>
  );
}
