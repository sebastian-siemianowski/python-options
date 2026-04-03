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
          <h1 className="text-[28px] font-bold text-[#f1f5f9] tracking-tight leading-none">{title}</h1>
          <p className="text-[13px] text-[#64748b] mt-2 leading-relaxed">{children}</p>
        </div>
        {action && <div>{action}</div>}
      </div>
      <div className="divider-fade mt-6" />
    </div>
  );
}
