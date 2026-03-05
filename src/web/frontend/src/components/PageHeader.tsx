interface Props {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}

export default function PageHeader({ title, children, action }: Props) {
  return (
    <div className="mb-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#e2e8f0]">{title}</h1>
          <p className="text-sm text-[#64748b] mt-1">{children}</p>
        </div>
        {action && <div>{action}</div>}
      </div>
    </div>
  );
}
