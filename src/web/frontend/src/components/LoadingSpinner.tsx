import { Loader2 } from 'lucide-react';

export default function LoadingSpinner({ text = 'Loading...' }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-3">
      <Loader2 className="w-8 h-8 text-[#42A5F5] animate-spin" />
      <p className="text-sm text-[#64748b]">{text}</p>
    </div>
  );
}
