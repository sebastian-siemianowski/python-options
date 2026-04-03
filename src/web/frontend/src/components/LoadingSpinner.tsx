import { Loader2 } from 'lucide-react';

export default function LoadingSpinner({ text = 'Loading...' }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4 fade-up">
      <div className="relative">
        <div className="w-12 h-12 rounded-full bg-[#42A5F5]/[0.06]" />
        <Loader2 className="w-6 h-6 text-[#42A5F5] animate-spin absolute inset-0 m-auto" />
      </div>
      <p className="text-[13px] text-[#475569] font-medium">{text}</p>
    </div>
  );
}
