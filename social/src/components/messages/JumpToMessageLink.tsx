import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

export default function JumpToMessageLink({
  href,
  label,
  channelLabel,
}: {
  href: string;
  label: string;
  channelLabel: string;
}) {
  return (
    <Link
      href={href}
      data-testid="jump-to-message-link"
      aria-label={`${label} in ${channelLabel}`}
      className="inline-flex h-7 shrink-0 items-center gap-1 rounded-full border border-border bg-bg-elevated px-2.5 text-2xs font-semibold text-white transition-colors hover:border-accent hover:bg-accent-muted hover:text-accent focus:outline-none focus:ring-2 focus:ring-accent/30"
    >
      <span>{label}</span>
      <ArrowUpRight size={12} />
    </Link>
  );
}
