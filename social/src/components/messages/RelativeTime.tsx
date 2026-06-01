"use client";

import { useEffect, useState } from "react";
import { formatDistanceToNow } from "date-fns";

export default function RelativeTime({
  value,
  addSuffix = true,
  className,
}: {
  value: string | Date;
  addSuffix?: boolean;
  className?: string;
}) {
  const isoValue =
    typeof value === "string"
      ? new Date(value).toISOString()
      : value.toISOString();
  const [label, setLabel] = useState<string | null>(null);

  useEffect(() => {
    const date = typeof value === "string" ? new Date(value) : value;
    const updateLabel = () =>
      setLabel(formatDistanceToNow(date, { addSuffix }));

    updateLabel();
    const interval = window.setInterval(updateLabel, 60_000);

    return () => window.clearInterval(interval);
  }, [addSuffix, value]);

  return (
    <time dateTime={isoValue} title={isoValue} className={className}>
      {label ?? "\u00a0"}
    </time>
  );
}
