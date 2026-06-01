"use client";

import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { withEmbedParam } from "@/lib/embeddedNavigation";

export function useEmbeddedNavigation() {
  const searchParams = useSearchParams();
  const [isFramed, setIsFramed] = useState(false);
  const shouldPreserveEmbed =
    searchParams.get("embed") === "true" || isFramed;

  useEffect(() => {
    try {
      setIsFramed(window.self !== window.top);
    } catch {
      setIsFramed(true);
    }
  }, []);

  const withEmbed = useCallback(
    (href: string) => withEmbedParam(href, shouldPreserveEmbed),
    [shouldPreserveEmbed],
  );

  return { shouldPreserveEmbed, withEmbed };
}
