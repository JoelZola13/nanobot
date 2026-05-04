type SocialLogLevel = "info" | "warn" | "error";

const SERVICE_NAME = "street-voices-social";

function compactFields(fields: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(fields).filter(([, value]) => value !== undefined),
  );
}

export function socialLog(
  level: SocialLogLevel,
  event: string,
  fields: Record<string, unknown> = {},
) {
  const line = JSON.stringify({
    timestamp: new Date().toISOString(),
    level,
    service: SERVICE_NAME,
    event,
    ...compactFields(fields),
  });

  if (level === "error") {
    console.error(line);
    return;
  }

  if (level === "warn") {
    console.warn(line);
    return;
  }

  console.log(line);
}
