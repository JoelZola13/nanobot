/**
 * Returns the base path for client-side API calls.
 * Next.js basePath ("/social") is not automatically prepended to fetch() URLs,
 * only to <Link> and router.push(). All client-side fetch("/api/...") calls
 * must use this prefix so requests go through nginx to the Social app
 * rather than to LibreChat or Paperclip.
 */
export const API_BASE = "/social";

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}
