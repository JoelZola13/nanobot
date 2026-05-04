export default function BridgeUnavailablePage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-bg px-6 py-10 text-text-primary">
      <section className="w-full max-w-lg rounded-xl border border-border bg-bg-surface p-6 shadow-xl">
        <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-lg border border-border bg-bg-elevated text-accent">
          <span className="font-heading text-lg font-bold">SV</span>
        </div>
        <h1 className="font-heading text-2xl font-semibold">
          Messages cannot reach LibreChat sign-in
        </h1>
        <p className="mt-3 text-sm leading-6 text-text-muted">
          Your Messages workspace is running, but it could not verify your
          existing LibreChat session. This usually means the LibreChat API,
          bridge secret, or local Docker network needs attention.
        </p>

        <div className="mt-5 rounded-lg border border-border bg-bg-elevated px-4 py-3 text-sm text-text-secondary">
          Run <code className="font-mono text-accent">npm run health</code> in
          <code className="font-mono text-accent"> social</code> to check auth,
          database, socket, and Social routes.
        </div>

        <div className="mt-6 flex flex-wrap gap-3">
          <a
            href="/messages"
            className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
          >
            Try again
          </a>
          <a
            href="/social/login"
            className="rounded-md border border-border px-4 py-2 text-sm font-semibold text-text-secondary hover:border-accent hover:text-accent"
          >
            Open Social sign-in
          </a>
        </div>
      </section>
    </main>
  );
}
