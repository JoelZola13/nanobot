# Street Voices Messages

## Health Check

After the local Docker stack is running, use one command to check the Messages dependencies:

```bash
npm run health
```

The check verifies the LibreChat API health route, the Social auth bridge, NextAuth providers, Social page routes, Social setup diagnostics, `social-postgres`, and the `/ws-social` Socket.IO proxy.
The same teammate-safe setup signal is available at `/social/api/setup/diagnostics` for the LibreChat `/messages` wrapper, so missing `sv-social`, missing Social tables, or bridge-secret mismatches become visible in the UI.

Useful overrides:

```bash
SOCIAL_HEALTH_BASE_URL=http://localhost:3180 npm run health
SOCIAL_POSTGRES_CONTAINER=nanobot-social-postgres npm run health
SOCIAL_HEALTH_BRIDGE_SECRET=street-voices-social-bridge-2026 npm run health
```

## API Tests

Run the route-level Messages API coverage without touching live workspace data:

```bash
npm run test:api
```

The test runner covers DM creation, channel membership join/leave behavior, and the LibreChat session bridge fallback.

## UI Regression Checks

Use the [Messages No-Dead-Controls Checklist](../docs/messages-no-dead-controls-checklist.md) before merging shell, sidebar, composer, panel, or Activity changes. Any visible control must navigate, open/close UI, update visible state, submit with feedback, be clearly disabled, or not render.

## Production Deployment

Use the production checklist in [Messages Production Deployment](../docs/messages-production-deployment.md) before promoting Social outside the local stack. The important pieces are:

- `sv-social` runs the Social Next.js app on `3182` and Socket.IO on `3183`.
- `social-postgres` is the only production database for Messages data.
- nginx must route `/social` to Social HTTP and `/ws-social` to Social Socket.IO.
- LibreChat and Social must share `LIBRECHAT_AUTH_BRIDGE_SECRET`.

## Getting Started

This is a [Next.js](https://nextjs.org) app. For local development, run:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!
