# Street Voices Messages

## Health Check

After the local Docker stack is running, use one command to check the Messages dependencies:

```bash
npm run health
```

The check verifies the LibreChat API health route, the Social auth bridge, NextAuth providers, Social page routes, `social-postgres`, and the `/ws-social` Socket.IO proxy.

Useful overrides:

```bash
SOCIAL_HEALTH_BASE_URL=http://localhost:3180 npm run health
SOCIAL_POSTGRES_CONTAINER=nanobot-social-postgres npm run health
SOCIAL_HEALTH_BRIDGE_SECRET=street-voices-social-bridge-2026 npm run health
```

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

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
