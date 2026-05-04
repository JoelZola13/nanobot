# Messages Production Deployment

Use this checklist when promoting Street Voices Messages outside the local Docker stack. Messages is not a standalone auth island: it runs as the Social service behind the LibreChat domain and reuses the main LibreChat session through the internal auth bridge.

## Runtime Shape

Traffic should stay on one public origin:

```text
Browser
  -> nginx /messages      -> LibreChat shell with embedded Social iframe
  -> nginx /social/*      -> Social Next.js app on sv-social:3182
  -> nginx /ws-social     -> Social Socket.IO server on sv-social:3183
  -> nginx /api/auth/*    -> LibreChat API auth bridge
```

Production should run these services together:

- `api` or `librechat`: the main LibreChat API and shell.
- `sv-social`: the Social Next.js app and Socket.IO server.
- `social-postgres`: the dedicated Social database.
- `casdoor`: OAuth issuer shared by LibreChat and Social.
- `nginx`: the single public entrypoint.

## Required Environment

Set these before building and running `sv-social`. Several values are read by Next.js during `next build`, so changing them after image build is not enough; rebuild the image when they change.

| Variable | Service | Purpose |
| --- | --- | --- |
| `NEXTAUTH_URL` | `sv-social` | Must be the public Social auth callback URL, for example `https://chat.example.com/social/api/auth`. |
| `AUTH_SECRET` | `sv-social` | NextAuth JWT/session secret. Use a long random value. |
| `AUTH_CASDOOR_ISSUER` | `sv-social` | Public or internal Casdoor issuer URL reachable from Social. |
| `AUTH_CASDOOR_ID` | `sv-social` | Casdoor application client id for Social. |
| `AUTH_CASDOOR_SECRET` | `sv-social` | Casdoor application client secret for Social. |
| `DATABASE_URL` | `sv-social` | Must point at `social-postgres`, not LobeHub/Postiz/LibreChat Postgres. |
| `LIBRECHAT_AUTH_BRIDGE_URL` | `sv-social` | Internal URL for LibreChat session verification, usually `http://api:3080/api/auth/social-session` or the compose service equivalent. |
| `LIBRECHAT_AUTH_BRIDGE_SECRET` | `api`, `sv-social` | Shared secret accepted by LibreChat and sent by Social. Generate once with `openssl rand -hex 32`. |
| `NANOBOT_API_URL` | `sv-social` | Internal Nanobot API URL used by agent DMs. |
| `NANOBOT_API_KEY` | `sv-social` | Nanobot API key. |
| `PORT` | `sv-social` | Next/custom server HTTP port. Current stack uses `3182`. |
| `WS_PORT` | `sv-social` | Socket.IO port. Current stack uses `3183`. |

## Build Notes

`social/Dockerfile` builds a production Next.js bundle. Because `social/next.config.mjs` exposes environment values through `env`, build the image in the same secret context that will run it.

For local compose, the current command is:

```bash
cd LibreChat
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build sv-social
```

For production images, pass the production values as build-time environment or build args in the deployment platform. Do not promote an image built with localhost auth URLs to a public host.

## Database And Migrations

Messages must use its own database:

```text
postgresql://social:<password>@social-postgres:5432/social
```

Before opening traffic after a schema change:

```bash
cd social
DATABASE_URL=postgresql://social:<password>@social-postgres:5432/social npx prisma migrate deploy
```

Keep `social-postgres` backups separate from LibreChat MongoDB and any other app Postgres volumes. Messages stores users, channels, memberships, messages, reactions, saved items, files metadata, notifications, and gallery/profile data in this database.

## Nginx Routing

Keep `/social` and `/ws-social` separate.

The Social app route:

```nginx
location /social {
    proxy_pass http://social;
    proxy_set_header Host $host:$server_port;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host:$server_port;
    proxy_set_header Accept-Encoding "";
}
```

The Socket.IO route:

```nginx
location /ws-social {
    proxy_pass http://social-ws;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_read_timeout 86400s;
}
```

Do not route `/ws-social` through the Next.js `/social` app. Socket.IO needs the separate upstream so websocket and polling handshakes do not get intercepted by App Router.

## Release Checklist

1. Confirm `LIBRECHAT_AUTH_BRIDGE_SECRET` is identical in LibreChat and Social.
2. Confirm `DATABASE_URL` points at `social-postgres`.
3. Build the Social image with production auth URLs and secrets.
4. Run `prisma migrate deploy` against `social-postgres`.
5. Start or roll `sv-social`.
6. Start or reload nginx with `/social` and `/ws-social` routes.
7. Run the health check against the public origin:

```bash
cd social
SOCIAL_HEALTH_BASE_URL=https://chat.example.com \
SOCIAL_POSTGRES_CONTAINER=<production-social-postgres-container> \
SOCIAL_HEALTH_BRIDGE_SECRET=<shared-bridge-secret> \
npm run health
```

8. Sign in once through LibreChat, open `/messages`, and confirm there is no `/social/login` or `/api/auth/error` hop.
9. Open a DM or channel and confirm sending, reactions, thread panel, and presence updates.

## Troubleshooting

| Symptom | Likely cause | Check |
| --- | --- | --- |
| `/messages` loads but iframe redirects to `/social/login` | Social cannot verify the LibreChat session. | Check `LIBRECHAT_AUTH_BRIDGE_URL`, shared secret, cookie forwarding, and `NEXTAUTH_URL`. |
| `/social/bridge-unavailable` appears | Bridge request failed or LibreChat rejected the bridge secret. | Run `npm run health`; inspect Social logs for `[AUTH] LibreChat bridge`. |
| Messages load but no live updates | `/ws-social` is not proxied to `social-ws` or Upgrade headers are missing. | Fetch `/ws-social/?EIO=4&transport=polling`; it should return a Socket.IO open packet starting with `0`. |
| DMs/channels work locally but not after deploy | Image was built with localhost values. | Rebuild `sv-social` with production `NEXTAUTH_URL`, Casdoor issuer, and database URL. |
| Teammates see missing channels or 409 user errors | Social user identity was not upserted from the LibreChat bridge. | Confirm the LibreChat bridge returns `openidId`, `idOnTheSource`, `id`, or email for the signed-in user. |
