# Street Voices — Reviewer Access

Credentials and URLs for checking out the running Street Voices stack.

## Email marketing admin (Listmonk)

- **URL:** https://concerns-ireland-nitrogen-sims.trycloudflare.com/admin
- **Username:** `admin`
- **Password:** `$treetvoices26`

## Public subscription form

- https://concerns-ireland-nitrogen-sims.trycloudflare.com/subscription/form

## ⚠ URL rotation notice

The public URL above is served through a **Cloudflare Quick Tunnel**, which rotates every time the host machine reboots or the tunnel process restarts. If the link returns `ERR_CONNECTION_REFUSED`, ask the maintainer for the current URL.

## Other services (local-only, not publicly reachable)

Running on the maintainer's machine only. Included for reference.

| Service | URL | Notes |
|---|---|---|
| LibreChat / Street Voices platform | http://localhost:3180/home | The AI chat + community platform |
| AI Email Composer | http://localhost:3001 | Generates branded email drafts with Groq; saves as drafts into Listmonk |
| Listmonk local | http://localhost:9001/admin | Same admin above, unproxied |
