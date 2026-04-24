# Street Voices Listmonk — local setup

Reproducible local email-marketing stack. Spins up Listmonk + Postgres with Street Voices branding, templates, lists, and custom CSS/JS pre-provisioned.

## One-time setup (~3 minutes)

**1. Copy env template and fill it in:**

```
cp .env.example .env
```

Open `.env` in any editor. The admin user/password are pre-filled; you only need to add a **SendGrid API key** if you want email sending to work locally. Get one at https://app.sendgrid.com/settings/api_keys (Restricted Access → Mail Send: Full Access).

**2. Start the containers:**

```
docker compose up -d
```

This starts:
- `listmonk_app` — Listmonk web UI + API on port `${LISTMONK_PORT:-9001}`
- `listmonk_db` — Postgres 17 on `127.0.0.1:${LISTMONK_DB_PORT:-15432}`

On first boot the admin user is auto-created from your `.env` values.

**3. Seed branded content (templates, lists, custom CSS/JS, images):**

```
node seed.js
```

The seeder is **idempotent** — safe to re-run. It will:
- Wait for Listmonk to be reachable
- Log in as the admin user
- Upload logo + social icons to `/uploads`
- Create the list taxonomy (Newsletter, Members, Volunteers, Donors, Partners & Press, Internal/Team) — skips ones that already exist
- Create the "Street Voices base" campaign template + "Welcome" transactional template
- Apply admin + public custom CSS/JS (the yellow branding, 2×2 dashboard, AI FAB, etc.)
- Configure SMTP with your SendGrid key (if present)

**4. Log in:**

- URL: http://localhost:9001/admin
- Username: `admin`
- Password: `$treetvoices26` (or whatever you set in `.env`)

## Updating content later

Edit the source files, then re-seed:

- Templates: `seed/templates/*.html`
- Admin CSS/JS: `seed/styles/admin.css`, `seed/styles/admin.js`
- Public CSS/JS: `seed/styles/public.css`, `seed/styles/public.js`
- Images: `uploads/*.png`

Then:

```
# Force overwrite of existing template bodies
SEED_FORCE_UPDATE=1 node seed.js
```

By default the seeder **skips** templates that already exist. Set `SEED_FORCE_UPDATE=1` in the environment (or `.env`) to overwrite.

## Start fresh

Wipe all local data (subscribers, campaigns, everything) and start over:

```
docker compose down -v
docker compose up -d
node seed.js
```

## File layout

```
listmonk/
├── docker-compose.yml      # listmonk + postgres
├── .env.example            # env template (copy to .env)
├── .gitignore              # ignores .env, node_modules, backups/, logs
├── seed.js                 # idempotent seeder (run after docker compose up)
├── uploads/                # logo + social icons (copied into Listmonk on seed)
│   ├── Street-voices-logo.png
│   ├── thumb_Street-voices-logo.png
│   ├── facebook.png / twitter.png / instagram.png
└── seed/
    ├── templates/
    │   ├── campaign-base.html  # main newsletter wrapper
    │   └── welcome.html        # transactional welcome email
    └── styles/
        ├── admin.css           # Listmonk admin theming (yellow, Rubik, pill buttons)
        ├── admin.js            # Logo replacement, 2x2 dashboard, AI FAB, blue-killer
        ├── public.css          # Subscription form theming
        └── public.js           # Makes Name required, rebrands footer
```

## Troubleshooting

**Admin login fails on first boot:** make sure `.env` is complete before running `docker compose up -d`. If you already started without one, either wipe and restart (`docker compose down -v`) or rotate the admin password via the UI.

**Images don't show in emails sent to real inboxes:** the `SV_ROOT_URL` in `.env` has to be publicly reachable (not `localhost`). For local-only testing, install `cloudflared` and run a Quick Tunnel; update `SV_ROOT_URL` and re-run `node seed.js`.

**Port conflict on 9001 or 15432:** change `LISTMONK_PORT` / `LISTMONK_DB_PORT` in `.env` and re-up.
