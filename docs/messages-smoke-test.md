# Messages Smoke Test

Use this when validating the teammate install path for `LibChatMain + NanobotMain`.

## Goal

Confirm that a teammate can sign in once with LibreChat OAuth and access Messages without a second Social/LobeHub login.

## Secret Prerequisite

LibreChat and Social must share the same internal auth bridge secret:

```bash
openssl rand -hex 32
```

Set the generated value as `LIBRECHAT_AUTH_BRIDGE_SECRET` in the teammate secrets bundle. The value must be identical for LibreChat and the Social service.

## Manual Check

1. Start the local stack.
2. Open `http://localhost:3180`.
3. Sign in with LibreChat OAuth.
4. Open `http://localhost:3180/messages`.
5. Confirm the LibreChat sidebar stays on the far left.
6. Confirm the Messages workspace sidebar appears immediately to the right.
7. Confirm there is no `/api/auth/error`, `/social/login`, or second OAuth screen.
8. Toggle light/dark mode and confirm the Messages sidebar changes with LibreChat.
9. Start or open a DM, send a test message, reload, and confirm it persists.
10. Open `/social/api/setup/diagnostics` and confirm every check is `ok`; if `/messages` cannot reach Social, confirm the diagnostics card appears in the wrapper.
11. Run the [Messages No-Dead-Controls Checklist](messages-no-dead-controls-checklist.md) for any shell, sidebar, composer, panel, or Activity UI change.
12. For composer changes, include failure and retry coverage for text sends, file uploads, mention search, and voice sends.

## Local Mongo Stability

The teammate compose override pins LibreChat Mongo to `mongo:7`, caps WiredTiger cache for local machines, disables diagnostic data collection, and stores the active local database in `LibreChat/data-node-mongo7`. Fresh installs do not need any migration.

If an older local clone has a crashing `LibreChat/data-node` database from the previous Mongo 8 setup, preserve it and migrate it instead of deleting it:

```bash
cd LibreChat
mkdir -p /tmp/nanobot-mongo-migration

docker compose -p nanobot -f docker-compose.yml -f ../deploy/docker-compose.override.yml stop api mongodb
docker rm -f nanobot-mongo8-dump 2>/dev/null || true

docker run -d \
  --name nanobot-mongo8-dump \
  --network nanobot_default \
  -v "$PWD/data-node:/data/db" \
  mongo:8.0.16 \
  mongod --noauth --wiredTigerCacheSizeGB 0.5

sleep 5

docker run --rm \
  --network nanobot_default \
  -v /tmp/nanobot-mongo-migration:/dump \
  mongo:8.0.16 \
  mongodump --host nanobot-mongo8-dump --archive=/dump/librechat-mongo8.archive.gz --gzip

docker rm -f nanobot-mongo8-dump
mkdir -p data-node-mongo7

docker compose -p nanobot -f docker-compose.yml -f ../deploy/docker-compose.override.yml up -d --force-recreate mongodb

docker run --rm \
  --network nanobot_default \
  -v /tmp/nanobot-mongo-migration:/dump \
  mongo:7 \
  mongorestore --host nanobot-mongodb --archive=/dump/librechat-mongo8.archive.gz --gzip --drop --nsExclude='admin.system.version'

docker compose -p nanobot -f docker-compose.yml -f ../deploy/docker-compose.override.yml up -d --force-recreate api
```

## Automated Smoke

Create a Playwright storage state after signing in manually:

```bash
cd LibreChat
LIBRECHAT_BASE_URL=http://localhost:3180 npm run e2e:messages-auth:capture
```

In the opened browser, complete LibreChat OAuth manually. The script waits until `/messages` renders the embedded Messages iframe and the Messages workspace sidebar inside it, writes `e2e/.auth/messages-storage-state.json`, and prints a base64 value that can be saved as `MESSAGES_STORAGE_STATE_BASE64` in GitHub secrets.

Run the smoke test:

```bash
cd LibreChat
LIBRECHAT_BASE_URL=http://localhost:3180 \
npm run e2e:messages-smoke
```

To run the signed-in smoke lane locally without manual OAuth, seed a dedicated local smoke user/session:

```bash
cd LibreChat
LIBRECHAT_BASE_URL=http://localhost:3180 npm run e2e:messages-ci:local
```

That command creates or reuses `messages-smoke@streetvoices.local` in the local LibreChat Mongo database, writes the ignored `e2e/.auth/messages-storage-state.json`, verifies `/messages`, and then runs the full smoke spec. The verifier and smoke spec write refreshed cookies back to that file because LibreChat rotates refresh tokens during silent auth.

Expected local result: all 36 Messages smoke tests pass with 0 skips. The spec creates its own temporary channel/message/member fixtures for data-dependent coverage and archives temporary channels after each test.

For CI, store either the raw Playwright storage JSON in `MESSAGES_STORAGE_STATE_JSON` or a base64-encoded version in `MESSAGES_STORAGE_STATE_BASE64`. The bootstrap script writes the ignored default file at `e2e/.auth/messages-storage-state.json`, and the Messages Playwright config uses it automatically:

```bash
cd LibreChat
npm run e2e:messages-auth:bootstrap
npm run e2e:messages-smoke
```

The GitHub workflow at `LibreChat/.github/workflows/messages-smoke.yml` runs the same smoke lane when `MESSAGES_SMOKE_BASE_URL` is configured as a repository variable, or when `base_url` is provided to the manual workflow dispatch.

The workflow uses the stricter CI command, so it fails when a smoke target is configured but no authenticated storage-state secret is present:

```bash
cd LibreChat
npm run e2e:messages-ci:required
```

That strict command also checks that the storage-state cookie domains or localStorage origins match `LIBRECHAT_BASE_URL`, fails fast if every matching cookie is expired, verifies the state against `/messages`, persists the refreshed cookie state, then runs the full smoke spec. Stale, expired, or wrong-domain auth secrets fail with a focused error.

To require a fresher persistent-cookie secret in CI, set `MESSAGES_AUTH_MIN_COOKIE_TTL_MINUTES`, or pass it directly:

```bash
cd LibreChat
npm run e2e:messages-auth:verify -- --min-cookie-ttl-minutes 1440
```

Run just the setup diagnostics regression without a saved login:

```bash
cd LibreChat
LIBRECHAT_BASE_URL=http://localhost:3180 \
npx playwright test --config=e2e/playwright.messages.config.ts \
  e2e/specs/social-messages-smoke.spec.ts -g "setup diagnostics"
```

The test checks:

- The Social setup diagnostics endpoint reports every teammate dependency as `ok`.
- `/messages` stays inside the LibreChat shell.
- The standalone cloned sidebar is absent.
- The embedded Social Messages sidebar renders.
- `/messages` permalink queries open and highlight a deterministic message inside the embedded Social iframe.
- Visible Messages controls have accessible labels and links do not use placeholder targets.
- The desktop alerts prompt gives permission-denied feedback and dismisses cleanly without opening a real browser permission prompt.
- The Messages sidebar Add channel button opens the channel form, submits a real channel create request, and lands in the created channel.
- Conversation header buttons open search, pinned messages, files, notifications, and details panels.
- Search panels surface filter/result load failures and recover through retry controls.
- Pinned message panels surface load/unpin failures and recover through retryable controls.
- Files panels surface load failures and recover through retry while preserving filters/context links.
- Members panels recover from load, teammate search, add-member, role-update, and remove-member failures.
- Details panels recover from summary and topic-save failures while shortcut rows open their target panels.
- Removed-message panels recover from audit-load failures and show removed-message audit rows for channel managers.
- Notification preference panels recover from load/save failures, and Activity notification shortcuts update their row label.
- Activity filters expose selected/count labels and switch visible results between All, Later, and Mentions.
- Channel browser controls recover from load failures and exercise search, archived rows, join, leave, edit, archive, and restore.
- DM directory controls recover from DM-start failures, preserve embedded navigation, and expose clearable people/agent search.
- Saved/Later items preserve embedded jump links and recover from remove failures with row-level feedback.
- Composer and thread reply send failures surface Social API error text, keep drafts in place, and recover on retry.
- Composer mention search failures surface Social API error text and retry without closing the picker or losing draft text.
- Composer file upload failures surface Social API error text and retry the retained file without re-opening the picker.
- Composer toolbar controls insert emoji, mentions, slash commands, and uploaded file messages.
- Voice message playback controls expose play, seek, speed, and transcript state.
- Profile popovers surface profile-load failures and recover through a retry control.
- Message row actions surface API errors for save, reaction, and pin attempts instead of failing silently.
- Thread panels surface reply-load failures and recover through a retry control.
- Direct Social `/social/dm?embed=true` accepts both `theme=light` and `theme=dark`.
- The `/messages` wrapper surfaces setup guidance and a working retry button when Social is unreachable.
- The mobile Messages quick switcher stays visible when the Messages sidebar slides in.
- Existing DM conversations expose visible voice and video call controls.
