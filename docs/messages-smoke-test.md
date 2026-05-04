# Messages Smoke Test

Use this when validating the teammate install path for `LibChatMain + NanobotMain`.

## Goal

Confirm that a teammate can sign in once with LibreChat OAuth and access Messages without a second Social/LobeHub login.

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

## Automated Smoke

Create a Playwright storage state after signing in manually:

```bash
cd LibreChat
LIBRECHAT_BASE_URL=http://localhost:3180 npx playwright codegen \
  --save-storage=e2e/messages-auth.json \
  http://localhost:3180/login
```

In the opened browser, complete LibreChat OAuth manually, then close codegen.

Run the smoke test:

```bash
cd LibreChat
LIBRECHAT_BASE_URL=http://localhost:3180 \
MESSAGES_STORAGE_STATE=e2e/messages-auth.json \
npm run e2e:messages-smoke
```

The test checks:

- `/messages` stays inside the LibreChat shell.
- The standalone cloned sidebar is absent.
- The embedded Social Messages sidebar renders.
- Direct Social `/social/dm?embed=true` accepts both `theme=light` and `theme=dark`.
