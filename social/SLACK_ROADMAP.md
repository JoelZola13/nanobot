# Messages Slack-Style Roadmap

## Goal

Make Social Messages feel like a lightweight Slack workspace for the team: one sign-in, clear team spaces, fast direct messages, readable conversations, and predictable admin/setup behavior when teammates clone LibChatMain and NanobotMain.

## Phase 0: Access Foundation

- [x] Use the main LibreChat/LibaChat session to enter Messages without a second login.
- [x] Document teammate setup so Messages is included in the normal local stack.
- [x] Remove hard-coded host assumptions that break inside Docker.
- [x] Route production Social data to `social-postgres` instead of the LobeHub database.
- [x] Resolve Casdoor/LibaChat identities to local Social user ids on every session read.
- [x] Add a one-command health check for auth, database, socket, and Social routes.
- [x] Add a friendly error screen when the Social session bridge is unavailable.

## Phase 1: Slack-Like Shell

- [x] Replace the generic sidebar with a workspace-style left navigation.
- [x] Surface channels, direct messages, agents, unreads, and presence in one scan-friendly rail.
- [x] Add a compact compose/search entry point for starting DMs.
- [x] Let the Social app own its Slack-style sidebar on `/social` routes instead of the injected platform sidebar.
- [x] Add a real quick switcher for channels, DMs, and agents.
- [x] Add keyboard shortcuts for search, compose, and channel switching.
- [x] Add responsive behavior for narrow browser widths.

## Phase 2: Conversation Experience

- [x] Group messages by sender and date.
- [x] Keep hover actions for reactions, replies, edits, pins, and more actions.
- [x] Use a Slack-style header with channel/DM identity and action buttons.
- [x] Make the composer read naturally for channels and DMs.
- [x] Add richer empty states per channel and DM.
- [x] Add threaded reply previews beside parent messages.
- [x] Add message permalink/copy-link actions.
- [x] Add slash command scaffolding.

## Phase 3: Team Directory And Onboarding

- [x] Turn the DM landing page into a denser people directory.
- [x] Separate online teammates, offline teammates, and AI agents.
- [x] Add invite/onboarding copy for newly added teammates.
- [x] Add admin-managed default channels.
- [x] Add profile popovers from DMs and mentions.

## Phase 4: Slack Workflows

- [x] Mentions view.
- [x] Saved items / Later view.
- [x] Channel browser with join/leave and private channel indicators.
- [x] Notification preferences per channel and DM.
- [x] File browser for shared attachments.
- [x] Search filters by sender, channel, date, and attachment type.

## Phase 5: Reliability

- [x] Fix websocket proxy routing so `/ws-social` reaches the Social Socket.IO server.
- [x] Add the internal socket broadcast bridge used by message API routes.
- [x] Add Playwright coverage for sign-in, DM start, message send, reaction, thread, and call buttons.
- [x] Add API tests for DM creation, channel membership, and Social session bridge.
- [x] Add production deployment notes for the Social service and websocket routing.
- [ ] Add telemetry or structured logs for auth bridge failures and socket disconnects.
