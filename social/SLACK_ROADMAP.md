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
- [x] Add telemetry or structured logs for auth bridge failures and socket disconnects.

## Phase 6: Persistence And Polish

- [x] Hydrate sidebar unread counts from persisted read receipts after reloads.
- [x] Add draft persistence per channel and DM.
- [x] Add browser notifications with a quiet permission prompt.
- [x] Add channel creation and edit controls for workspace admins.

## Phase 7: Workspace Administration

- [x] Add member management for channel admins.
- [x] Add channel archive and restore controls.
- [x] Add workspace-level defaults for notification and channel policies.
- [x] Add lightweight moderation controls for message removal auditability.

## Phase 8: Conversation History

- [x] Open channels and DMs on the latest message window.
- [x] Add a load-older control for long conversations.
- [x] Add jump-to-latest behavior after reading older history.
- [x] Add unread dividers inside long conversations.

## Phase 9: Conversation Details

- [x] Wire the header details button to a Slack-style conversation details panel.
- [x] Add pinned/file/member counts to the details panel.
- [x] Add channel topic editing from the details panel for channel managers.
- [x] Add a shared media preview strip for DMs and channels.

## Phase 10: Permalinks And Polish

- [x] Keep the LibreChat Messages wrapper synced with channel/message permalink query changes.
- [x] Add a visible copied-link recovery affordance when clipboard access is blocked.
- [x] Add permalink smoke coverage for highlighted messages inside the embedded Social app.
- [x] Add a compact jump-to-mentioned-message treatment for mentions and saved items.

## Phase 11: Activity Inbox

- [x] Add a Slack-style Activity page that combines mentions and saved follow-ups.
- [x] Add activity filters for mentions, saved items, threads, and reactions.
- [x] Add unread activity badges backed by persisted activity state.
- [x] Add notification preference shortcuts from Activity rows.

## Phase 12: Slack Polish And QA

- [x] Show current notification levels on Activity row shortcuts and header icons.
- [x] Add date-grouped Activity sections for faster scanning.
- [x] Add a no-dead-controls regression checklist for the Messages shell.
- [x] Add teammate setup diagnostics for missing Social services.

## Phase 13: Automated Regression Guardrails

- [x] Add always-runnable Playwright coverage for the teammate setup diagnostics endpoint.
- [x] Add signed-in Playwright coverage for the `/messages` setup diagnostics failure card and retry button.
- [x] Add mobile Playwright coverage that keeps the Messages quick switcher visible outside the translated sidebar.
- [x] Wire the authenticated Messages smoke storage-state bootstrap into CI.

## Phase 14: Auth Smoke Operations

- [x] Add a one-command local capture flow for authenticated Messages Playwright storage state.
- [x] Print a GitHub-secret-ready base64 storage-state value from the capture flow.

## Phase 15: Fail-Closed Smoke CI

- [x] Add a required CI command that fails when authenticated Messages storage state is missing.
- [x] Make the GitHub Messages smoke workflow use the required authenticated smoke command.

## Phase 16: Auth State Preflight

- [x] Add a headless verifier for captured Messages storage state before the full smoke spec runs.
- [x] Make auth capture wait for the embedded Messages workspace sidebar, not just the wrapper iframe.

## Phase 17: Auth State Host Diagnostics

- [x] Add a fast storage-state host/domain preflight for wrong-target auth secrets.
- [x] Keep auth verification logs redacted to domains and origins only.

## Phase 18: Auth State Expiry Diagnostics

- [x] Fail fast when every matching Messages storage-state cookie is expired.
- [x] Add a configurable minimum persistent-cookie TTL check for stricter CI freshness.

## Phase 19: Control Hygiene Automation

- [x] Add authenticated smoke coverage for visible Messages controls with missing accessible labels or placeholder links.
- [x] Replace obvious profile/call/audio dead controls with real actions, disabled states, or accessible labels.

## Phase 20: Channel Creation Reliability

- [x] Verify the live Messages sidebar can open the channel creation form and create a channel.
- [x] Add authenticated smoke coverage for the Add channel button and create-channel flow.
- [x] Surface API error details in the channel create/edit/join/leave/archive UI instead of a generic failure.

## Phase 21: Header Panel Reliability

- [x] Add authenticated smoke coverage for conversation header search, pins, files, notifications, and details panels.
- [x] Add explicit button types and accessible labels to header controls.
- [x] Preserve embedded navigation when opening file context links from the Files panel.
- [x] Surface API error details in member, notification, pinned-message, and topic panel actions.

## Phase 22: Composer Control Reliability

- [x] Replace the mention button's raw `@` insertion with a searchable teammate/agent mention picker.
- [x] Add composer error states for failed sends, file uploads, and voice sends.
- [x] Add explicit button types and accessible labels to voice recorder controls.
- [x] Add authenticated smoke coverage for emoji, mention, slash-command, and file-upload composer controls.

## Phase 23: Sidebar DM Control Reliability

- [x] Add visible search and start-DM error feedback to the sidebar People add flow.
- [x] Add opening/disabled feedback while a sidebar or quick-switcher DM request is in flight.
- [x] Add explicit labels to the sidebar compose, people add, quick-switcher input, and quick-switcher close controls.
- [x] Add authenticated smoke coverage for sidebar and quick-switcher DM request feedback.

## Phase 24: Message Row Action Reliability

- [x] Add loading and error feedback for message row save, reaction, pin, edit, and delete actions.
- [x] Make message row action handlers surface Social API error text instead of failing silently.
- [x] Keep row action controls keyboard-visible with focus-within styling and explicit button types.
- [x] Add authenticated smoke coverage for message row save, reaction, and pin API failures.

## Phase 25: Thread Panel Reliability

- [x] Add explicit loading, empty, and retryable error states for thread reply loading.
- [x] Surface reply-send API errors through the composer instead of a generic failure.
- [x] Add complementary landmark labeling for the thread drawer.
- [x] Add authenticated smoke coverage for thread reply-load failure and retry recovery.

## Phase 26: Voice Playback Reliability

- [x] Add visible loading, playback-error, and retry states to voice messages.
- [x] Replace the mouse-only waveform seek target with a keyboard-accessible seek slider.
- [x] Keep speed and transcript controls stateful, labelled, and disabled when playback is unavailable.
- [x] Add authenticated smoke coverage for voice player seek, speed, and transcript controls.

## Phase 27: Profile Popover Reliability

- [x] Add explicit profile-card labels, close controls, and dialog wiring to profile popovers.
- [x] Surface profile API error text with a retry action instead of a static unavailable message.
- [x] Surface direct-message API error details from profile actions.
- [x] Add authenticated smoke coverage for profile popover load failure and retry recovery.

## Phase 28: Search Panel Reliability

- [x] Add explicit loading and retryable API-error states for message search results.
- [x] Add retryable filter-load errors for channel and sender filter options.
- [x] Label search input, result rows, and clear-filter controls for keyboard/screen-reader use.
- [x] Add authenticated smoke coverage for search filter failure, result failure, and retry recovery.

## Phase 29: Pinned Messages Panel Reliability

- [x] Add explicit loading, empty, and retryable load-error states for pinned messages.
- [x] Surface unpin API errors at the affected pinned row with dismissible feedback.
- [x] Keep unpin controls keyboard-visible with pending/disabled feedback.
- [x] Add authenticated smoke coverage for pinned-message load failure, retry, unpin failure, and unpin recovery.

## Phase 30: Files Panel Reliability

- [x] Add explicit loading, empty, and retryable load-error states for shared files.
- [x] Surface Social API error text when files cannot load.
- [x] Label file filters, open links, and context links for keyboard/screen-reader use.
- [x] Add authenticated smoke coverage for files load failure, retry, filters, and embedded context links.

## Phase 31: Members Panel Reliability

- [x] Add explicit loading, empty, and retryable load-error states for channel members.
- [x] Surface teammate search API errors with retry instead of silently clearing results.
- [x] Keep add-member, role-update, and remove-member failures visible at the affected control or row.
- [x] Add authenticated smoke coverage for member load, search, add, role, and remove failure recovery.

## Phase 32: Details Panel Reliability

- [x] Add explicit retryable errors for pinned/file summary loading in conversation details.
- [x] Keep topic edit/save/cancel labelled and surface topic-save API errors inline.
- [x] Label shared-media previews and details shortcut rows for keyboard/screen-reader use.
- [x] Add authenticated smoke coverage for details summary retry, topic-save retry, and shortcut panel routing.

## Phase 33: Notification Preferences Reliability

- [x] Add explicit loading, retryable load-error, and inline save-error states for notification preferences.
- [x] Label notification preference options and expose the panel as a dialog.
- [x] Keep Activity row notification shortcuts wired to the same preference flow and visible row label updates.
- [x] Add authenticated smoke coverage for notification load retry, save retry, and Activity shortcut updates.

## Phase 34: Removed Messages Reliability

- [x] Add explicit loading, empty, retryable load-error, and dialog states for removed-message audit panels.
- [x] Surface Social API error text when the manager-only audit cannot load.
- [x] Label removed-message counts and audit rows for smoke coverage.
- [x] Add authenticated smoke coverage for removed-message audit retry and recovered audit rows.

## Phase 35: Browser Notification Prompt Reliability

- [x] Add explicit permission-denied, unsupported-browser, cancelled, and request-failure feedback for desktop alerts.
- [x] Keep the desktop alerts prompt labelled, dismissible, and visibly pending while permission is requested.
- [x] Add authenticated smoke coverage for the desktop alerts prompt using a stubbed browser Notification API.

## Phase 36: Activity Filter Reliability

- [x] Add stronger tab labels with visible count/unread state for Activity filters.
- [x] Expose Activity results and empty states as a labelled tab panel.
- [x] Add authenticated smoke coverage for All, Later, and Mentions filter switching with a real saved message.

## Phase 37: Channel Browser Reliability

- [x] Add retryable channel-browser load errors with Social API error text.
- [x] Label and wire search clear, archived toggle, row actions, and channel forms for reliable control checks.
- [x] Add authenticated smoke coverage for channel search, archived rows, join, leave, edit, archive, and restore.
- [x] Keep channel join/leave updates client-side so embedded membership controls do not bounce through the auth bridge.

## Phase 38: DM Directory Reliability

- [x] Add clearable people/agent search with empty-state feedback.
- [x] Surface direct-message start API failures on the affected directory row.
- [x] Preserve embedded navigation and add authenticated smoke coverage for DM directory error recovery.

## Phase 39: Later And Mention Link Reliability

- [x] Preserve embedded navigation for Later and Mentions source/jump links.
- [x] Surface saved-item removal failures on the affected Later row with dismissible feedback.
- [x] Move Later and Mentions relative timestamps behind client-side hydration-safe rendering.
- [x] Add authenticated smoke coverage for saved-item jump links, remove failure, dismiss, and retry success.

## Phase 40: Composer Send Failure Reliability

- [x] Surface Social API error text for failed top-level message sends instead of a generic composer error.
- [x] Keep failed top-level and thread reply drafts in place with visible retry feedback.
- [x] Add authenticated smoke coverage for top-level send failure, thread reply send failure, and retry success.

## Phase 41: Composer Mention Search Reliability

- [x] Surface Social API error text when teammate/agent mention search fails.
- [x] Keep the mention picker open and retry the same search without losing draft text.
- [x] Add authenticated smoke coverage for mention search failure, retry, and successful insertion.

## Phase 42: Composer Attachment Retry Reliability

- [x] Surface Social API error text when an attachment upload or file-message send fails.
- [x] Keep the failed file available and expose an inline retry action without reopening the file picker.
- [x] Add authenticated smoke coverage for attachment upload failure, retained-file retry, and success cleanup.

## Phase 43: Composer Voice Retry Reliability

- [x] Surface Social API error text when a voice upload or voice-message send fails.
- [x] Keep the recorded audio available and expose an inline retry action without forcing a re-record.
- [x] Add authenticated smoke coverage for voice upload failure, retained-audio retry, and success cleanup.

## Phase 44: Local Auth Smoke Reliability

- [x] Add a local-only Messages auth seed command that creates a dedicated LibreChat smoke user/session.
- [x] Add a one-command local signed-in smoke lane that seeds auth, verifies `/messages`, and runs the full smoke spec.
- [x] Persist rotated LibreChat refresh-token cookies during verification and between smoke-test contexts.
- [x] Keep CI on the stricter secret-backed required smoke command while documenting the local seeded path.
- [x] Guard embedded mobile/sidebar controls until client interactivity is ready, avoiding dead clicks during iframe auth settle.
- [x] Keep the framed Messages mobile sidebar toggle clear of LibreChat's own mobile nav toggle.
- [x] Make permalink, members-panel, and DM call-button smoke coverage create deterministic fixtures instead of skipping.
- [x] Convert remaining seeded-data smoke assumptions into hard failures so missing conversations or DM targets cannot pass silently.
- [x] Confirm the seeded local lane passes end-to-end: 36 passed, 0 skipped.

## Phase 45: Call Control Reliability

- [x] Surface microphone/camera permission failures inline when voice or video call startup fails.
- [x] Give the active call overlay a dialog label and stable smoke-test hook.
- [x] Add authenticated smoke coverage that clicks voice and video call controls, verifies overlay controls, and ends the calls.

## Phase 46: Message Row Secondary Action Reliability

- [x] Label message edit and permalink fallback controls for reliable keyboard and smoke-test access.
- [x] Add authenticated smoke coverage for blocked clipboard fallback and manual permalink selection.
- [x] Add authenticated smoke coverage for edit and delete failure recovery followed by successful retry.

## Phase 47: Activity Shortcut State Reliability

- [x] Keep Activity notification shortcut labels updated immediately after preference saves, even during background Activity refreshes.
- [x] Reconfirm authenticated smoke coverage for Activity row notification shortcut label updates.

## Phase 48: Imported Email Reply Reliability

- [x] Label imported-email reply controls for reliable keyboard and smoke-test access.
- [x] Add authenticated smoke coverage for empty email reply validation.
- [x] Add authenticated smoke coverage for email reply send failure, retained draft retry, and success feedback.

## Phase 49: Conversation History Control Reliability

- [x] Surface retryable errors when loading older conversation history fails.
- [x] Label load-older and jump-to-latest controls for reliable keyboard and smoke-test access.
- [x] Add authenticated smoke coverage for load-older failure, retry success, and jump-to-latest cleanup.

## Phase 50: Voice Transcription Reliability

- [x] Show voice transcription pending, failed, and empty-result states inline on the voice player.
- [x] Add a retry action for failed voice transcription requests.
- [x] Add authenticated smoke coverage for transcription failure, retry, and recovered transcript display.

## Phase 51: Voice Recorder Permission Reliability

- [x] Surface unsupported-browser and microphone-denied recorder failures inline.
- [x] Keep microphone retry controls labelled and recoverable after a failed recorder start.
- [x] Add authenticated smoke coverage for microphone denial, retry, ready state, and cancel cleanup.

## Phase 52: Call Screen Share Reliability

- [x] Replace the disabled screen-share placeholder with a functional video-call control.
- [x] Surface screen-share permission errors inline with a dismiss action.
- [x] Add authenticated smoke coverage for screen-share denial, retry success, stop sharing, and call cleanup.

## Phase 53: Profile Edit Reliability

- [x] Replace the disabled own-profile edit placeholder with a working profile editor.
- [x] Add authenticated profile update support for display name, bio, location, and website.
- [x] Add authenticated smoke coverage for profile edit save failure, retry success, and restored profile state.

## Phase 54: Channel Membership Retry Reliability

- [x] Show channel join and leave failures on the affected channel row instead of only as a page-level error.
- [x] Keep failed join and leave controls in their original state so teammates can retry immediately.
- [x] Add authenticated smoke coverage for join failure retry, leave failure retry, and cleared row errors.

## Phase 55: Channel Form And Archive Retry Reliability

- [x] Show channel create and edit failures inside the active form while preserving draft fields.
- [x] Show channel archive and restore failures on the affected row while preserving the current action state.
- [x] Add authenticated smoke coverage for create, edit, archive, and restore failure retries.

## Phase 56: Workspace Defaults Retry Reliability

- [x] Show an explicit workspace-defaults loading state in the channel browser.
- [x] Surface workspace-policy load failures with a retry action instead of silently hiding the defaults panel.
- [x] Add authenticated smoke coverage for workspace-defaults failure and retry recovery.

## Phase 57: Teammate Mongo Stability

- [x] Move the tracked teammate compose path to the stable local Mongo image and data directory.
- [x] Cap local Mongo memory and disable diagnostic data collection to avoid the LibreChat auth DB crash loop.
- [x] Document the safe migration path from the old Mongo 8 `data-node` folder to `data-node-mongo7`.

## Phase 58: Channel Form Default Reliability

- [x] Make the New Channel form respect workspace default channel visibility.
- [x] Reset create-form drafts and visibility state when the form is closed or cancelled.
- [x] Add authenticated smoke coverage for create-form open, close, cancel, and visibility controls.

## Phase 59: Workspace Switcher Action Reliability

- [x] Replace the workspace-name quick-switch shortcut with a real workspace action menu.
- [x] Keep workspace search, new-DM compose, channel browser, DM, Activity, and Later actions functional from the menu.
- [x] Add authenticated smoke coverage for opening the menu, launching both switcher modes, and navigating from the menu.

## Phase 60: Quick Switcher DM Retry Reliability

- [x] Give quick-switcher result rows stable labels and smoke-test hooks.
- [x] Surface quick-switcher DM-start failures through an explicit status region.
- [x] Add authenticated smoke coverage for quick-switcher DM failure followed by retry success.

## Phase 61: People Search Retry Reliability

- [x] Add explicit retry controls for failed sidebar people searches.
- [x] Add explicit retry controls for failed quick-switcher people searches.
- [x] Add authenticated smoke coverage for sidebar and quick-switcher search failure recovery.

## Phase 62: Agent Directory Filter Reliability

- [x] Make the sidebar Browse agents action open an agents-only DM directory.
- [x] Add a functional DM directory filter for switching between all contacts and AI agents.
- [x] Add authenticated smoke coverage for the Browse agents navigation and filter reset.

## Phase 63: Quick Switcher Agent Browse Reliability

- [x] Add a dedicated Browse AI agents action to the quick switcher.
- [x] Route that action into the agents-only DM directory while preserving embedded mode.
- [x] Add authenticated smoke coverage for quick-switcher agent directory navigation.

## Phase 64: Teammate Directory Filter Reliability

- [x] Add a teammates-only filter to the direct-message directory.
- [x] Add a dedicated Browse teammates action to the quick switcher.
- [x] Add authenticated smoke coverage for quick-switcher teammate directory navigation.

## Phase 65: DM Directory Empty-State Recovery

- [x] Add a clear-search action directly inside the DM directory empty state.
- [x] Reuse that recovery action across all DM directory filters.
- [x] Add authenticated smoke coverage for clearing a no-match directory search from the empty state.

## Phase 66: Message Search Empty-State Recovery

- [x] Add a clear-search action to the message search dialog header.
- [x] Add a clear-search action directly inside the no-results search empty state.
- [x] Add authenticated smoke coverage for clearing a no-match message search from the empty state.

## Phase 67: Files Panel Filter Empty-State Recovery

- [x] Add a recovery action directly inside filtered Files panel empty states.
- [x] Reset filtered empty states back to the all-files view without closing the panel.
- [x] Add authenticated smoke coverage for filtering to an empty file type and recovering to all files.

## Phase 68: Live Upload And Voice Storage Reliability

- [x] Add a project-local RustFS/S3 service for teammate Messages installs.
- [x] Split internal S3 writes from same-origin public attachment URLs.
- [x] Add a same-origin attachment streaming route with range support for audio playback.
- [x] Preserve recorded voice MIME types when uploading voice messages.
- [x] Add authenticated smoke coverage for real file upload storage and real voice attachment retrieval.

## Phase 69: Read Receipt Stale Message Reliability

- [x] Validate channel membership before reading or writing read receipts.
- [x] Ignore stale, deleted, or cross-channel read receipt message ids without crashing Prisma.
- [x] Add authenticated smoke coverage for stale read receipt posts and valid read receipt recovery.
