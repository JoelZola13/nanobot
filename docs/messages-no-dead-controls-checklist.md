# Messages No-Dead-Controls Checklist

Use this checklist before merging work that changes the `/messages` wrapper, any `/social/*` Messages route, the Messages sidebar, conversation header, composer, Activity inbox, or any flyout panel.

## Gate

A visible control passes only when it does at least one of these things:

- Navigates to the expected route and preserves `embed=true` inside the Messages shell.
- Opens or closes a visible panel, modal, picker, menu, browser prompt, or file picker.
- Changes visible UI state, such as inserting text, toggling a picker, changing a filter, or changing a notification label.
- Performs a request and gives success, loading, disabled, or error feedback.
- Is disabled with an obvious disabled state when the action is unavailable.
- Is not rendered when the backing action is not implemented for that context.

No visible button, icon button, link, row action, panel action, or composer action should be inert.

## Automated Guard

The authenticated Messages smoke suite includes a rendered control hygiene pass. It checks the embedded `/messages` workspace and quick switcher for visible buttons without accessible names and links that still point to empty, hash-only, or `javascript:` targets.

```bash
cd LibreChat
LIBRECHAT_BASE_URL=http://localhost:3180 npm run e2e:messages-ci:local
```

The local lane seeds an authenticated smoke user, verifies `/messages`, and should finish with all 36 Messages smoke tests passing and 0 skips. This guard does not replace the manual checklist below; it catches the easy regressions quickly, while the manual pass still confirms that each visible action changes state, navigates, sends a request, or is explicitly disabled.

## Setup

1. Start the normal local stack.
2. Sign in once through LibreChat at `http://localhost:3180`.
3. Open `http://localhost:3180/messages`.
4. Confirm the LibreChat sidebar remains on the far left and the Messages sidebar is directly beside it.
5. Open `http://localhost:3180/social/dm?embed=true` for direct Social shell checks.
6. Keep the browser console open or collect console errors after the pass.

## Shell And Sidebar

| Control                         | Expected result                                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Open messages sidebar           | Opens the Messages sidebar on narrow widths.                                                                            |
| Close messages sidebar/backdrop | Closes the Messages sidebar on narrow widths.                                                                           |
| Workspace switcher button       | Opens the quick switcher in jump mode, including narrow embedded widths where the sidebar is translated.                |
| Pencil quick switcher button    | Opens the quick switcher in new-message/compose mode, including narrow embedded widths where the sidebar is translated. |
| Jump to channel, DM, or agent   | Opens the quick switcher in jump mode, keeps it visible above the shell, and accepts search input.                      |
| Direct messages                 | Navigates to `/social/dm?embed=true`.                                                                                   |
| Activity                        | Navigates to `/social/activity?embed=true`.                                                                             |
| Mentions                        | Navigates to `/social/mentions?embed=true`.                                                                             |
| Later                           | Navigates to `/social/saved?embed=true`.                                                                                |
| Channel browser                 | Navigates to `/social/channels?embed=true`.                                                                             |
| Add channel                     | Opens channel creation at `/social/channels?create=true&embed=true`.                                                    |
| New message                     | Toggles the people/agent search box with an accessible expanded state.                                                  |
| New message search result       | Starts or opens a DM, shows loading/error feedback, and clears the search UI on success.                                |
| Channel rows                    | Navigate to the selected channel and update active state.                                                               |
| DM rows                         | Navigate to the selected DM and update active state.                                                                    |
| Desktop alerts prompt           | Enable shows waiting plus permission feedback; dismiss hides the prompt.                                                |

## DM Directory

| Control                   | Expected result                                                                  |
| ------------------------- | -------------------------------------------------------------------------------- |
| Search people and agents  | Filters teammate/agent rows, shows empty state, and exposes clear search.        |
| Teammate/agent profile    | Opens profile popover with loading, retryable error, and profile link states.    |
| Message teammate or agent | Shows opening feedback, navigates with `embed=true`, or surfaces API error text. |
| Dismiss DM error          | Clears the visible row-level DM start error without changing directory results.  |

## Conversation Header

| Control                       | Expected result                                                                          |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| Channel title/details trigger | Opens the conversation details panel.                                                    |
| DM title/profile trigger      | Opens the profile popover.                                                               |
| Members count                 | Opens the members panel.                                                                 |
| Voice call                    | Starts a call flow or is visibly disabled when socket/call prerequisites are missing.    |
| Video call                    | Starts a call flow or is visibly disabled when socket/call prerequisites are missing.    |
| Pinned messages               | Opens the pinned messages panel.                                                         |
| Files                         | Opens the files panel.                                                                   |
| Removed messages              | Opens the moderation/audit panel for channel managers only.                              |
| Search messages               | Opens the search panel.                                                                  |
| Details                       | Opens the details panel.                                                                 |
| Notifications                 | Opens notification preferences and reflects preference changes in the header icon state. |

## Composer

| Control                   | Expected result                                                                                       |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| Text area                 | Accepts text, slash commands, and line breaks.                                                        |
| Send                      | Disabled when empty; sends non-empty content; sent message appears after send or clear error appears. |
| Send failure              | Shows the Social API error text, keeps the draft in place, and allows retry.                          |
| Mention someone           | Opens a searchable teammate/agent picker; selecting a result inserts `@username` at the cursor.       |
| Mention search failure    | Shows the Social API error text, keeps the picker open, and retries the same search.                  |
| Add emoji                 | Opens the emoji picker; selecting an emoji inserts it and closes the picker.                          |
| Attach file               | Opens the file picker when uploads are available; otherwise the button should not render.             |
| Attach file failure       | Shows the Social API error text, keeps the failed file available, and retries without re-picking it.  |
| Record voice message      | Opens recorder controls when voice send is available; otherwise the button should not render.         |
| Recorder stop/send/cancel | Each visible recorder action changes recorder state or sends/cancels the draft.                       |
| Voice send failure        | Shows the Social API error text, keeps the recorded audio, and retries without recording again.       |
| Voice playback            | Play/pause, seek, playback speed, retry, and transcript controls update visible state or error text.  |

## Message Rows

| Control                  | Expected result                                                  |
| ------------------------ | ---------------------------------------------------------------- |
| Reaction button          | Opens emoji reactions and selected reaction appears on the row.  |
| Reply/thread button      | Opens the thread panel.                                          |
| Pin/unpin                | Shows loading feedback, changes pinned state, or shows an error. |
| Copy permalink           | Copies the link or shows the blocked-copy recovery affordance.   |
| More actions             | Opens a menu or is not rendered.                                 |
| Removed-message controls | Only render for allowed users and show success/error feedback.   |
| Save for later           | Shows loading feedback, changes saved state, or shows an error.  |

## Panels And Drawers

Every panel must have a working close control and a non-blank loading, empty, success, or error state.

| Panel                      | Required checks                                                                                                        |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Quick switcher             | Close works; search narrows results; Enter/click navigates or starts a DM.                                             |
| Search                     | Close works; query, channel, sender, date, and attachment filters trigger results, empty state, or retryable error.    |
| Pinned messages            | Close works; loading, empty, retryable load-error, and unpin success/error states appear.                              |
| Files                      | Close works; loading, empty, retryable load-error, filters, file links, and context links work.                        |
| Members                    | Close works; loading, retryable load/search errors, add-member errors, and role/remove row errors recover cleanly.     |
| Details                    | Close works; summary load errors retry; topic save errors recover; shortcuts open members, pins, files, notifications. |
| Removed messages           | Close works; loading, empty, retryable load-error, and removed-message audit rows appear for managers only.            |
| Notifications              | Close works; load errors retry; save errors recover; All activity, Mentions, and Muted update visible state.           |
| Activity row notifications | Opens the same notification preferences panel and updates the row label after saving.                                  |
| Thread                     | Close works; replies show loading, empty, retryable error, and sent-reply states.                                      |
| Thread reply send          | Shows the Social API error text, keeps the reply draft in place, and allows retry.                                     |
| Voice player               | Playback errors show a retry control; seek and transcript toggles are keyboard reachable.                              |
| Profile popover            | Close works; profile fetch failures show API error text and a retry control.                                           |

## Activity Inbox

| Control                                      | Expected result                                                                        |
| -------------------------------------------- | -------------------------------------------------------------------------------------- |
| All/Mentions/Later/Threads/Reactions filters | Switch selected state, expose count/unread labels, and update the list or empty state. |
| Date section headers                         | Activity remains grouped by Today, Yesterday, or absolute date after filtering.        |
| Jump links                                   | Navigate to the target message and preserve embedded Messages context.                 |
| Notification shortcut                        | Opens preferences for that row's conversation and updates the row label.               |

## Mentions And Later

| Control                  | Expected result                                                                  |
| ------------------------ | -------------------------------------------------------------------------------- |
| Mention jump links       | Navigate to the target message and preserve `embed=true`.                        |
| Later channel links      | Navigate to the source conversation and preserve `embed=true`.                   |
| Later jump links         | Navigate to the saved message and preserve `embed=true`.                         |
| Remove saved item        | Shows removing feedback, removes the row on success, or surfaces API error text. |
| Dismiss saved item error | Clears the visible row-level removal error without removing the saved item.      |
| Relative timestamps      | Hydrate cleanly without render-time clock drift between server and browser.      |

## Channel Browser

| Control               | Expected result                                                                  |
| --------------------- | -------------------------------------------------------------------------------- |
| Retry channel loading | Re-runs the channel list request and keeps the API error visible until recovery. |
| Search channels       | Filters the channel rows and exposes a clear-search action when populated.       |
| Show archived         | Reloads the list with archived rows and updates pressed state.                   |
| Join/leave channel    | Shows busy feedback and moves the row between joined and browse sections.        |
| Edit channel          | Opens the edit form, saves changes, or shows the API error text.                 |
| Archive/restore       | Shows confirmation/busy feedback and updates the archived row state.             |

## Responsive And Theme Pass

1. Test at desktop width and narrow mobile width.
2. Confirm text does not overlap inside buttons, filter pills, sidebars, headers, or composer controls.
3. Toggle LibreChat light/dark mode and confirm the Messages sidebar, panels, cards, and controls keep readable contrast.
4. Confirm mobile sidebar controls do not cover the composer or conversation header after close.

## Regression Record

Use this template in PR notes or handoff comments:

```text
Messages no-dead-controls pass:
- Date:
- Build/container:
- Account:
- Browser/viewport:
- Routes checked:
- Controls checked:
- Known disabled controls and why:
- Console errors:
- Network/API failures:
- Screenshots or notes:
```

The pass is not complete if any visible control has no observable outcome.
