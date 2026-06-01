# Nanobot Messages and Documents Extension

This is an unpacked Chrome/Edge extension MVP for sending the currently open
Gmail or Outlook web email into Nanobot Messages, plus saving web pages into
Nanobot Documents.

## Install locally

1. Open `chrome://extensions` or `edge://extensions`.
2. Enable Developer mode.
3. Choose **Load unpacked**.
4. Select this folder: `extensions/email-to-messages`.

## Configure

### Messages

1. Sign in to Messages in the same browser.
2. Open the extension popup.
3. Set the Messages URL, for example `http://localhost:3180/social`.
4. Search for a channel or direct-message user.
5. Select the destination.

Once configured, open a Gmail or Outlook web email and click the inline
**Choose destination** button injected into the email itself. The button opens a
destination picker so you can choose a channel or direct-message target for
that specific email before sending. The popup still has a fallback
**Send to Selected Destination** button.

### Documents

1. Set the Documents URL to `http://localhost:3180/documents`.
2. Leave User ID blank to auto-detect the signed-in Documents session in this
   browser. Enter a User ID only when you intentionally want to override that.
3. Open any normal web page and choose **Save Page to Documents**.

The extension captures the selected text if text is highlighted; otherwise it
captures the readable page body, page metadata, article outline, JSON-LD
structured data, links, image references, and HTML tables. Tables are saved as
editable Tiptap tables when the Documents editor supports them. It creates a
standard editable Documents item through
`POST /sbapi/api/documents?user_id=...` and opens
`http://localhost:3180/documents?documentId=...` after saving when enabled.
The created document ID is also remembered in the Documents origin local
storage so it appears in the Documents list for the same user.

## Current scope

- Injects a native-feeling action button into open Gmail and Outlook messages.
- Captures the visible email subject, sender, date, body text, body HTML,
  visible attachment metadata, and source URL.
- Sends the email through `POST /social/api/email-import`.
- Stores the original email details in message metadata.
- Imported emails show a **Reply by email** action inside Messages. Replies are
  sent through Nanobot's configured SMTP account and recorded in the message
  thread.
- Clips web pages on demand with the popup using Chrome's active-tab permission.
- Saves page clips to Nanobot Documents as editable documents tagged
  `web-clip`.
- Uses the existing Messages browser session; if the API returns `401`, sign in
  to Messages and retry.

Attachment import is intentionally not included in this DOM-based MVP. Pulling
real attachment bytes from Gmail or Outlook requires provider OAuth/API work
rather than only reading the currently visible web page.

For production packaging, narrow `host_permissions` in `manifest.json` to the
actual Messages host instead of the broad development-friendly `https://*/*`.
