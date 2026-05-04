const MESSAGE_DRAFT_PREFIX = "street-voices:message-draft:";
const MAX_DRAFT_LENGTH = 12000;

type MessageDraftStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export function getMessageDraftStorageKey(draftId: string) {
  return `${MESSAGE_DRAFT_PREFIX}${encodeURIComponent(draftId)}`;
}

export function getBrowserMessageDraftStorage(): MessageDraftStorage | null {
  if (typeof window === "undefined") return null;

  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function readMessageDraft(
  draftId: string,
  storage: MessageDraftStorage | null,
) {
  if (!draftId || !storage) return "";

  try {
    return storage.getItem(getMessageDraftStorageKey(draftId)) || "";
  } catch {
    return "";
  }
}

export function writeMessageDraft(
  draftId: string,
  content: string,
  storage: MessageDraftStorage | null,
) {
  if (!draftId || !storage) return;

  try {
    const draft = content.length > MAX_DRAFT_LENGTH
      ? content.slice(0, MAX_DRAFT_LENGTH)
      : content;

    if (!draft.trim()) {
      storage.removeItem(getMessageDraftStorageKey(draftId));
      return;
    }

    storage.setItem(getMessageDraftStorageKey(draftId), draft);
  } catch {
    // Draft persistence should never block composing a message.
  }
}

export function clearMessageDraft(
  draftId: string,
  storage: MessageDraftStorage | null,
) {
  if (!draftId || !storage) return;

  try {
    storage.removeItem(getMessageDraftStorageKey(draftId));
  } catch {
    // best-effort
  }
}
