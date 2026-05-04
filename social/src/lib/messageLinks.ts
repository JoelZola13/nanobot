type ConversationChannelType = "PUBLIC" | "PRIVATE" | "DM" | "GROUP_DM";

type MessageHrefInput = {
  channelId: string;
  messageId: string;
  channelType: ConversationChannelType;
};

export function getConversationPath(channelType: ConversationChannelType, channelId: string) {
  return `/${channelType === "DM" ? "dm" : "channels"}/${channelId}`;
}

export function getMessageHref({ channelId, messageId, channelType }: MessageHrefInput) {
  const params = new URLSearchParams({ message: messageId });
  return `${getConversationPath(channelType, channelId)}?${params.toString()}`;
}

export function getJumpToMessageLabel(kind: "mention" | "saved") {
  return kind === "mention" ? "Jump to mention" : "Jump to saved message";
}
