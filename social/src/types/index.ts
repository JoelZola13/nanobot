export interface UserProfile {
  id: string;
  username: string;
  displayName: string;
  email: string;
  avatarUrl: string | null;
  bannerUrl: string | null;
  bio: string | null;
  location: string | null;
  website: string | null;
  isAgent: boolean;
  agentModel: string | null;
  status: string;
}

export interface ChannelInfo {
  id: string;
  name: string | null;
  slug: string | null;
  description: string | null;
  type: "PUBLIC" | "PRIVATE" | "DM" | "GROUP_DM";
  iconEmoji: string | null;
  isDefault?: boolean;
  memberCount?: number;
  messageCount?: number;
  role?: string;
  unreadCount?: number;
  canCreate?: boolean;
  canManage?: boolean;
}

export interface MessageData {
  id: string;
  channelId: string;
  content: string;
  createdAt: string;
  isEdited: boolean;
  isPinned: boolean;
  isSaved?: boolean;
  parentId: string | null;
  replyCount?: number;
  threadPreview?: {
    participants: {
      id: string;
      displayName: string;
      avatarUrl: string | null;
      isAgent: boolean;
    }[];
    latestReply: {
      id: string;
      content: string;
      createdAt: string;
      author: {
        id: string;
        displayName: string;
        avatarUrl: string | null;
        isAgent: boolean;
      };
    };
  };
  author: {
    id: string;
    username: string;
    displayName: string;
    avatarUrl: string | null;
    isAgent: boolean;
  };
  reactions: {
    emoji: string;
    count: number;
    userReacted: boolean;
  }[];
  attachments: {
    id: string;
    fileName: string;
    mimeType: string;
    url: string;
    width: number | null;
    height: number | null;
  }[];
  metadata?: {
    type?: "voice" | "email_import" | "email_reply";
    duration?: number;
    transcription?: string;
    transcriptionStatus?: "pending" | "complete" | "failed";
    transcriptionError?: string;
    email?: {
      provider?: string;
      subject: string;
      from?: {
        name?: string;
        email?: string;
      };
      to?: {
        name?: string;
        email?: string;
      }[];
      cc?: {
        name?: string;
        email?: string;
      }[];
      sentAt?: string;
      messageId?: string;
      sourceUrl?: string;
      capturedAt?: string;
      bodyPreview?: string;
      bodyHtml?: string;
      bodyTruncated?: boolean;
      htmlTruncated?: boolean;
      attachments?: {
        name?: string;
        url?: string;
        mimeType?: string;
        sizeLabel?: string;
      }[];
    };
    emailReply?: {
      sourceMessageId: string;
      provider?: string;
      sourceUrl?: string;
      to: {
        name?: string;
        email: string;
      };
      subject: string;
      sentAt: string;
      inReplyTo?: string;
      messageId?: string;
    };
    deletionAudit?: {
      actorId: string;
      actorName: string;
      mode: "author" | "moderator";
      reason: string | null;
      deletedAt: string;
    };
  };
}

export interface FeedPostData {
  id: string;
  content: string;
  createdAt: string;
  author: {
    id: string;
    username: string;
    displayName: string;
    avatarUrl: string | null;
  };
  likeCount: number;
  commentCount: number;
  userLiked: boolean;
  media: {
    url: string;
    mimeType: string;
    width: number | null;
    height: number | null;
  }[];
}
