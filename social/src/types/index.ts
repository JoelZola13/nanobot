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
  memberCount?: number;
  unreadCount?: number;
}

export interface MessageData {
  id: string;
  channelId: string;
  content: string;
  createdAt: string;
  isEdited: boolean;
  isPinned: boolean;
  parentId: string | null;
  replyCount?: number;
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
    type?: "voice";
    duration?: number;
    transcription?: string;
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
