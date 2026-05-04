import { create } from "zustand";

interface UnreadState {
  counts: Map<string, number>;
  activeChannelId: string | null;
  hydrate: (counts: Record<string, number>) => void;
  increment: (channelId: string) => void;
  clear: (channelId: string) => void;
  setActive: (channelId: string | null) => void;
  getCount: (channelId: string) => number;
  totalUnread: () => number;
}

export const useUnreadStore = create<UnreadState>((set, get) => ({
  counts: new Map(),
  activeChannelId: null,
  hydrate: (counts) =>
    set((state) => {
      const next = new Map<string, number>();
      for (const [channelId, count] of Object.entries(counts)) {
        if (count > 0 && channelId !== state.activeChannelId) {
          next.set(channelId, count);
        }
      }
      return { counts: next };
    }),
  increment: (channelId) =>
    set((state) => {
      // Don't increment for the active channel
      if (state.activeChannelId === channelId) return state;
      const next = new Map(state.counts);
      next.set(channelId, (next.get(channelId) || 0) + 1);
      return { counts: next };
    }),
  clear: (channelId) =>
    set((state) => {
      const next = new Map(state.counts);
      next.delete(channelId);
      return { counts: next };
    }),
  setActive: (channelId) =>
    set((state) => {
      if (channelId) {
        const next = new Map(state.counts);
        next.delete(channelId);
        return { activeChannelId: channelId, counts: next };
      }
      return { activeChannelId: channelId };
    }),
  getCount: (channelId) => get().counts.get(channelId) || 0,
  totalUnread: () => {
    let total = 0;
    get().counts.forEach((count) => { total += count; });
    return total;
  },
}));
