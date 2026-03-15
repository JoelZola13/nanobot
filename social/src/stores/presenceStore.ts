import { create } from "zustand";

interface PresenceState {
  statuses: Map<string, "online" | "away" | "offline">;
  setStatus: (userId: string, status: "online" | "away" | "offline") => void;
  setAll: (entries: { userId: string; status: string }[]) => void;
  getStatus: (userId: string) => "online" | "away" | "offline";
}

export const usePresenceStore = create<PresenceState>((set, get) => ({
  statuses: new Map(),
  setStatus: (userId, status) =>
    set((state) => {
      const next = new Map(state.statuses);
      if (status === "offline") next.delete(userId);
      else next.set(userId, status);
      return { statuses: next };
    }),
  setAll: (entries) =>
    set(() => {
      const next = new Map<string, "online" | "away" | "offline">();
      for (const e of entries) {
        if (e.status === "online" || e.status === "away") {
          next.set(e.userId, e.status as "online" | "away");
        }
      }
      return { statuses: next };
    }),
  getStatus: (userId) => get().statuses.get(userId) || "offline",
}));
