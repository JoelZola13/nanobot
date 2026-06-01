import { redirect } from "next/navigation";
import ActivityView from "@/components/activity/ActivityView";
import TopBar from "@/components/layout/TopBar";
import { getActivityForUser } from "@/lib/activity";
import { auth } from "@/lib/session";

export default async function ActivityPage() {
  const session = await auth();
  if (!session?.user?.id) redirect("/login");

  const activity = await getActivityForUser(session.user.id);

  return (
    <>
      <TopBar
        title="Activity"
        type="activity"
        description="Mentions and saved follow-ups"
      />
      <ActivityView
        username={activity.username}
        items={activity.items}
        counts={activity.counts}
        unreadCounts={activity.unreadCounts}
      />
    </>
  );
}
