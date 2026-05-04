import { redirect } from "next/navigation";
import TopBar from "@/components/layout/TopBar";
import MentionsView from "@/components/mentions/MentionsView";
import { auth } from "@/lib/session";
import { getMentionMessagesForUser } from "@/lib/mentions";

export default async function MentionsPage() {
  const session = await auth();
  if (!session?.user?.id) redirect("/login");

  const { username, mentions } = await getMentionMessagesForUser(session.user.id);

  return (
    <>
      <TopBar
        title="Mentions"
        type="mentions"
        description={username ? `Messages that mention @${username}` : undefined}
      />
      <MentionsView username={username} mentions={mentions} />
    </>
  );
}
