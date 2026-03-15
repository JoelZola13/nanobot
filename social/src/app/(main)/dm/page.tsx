import { redirect } from "next/navigation";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import TopBar from "@/components/layout/TopBar";
import PeopleList from "@/components/dm/PeopleList";

export default async function DMPage() {
  const session = await auth();
  if (!session?.user) redirect("/login");

  // Fetch all users — people first, then agents
  const people = await prisma.user.findMany({
    where: {
      isAgent: false,
      id: { not: session.user.id },
    },
    select: {
      id: true,
      username: true,
      displayName: true,
      avatarUrl: true,
      bio: true,
      status: true,
      location: true,
      isAgent: true,
    },
    orderBy: [{ status: "asc" }, { displayName: "asc" }],
  });

  const agents = await prisma.user.findMany({
    where: { isAgent: true },
    select: {
      id: true,
      username: true,
      displayName: true,
      avatarUrl: true,
      bio: true,
      status: true,
      location: true,
      isAgent: true,
    },
    orderBy: { displayName: "asc" },
  });

  return (
    <>
      <TopBar title="Messages" type="dm" description="Send direct messages to people and AI agents" />
      <PeopleList people={people} agents={agents} currentUserId={session.user.id} />
    </>
  );
}
