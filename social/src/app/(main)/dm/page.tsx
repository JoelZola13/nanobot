import { redirect } from "next/navigation";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import TopBar from "@/components/layout/TopBar";
import PeopleList from "@/components/dm/PeopleList";

export default async function DMPage({
  searchParams,
}: {
  searchParams?: { filter?: string };
}) {
  const session = await auth();
  if (!session?.user) redirect("/login");
  const initialFilter =
    searchParams?.filter === "agents" || searchParams?.filter === "teammates"
      ? searchParams.filter
      : "all";

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
      lastSeenAt: true,
      createdAt: true,
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
      lastSeenAt: true,
      createdAt: true,
    },
    orderBy: { displayName: "asc" },
  });

  const serializePerson = (person: (typeof people)[number]) => ({
    ...person,
    lastSeenAt: person.lastSeenAt?.toISOString() ?? null,
    createdAt: person.createdAt.toISOString(),
  });

  return (
    <>
      <TopBar
        title="Messages"
        type="dm"
        description="Send direct messages to people and AI agents"
      />
      <PeopleList
        people={people.map(serializePerson)}
        agents={agents.map(serializePerson)}
        currentUserId={session.user.id}
        initialFilter={initialFilter}
      />
    </>
  );
}
