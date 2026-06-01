import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

const PROFILE_LIMITS = {
  displayName: 80,
  bio: 280,
  location: 80,
  website: 200,
};

type ProfileResponseUser = {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
  bio: string | null;
  location: string | null;
  website: string | null;
  status: string;
  isAgent: boolean;
  createdAt: Date;
  _count: {
    channelMembers: number;
    feedPosts: number;
  };
};

function profileResponse(user: ProfileResponseUser) {
  return {
    id: user.id,
    username: user.username,
    displayName: user.displayName,
    avatarUrl: user.avatarUrl,
    bio: user.bio,
    location: user.location,
    website: user.website,
    status: user.status,
    isAgent: user.isAgent,
    createdAt: user.createdAt.toISOString(),
    channelCount: user._count.channelMembers,
    postCount: user._count.feedPosts,
  };
}

function optionalText(value: unknown, field: keyof typeof PROFILE_LIMITS) {
  if (value === undefined) return undefined;
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.length > PROFILE_LIMITS[field]) {
    throw new Error(
      `${field} must be ${PROFILE_LIMITS[field]} characters or less.`,
    );
  }
  return trimmed;
}

function normalizeWebsite(value: unknown) {
  const text = optionalText(value, "website");
  if (!text) return text;

  const withProtocol = /^https?:\/\//i.test(text) ? text : `https://${text}`;
  const url = new URL(withProtocol);
  if (!["http:", "https:"].includes(url.protocol)) {
    throw new Error("Website must use http or https.");
  }
  return url.toString();
}

// GET /api/users/profile?userId=id or ?username=name — fetch a compact profile card
export async function GET(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const userId = req.nextUrl.searchParams.get("userId")?.trim();
  const username = req.nextUrl.searchParams.get("username")?.trim();

  if (!userId && !username) {
    return NextResponse.json(
      { error: "userId or username is required" },
      { status: 400 },
    );
  }

  const user = await prisma.user.findFirst({
    where: userId ? { id: userId } : { username },
    select: {
      id: true,
      username: true,
      displayName: true,
      avatarUrl: true,
      bio: true,
      location: true,
      website: true,
      status: true,
      isAgent: true,
      createdAt: true,
      _count: {
        select: {
          channelMembers: true,
          feedPosts: true,
        },
      },
    },
  });

  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  return NextResponse.json(profileResponse(user));
}

// PATCH /api/users/profile — update the signed-in user's profile
export async function PATCH(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = (await req.json().catch(() => null)) as Record<
    string,
    unknown
  > | null;
  if (!body) {
    return NextResponse.json(
      { error: "Invalid profile payload" },
      { status: 400 },
    );
  }

  try {
    const displayName = optionalText(body.displayName, "displayName");
    if (displayName !== undefined && !displayName) {
      return NextResponse.json(
        { error: "Display name is required" },
        { status: 400 },
      );
    }

    const bio = optionalText(body.bio, "bio");
    const location = optionalText(body.location, "location");
    const website = normalizeWebsite(body.website);

    const user = await prisma.user.update({
      where: { id: session.user.id },
      data: {
        ...(displayName !== undefined ? { displayName } : {}),
        ...(bio !== undefined ? { bio } : {}),
        ...(location !== undefined ? { location } : {}),
        ...(website !== undefined ? { website } : {}),
      },
      select: {
        id: true,
        username: true,
        displayName: true,
        avatarUrl: true,
        bio: true,
        location: true,
        website: true,
        status: true,
        isAgent: true,
        createdAt: true,
        _count: {
          select: {
            channelMembers: true,
            feedPosts: true,
          },
        },
      },
    });

    return NextResponse.json(profileResponse(user));
  } catch (error) {
    if (error instanceof Error) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }
    return NextResponse.json(
      { error: "Profile could not be updated" },
      { status: 500 },
    );
  }
}
