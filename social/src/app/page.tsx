import { redirect } from "next/navigation";
import { auth, isLibreChatBridgeUnavailableError } from "@/lib/session";

export default async function Home() {
  let session;
  try {
    session = await auth({ bridgeUnavailable: "throw" });
  } catch (error) {
    if (isLibreChatBridgeUnavailableError(error)) {
      redirect("/bridge-unavailable");
    }
    throw error;
  }

  if (session?.user) {
    redirect("/dm");
  } else {
    redirect("/login");
  }
}
