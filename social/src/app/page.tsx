import { redirect } from "next/navigation";
import { auth } from "@/lib/session";

export default async function Home() {
  const session = await auth();
  if (session?.user) {
    redirect("/dm");
  } else {
    redirect("/login");
  }
}
