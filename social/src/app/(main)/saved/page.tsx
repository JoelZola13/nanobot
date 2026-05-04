import { redirect } from "next/navigation";
import TopBar from "@/components/layout/TopBar";
import SavedItemsView from "@/components/saved/SavedItemsView";
import { auth } from "@/lib/session";
import { getSavedItemsForUser } from "@/lib/savedItems";

export default async function SavedPage() {
  const session = await auth();
  if (!session?.user?.id) redirect("/login");

  const savedItems = await getSavedItemsForUser(session.user.id);

  return (
    <>
      <TopBar
        title="Later"
        type="saved"
        description="Messages you saved for follow-up"
      />
      <SavedItemsView initialSavedItems={savedItems} />
    </>
  );
}
