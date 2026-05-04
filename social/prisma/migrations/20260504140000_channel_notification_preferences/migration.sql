ALTER TABLE "channel_members"
  ADD COLUMN "notification_level" TEXT NOT NULL DEFAULT 'ALL';

UPDATE "channel_members"
SET "notification_level" = 'MUTED'
WHERE "muted_at" IS NOT NULL;
