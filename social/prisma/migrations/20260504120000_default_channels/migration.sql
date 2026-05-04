ALTER TABLE "channels"
  ADD COLUMN "is_default" BOOLEAN NOT NULL DEFAULT false;

UPDATE "channels"
SET "is_default" = true
WHERE "slug" IN ('announcements', 'general', 'help');
