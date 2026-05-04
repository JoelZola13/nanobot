CREATE TABLE "saved_items" (
  "id" TEXT NOT NULL,
  "user_id" TEXT NOT NULL,
  "message_id" TEXT NOT NULL,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

  CONSTRAINT "saved_items_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "saved_items_user_id_message_id_key"
  ON "saved_items"("user_id", "message_id");

CREATE INDEX "saved_items_user_id_created_at_idx"
  ON "saved_items"("user_id", "created_at");

ALTER TABLE "saved_items"
  ADD CONSTRAINT "saved_items_user_id_fkey"
  FOREIGN KEY ("user_id") REFERENCES "users"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "saved_items"
  ADD CONSTRAINT "saved_items_message_id_fkey"
  FOREIGN KEY ("message_id") REFERENCES "messages"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
