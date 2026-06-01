CREATE TABLE "activity_read_states" (
  "id" TEXT NOT NULL,
  "user_id" TEXT NOT NULL,
  "read_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL,

  CONSTRAINT "activity_read_states_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "activity_read_states_user_id_key"
  ON "activity_read_states"("user_id");

ALTER TABLE "activity_read_states"
  ADD CONSTRAINT "activity_read_states_user_id_fkey"
  FOREIGN KEY ("user_id") REFERENCES "users"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
