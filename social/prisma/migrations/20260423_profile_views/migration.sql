-- Add a counter for Street Profile page views.
-- Incremented by POST /street-profiles/{username}/view each time someone
-- other than the owner loads the page.

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS profile_views INTEGER NOT NULL DEFAULT 0;
