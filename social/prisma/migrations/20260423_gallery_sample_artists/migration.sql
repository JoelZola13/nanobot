-- Street Profile rows for the sample artists referenced by ga-001..ga-015.
-- Makes their clickable names in the gallery resolve to a real profile page.
-- Idempotent: ON CONFLICT DO NOTHING; re-running is safe.

INSERT INTO users (id, casdoor_id, username, display_name, email, profile_complete, created_at, updated_at) VALUES
  ('sample-maria-reyes',     'sample-maria-reyes',     'maria_reyes',     'Maria Reyes',     'maria_reyes@sample.streetvoices',     true, NOW(), NOW()),
  ('sample-jamal-carter',    'sample-jamal-carter',    'jamal_carter',    'Jamal Carter',    'jamal_carter@sample.streetvoices',    true, NOW(), NOW()),
  ('sample-yuki-tanaka',     'sample-yuki-tanaka',     'yuki_tanaka',     'Yuki Tanaka',     'yuki_tanaka@sample.streetvoices',     true, NOW(), NOW()),
  ('sample-elena-dubois',    'sample-elena-dubois',    'elena_dubois',    'Elena Dubois',    'elena_dubois@sample.streetvoices',    true, NOW(), NOW()),
  ('sample-marcus-webb',     'sample-marcus-webb',     'marcus_webb',     'Marcus Webb',     'marcus_webb@sample.streetvoices',     true, NOW(), NOW()),
  ('sample-priya-sharma',    'sample-priya-sharma',    'priya_sharma',    'Priya Sharma',    'priya_sharma@sample.streetvoices',    true, NOW(), NOW()),
  ('sample-diego-morales',   'sample-diego-morales',   'diego_morales',   'Diego Morales',   'diego_morales@sample.streetvoices',   true, NOW(), NOW()),
  ('sample-aisha-okonkwo',   'sample-aisha-okonkwo',   'aisha_okonkwo',   'Aisha Okonkwo',   'aisha_okonkwo@sample.streetvoices',   true, NOW(), NOW()),
  ('sample-tommy-chen',      'sample-tommy-chen',      'tommy_chen',      'Tommy Chen',      'tommy_chen@sample.streetvoices',      true, NOW(), NOW()),
  ('sample-sofia-restrepo',  'sample-sofia-restrepo',  'sofia_restrepo',  'Sofia Restrepo',  'sofia_restrepo@sample.streetvoices',  true, NOW(), NOW()),
  ('sample-alex-rivera',     'sample-alex-rivera',     'alex_rivera',     'Alex Rivera',     'alex_rivera@sample.streetvoices',     true, NOW(), NOW()),
  ('sample-nina-petrova',    'sample-nina-petrova',    'nina_petrova',    'Nina Petrova',    'nina_petrova@sample.streetvoices',    true, NOW(), NOW()),
  ('sample-kwame-asante',    'sample-kwame-asante',    'kwame_asante',    'Kwame Asante',    'kwame_asante@sample.streetvoices',    true, NOW(), NOW()),
  ('sample-liam-osullivan',  'sample-liam-osullivan',  'liam_osullivan',  'Liam OSullivan',  'liam_osullivan@sample.streetvoices',  true, NOW(), NOW()),
  ('sample-rosa-gutierrez',  'sample-rosa-gutierrez',  'rosa_gutierrez',  'Rosa Gutierrez',  'rosa_gutierrez@sample.streetvoices',  true, NOW(), NOW())
ON CONFLICT (id) DO NOTHING;

-- Link the sample artworks to the new profile rows so the InlineProfileBadge
-- (which needs artist_id + matching users row) renders automatically.
UPDATE gallery_artworks SET artist_id = 'sample-maria-reyes'     WHERE id = 'ga-001' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-jamal-carter'    WHERE id = 'ga-002' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-yuki-tanaka'     WHERE id = 'ga-003' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-elena-dubois'    WHERE id = 'ga-004' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-marcus-webb'     WHERE id = 'ga-005' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-priya-sharma'    WHERE id = 'ga-006' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-diego-morales'   WHERE id = 'ga-007' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-aisha-okonkwo'   WHERE id = 'ga-008' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-tommy-chen'      WHERE id = 'ga-009' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-sofia-restrepo'  WHERE id = 'ga-010' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-alex-rivera'     WHERE id = 'ga-011' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-nina-petrova'    WHERE id = 'ga-012' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-kwame-asante'    WHERE id = 'ga-013' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-liam-osullivan'  WHERE id = 'ga-014' AND artist_id IS NULL;
UPDATE gallery_artworks SET artist_id = 'sample-rosa-gutierrez'  WHERE id = 'ga-015' AND artist_id IS NULL;
