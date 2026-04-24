// Street Voices — idempotent Listmonk seeder
// Run once after `docker compose up -d` (or any time to refresh content):
//   node seed.js
//
// What it does:
//   1. Waits for Listmonk to be reachable
//   2. Logs in as the admin user (created by Docker env vars)
//   3. Uploads brand logo + social icons to /listmonk/uploads
//   4. Creates/updates the list taxonomy (Newsletter, Members, Volunteers, ...)
//   5. Creates/updates the "Street Voices base" campaign template + "Welcome" tx template
//   6. Applies admin & public custom CSS/JS (branding)
//   7. Updates SMTP (SendGrid) credentials from .env
//   8. Sets site name, logo URL, root URL
//
// Idempotent: safe to re-run. Set SEED_FORCE_UPDATE=1 to overwrite existing template bodies.

const http = require('http');
const fs = require('fs');
const path = require('path');

// ----- Load .env -----
const envPath = path.join(__dirname, '.env');
if (!fs.existsSync(envPath)) {
  console.error('ERROR: .env not found. Copy .env.example to .env and fill it in.');
  process.exit(1);
}
const env = Object.fromEntries(
  fs.readFileSync(envPath, 'utf8')
    .split(/\r?\n/).filter((l) => l && !l.startsWith('#') && l.includes('='))
    .map((l) => { const i = l.indexOf('='); return [l.slice(0, i).trim(), l.slice(i + 1).trim()]; })
);

const HOST = 'localhost';
const PORT = Number(env.LISTMONK_PORT || 9001);
const ADMIN_USER = env.LISTMONK_ADMIN_USER || 'admin';
const ADMIN_PASS = env.LISTMONK_ADMIN_PASSWORD || '$treetvoices26';
const ROOT_URL = env.SV_ROOT_URL || `http://localhost:${PORT}`;
const FROM_EMAIL = env.SV_FROM_EMAIL || 'StreetVoices <communications@streetvoices.ca>';
const SENDGRID_KEY = env.SENDGRID_API_KEY || '';
const FORCE = env.SEED_FORCE_UPDATE === '1';

// ----- HTTP helpers -----
function req(method, p, { body, cookie, headers } = {}) {
  return new Promise((resolve, reject) => {
    const opts = { method, hostname: HOST, port: PORT, path: p, headers: { 'Content-Type': 'application/json', ...(headers || {}) } };
    if (cookie) opts.headers['Cookie'] = cookie;
    const r = http.request(opts, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks).toString() }));
    });
    r.on('error', reject);
    if (body) r.write(typeof body === 'string' ? body : JSON.stringify(body));
    r.end();
  });
}

function formLogin() {
  return new Promise((resolve, reject) => {
    const data = `username=${encodeURIComponent(ADMIN_USER)}&password=${encodeURIComponent(ADMIN_PASS)}`;
    const opts = { method: 'POST', hostname: HOST, port: PORT, path: '/admin/login', headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'Content-Length': Buffer.byteLength(data) } };
    const r = http.request(opts, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks).toString() }));
    });
    r.on('error', reject);
    r.write(data); r.end();
  });
}

async function login() {
  const r = await formLogin();
  if (!r.headers['set-cookie']) throw new Error(`Admin login failed (${r.status}). Check LISTMONK_ADMIN_USER/PASSWORD in .env match what Listmonk was provisioned with.`);
  return r.headers['set-cookie'].map((c) => c.split(';')[0]).join('; ');
}

async function waitForListmonk(maxTries = 60) {
  for (let i = 0; i < maxTries; i++) {
    try {
      const r = await req('GET', '/');
      if (r.status === 200) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error(`Listmonk did not become reachable at http://${HOST}:${PORT} after ${maxTries * 2}s.`);
}

// ----- Upload images via multipart -----
async function uploadMedia(cookie, filename) {
  const filePath = path.join(__dirname, 'uploads', filename);
  if (!fs.existsSync(filePath)) return null;
  const buf = fs.readFileSync(filePath);
  const boundary = '----svseed' + Date.now();
  const headers = { 'Content-Type': `multipart/form-data; boundary=${boundary}`, Cookie: cookie };
  const head = Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${filename}"\r\nContent-Type: image/png\r\n\r\n`);
  const tail = Buffer.from(`\r\n--${boundary}--\r\n`);
  const body = Buffer.concat([head, buf, tail]);

  return new Promise((resolve, reject) => {
    const r = http.request({ method: 'POST', hostname: HOST, port: PORT, path: '/api/media', headers: { ...headers, 'Content-Length': body.length } }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        try {
          const data = JSON.parse(Buffer.concat(chunks).toString());
          resolve(data?.data || null);
        } catch { resolve(null); }
      });
    });
    r.on('error', reject);
    r.write(body); r.end();
  });
}

// ----- Data fetchers -----
async function listAllMedia(cookie) {
  const r = await req('GET', '/api/media?per_page=all', { cookie });
  return (JSON.parse(r.body)?.data?.results) || [];
}
async function listAllLists(cookie) {
  const r = await req('GET', '/api/lists?per_page=all', { cookie });
  return (JSON.parse(r.body)?.data?.results) || [];
}
async function listAllTemplates(cookie) {
  const r = await req('GET', '/api/templates', { cookie });
  return JSON.parse(r.body)?.data || [];
}

// ----- Lists -----
const DEFAULT_LISTS = [
  { name: 'Newsletter', type: 'public', optin: 'single', tags: ['newsletter', 'public'], description: 'Main public newsletter — community updates, stories, and announcements from Street Voices.' },
  { name: 'Community / Members', type: 'public', optin: 'single', tags: ['members', 'public'], description: 'Active community participants — deeper updates, member-only content.' },
  { name: 'Volunteers', type: 'private', optin: 'single', tags: ['volunteers', 'internal'], description: 'Volunteer roster managed by Street Voices staff.' },
  { name: 'Donors', type: 'private', optin: 'single', tags: ['donors', 'internal'], description: 'Donor communications, impact reports, and stewardship.' },
  { name: 'Partners & Press', type: 'private', optin: 'single', tags: ['partners', 'press', 'internal'], description: 'Partner organizations, journalists, and external stakeholders.' },
  { name: 'Internal / Team', type: 'private', optin: 'single', tags: ['internal', 'team', 'test'], description: 'Staff, board, and contractors. Used for QA test sends and internal comms.' },
];

async function seedLists(cookie) {
  const existing = await listAllLists(cookie);
  const byName = Object.fromEntries(existing.map((l) => [l.name, l]));
  for (const l of DEFAULT_LISTS) {
    if (byName[l.name]) {
      console.log(`  list "${l.name}" exists (id ${byName[l.name].id}) — skipping`);
      continue;
    }
    const r = await req('POST', '/api/lists', { cookie, body: l });
    if (r.status === 200) console.log(`  list "${l.name}" created`);
    else console.log(`  list "${l.name}" FAIL ${r.status}: ${r.body.slice(0, 120)}`);
  }
}

// ----- Templates -----
function loadTemplate(name, imageUrls) {
  const src = fs.readFileSync(path.join(__dirname, 'seed', 'templates', name), 'utf8');
  return src
    .replace(/__LOGO_URL__/g, imageUrls.logo)
    .replace(/__FB_URL__/g, imageUrls.fb)
    .replace(/__X_URL__/g, imageUrls.x)
    .replace(/__IG_URL__/g, imageUrls.ig);
}

async function ensureTemplate(cookie, { name, type, subject, body }) {
  const all = await listAllTemplates(cookie);
  const existing = all.find((t) => t.name === name);
  if (existing) {
    if (!FORCE) {
      console.log(`  template "${name}" exists (id ${existing.id}) — skipping (SEED_FORCE_UPDATE=1 to overwrite)`);
      return existing.id;
    }
    const r = await req('PUT', `/api/templates/${existing.id}`, { cookie, body: { name, type, subject: subject || '', body } });
    console.log(`  template "${name}" updated (id ${existing.id}, HTTP ${r.status})`);
    return existing.id;
  }
  const r = await req('POST', '/api/templates', { cookie, body: { name, type, subject: subject || '', body } });
  if (r.status !== 200) { console.log(`  template "${name}" FAIL ${r.status}: ${r.body.slice(0, 120)}`); return null; }
  const id = JSON.parse(r.body).data.id;
  console.log(`  template "${name}" created (id ${id})`);
  return id;
}

async function setDefaultTemplate(cookie, id) {
  if (!id) return;
  const r = await req('PUT', `/api/templates/${id}/default`, { cookie });
  console.log(`  default template id ${id} (HTTP ${r.status})`);
}

// ----- Settings -----
function loadAsset(rel) {
  return fs.readFileSync(path.join(__dirname, 'seed', 'styles', rel), 'utf8');
}

async function applySettings(cookie, imageUrls) {
  const cur = await req('GET', '/api/settings', { cookie });
  const s = JSON.parse(cur.body).data;

  s['app.site_name'] = 'Street Voices';
  s['app.root_url'] = ROOT_URL;
  s['app.logo_url'] = imageUrls.logo;
  s['app.favicon_url'] = imageUrls.logo;
  s['app.from_email'] = FROM_EMAIL;
  s['app.send_optin_confirmation'] = true;

  // Base64-embed logo into admin.js
  const logoB64 = fs.readFileSync(path.join(__dirname, 'uploads', 'thumb_Street-voices-logo.png')).toString('base64');
  const logoDataUri = `data:image/png;base64,${logoB64}`;
  let adminJs = loadAsset('admin.js');
  adminJs = adminJs.replace(/__LOGO_URI__/g, logoDataUri);
  adminJs = adminJs.replace(/__COMPOSER_URL__/g, env.AI_COMPOSER_URL || 'http://localhost:3001');

  s['appearance.admin.custom_css'] = loadAsset('admin.css');
  s['appearance.admin.custom_js'] = adminJs;
  s['appearance.public.custom_css'] = loadAsset('public.css');
  s['appearance.public.custom_js'] = loadAsset('public.js');

  if (SENDGRID_KEY && SENDGRID_KEY.startsWith('SG.')) {
    s['smtp'] = [{
      name: 'email-sendgrid',
      uuid: '',
      enabled: true,
      host: 'smtp.sendgrid.net',
      port: 587,
      auth_protocol: 'login',
      username: 'apikey',
      password: SENDGRID_KEY,
      hello_hostname: '',
      email_headers: [],
      max_conns: 10,
      max_msg_retries: 2,
      idle_timeout: '15s',
      wait_timeout: '5s',
      tls_type: 'STARTTLS',
      tls_skip_verify: false,
    }];
  } else {
    console.log('  (SendGrid key not set — leaving SMTP untouched; add SENDGRID_API_KEY to .env to configure)');
    // Preserve existing SMTP; just ensure masked passwords aren't written back
    for (const smtp of s['smtp'] || []) {
      if (smtp.password && smtp.password.includes('\u2022')) {
        // keep as-is; Listmonk will preserve the real password
      }
    }
  }

  const put = await req('PUT', '/api/settings', { cookie, body: s });
  console.log(`  settings PUT ${put.status}`);
}

// ----- Main -----
(async () => {
  console.log('Street Voices Listmonk seeder');
  console.log(`  target: http://${HOST}:${PORT}`);
  console.log(`  admin : ${ADMIN_USER}`);

  console.log('\n1. Waiting for Listmonk to be reachable...');
  await waitForListmonk();
  console.log('   OK');

  console.log('\n2. Logging in as admin...');
  const cookie = await login();
  console.log('   OK');

  console.log('\n3. Uploading brand images...');
  const needed = ['Street-voices-logo.png', 'thumb_Street-voices-logo.png', 'facebook.png', 'twitter.png', 'instagram.png'];
  const existing = await listAllMedia(cookie);
  const byName = Object.fromEntries(existing.map((m) => [m.filename, m]));
  const results = {};
  for (const f of needed) {
    if (byName[f]) {
      console.log(`  ${f} already uploaded — reusing`);
      results[f] = byName[f];
    } else {
      const uploaded = await uploadMedia(cookie, f);
      if (uploaded) { results[f] = uploaded; console.log(`  ${f} uploaded`); }
      else { console.log(`  ${f} FAILED or missing`); }
    }
  }
  const imageUrls = {
    logo: `${ROOT_URL}/uploads/Street-voices-logo.png`,
    fb: `${ROOT_URL}/uploads/facebook.png`,
    x: `${ROOT_URL}/uploads/twitter.png`,
    ig: `${ROOT_URL}/uploads/instagram.png`,
  };

  console.log('\n4. Seeding lists...');
  await seedLists(cookie);

  console.log('\n5. Seeding templates...');
  const baseId = await ensureTemplate(cookie, {
    name: 'Street Voices base',
    type: 'campaign',
    body: loadTemplate('campaign-base.html', imageUrls),
  });
  await setDefaultTemplate(cookie, baseId);
  await ensureTemplate(cookie, {
    name: 'Welcome',
    type: 'tx',
    subject: 'Welcome to Street Voices{{ if .Subscriber.FirstName }}, {{ .Subscriber.FirstName }}{{ end }}!',
    body: loadTemplate('welcome.html', imageUrls),
  });

  console.log('\n6. Applying settings (branding, SMTP, URLs)...');
  await applySettings(cookie, imageUrls);

  console.log('\nDone. Visit', ROOT_URL + '/admin');
})().catch((e) => { console.error('\nFATAL:', e.message); process.exit(1); });
