// Decode the selected native Zcode enc:v1 values for projection into a private
// home. Zcode accepts plaintext values and encrypts subsequent native writes.
// No authentication requests or credential refreshes occur here.
const crypto = require('node:crypto');
const os = require('node:os');
const fs = require('node:fs');
try {
  const values = JSON.parse(fs.readFileSync(0, 'utf8'));
  let username = 'unknown';
  try { username = os.userInfo().username; } catch {}
  const secret = process.env.ZCODE_CREDENTIAL_SECRET?.trim()
    || `zcode-credential-fallback:${os.platform()}:${os.homedir()}:${username}`;
  const key = crypto.createHash('sha256').update(secret).digest();
  const decoded = {};
  for (const [name, value] of Object.entries(values)) {
    if (typeof value !== 'string') throw new Error('Invalid credential value');
    if (!value.startsWith('enc:v1:')) { decoded[name] = value; continue; }
    const parts = value.slice(7).split('.');
    if (parts.length !== 3) throw new Error('Invalid ciphertext');
    const [iv, tag, body] = parts.map(part => Buffer.from(part, 'base64url'));
    if (iv.length !== 12 || tag.length !== 16) throw new Error('Invalid ciphertext');
    const cipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
    cipher.setAuthTag(tag);
    decoded[name] = Buffer.concat([cipher.update(body), cipher.final()]).toString('utf8');
  }
  process.stdout.write(JSON.stringify(decoded));
} catch {
  process.stderr.write('Cannot decode selected native Zcode credentials\n');
  process.exitCode = 1;
}
