# Secret Rotation — .env.vault.gpg

## What is .env.vault.gpg?

Encrypted vault containing production API keys and secrets. Encrypted with GPG
so credentials are never stored in plaintext in the repository.

## Decryption

```bash
gpg --decrypt .env.vault.gpg > .env
```

You will be prompted for the GPG passphrase. The decrypted `.env` file contains
all production credentials and must NEVER be committed to git.

## Rotation Schedule

| Secret | Rotation Frequency | How to Rotate |
|--------|-------------------|---------------|
| IBKR credentials | On compromise only | Regenerate in TWS → Settings → API → Reset |
| FMP_API_KEY | Annually | financialmodelingprep.com → Dashboard → API Keys |
| OPENBB_TOKEN | Annually | my.openbb.co → Settings → API Keys |
| ANTHROPIC_API_KEY | Quarterly | console.anthropic.com → API Keys |
| XIAOMI_MIMO_API_KEY | On compromise only | Xiaomi developer portal |
| ZEP_API_KEY | Annually | Zep dashboard |
| GRAFANA_ADMIN_PASSWORD | On deployment | Change in docker-compose or Grafana UI |

## After Rotation

1. Update the plaintext `.env` file with new credentials
2. Re-encrypt the vault:
   ```bash
   gpg --symmetric --cipher-algo AES256 -o .env.vault.gpg .env
   ```
3. Verify decryption works:
   ```bash
   gpg --decrypt .env.vault.gpg | head -5
   ```
4. Commit the updated `.env.vault.gpg` (encrypted file only — NEVER `.env`)
5. Restart all PM2 services:
   ```bash
   pm2 restart all
   ```

## Security Rules

- `.env` is in `.gitignore` — never committed
- `.env.vault.gpg` is committed — encrypted, safe in repo
- GPG passphrase shared via secure channel (1Password, Signal) — never in git/slack
- If any secret is compromised: rotate immediately, re-encrypt vault, force-restart all services
- Quarterly audit: verify `.env` matches `.env.vault.gpg` decryption
