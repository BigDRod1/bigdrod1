# Public tip / donation payment links

Openly published Cash App, Venmo, PayPal.me, Ko-fi, Buy Me a Coffee, Furipay, Zelle, and similar tip links for public creators, musicians, journalists, entertainers, streamers, and mutual-aid orgs.

## Files

| File | Description |
|------|-------------|
| `public_tip_links.json` | Full dataset + metadata |
| `public_tip_links.csv` | Flat rows for spreadsheets |

## Schema (CSV / `rows[]`)

- `name` — person or org
- `category` — role (musician, journalist, streamer, etc.)
- `platform` — Cash App / Venmo / PayPal / Ko-fi / Buy Me a Coffee / Furipay / Zelle / busk.co
- `handle_or_url` — normalized URL when possible (`cash.app/$Handle`, `venmo.com/u/Handle`, `paypal.me/Handle`)
- `source_url` — where the handle was published
- `verification` — `self-published` \| `directory-listed` \| `directory-verified` \| `secondary-source` \| `likely-self-published` \| `unverified-page-changed` \| `mentioned-no-handle`
- `notes` — context / caveats

## Caveat

**Payment handles change. Imposters are common. Always verify against the person’s official website, verified social bio, or Linktree before sending money.** This list is an open-web aggregation, not an endorsement or guarantee of authenticity.
