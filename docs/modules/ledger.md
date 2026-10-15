# ledger.rs — receipts banked

Token usage and the prompt/response receipts, banked exact to 18 decimal places and ready for a blockchain by default.

- **Amounts** (`Amount`): integers of 10⁻¹⁸ in a `u128` (an ERC-20 balance with 18 decimals; it fits a `uint256`),
  parsed and printed exactly — always 18 decimals — with checked arithmetic. `0.1 + 0.2 = 0.3` here.
- **Entries**: version, seq, time (ms), source (`bankml`, or `external:<provider>` for inference elsewhere, its counts
  as reported), model (sha256 where bankML verified it), request and response sha256, prompt / completion tokens, the
  total as an amount, the rate per token and the cost (rate × tokens, exact), and the previous entry's hash.
- **The chain**: each entry's hash is keccak256 (the EVM's) of its canonical bytes — JSON in a fixed key order, no
  spaces — which include the previous hash, so the head commits to the whole ledger; `verify` names the first entry
  that does not hold. keccak256 here agrees with `sAGI/agents.py` and Solidity (`transfer(address,uint256)` →
  `a9059cbb`); the shared vector (two entries, head `7b6663da…0a65`) is pinned in `the_shared_vector`, and browseML's
  browser ledger must reproduce it byte for byte.
- **Serving**: `bankml serve … --ledger FILE` (or `BANKML_LEDGER=FILE`) banks every receipt it writes, as one JSON line,
  appended; the file is verified in full first and a broken chain is never appended to. `BANKML_RATE` sets the rate
  per token (default 0). Nothing is written unless a ledger is asked for.
- **The command**: `bankml ledger verify FILE` (exit 1 if the chain breaks) and `bankml ledger totals FILE` — exact
  totals per source and over all, with the head.
- **The blockchain, as a module** (`Anchor`): `NoAnchor` by default (the ledger stands alone); THOT (the head as a
  commitment in an agent's bundle, `sAGI/thot.py`) and iNFT (an unsigned anchoring transaction for the owner to sign,
  `sAGI/chain.py`) connect through it. bankML never signs or sends a transaction.

Checked live (2026-10-14): `serve --native --ledger`, two answers, `ledger verify` ok, totals 47 tokens
(`47.000000000000000000`). Next: the same ledger in browseML (BigInt, the shared vector), THOT and iNFT connectors in
`sAGI`, external providers' counts banked where bankML proxies them.
