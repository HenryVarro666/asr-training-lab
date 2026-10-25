# Fixed128 error transitions

CPU-only descriptive analysis; no model calls, GPU or training. Input totals verified: 128 unique IDs, 2905 reference words, base/25/50/100 errors 46/122/108/93.

State means improved/tied/worse edit count relative to base. Recovery means leaving worse, not necessarily becoming error-free. Same state does not mean unchanged error count or unchanged hypothesis.

| Interval | Errors decreased / unchanged / increased (IDs) | Net error change | Persistent worse / recovered / newly worse |
|---|---|---|---|
| 25->50 | 23 / 88 / 17 | -14 | 35 / 12 / 10 |
| 50->100 | 26 / 87 / 15 | -15 | 27 / 18 / 7 |
| 25->100 | 31 / 81 / 16 | -29 | 27 / 20 / 7 |

Persistently worse across all three checkpoints: 23 IDs. All exact IDs and complete 3×3 tables are in transitions.json and summary.json.

| Candidate | Leave-one-speaker-out corpus WER delta range vs base (pp) |
|---|---|
| 25 | 2.462876 to 2.711381 |
| 50 | 1.919594 to 2.259887 |
| 100 | 1.514085 to 1.694915 |

Per-speaker absolute error contributions and every excluded-speaker calculation are saved separately. Positive ranges describe this subset after any one speaker is removed; they do not prove significance, independence or a training mechanism.

Reproduce: `python3 reports/error-transitions/analyze.py` (standard library; internal 60-second deadline; overwrites only this derived directory).

Analysis computation completed in 0.005497 seconds; 0 generate, 0 teacher forward, 0 updates. Output manifests exclude themselves to avoid self-hashing.
