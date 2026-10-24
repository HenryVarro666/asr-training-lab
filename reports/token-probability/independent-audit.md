# Independent evidence audit preparation

Status: **prepared, not run**. No completed token-probability run exists. GPU1 was occupied according to the parent live preflight; no model call, waiting/retry, weight read or actual probability audit was performed.

Independent FP64 CPU toy checks passed: stable normalization, exact rank/ties/top10, entropy, distinct EOS sum, finite log probability under probability underflow, and teacher first/last/EOS/L=1/off-by-one alignment. Pinned tokenizer JSON hashes and C34/CA5049/UL1094/aul4943 pieces passed. Plot helper syntax passed; **no real or synthetic result figures generated**.

The saved-evidence audit has a hard600-second alarm and predeclared FP64-vs-worker tolerances. Each PT is read once, hashed and loaded from the same bytes. Run only after explicit parent completed-run signal and local evidence mirror. Plotting reads derived JSON only and emits PNG/SVG/PDF, preserving log probabilities and identical-prefix conditioning.
