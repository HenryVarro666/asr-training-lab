# Independent saved-evidence CPU audit

Status: **passed**. Audited 2304 complete vocabulary rows across 12 matrices.

Independent FP64 recomputation verifies every target/candidate, exact rank/tie/top10 order, entropy, margins, distinct EOS sum, causal prefix/teacher row alignment and saved TF/generation differences against predeclared tolerances. Every PT was read, hashed and loaded once. No model calls or weight reads.

Saved evidence audit; saved model SHA/stat receipts are not a new weight rehash. Fixed abnormal prefixes change conditioning; low EOS is not a training cause. Finite cached-generation/full-forward differences are observations.

This pass applies to the corrected derived overlay only. Every original artifact was independently verified against its immutable worker result manifest before selecting corrected derived JSON. Original matrices, raw run and failed original audit remain unchanged. FP64 arithmetic, exact rank/tie/top10 checks and tolerances are from the SHA-pinned original auditor. Cross-model comparisons were independently reconstructed. Change-summary diagnostic counts are only hash-verified, not independently certified.
