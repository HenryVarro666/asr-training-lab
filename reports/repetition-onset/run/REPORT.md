# Four-model singleton repetition onset probe

Status: complete; stage: four_preset_singletons_complete.

base → step025 → step050 → step100; four fresh singleton calls at most, zero training; source worker unchanged.
Strong repetition and512/no-EOS budget exhaustion are separate observations. No retries or follow-up queue.
Each current model is content-hashed once within its240-second stage, then protected by stat checks; base shard capture has no historical file-SHA anchor.
- base: eos_terminated; strong=False; loop=False.
- step025: eos_terminated; strong=False; loop=False.
- step050: eos_terminated; strong=False; loop=False.
- step100: budget_exhausted_with_periodic_repetition; strong=True; loop=True.

Only saved sampled points are observed; exact first update, monotonicity and training cause are not established.
