# Fixed checkpoint100 truncation probe

Status: complete; stage: three_preset_probes_complete.

Three fresh workers only: original batch137 twice, then the target alone;33 sample consumptions, zero optimizer updates.
Original checkpoint,512-token budget and generation kwargs retained. No scores/processors/stopping criteria added.
No previous137 batches replayed. Probe completion does not repair the original failed quality gate.

Repeated batch full tokens identical: True.
- batch16-first: returned width512; legacy truncated=True; parser output=['English', 'CULINARY CUL'].
- batch16-repeat: returned width512; legacy truncated=True; parser output=['English', 'CULINARY CUL'].
- target-single: returned width512; legacy truncated=True; parser output=['English', 'CULINARY CUL'].
