# Frozen-trajectory token probability probe

Status: complete; stage: three_models_nine_teacher_forwards_complete.

Planned maximum: 3 native-observation generations, 9 empty-cache teacher forwards, 3 generation matrices and 9 teacher matrices; 0 optimizer updates. No retries or additional samples.
Observed generate calls: 3; Observed teacher forward calls: 9 (recorded invocation counters).
Workers with unknown invocation counts: []; these are not counted as zero.
Present generation matrix files: 3; present teacher matrix files: 9.
Archived generation matrices: 3; Archived teacher matrices: 9 (file plus size/SHA receipt verified).
Fully validated workers: ['base', 'step050', 'step100']. Partial files and unchecked receipts do not establish a complete matrix.
Frozen trajectory lengths: 27/37/512 IDs. Same-prefix probabilities describe model behavior; they do not identify training causes.
Planned matrix bytes: 1400242176; this is an estimate, not observed storage.
