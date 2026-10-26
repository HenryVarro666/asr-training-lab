# Independent CPU numerical audit

Status: **failed predeclared tolerance**. One remote CPU-only attempt exited1 after15.597seconds; no retry and no tolerance changes. No model, GPU or weight-content read by the audit.

At row j=2, target.prob differs from the independent FP64 reference by1.1167530824529592e-6, exceeding the frozen1e-6 absolute probability tolerance. The fixed source iteration identifies the first dataset as base/generation; the exception prints only the row.

This is a derived FP32-versus-FP64 acceptance failure. GPU inference completion is separate. The full2304-row audit did not finish; no all-row pass or aggregate maxima is claimed. Raw evidence/campaign are unchanged. No result figures were generated under the pass-before-plot instruction.

Full command, tool SHA, elapsed time and exit status: audit-execution.json. Full traceback: audit-stderr.log. The remote auditor aborted before writing its own JSON/Markdown; these local files record the failure without fabricating a successful remote output.
