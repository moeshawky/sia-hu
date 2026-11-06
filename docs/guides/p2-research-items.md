# P2 Research Items — SIA-RELEASE-DECISION.md Promising Research / Optional Integrations

*P2 items are optional research or optional integrations — none are release blockers, none are imported as dependencies. Every claim cites SIA-RELEASE-DECISION.md, SIA-RECON.md, or SIA-DIAGNOSTIC-CONTRACT.md by file:line or section.*

**Source of the P2 set:** SIA-RELEASE-DECISION.md lines 139-149, "P2 — Promising Research or Optional Integrations" table (six items A1-A6).

**Consolidated decision authority:** SIA-RECON.md section 5 "Import/Borrow/Reject Decisions" (lines 176-198), bottom line at line 198: *"No candidate merits IMPORT. torch.func is the only ADAPT (pattern borrow, zero new deps). Everything else is STUDY or REJECT. Default dependency burden stays at torch only."*

**Release gating:** SIA-RELEASE-DECISION.md line 178 — "P1/P2 items can follow as time permits" only after all P0 items are complete. P2 items never block publication.

---

## A1 — WeightWatcher (heavy-tail fit / MP bulk edge as optional cross-check for R-03)

### What it does
WeightWatcher is a checkpoint-only spectral diagnostics library. Per SIA-RECON.md lines 57-61: it computes per-layer spectral metrics — spectral norm, alpha power-law exponent, MP bulk edge, correlation traps — requiring no training, only a checkpoint. Import probe confirmed at SIA-RECON.md line 137: `pip install weightwatcher` → 0.7.7; `import weightwatcher` OK; `WeightWatcher`, `analyze`, `get_summary`, `distances`, `SVDSharpness` present. Evidence ledger at SIA-RECON.md lines 149-150.

### Why it's P2 (not P0/P1)
SIA-RELEASE-DECISION.md line 143: "Different math from Sia/Hu R-03; wrong problem (generalization not headroom); optional augmentation only." Sia/Hu R-03 measures training headroom (remaining adaptation capacity); WeightWatcher's alpha<2.0 → "over-trained" is a generalization/test-accuracy heuristic, not headroom (SIA-RECON.md line 65). Contradiction documented at SIA-RECON.md line 210: WW's alpha heuristic contradicts Sia/Hu's headroom definition — a model can be over-trained yet still have headroom, or vice versa.

### Integration potential
Optional cross-check for R-03 effective_rank only. SIA-RECON.md line 70: "STUDY — borrow heavy-tail fit and MP bulk edge as optional cross-checks for effective_rank; Sia/Hu retains headroom authority." SIA-RELEASE-DECISION.md line 190: "WeightWatcher as primary... STUDY only, as optional cross-check." It would not replace R-03, add to the composite score, or affect decision-grade gating (scoring.py weights effective_rank at 0.15 per SIA-RECON.md line 33).

### Importable code exists?
Yes — confirmed importable at SIA-RECON.md line 137. But per section 5 line 180 the disposition is STUDY, not IMPORT: no code is imported; heavy-tail fit / MP bulk edge would be studied as reference math only.

### Disposition: STUDY
Justification: importable but solves the wrong problem (generalization, not headroom); missing 6 of 7 Sia/Hu signals (SIA-RECON.md line 180, line 68); heavy-tail fit and MP bulk edge are the only borrowable elements, as optional cross-checks for R-03 effective_rank.

**file:line references:** SIA-RECON.md lines 57-71, 65, 68, 70, 137, 149-150, 180, 210; SIA-RELEASE-DECISION.md lines 143, 190; SIA-DIAGNOSTIC-CONTRACT.md line 57 (representation condition = weight spectra / effective rank).

---

## A2 — EOS Overtraining (perturbation-based sharpness via output-KL)

### What it does
Per SIA-RECON.md lines 74-76: isotropic Gaussian parameter noise → output-KL sharpness estimation, corroborated against true Hessian eigenvalues on Pythia-14m (arxiv:2604.13627 Appendix B.2). It measures parameter sensitivity / adaptation fragility via perturbation — the condition listed at SIA-DIAGNOSTIC-CONTRACT.md line 59 ("Perturbation sensitivity (not yet implemented; EOS STUDY candidate)").

### Why it's P2 (not P0/P1)
SIA-RELEASE-DECISION.md line 144: "Fails under quantization/bfloat16/architecture limits; doesn't predict FT outcomes; requires calibration study first." SIA-RECON.md line 78: fails under quantization, bfloat16 (noise_eps=1e-5 swallowed), architecture limits; does not predict fine-tuning outcomes; validated only on small models (1–3B).

### Integration potential
SIA-RECON.md line 80 (section 2 exploration verdict): "STUDY — extract/re-implement with calibrated noise_eps before reuse; not a fine-tuning-outcome predictor." Any integration would be a re-implementation, not an import — there is no library to import. It would sit outside the fixed 7-signal set (R-01..R-09) and would not affect the composite score or decision-grade gating.

### Importable code exists?
No. SIA-RECON.md line 141: "No importable package located for: EOS Overtraining..." Section 3 line 141 confirms EOS Overtraining is paper-only, not found as a reusable implementation. Per the rule at SIA-RECON.md line 141: "a research paper with no reusable implementation is not an import candidate."

### Disposition: REJECT
Justification: no importable library exists (paper-only); fails under quantization/bfloat16; does not predict fine-tuning outcomes; validated only on small models. Consolidated decision table at SIA-RECON.md line 181: "EOS Overtraining | A2 | REJECT | No importable library exists; paper-only."

**Note on tension:** SIA-RECON.md section 2 line 80 records a STUDY verdict ("extract/re-implement with calibrated noise_eps before reuse"), while the consolidated section 5 line 181 records REJECT ("No importable library exists; paper-only"). The consolidated table is the disposition authority; the section 2 STUDY reflects an exploration hypothesis that was superseded by the no-importable-code finding. Both are reported; neither is silently resolved.

**file:line references:** SIA-RECON.md lines 72-81, 80, 141, 152-153, 181; SIA-RELEASE-DECISION.md line 144; SIA-DIAGNOSTIC-CONTRACT.md line 59.

---

## A3 — LogME / Choose Your Transformer (task-compatibility signals)

### What they do
Per SIA-RECON.md lines 84-89: LogME measures downstream task compatibility of frozen features (needs labels `y`, closed-form, no prior runs). Choose Your Transformer (CYT) measures model selection for classification (needs target labels, no prior runs). Both are task-conditioned transferability estimators. Key finding at SIA-RECON.md line 91: "No universal fine-tunability value exists. LogME correlation is 0.58 last-layer → 0.86 only after per-task layer aggregation. None measures internal adaptation potential — all measure downstream task compatibility." CYT evidence ledger at SIA-RECON.md line 156.

### Why they're P2 (not P0/P1)
SIA-RELEASE-DECISION.md line 145: "Task-conditioned, need labels; H4 rejected; would extend signal set beyond fixed 7." SIA-DIAGNOSTIC-CONTRACT.md lines 15-19: H4 (task-specific transferability estimator) is explicitly rejected — zero task-conditioned signals in code; all signals are architecture-conditioned; A3 tools need labels and measure downstream compatibility, not internal adaptation. SIA-DIAGNOSTIC-CONTRACT.md line 60: task compatibility is a separate condition that Sia/Hu does not measure.

### Integration potential
None as imported signals. SIA-RECON.md line 93 (section 2 verdict): "LogME=ADAPT, LensLLM=STUDY, CYT=ADAPT, TuneAhead=DEFER" — but these are per-tool exploration verdicts, not import decisions. Consolidated table at SIA-RECON.md lines 182-185 records LogME, LensLLM, CYT all as STUDY. SIA-RELEASE-DECISION.md line 163 (OUT list): "Task-conditioned transferability signals (H4)... Zero task-conditioned signals in code; all transferability estimators need labels and measure downstream compatibility (A3)." Line 188: "No LogME/LensLLM-style downstream compatibility measurement."

### Importable code exists?
LogME: import probe not individually confirmed in section 3; section 5 line 182 records LogME as STUDY. No package located for TuneAhead (line 185). LogME and CYT are not imported into Sia/Hu; they remain external reference tools only.

### Disposition: STUDY (LogME and CYT)
Justification: task-conditioned transferability estimators needing labels; H4 rejected; measure downstream compatibility not headroom; would extend the fixed 7-signal set beyond Sia/Hu's scope. Consolidated table at SIA-RECON.md lines 182-185: LogME=STUDY, CYT=STUDY (both "not headroom; needs labels").

**Note on tension:** SIA-RECON.md section 2 line 93 records LogME=ADAPT and CYT=ADAPT (borrow pattern), while the consolidated section 5 lines 182-185 records both as STUDY. The consolidated table is the disposition authority; section 2 reflects per-tool exploration hypotheses superseded by the no-headroom-signal determination. Both reported; neither silently resolved.

**file:line references:** SIA-RECON.md lines 82-94 (table at 84-89, verdict at 91-93), lines 154-157 (evidence ledger), lines 182-185 (decision table); SIA-RELEASE-DECISION.md lines 145, 163, 188; SIA-DIAGNOSTIC-CONTRACT.md lines 15-19, 28-34, 60.

---

## A4 — Curvlinops / BackPACK (matrix-free curvature primitives)

### What they do
Per SIA-RECON.md lines 97-104: curvlinops exposes matrix-free curvature operators (GGN, Hessian, EF, KFAC); BackPACK exposes per-sample gradients and second moments via a `backpack()` context manager. torch.func (PyTorch core) exposes `grad`, `jvp`, `vjp`, `vmap` functional transforms. None of these exposes a subspace-mismatch API (accepting a proposed trainable subspace Q and returning projection energy / principal angles / residual gradient energy) — SIA-RECON.md line 107.

### Why they're P2 (not P0/P1)
SIA-RELEASE-DECISION.md line 146: "Gradient-adjacent but not weight-pair ratio/effective rank; would add dependency without adding an Sia/Hu signal." These compute curvature/gradient primitives, not headroom signals. Sia/Hu's R-03 uses weight singular values + spectral entropy; R-04 uses grad/weight ratio from weight pairs — neither is a curvature primitive output.

### Integration potential
SIA-RECON.md line 107: "Sia/Hu could build one [subspace-mismatch diagnostic] using curvlinops/backpack/torch.func as primitives without becoming an optimizer or PEFT framework." torch.func is already in torch (zero new deps) — SIA-RECON.md line 138, section 5 line 188: "torch.func | A4 | ADAPT | Already in torch; borrow functional-grad pattern, zero new deps." Curvlinops import probe timed out at 180s (unmeasured, not reported) — SIA-RECON.md line 139.

### Importable code exists?
torch.func: yes, already available in system Python (torch 2.11.0+cpu) — SIA-RECON.md line 138. Curvlinops: probe timed out, not verified — line 139. BackPACK: no import probe result recorded; section 5 line 187 records BackPACK as STUDY ("Second-order extensions, memory-heavy with batch"). None are imported as dependencies into Sia/Hu.

### Disposition: STUDY (Curvlinops and BackPACK)
Justification: gradient-adjacent curvature primitives, not weight-pair ratio/effective rank signals; would add dependency without adding an Sia/Hu signal; torch.func is the only ADAPT (pattern borrow, zero new deps). Consolidated table at SIA-RECON.md lines 186-188: Curvlinops=STUDY, BackPACK=STUDY, torch.func=ADAPT.

**file:line references:** SIA-RECON.md lines 95-109 (table at 97-104, verdict at 109), line 138 (torch.func probe), line 139 (curvlinops probe timed out), lines 158-163 (evidence ledger), lines 186-188 (decision table); SIA-RELEASE-DECISION.md line 146; SIA-DIAGNOSTIC-CONTRACT.md line 59 (perturbation sensitivity condition, not yet implemented).

---

## A5 — Tinker Cookbook capture plumbing ideas (ContextVar scope, WAL dedupe, tri-state schema)

### What it does
Per SIA-RECON.md lines 113-117: Tinker Cookbook capture plumbing designs include ContextVar ambient scope with reserved coordinate keys, bounded-queue exporter with drop-newest accounting, SQLite WAL store with shared cursor + idempotent dedupe, tri-state schema classification, example_id content-hash provenance, pass@k, regrade_trajectories. Eval components require hosted Tinker service credentials and are unsuitable. Evidence ledger at SIA-RECON.md line 164.

### Why it's P2 (not P0/P1)
SIA-RELEASE-DECISION.md line 147: "Ideas only — already partially present in carrier (tri-state schema); no code import needed." The carrier already implements tri-state field classification (null vs absent are different claims) per SIA-DIAGNOSTIC-CONTRACT.md line 228 (section 7d). These are plumbing design ideas, not functional requirements for release.

### Integration potential
Adapt ideas only, no code import. SIA-RECON.md line 119: "Capture=ADAPT (ideas only, no code import), Eval=DEFER (study designs, reject import)." Consolidated table at SIA-RECON.md lines 193-194: "Tinker capture | A5 | ADAPT | Ideas only (ContextVar scope, WAL dedupe, tri-state schema); no code import" and "Tinker eval | A5 | DEFER | Requires hosted Tinker service; study designs." Specific ideas that could inform carrier plumbing without importing code:
- ContextVar ambient scope for capture coordination
- Bounded-queue exporter with drop-newest accounting
- SQLite WAL store with shared cursor + idempotent dedupe
- Tri-state schema classification (already present)
- example_id content-hash provenance

### Importable code exists?
No code import — ideas only. SIA-RECON.md line 119 explicitly states "no code import" for capture; eval requires hosted Tinker service credentials (line 117) and is DEFER.

### Disposition: ADAPT (ideas only)
Justification: capture plumbing ideas are adaptable design patterns (ContextVar scope, WAL dedupe, tri-state schema); no code is imported; tri-state schema is already partially present in the carrier. Eval components are DEFER (require hosted Tinker service). Consolidated table at SIA-RECON.md lines 193-194.

**file:line references:** SIA-RECON.md lines 111-120, line 117 (eval requires hosted Tinker), line 119 (capture ADAPT / eval DEFER), line 164 (evidence ledger), lines 193-194 (decision table); SIA-RELEASE-DECISION.md line 147; SIA-DIAGNOSTIC-CONTRACT.md line 228 (tri-state null vs absent distinction).

---

## A6 — massive-sft CSV (1,027-row SFT experiment corpus as outcome labels)

### What it does
Per SIA-RECON.md lines 123-125: `evaluation_result.csv` — 1,027 rows × 37 cols of downstream eval scores only, no weights or gradients. Model weight checkpoints (~80 TB Git LFS, selectively downloadable); training logs; no gradients anywhere. Accessible via github.com/llm-jp/massive-sft commit `ca323da`; model weights on GitLab project 23. Evidence ledger at SIA-RECON.md line 165.

### Why it's P2 (not P0/P1)
SIA-RELEASE-DECISION.md line 148: "CSV is outcomes-only, no weights/gradients; useful for validating effective_rank predictions against real SFT outcomes." It is not a headroom signal, not a loader format, and carries no gradients — so it cannot be consumed by the authority's weight-pairs loader. It is useful only as external outcome labels for heuristic validation (internal research, per SIA-DIAGNOSTIC-CONTRACT.md line 252).

### Integration potential
Validate effective_rank (R-03) predictions against real SFT outcomes. SIA-RELEASE-DECISION.md line 148 explicitly: "useful for validating effective_rank predictions against real SFT outcomes." SIA-DIAGNOSTIC-CONTRACT.md line 252: "Sia/Hu can score a model against user-supplied checkpoints and loss curves to validate its own heuristics (internal research). It cannot become a benchmarking product." SIA-DIAGNOSTIC-CONTRACT.md line 61: "Observed response to training — Actual fine-tuning outcomes (not measured by Sia/Hu) — External (massive-sft CSV as labels only)." Any integration would be offline heuristic validation, not a shipped signal.

### Importable code exists?
No. SIA-RECON.md line 141: "No importable package located for: ... massive-sft weights/CSV-as-library." The CSV is data, not a library; weights are ~80 TB Git LFS, not importable. Consolidated table at SIA-RECON.md lines 195-196: "massive-sft CSV | A6 | REJECT | Raw SFT outcomes, not headroom signal or loader format" and "massive-sft weights | A6 | REJECT | Weight checkpoints ≠ 2-D pair loader input."

### Disposition: REJECT (for import); STUDY (as external outcome labels for heuristic validation)
Justification: raw SFT outcomes only; not a headroom signal; not a loader format; no weights or gradients; cannot be consumed by the authority's 2-D pair loader. Useful only as external labels for validating effective_rank predictions against real SFT outcomes (internal research, not shipped functionality). Consolidated table at SIA-RECON.md lines 195-196 records both CSV and weights as REJECT.

**Note on tension:** SIA-RECON.md section 2 line 128 records "STUDY — CSV as outcome labels for heuristic validation; do not import weights or gradients," while the consolidated section 5 lines 195-196 records REJECT ("Raw SFT outcomes, not headroom signal or loader format"). The consolidated table is the disposition authority for import decisions; the section 2 STUDY reflects the external-validation use case. Both reported; neither silently resolved. The task-facing disposition for the CSV as an *integration candidate* is REJECT; its value as external outcome labels remains documented.

**file:line references:** SIA-RECON.md lines 121-128, line 141 (no importable CSV-as-library), lines 165-167 (evidence ledger), lines 195-196 (decision table); SIA-RELEASE-DECISION.md line 148; SIA-DIAGNOSTIC-CONTRACT.md lines 61, 252.

---

## Disposition Summary — All Six P2 Items

| # | Item | Source | Disposition | One-line justification (SIA-RECON.md §5) |
|---|------|--------|-------------|------------------------------------------|
| A1 | WeightWatcher heavy-tail fit / MP bulk edge | A1 | STUDY | Wrong problem (generalization, not headroom); missing 6/7 signals; line 180 |
| A2 | EOS Overtraining perturbation-based sharpness | A2 | REJECT | No importable library exists; paper-only; line 181 |
| A3 | LogME / Choose Your Transformer | A3 | STUDY | Task-conditioned, needs labels; not headroom; lines 182-185 |
| A4 | Curvlinops / BackPACK matrix-free curvature | A4 | STUDY | Gradient-adjacent, not weight-pair ratio/effective rank; lines 186-188 |
| A5 | Tinker capture plumbing ideas | A5 | ADAPT | Ideas only (ContextVar scope, WAL dedupe, tri-state schema); no code import; lines 193-194 |
| A6 | massive-sft CSV outcome labels | A6 | REJECT | Raw SFT outcomes, not headroom signal or loader format; lines 195-196 |

**Consolidated bottom line (SIA-RECON.md line 198):** No candidate merits IMPORT. torch.func is the only ADAPT (pattern borrow, zero new deps). Everything else is STUDY or REJECT. Default dependency burden stays at torch only.

**Release status:** All six items are P2 — optional research or optional integrations. Per SIA-RELEASE-DECISION.md line 178, P1/P2 items follow only after all P0 items are complete. None block publication.

---

## Cross-References

| Document | Section | Relevance |
|----------|---------|-----------|
| SIA-RELEASE-DECISION.md | lines 139-149 | P2 table (A1-A6) with why-P2 rationale |
| SIA-RELEASE-DECISION.md | lines 168-178 | Release readiness — P0 must complete before P1/P2 |
| SIA-RELEASE-DECISION.md | lines 182-194 | What we should deliberately not build (incl. WeightWatcher as primary, line 190) |
| SIA-RECON.md | section 2, lines 57-128 | Adjacent systems A1-A6 exploration findings |
| SIA-RECON.md | section 3, lines 131-142 | Import probes actually executed |
| SIA-RECON.md | section 4, lines 145-167 | Evidence ledger (A1-A6 source facts) |
| SIA-RECON.md | section 5, lines 176-198 | Import/Borrow/Reject consolidated decisions |
| SIA-RECON.md | section 6, lines 202-225 | Contradictions (incl. WW alpha vs headroom, line 210) |
| SIA-DIAGNOSTIC-CONTRACT.md | lines 15-19 | H4 task-conditioned transferability rejected |
| SIA-DIAGNOSTIC-CONTRACT.md | lines 50-63 | Evidentiary distinctions (representation, gradient, perturbation, task compatibility, training response) |
| SIA-DIAGNOSTIC-CONTRACT.md | line 252 | Internal research vs product boundary |

---

*Document end. All claims trace to SIA-RELEASE-DECISION.md, SIA-RECON.md, or SIA-DIAGNOSTIC-CONTRACT.md by file:line or section. No claims are summarized from memory.*
