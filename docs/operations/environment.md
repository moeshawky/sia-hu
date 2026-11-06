# Environment — GPU/CPU/TPU notes, T4 measurements, transformers ceiling

**Purpose:** Document the measured environment: hardware constraints, device
policy, TPU absence, and known software limitations. Cross-references
`concepts/frozen-contracts.md` (NFR-01 CPU-only weights), `operations/verify-chain.md`
(verify chain per device), and `guides/model-headroom.md` (`--device` opt-in).
No Python code changes are proposed here.

---

## Device policy

**CPU is the default.** Weights load to CPU RAM by design (NFR-01). The authority is
an offline analysis tool — vLLM HBM and GPU memory are untouched.

An opt-in `--device` flag exists (Q4 decision in the plan) with an explicit memory
budget, not auto-selection. A deliberately oversized tensor fails loudly, never
silently OOMs into a wrong answer. GPU probe: same tensor on CPU and GPU gives the
same R-03 within tolerance.

**TPU is absent on this box.** Verified four ways: no `/dev/accel*`, no `accel` in
`/proc/devices`. TPU work is code-level guards + a portability test — explicitly
not runnable here, explicitly not reported as verified.

---

## Measured hardware (2× Tesla T4 box)

| Resource | Measured | Consequence |
|----------|----------|-------------|
| GPU | 2× Tesla T4, 15360 MiB each (~30.7 GB total) | 27.1 GB bf16 model does not fit one T4; EMO needs device_map with CPU offload |
| CPU | 4 cores | SVD on 17408×5120 float32 is slow; probes use ≤512×1024 |
| RAM | 31 GB total, ~28 GB available | Bounds the full-matrix R-03 path |
| `/dev/shm` | 14 GB | Rules out the `/dev/shm` model-staging pattern for large models |
| Disk | ~1.1 TB free | Ample for model copies |
| TPU | **Absent** | Code guards only, not verified |

**fp16 used throughout.** T4 is compute capability 7.5 (Turing) with no native bf16,
so results are fp16-specific.

---

## Parameter budget (EMO_1b14b_1T, measured from safetensors index)

| Family | Params | Share |
|--------|--------|-------|
| experts | 12.885 B | 95.0% |
| embed + lm_head | 0.411 B | 3.0% |
| attention | 0.268 B | 2.0% |
| routers (16×) | 0.004 B | 0.03% |
| norms | ~0 | — |

**Total:** 13.569 B parameters (README's "14B" rounds up). fp32 = 54.3 GB —
1.8× over VRAM, impossible on one T4. bf16/fp16 = 27.1 GB — fits with ~3.6 GB
headroom (tight). int8 = 13.6 GB — fits. int4 = 6.8 GB — fits.

---

## Known software limitations

| Limitation | Detail |
|------------|--------|
| `transformers` broken on import | `huggingface-hub>=1.5,<2.0` required, `2.0.0` present. Deliberately preserved — do not fix, downgrade, or uninstall. Blocks any real-model step using transformers. |
| `gguf` extra not installed | `import gguf` → `ModuleNotFoundError`. GGUF code paths dead until `pip install -e ".[gguf]"` runs. |
| No real checkpoints on this box | Every number is synthetic. The 16 authority integration tests skip (no model files). |
| `mypy` not on PATH | Authority chain requires mypy; carrier keeps its own mypy in `.venv`. Toolchains stay separate. |
| No CI, no pre-commit | `.pre-commit-config.yaml` missing. Pre-commit listed as dev dep but not configured. |

---

## Cross-references

- `concepts/frozen-contracts.md` — NFR-01 CPU-only weights
- `operations/verify-chain.md` — verify chain (mypy src/, not mypy .)
- `guides/model-headroom.md` — `--device`, `--architecture` flags
- `reference/env-vars.md` — env vars for device/backend selection
