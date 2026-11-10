# Engine selection — serve vs measure

**Purpose:** Decide, at decision time, whether a task needs a *serving engine*
(llama.cpp / vLLM / ollama / SGLang) or *in-process measurement* (the
model_headroom authority), and which serving engine to pick. Cross-references
`AGENTS.md` "Engine Selection Policy" (the consulted-at-decision-time gate)
and `tools/engine_probe.py` (the probe that prints what is installed/running).

**Status:** reference only. No Python code changes are proposed here. The probe
is a workspace shim (`tools/`) importing neither product package.

---

## The two axes, never flattened

| Axis | What it needs | How it is satisfied |
|------|---------------|---------------------|
| Serving / capture | Tokens or a server-exposed extra | A serving engine (llama.cpp, vLLM, ollama, SGLang) |
| Measurement | Anything INTERNAL to the model | In-process, on the accelerator, via `--device` (`src/model_headroom/device.py`) |

**Do not flatten these into one axis.** A serving engine can emit tokens but
cannot expose a weight matrix, `named_modules()`, parameter counts, optimizer
state, or real activations from a forward pass. Measurement cannot be served.

## Hard constraints (physically impossible otherwise)

- **`routed_experts` capture and hidden-state residual capture require SGLang.**
  llama.cpp and vLLM expose neither:
  - `src/capture_proxy/routing.py` decodes `sglext.routed_experts`
    (SGLang `--enable-return-routed-experts`), and
  - `src/model_headroom/refusal/activations.py:capture_residuals_sglang`
    (`src/model_headroom/refusal/activations.py:266`) requires
    `--enable-return-hidden-states`.
  Report `routed_experts: false` and `hidden_states: false` for llama.cpp,
  vLLM, and ollama — never mark them true on a guess.

- **Measurement signals cannot be served.** Effective rank needs the weight
  MATRIX; module discovery needs `named_modules()`; training feasibility needs
  parameter counts and optimizer-state arithmetic; R-05 dead neurons needs real
  activations from a forward pass. The authority reads weights directly
  (safetensors/GGUF via `src/model_headroom/models/loader.py` and
  `src/model_headroom/models/gguf_loader.py`) and runs forward passes in-process
  (`src/model_headroom/analysis/`).

- **T4 is sm_75: fp8 requires sm_89+, so fp8 checkpoints dequantize to bf16 and
  never execute fp8 kernels.** Measured on this box: two Tesla T4s, compute
  capability 7.5, 14911 MiB each. Report this as a limit whenever an sm_75
  device is present.

- **The 18GB mimo-v2.6-distill-qwen-9b checkpoint does NOT fit in 15GB VRAM.**
  CPU-RAM loading plus GPU compute is the correct architecture, not a workaround.

## Capability table per engine

| Capability | llama.cpp | vLLM (≥ 0.7) | ollama | SGLang |
|------------|-----------|--------------|--------|--------|
| chat_completions | ✅ | ✅ | ✅ | ✅ |
| embeddings | ✅ | ✅ | ✅ | ✅ |
| rerank | ✅ | ✅ (/v1/rerank) | ❌ (501) | ✅ |
| routed_experts | ❌ | ❌ | ❌ | ✅ (*) |
| hidden_states | ❌ | ❌ | ❌ | ✅ (**) |

(*) `routed_experts` requires SGLang `--enable-return-routed-experts` plus a
configured `LxK` shape (`CAPTURE_ROUTED_EXPERTS_SHAPE`).
(**) `hidden_states` requires SGLang `--enable-return-hidden-states`.

Serving-engine preferences (defaults, overridable):
- **llama.cpp** — GGUF, low VRAM, FastMTP sidecar.
- **vLLM** — dense-transformers throughput, `/v1/rerank`.
- **SGLang** — only when the task needs `routed_experts` or hidden states.

## One-screen decision rule

- Need tokens or a server-exposed extra → serving engine (pick per the
  preferences above).
- Need anything INTERNAL to the model (matrix, activations, gradients, module
  names, parameter counts) → in-process, on the accelerator.
- Need `routed_experts` or hidden states → SGLang, no alternative.

## The probe: `tools/engine_probe.py`

`python tools/engine_probe.py` prints a JSON manifest (exit 0 even when nothing
is found); `--compact` prints a human-readable summary. It is best-effort,
observation only: it never raises, never starts or stops a server, never binds a
port, and never guesses a backend from a port number (it asks the server's own
`/v1/models` metadata).

Output shape (schema_version 1):

```json
{
  "schema_version": 1,
  "generated_at": "<ISO8601 UTC>",
  "hardware": {
    "devices": [ {"name": "...", "total_memory_mib": <int>, "compute_capability": null} ],
    "notes": [ "...constraint strings..." ]
  },
  "engines": {
    "installed": [
      {
        "engine": "llamacpp\|vllm\|ollama\|sglang",
        "binary": "<path or null>",
        "version": "<string or null>",
        "running": true\|false,
        "endpoints": [ {"url": "...", "healthy": true\|false, "models": [...]} ],
        "capabilities": {
          "chat_completions": bool, "embeddings": bool, "rerank": bool,
          "routed_experts": bool, "hidden_states": bool
        },
        "notes": [ "..." ]
      }
    ]
  },
  "limits": [ "...hardware/engine constraint strings..." ],
  "errors": [ "..." ]   // present only when something went wrong
}
```

### How to read it

- `engines.installed[*].running == true` means a server answered `/health` on a
  candidate endpoint (11434, 8000, 8001, 8080, 30000) and identified itself by
  its own metadata. Port alone never identifies an engine.
- `capabilities.routed_experts` / `capabilities.hidden_states` are `true` only
  for SGLang. Every other engine reports `false`.
- `limits` carries the T4/fp8 and 18GB-vs-15GB-VRAM constraints measured at
  runtime.

## Measured hardware limits (this box, 2026-11-10)

| Item | Value | Evidence |
|------|-------|----------|
| GPUs | 2 × Tesla T4 | `llama-server --list-devices` / nvidia-smi |
| Compute capability | sm_75 (7.5) | nvidia-smi |
| VRAM per device | 14911 MiB | `llama-server --list-devices` |
| fp8 on T4 | dequantizes to bf16, no fp8 kernels | sm_75 < sm_89 required |
| 18GB checkpoint vs 15GB VRAM | does not fit | measured VRAM 14911 MiB |
| llama-server | `/usr/local/bin/llama-server`, build 11395, commit 7f2dd88b0 | `llama-server --version` |
| vLLM venv | `/root/.venvs/vllm/bin/python`, vllm 0.32.0 | venv import probe |
| ollama / sglang | not present on PATH | `command -v` |

## Known limitation (do not fix here)

The stock llama.cpp build 11395 cannot read `Ternary-Bonsai-8B-Q2_0.gguf`
(`gguf_init_from_reader: tensor 'output_norm.weight' has offset 165015872,
expected 174722688` — legacy ternary quant). A PrismML llama.cpp is being
built for this; until then, treat that GGUF as unreadable by the serving
engine. The repo's own authority path handles that GGUF separately via
`src/model_headroom/models/gguf_loader.py` ternary dequant adapters — do not
conflate the two.

## Port-based backend detection is frozen

Do NOT change `src/capture_proxy/app.py`'s port-based backend auto-detection
(`"ollama" if "11434" in self._upstream_url else "llamacpp"` at line ~230).
The probe is the decision-time artifact; the runtime path stays as documented
and tested. This is not a bug to fix.

## File:line reference index

| Claim | File:line |
|-------|-----------|
| device resolver (accelerator preferred, CPU fallback) | `src/model_headroom/device.py` |
| `capture_residuals_sglang` requires `--enable-return-hidden-states` | `src/model_headroom/refusal/activations.py:266` |
| `sglext.routed_experts` decode | `src/capture_proxy/routing.py` |
| GGUF weight extraction | `src/model_headroom/models/gguf_loader.py` |
| HF weight extraction / architecture detection | `src/model_headroom/models/loader.py` |
| analysis signals (in-process forward passes) | `src/model_headroom/analysis/` |
| proxy port-based backend auto-detection (frozen) | `src/capture_proxy/app.py:~230` |
| probe shim (imports neither package) | `tools/engine_probe.py` |
