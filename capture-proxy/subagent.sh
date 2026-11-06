#!/usr/bin/env bash
# subagent.sh — puppeteer subagent launcher (pi one-shot dispatch)
#
# Usage:
#   ./subagent.sh <label> <prompt-file|-> [model] [timeout-sec] [--bg] [--resume <session-uuid>] [--thinking <lvl>]
#   ./subagent.sh list
#   ./subagent.sh status
#
#   label        snake_case id; prompt/output/logs are named after it
#   prompt-file  text file with the dispatch prompt, or - for stdin
#   model        default: local/qwen3.8-27b-fp8
#   timeout-sec  default: 1800 (0 = no timeout, foreground only)
#   --bg         nohup + disown (survives ssh drop); prints PID, use 'status'
#   --resume     continue an existing pi session (uuid or file path)
#   --thinking   default: medium (xhigh caused thinking-only turn ends 2026-08-20)
#                levels: off|minimal|low|medium|high|xhigh|max
#   guard        auto-continues up to 3x when a run ends on a thinking-only turn
#
# Everything is stored under /kaggle/working/fglob/out/subagents/ (persistent volume).
# Subagents run with cwd=/kaggle/working/fglob so they inherit skills, .mcp.json
# (hive-mind, searxng, sequential-thinking), RULES.md, and the workspace contract.
set -euo pipefail

WS=/kaggle/working/fglob
SUB=$WS/out/subagents
MANIFEST=$SUB/dispatches.jsonl
DEFAULT_MODEL=local/qwen3.8-27b-fp8

mkdir -p "$SUB"

status() {
  echo "=== subagent runs (cwd $WS) ==="
  pgrep -af "pi --model" | grep -v pgrep || echo "(none running)"
  echo
  echo "=== stored runs ==="
  for f in "$SUB"/*.out.md; do
    [ -e "$f" ] || continue
    label=$(basename "$f" .out.md)
    pf="$SUB/$label.prompt.md"; pl=$([ -f "$pf" ] && wc -c <"$pf" || echo 0)
    ol=$(wc -c <"$f")
    age=$(date -r "$f" '+%H:%M')
    echo "  $label  prompt:${pl}B  out:${ol}B  last:$age"
  done | sort || true
}

[ "${1:-}" = "list" ] || [ "${1:-}" = "status" ] && { status; exit 0; }

label="${1:-}"; promptsrc="${2:-}"; model="$DEFAULT_MODEL"; timeout=1800; bg=0; resume=""; thinking="medium"
shift 2 2>/dev/null || true
while [ $# -gt 0 ]; do
  case "$1" in
    --bg) bg=1 ;;
    --model) model="${2:-}"; shift ;;
    --resume) resume="${2:-}"; shift ;;
    --thinking) thinking="${2:-}"; shift ;;
    *) case "$1" in
         *[!0-9]*) model="$1" ;;
         *) if [ "$timeout" = "1800" ]; then timeout="$1"; else echo "unexpected arg: $1" >&2; exit 2; fi ;;
       esac ;;
  esac
  shift
done

[ -n "$label" ] || { echo "usage: $0 <label> <prompt-file|-> [model] [timeout-sec] [--bg]" >&2; exit 2; }
[ -n "$promptsrc" ] || { echo "usage: $0 <label> <prompt-file|-> [model] [timeout-sec] [--bg]" >&2; exit 2; }
case "$label" in *[!a-zA-Z0-9_-]*) echo "label must be [a-zA-Z0-9_-]" >&2; exit 2;; esac

prompt_file="$SUB/$label.prompt.md"
out_file="$SUB/$label.out.md"
log_file="$SUB/$label.log"

if [ "$promptsrc" = "-" ]; then
  cat > "$prompt_file"
else
  [ -f "$promptsrc" ] || { echo "prompt file not found: $promptsrc" >&2; exit 2; }
  cp "$promptsrc" "$prompt_file"
fi

ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
pbytes=$(wc -c < "$prompt_file")
printf '{"label":"%s","model":"%s","prompt_file":"%s","prompt_bytes":%s,"ts":"%s","mode":"%s"}\n' \
  "$label" "$model" "$prompt_file" "$pbytes" "$ts" "${SUBAGENT_MODE:-direct}" >> "$MANIFEST"

run() {
  cd "$WS"
  local sess=""
  [ -n "$resume" ] && sess="--session $resume"
  local tflag=""
  [ -n "$thinking" ] && tflag="--thinking $thinking"
  local nudge="" prompt_arg="$prompt_file" rc=0 cont=0
  while [ "$cont" -lt 3 ]; do
    if [ -n "$nudge" ]; then prompt_arg="$nudge"; fi
    if [ "$timeout" -gt 0 ]; then
      timeout "$timeout" pi --model "$model" $sess $tflag -p "$(cat "$prompt_arg")" > "$out_file" 2> "$log_file"
    else
      pi --model "$model" $sess $tflag -p "$(cat "$prompt_arg")" > "$out_file" 2> "$log_file"
    fi
    rc=$?
    if [ $rc -eq 124 ]; then echo "TIMEOUT after ${timeout}s" >> "$log_file"; break; fi
    # thinking-only termination guard: qwen3 can end a turn on a thinking block
    # with no text and no tool call; pi then exits 0 with empty stdout.
    local sid to
    sid=$(ls -t ~/.pi/agent/sessions/*-kaggle-working-fglob--/2026-*.jsonl 2>/dev/null | head -1)
    to=$(python3 - "$sid" 2>/dev/null <<'PYEOF'
import json, sys
try:
    f = sys.argv[1]
    last = None
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        m = e.get("message", {})
        if m.get("role") == "assistant":
            last = m
    c = last.get("content") if last else None
    if isinstance(c, str):
        c = [{"type": "text", "text": c}]
    if not c:
        print("no")
    else:
        has_text = any(x.get("type") == "text" and (x.get("text") or "").strip() for x in c if isinstance(x, dict))
        has_think = any(x.get("type") == "thinking" for x in c if isinstance(x, dict))
        print("yes" if has_think and not has_text else "no")
except Exception:
    print("unknown")
PYEOF
    )
    if [ "$to" = "yes" ] && [ -n "$sid" ]; then
      cont=$((cont+1))
      echo "GUARD: thinking-only turn end (attempt $cont/3) — auto-continuing session $(basename "$sid" | sed -E 's/.*_//; s/\.jsonl$//')" >> "$log_file"
      sess="--session $(basename "$sid" | sed -E 's/.*_//; s/\.jsonl$//')"
      nudge="$SUB/$label.continue.md"
      printf 'CONTINUE — your previous turn ended with a thinking-only assistant message (no tool call, no text). Pick up exactly where you stopped; consult your scratchpad if needed. Rule: every turn must end with a tool call or a non-empty text message. Do not restart the task.\n' > "$nudge"
      continue
    fi
    [ "$to" = "yes" ] && echo "GUARD: thinking-only end but no session file found — cannot auto-continue" >> "$log_file"
    break
  done
  if [ $rc -eq 0 ] && [ ! -s "$out_file" ]; then
    echo "WARNING: rc=0 but EMPTY output — final assistant message had no text (likely ended on a thinking-only turn). Check session transcript." >> "$log_file"
  fi
  echo "$rc" > "$SUB/$label.rc"
}

if [ "$bg" -eq 1 ]; then
  nohup bash -c "export WS='$WS' SUB='$SUB' timeout='$timeout' label='$label' model='$model' resume='$resume' thinking='$thinking' prompt_file='$prompt_file' out_file='$out_file' log_file='$log_file'; $(declare -f run); run" \
    > "$log_file.no" 2>&1 & disown
  echo "dispatched: label=$label model=$model prompt=${pbytes}B (bg)"
  echo "pid=$!  out=$out_file  log=$log_file"
  echo "check: $0 status"
else
  run
  rc=$(cat "$SUB/$label.rc" 2>/dev/null || echo "?")
  echo "=== $label done (rc=$rc) ==="
  echo "out: $out_file ($(wc -c <"$out_file") B)"
  echo "--- tail ---"
  tail -c 2000 "$out_file"
fi
