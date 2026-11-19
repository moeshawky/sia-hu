#!/usr/bin/env bash
set -euo pipefail

# bootstrap-capture.sh — make capture-proxy real on a freshly (re)started
# instance, in one idempotent command.
#
# Why this exists: capture-proxy being INSTALLED is not the same as it being
# CONFIGURED or SYNCING. Measured 2026-10-13 on a restarted Kaggle GPU box —
# the package was installed and a config existed, but no proxy process, no PID
# file, no sync loop, and an empty capture dir. Everything below is
# re-runnable: each step either creates what is missing or reports that it is
# already in place.
#
# Steps:
#   1. ensure the config file exists (never overwrites an existing one)
#   2. start the proxy daemon (skips when already running)
#   3. start the capture sync loop (skips when already running; skipped
#      entirely when rclone or its config is absent)
#
# Usage:
#   bash bootstrap-capture.sh                 # all three steps
#   bash bootstrap-capture.sh --no-proxy      # config + sync only
#   bash bootstrap-capture.sh --no-sync       # config + proxy only
#   bash bootstrap-capture.sh --check         # report state, change nothing
#
# Exit: 0 when the requested steps are in place (including "already were"),
# 1 when a requested step could not be completed. A boot calling this must
# not be killed by a missing rclone — that is a warning, not a failure.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="${CAPTURE_SYNC_SCRIPT:-${HERE}/rclone-sync-captures.sh}"

START_PROXY=1
START_SYNC=1
CHECK_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --no-proxy) START_PROXY=0 ;;
    --no-sync) START_SYNC=0 ;;
    --check) CHECK_ONLY=1 ;;
    -h | --help)
      sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

# The console script when installed; the module otherwise (editable or
# source checkout). Probing the module keeps this working on a box where the
# pip entry point was never created.
if command -v capture-proxy >/dev/null 2>&1; then
  CAPTURE=(capture-proxy)
elif python3 -c 'import capture_proxy' >/dev/null 2>&1; then
  CAPTURE=(python3 -m capture_proxy)
else
  echo "ERROR: capture-proxy is not installed and not importable" >&2
  exit 1
fi

rc=0

# --- 1. config -------------------------------------------------------------
# Runs even under --check: reading the effective config is the report.
ensure_config() {
  "${CAPTURE[@]}" --ensure-config
}
ensure_config || rc=1

# --- 2. proxy daemon -------------------------------------------------------
proxy_state() { "${CAPTURE[@]}" --status >/dev/null 2>&1; }

if [ "${START_PROXY}" -eq 1 ]; then
  if proxy_state; then
    echo "==> proxy: already running"
  elif [ "${CHECK_ONLY}" -eq 1 ]; then
    echo "==> proxy: not running (check only — not starting)"
  else
    echo "==> proxy: starting daemon"
    # --daemon exits 1 when it is already running; a race between our status
    # probe and the start is not a boot failure.
    "${CAPTURE[@]}" --daemon || echo "==> proxy: daemon start returned non-zero (may be a race; verify with --status)"
    sleep 1
    if proxy_state; then
      echo "==> proxy: up"
    else
      echo "ERROR: proxy did not come up — see $HOME/.cache/capture-proxy/proxy.log" >&2
      rc=1
    fi
  fi
fi

# --- 3. sync loop ----------------------------------------------------------
if [ "${START_SYNC}" -eq 1 ]; then
  if [ ! -f "${SYNC_SCRIPT}" ]; then
    echo "WARNING: sync script not found at ${SYNC_SCRIPT} — captures stay local" >&2
  elif ! command -v rclone >/dev/null 2>&1; then
    echo "WARNING: rclone not installed — captures stay local" >&2
  elif [ "${CHECK_ONLY}" -eq 1 ]; then
    echo "==> sync: ${SYNC_SCRIPT} (check only)"
  else
    echo "==> sync: starting capture sync loop"
    bash "${SYNC_SCRIPT}" --daemon || {
      echo "WARNING: sync loop failed to start — captures stay local" >&2
      rc=1
    }
  fi
fi

echo "==> effective config"
"${CAPTURE[@]}" --print-effective-config || rc=1

exit "${rc}"