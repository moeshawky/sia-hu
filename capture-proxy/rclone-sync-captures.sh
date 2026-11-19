#!/usr/bin/env bash
set -euo pipefail

# Syncs the capture-proxy base path to gdrive:captures/<instance-id>/.
#
# Capture files are append-only and are still being appended while a
# streaming response is in flight, so syncing one mid-write fails with
# "source file is being updated (size changed)" or "md5 hashes differ".
# --min-age skips files modified within the last MIN_AGE seconds; a file
# is picked up on a later cycle once its mtime stops changing.

REMOTE="${REMOTE:-gdrive:captures}"

# Instance key: hostname FIRST, /etc/machine-id only as a last resort.
#
# Measured 2026-10-13 on the Kaggle GPU image: /etc/machine-id is baked into
# the container image (mtime May 12, value 7375099dc88b47cca8818bdeddd1f06c)
# and is IDENTICAL on every instance booted from it — the earlier
# machine-id-first order pointed every instance at one shared gdrive prefix,
# which already holds another session's 2026-10-06 captures. Hostname
# (3e466401f9c2) is per-instance, so it is the safer default here.
#
# Collisions are destructive only under `sync` (mirror deletes), which is why
# SYNC_CMD now defaults to `copy`. Uniqueness outranks stability: a renamed
# host leaves an orphaned prefix, while a shared prefix under `sync` deletes
# another live instance's captures.
HOSTNAME_ID="$(hostname 2>/dev/null || true)"
HOSTNAME_ID="${HOSTNAME_ID//[[:space:]]/}"
MACHINE_ID_FALLBACK="$(cat /etc/machine-id 2>/dev/null || true)"
MACHINE_ID_FALLBACK="${MACHINE_ID_FALLBACK//[[:space:]]/}"

if [ -n "${CAPTURE_MACHINE_ID:-}" ]; then
  # Explicit operator override wins over both, so a deliberate shared prefix
  # (e.g. a stable dev box) stays possible.
  MACHINE_ID="${CAPTURE_MACHINE_ID//[[:space:]]/}"
elif [ -n "${HOSTNAME_ID}" ]; then
  MACHINE_ID="${HOSTNAME_ID}"
elif [ -n "${MACHINE_ID_FALLBACK}" ]; then
  MACHINE_ID="${MACHINE_ID_FALLBACK}"
  echo "WARNING: hostname unavailable; falling back to /etc/machine-id (${MACHINE_ID})" >&2
  echo "WARNING: image-baked machine-ids are shared across instances — verify before using SYNC_CMD=sync" >&2
else
  echo "ERROR: no instance id (CAPTURE_MACHINE_ID, hostname, and /etc/machine-id all empty)" >&2
  exit 1
fi
DEST="${REMOTE}/${MACHINE_ID}"

CONFIG_FILE="${CONFIG_FILE:-${HOME}/.config/capture-proxy/config.toml}"
# `|| true`: a missing config file must not kill the script (sed exits 2,
# pipefail would propagate it under set -e) — fall through to defaults.
BASE_PATH="$(sed -nE 's/^[[:space:]]*base_path[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/p' "${CONFIG_FILE}" 2>/dev/null | head -n 1 || true)"
SOURCE="${CAPTURE_BASE_PATH:-${BASE_PATH:-${HOME}/captures}}"

DRY_RUN="${DRY_RUN:-false}"
RCLONE_OPTS="${RCLONE_OPTS:-}"
DAEMON="${DAEMON:-false}"
# Default is `copy` (never deletes). `sync` is a mirror: it removes remote files
# absent locally, so any two instances that share an instance id erase each
# other's captures. Opt in only with a key you own alone.
SYNC_CMD="${SYNC_CMD:-copy}"
MIN_AGE="${MIN_AGE:-60s}"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"
LOG_DIR="${LOG_DIR:-${HOME}/.local/var/log}"
LOG_FILE="${LOG_DIR}/rclone-sync-captures.log"
# PID file for the daemon loop. Re-running --daemon on every boot must not
# spawn a second loop: two loops on one instance race each other's rclone
# state and double the egress. Kept next to the log so one directory holds
# both.
PID_FILE="${PID_FILE:-${LOG_DIR}/rclone-sync-captures.pid}"

case "${SYNC_CMD}" in
  sync | copy) ;;
  *) echo "ERROR: SYNC_CMD must be 'sync' or 'copy' (got: ${SYNC_CMD})" >&2; exit 1 ;;
esac

case "${SYNC_INTERVAL}" in
  '' | *[!0-9]*) echo "ERROR: SYNC_INTERVAL must be a whole number of seconds (got: ${SYNC_INTERVAL})" >&2; exit 1 ;;
esac

AGE_OPTS=""
if [ -n "${MIN_AGE}" ] && [ "${MIN_AGE}" != "0" ] && [ "${MIN_AGE}" != "0s" ]; then
  AGE_OPTS="--min-age ${MIN_AGE}"
fi

DRY_RUN_FLAG=""
if [ "${DRY_RUN}" = "true" ]; then
  DRY_RUN_FLAG="--dry-run"
fi

usage() {
  echo "Usage: rclone-sync-captures [--daemon]"
  echo ""
  echo "Sync <capture base path> to gdrive:captures/<instance-id>/"
  echo ""
  echo "Instance id: hostname, else /etc/machine-id. Override with CAPTURE_MACHINE_ID."
  echo ""
  echo "Options:"
  echo "  --daemon    Run as a background loop, disowned from the terminal"
  echo ""
  echo "Environment variables:"
  echo "  DRY_RUN=true         Preview without syncing (honored in daemon mode too)"
  echo "  RCLONE_OPTS=\"...\"   Extra rclone flags"
  echo "  REMOTE=gdrive:captures   Remote prefix"
  echo "  CAPTURE_BASE_PATH=/path  Source override (default: config base_path, else ~/captures)"
  echo "  CONFIG_FILE=/path        Config file (default: ~/.config/capture-proxy/config.toml)"
  echo "  CAPTURE_MACHINE_ID=id    Instance key override (default: hostname)"
  echo "  MIN_AGE=60s              Skip files modified within the last N (still being"
  echo "                           written); files are picked up once settled. 0 disables"
  echo "  SYNC_INTERVAL=300        Daemon loop period in seconds"
  echo "  SYNC_CMD=copy            'copy' (default) never deletes; 'sync' MIRRORS and"
  echo "                           deletes remote files absent locally — only safe with"
  echo "                           an instance id that belongs to you alone"
  exit 0
}

for arg in "$@"; do
  case "${arg}" in
    --daemon) DAEMON="true" ;;
    --help|-h) usage ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 1 ;;
  esac
done

if [ ! -d "${SOURCE}" ]; then
  # The capture base path is created lazily by the proxy on its first capture,
  # so at boot it legitimately does not exist yet. Creating it here (empty,
  # harmless) keeps a boot-time daemon alive instead of exiting before the
  # first response is ever captured.
  mkdir -p "${SOURCE}"
  echo "WARNING: created missing source directory: ${SOURCE}"
fi

if [ -z "$(ls -A "${SOURCE}" 2>/dev/null)" ]; then
  echo "WARNING: Source directory is empty: ${SOURCE}" >&2
fi

mkdir -p "${LOG_DIR}"

echo "Syncing ${SOURCE} -> ${DEST}"
echo "System identifier: ${MACHINE_ID}"

if [ "${DAEMON}" = "true" ]; then
  if [ -z "${_SYNC_DAEMONIZED:-}" ]; then
    # Idempotent start: a live loop recorded in PID_FILE means this boot is a
    # re-run, not a first run. os.kill -0 equivalent (`kill -0`) is the same
    # liveness probe the proxy's own --status uses; a stale file is overwritten.
    if [ -f "${PID_FILE}" ]; then
      existing="$(cat "${PID_FILE}" 2>/dev/null || true)"
      if [ -n "${existing}" ] && kill -0 "${existing}" 2>/dev/null; then
        echo "Daemon already running (PID ${existing}) — nothing to do"
        echo "Log: ${LOG_FILE}"
        exit 0
      fi
      echo "Stale PID file (${PID_FILE}) — starting a new loop"
    fi
    # Re-exec self once under nohup so the loop survives the terminal;
    # the guard env var stops the inner process from re-execing again.
    export _SYNC_DAEMONIZED=1
    nohup bash "$0" "$@" >> "${LOG_FILE}" 2>&1 &
    disown
    echo "Daemon mode — background loop started (PID: $!)"
    echo "Interval: ${SYNC_INTERVAL}s   Log: ${LOG_FILE}"
    echo "Stop with: kill $!"
    exit 0
  fi
  # Inner process: stdout already points at LOG_FILE via nohup above. Record
  # the loop's OWN pid — the outer PID above belongs to the short-lived
  # re-exec parent, so recording it would leave a PID file that goes stale
  # immediately and defeat the idempotency check on the next boot.
  printf '%s\n' "$$" > "${PID_FILE}"
  echo "Daemon loop running (PID $$), interval ${SYNC_INTERVAL}s"
  while true; do
    # if/else keeps set -e from killing the loop on a failed cycle.
    # rc is captured in its own statement: a $(date) substitution in the
    # same word would clobber $? before it is expanded.
    if rclone "${SYNC_CMD}" "${SOURCE}" "${DEST}" ${AGE_OPTS} ${RCLONE_OPTS} ${DRY_RUN_FLAG} >> "${LOG_FILE}" 2>&1; then
      rc=0
    else
      rc=$?
    fi
    if [ "${rc}" -eq 0 ]; then
      echo "$(date -Is) ${SYNC_CMD} ok" >> "${LOG_FILE}"
    else
      echo "$(date -Is) ${SYNC_CMD} failed (exit ${rc})" >> "${LOG_FILE}"
    fi
    sleep "${SYNC_INTERVAL}"
  done
else
  rclone "${SYNC_CMD}" "${SOURCE}" "${DEST}" ${AGE_OPTS} ${RCLONE_OPTS} ${DRY_RUN_FLAG}
  echo "Sync complete."
fi
