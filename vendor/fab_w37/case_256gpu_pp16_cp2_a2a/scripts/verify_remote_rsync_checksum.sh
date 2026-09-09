#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 REMOTE REMOTE_ROOT FILE_LIST LOCAL_ROOT [SSH_PORT]" >&2
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage
  exit 2
fi

remote=$1
remote_root=$2
file_list=$3
local_root=$4
ssh_port=${5:-22}

if [[ ! -s "$file_list" ]]; then
  echo "FILE_LIST does not exist or is empty: $file_list" >&2
  exit 2
fi
if [[ ! -d "$local_root" ]]; then
  echo "LOCAL_ROOT does not exist: $local_root" >&2
  exit 2
fi

temporary_report=$(mktemp)
trap 'rm -f -- "$temporary_report"' EXIT
ssh_command="ssh -p $ssh_port -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

rsync \
  -rtcn \
  --protect-args \
  --relative \
  --files-from="$file_list" \
  --itemize-changes \
  --timeout=120 \
  --contimeout=30 \
  -e "$ssh_command" \
  "$remote:${remote_root%/}/" \
  "$local_root/" > "$temporary_report"

if [[ -s "$temporary_report" ]]; then
  echo "Checksum verification FAILED. Differing or missing paths:" >&2
  sed -n '1,200p' "$temporary_report" >&2
  exit 1
fi

echo "Checksum verification PASS: remote and local selected files match."
