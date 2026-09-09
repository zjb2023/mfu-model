#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 REMOTE REMOTE_ROOT OUTPUT_TSV [SSH_PORT]" >&2
  echo "Example: $0 user@jump /data/0731 /local/manifests/remote_inventory.tsv 22" >&2
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
  exit 2
fi

remote=$1
remote_root=$2
output_tsv=$3
ssh_port=${4:-22}

if [[ "$remote_root" != /* ]]; then
  echo "REMOTE_ROOT must be an absolute path: $remote_root" >&2
  exit 2
fi

output_dir=$(dirname -- "$output_tsv")
mkdir -p -- "$output_dir"
temporary_output="${output_tsv}.tmp.$$"
trap 'rm -f -- "$temporary_output"' EXIT

ssh \
  -p "$ssh_port" \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  "$remote" \
  bash -s -- "$remote_root" > "$temporary_output" <<'REMOTE_SCRIPT'
set -euo pipefail
root=$1
if [[ ! -d "$root" ]]; then
  echo "Remote root does not exist or is not a directory: $root" >&2
  exit 3
fi
printf 'relative_path\tsize_bytes\tmtime_epoch\n'
find "$root" -type f -printf '%P\t%s\t%T@\n' | LC_ALL=C sort
REMOTE_SCRIPT

mv -- "$temporary_output" "$output_tsv"
trap - EXIT
echo "Inventory written to $output_tsv"
