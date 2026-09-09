#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 REMOTE REMOTE_ROOT FILE_LIST DEST_ROOT [SSH_PORT] [BWLIMIT_KIBPS]" >&2
  echo "Files are written under DEST_ROOT/.incoming with their relative paths preserved." >&2
}

if [[ $# -lt 4 || $# -gt 6 ]]; then
  usage
  exit 2
fi

remote=$1
remote_root=$2
file_list=$3
dest_root=$4
ssh_port=${5:-22}
bwlimit_kibps=${6:-0}

if [[ "$remote_root" != /* ]]; then
  echo "REMOTE_ROOT must be absolute: $remote_root" >&2
  exit 2
fi
if [[ ! -s "$file_list" ]]; then
  echo "FILE_LIST does not exist or is empty: $file_list" >&2
  exit 2
fi
if awk '
  /^\// { bad=1 }
  /(^|\/)\.\.(\/|$)/ { bad=1 }
  END { exit bad ? 0 : 1 }
' "$file_list"; then
  echo "FILE_LIST contains an absolute path or '..': $file_list" >&2
  exit 2
fi

incoming="$dest_root/.incoming"
state_dir="$dest_root/manifests/transfer_state"
mkdir -p -- "$incoming" "$state_dir"

list_name=$(basename -- "$file_list")
timestamp=$(date -u '+%Y%m%dT%H%M%SZ')
log_file="$state_dir/${list_name}.${timestamp}.rsync.log"
complete_marker="$state_dir/${list_name}.transfer_complete"

ssh_command="ssh -p $ssh_port -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
rsync_args=(
  -rt
  --protect-args
  --relative
  --files-from="$file_list"
  --partial
  --partial-dir=.rsync-partial
  --info=progress2,stats2
  --human-readable
  --timeout=120
  --contimeout=30
  --log-file="$log_file"
  -e "$ssh_command"
)

if [[ "$bwlimit_kibps" != "0" ]]; then
  if [[ ! "$bwlimit_kibps" =~ ^[0-9]+$ ]]; then
    echo "BWLIMIT_KIBPS must be a non-negative integer" >&2
    exit 2
  fi
  rsync_args+=("--bwlimit=$bwlimit_kibps")
fi

echo "Pulling files listed in $file_list"
echo "Destination staging root: $incoming"
echo "Rsync log: $log_file"
rsync "${rsync_args[@]}" "$remote:${remote_root%/}/" "$incoming/"

{
  printf 'completed_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'remote=%s\n' "$remote"
  printf 'remote_root=%s\n' "$remote_root"
  printf 'file_list=%s\n' "$file_list"
} > "$complete_marker"
echo "Transfer batch completed. Size/checksum verification is still required."
