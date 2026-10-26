#!/bin/bash
# Pull DAS-6 ~/FlexLLaVA/jobs (Slurm outputs) into FlexLLaVA/jobs, fetching files that are new or whose size changed
# (running jobs keep growing). jobs/ is gitignored upstream, so this stays local and is never committed.
export SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-$HOME/.ssh/agent.sock}"
DEST="$(cd "$(dirname "$0")" && pwd)/FlexLLaVA/jobs"
mkdir -p "$DEST"
ssh -o BatchMode=yes das6 'cd ~/FlexLLaVA/jobs && find . -type f -printf "%P\t%s\n"' 2>/dev/null |
while IFS=$'\t' read -r f s; do
    [ "$(stat -c %s "$DEST/$f" 2>/dev/null)" = "$s" ] || printf '%s\0' "$f"
done > /tmp/das6_jobs_need.lst
n=$(tr -cd '\0' < /tmp/das6_jobs_need.lst | wc -c)
[ "$n" -gt 0 ] && ssh -o BatchMode=yes das6 'cd ~/FlexLLaVA/jobs && tar --null -T - -cf -' < /tmp/das6_jobs_need.lst 2>/dev/null | tar -xf - -C "$DEST"
echo "synced $n job files into $DEST"
