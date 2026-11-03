#!/bin/bash
# Master-repo sync: this repo (FlexLLaVA/ + docs/) is the MASTER; the DAS-6 and hipster ~/FlexLLaVA clones are
# running copies and must always match it.
#
#   DAS-6   branch main     <- git tree  main:FlexLLaVA
#   hipster branch hipster  <- git tree  hipster:FlexLLaVA   (= main + the hipster-only launchers)
#
#   ./sync_clusters.sh status [das6|hipster|all]   what differs (exit 1 if anything does); changes nothing
#   ./sync_clusters.sh push   [das6|hipster|all]   make the cluster match master, commit + push there, tag master-sync
#   ./sync_clusters.sh pull   [das6|hipster|all]   copy cluster-side edits (hot fixes) to ./.cluster_pull/<cluster>/
#   ./sync_clusters.sh init   [das6|hipster|all]   record "cluster == master" (tag master-sync) after verifying it
#   add --dry-run to push to list the plan only; FORCE=1 overrides the cluster-edit guard (clobbers them!)
#
# Rules it enforces:
#  * Only COMMITTED master content is pushed (git archive), never the working tree.
#  * Files are replaced atomically (tmp + mv), so jobs that are running a script from the cluster checkout are safe.
#  * The cluster branch must be the expected one; cluster-side edits made since the last sync are NOT overwritten
#    (abort, run `pull`, fold them into master, then push).
#  * docs/SUBMITTED_RUNS.tsv is written by submit_elastic_run.sh on DAS-6, so it flows cluster -> master only:
#    `pull das6` refreshes both local copies (commit them), and push never touches it.
#  * The local hipster branch is kept = main + hipster-only files (plumbing merge; never switches your checkout).
#    To EDIT hipster-only launchers: git switch hipster; edit; commit; git switch main  (stay on main otherwise).
set -uo pipefail
export SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-$HOME/.ssh/agent.sock}"
cd "$(dirname "$0")"
CMD="${1:-status}"; WHICH="${2:-all}"; DRY=""; [ "${3:-}" = "--dry-run" ] || [ "${2:-}" = "--dry-run" ] && DRY=1
[ "$WHICH" = "--dry-run" ] && WHICH=all
LEDGER="docs/SUBMITTED_RUNS.tsv"          # cluster-written (DAS-6); path relative to FlexLLaVA/
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -o LogLevel=QUIET"

cfg() {   # $1=cluster -> HOST BR
    case "$1" in
        das6)    HOST=das6;    BR=main ;;
        hipster) HOST=hipster; BR=hipster ;;
        *) echo "unknown cluster '$1'"; exit 2 ;;
    esac
}
clusters() { [ "$WHICH" = all ] && echo "das6 hipster" || echo "$WHICH"; }

update_hipster_branch() {   # keep local hipster = main + hipster-only files, without touching the checkout
    git merge-base --is-ancestor main hipster 2>/dev/null && return 0
    [ "$(git branch --show-current)" = hipster ] && { echo "hipster is checked out: git switch main first (or git merge main)"; exit 2; }
    local base idx tmp T C f m h
    base=$(git merge-base main hipster)
    tmp=$(mktemp); idx=$(mktemp -u)
    git diff --name-only --diff-filter=AM "$base" hipster -- FlexLLaVA > "$tmp"
    GIT_INDEX_FILE="$idx" git read-tree main
    while read -r f; do
        read -r m _ h _ < <(git ls-tree hipster -- "$f")
        GIT_INDEX_FILE="$idx" git update-index --add --cacheinfo "$m,$h,$f"
    done < "$tmp"
    T=$(GIT_INDEX_FILE="$idx" git write-tree)
    C=$(git commit-tree "$T" -p hipster -p main -m "Merge main into hipster (sync_clusters.sh)" \
        -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>")
    git branch -f hipster "$C"; rm -f "$tmp" "$idx"
    echo "hipster branch advanced to include main ($(git rev-parse --short hipster))"
}

local_ready() {
    if ! git diff --quiet HEAD -- FlexLLaVA ":!FlexLLaVA/$LEDGER" ":!docs/SUBMITTED_RUNS.tsv"; then
        echo "master has UNCOMMITTED changes under FlexLLaVA/ -- commit first (only committed content is pushed):"
        git diff --stat HEAD -- FlexLLaVA ":!FlexLLaVA/$LEDGER" | tail -5; exit 2
    fi
}

# the program run on the cluster; args: BR MODE SHA
read -r -d '' REMOTE <<'EOF'
set -uo pipefail
BR=$1; MODE=$2; SHA=$3; EXCL=docs/SUBMITTED_RUNS.tsv; STAGE=$HOME/.sync_stage
cd ~/FlexLLaVA || exit 3
[ "$(git branch --show-current)" = "$BR" ] || { echo "cluster is on branch '$(git branch --show-current)', expected '$BR'"; exit 3; }
rm -f "$STAGE/$EXCL"
# cluster-side edits since the last sync (committed or not, plus new untracked files)
if git rev-parse -q --verify refs/tags/master-sync >/dev/null; then
    { git diff --name-only master-sync -- . ":!$EXCL"; git ls-files --others --exclude-standard; } | sort -u > /tmp/drift.$$
else : > /tmp/drift.$$; fi
# plan: stage vs cluster working tree
( cd "$STAGE" && find . -type f | sed 's#^\./##' | sort ) > /tmp/stage.$$
git ls-files | grep -vx "$EXCL" | sort > /tmp/tracked.$$
: > /tmp/changed.$$
while read -r f; do cmp -s "$STAGE/$f" "$f" 2>/dev/null || echo "$f"; done < /tmp/stage.$$ > /tmp/changed.$$
comm -13 /tmp/stage.$$ /tmp/tracked.$$ > /tmp/deleted.$$
echo "branch $BR @ $(git rev-parse --short HEAD): $(wc -l < /tmp/changed.$$) to write, $(wc -l < /tmp/deleted.$$) to delete, $(wc -l < /tmp/drift.$$) cluster-side edits since last sync"
head -15 /tmp/changed.$$ | sed 's/^/  write  /'; head -10 /tmp/deleted.$$ | sed 's/^/  delete /'
case "$MODE" in
  pull)
    cat /tmp/drift.$$ /tmp/changed.$$ | sort -u > /tmp/pull.$$
    # edits made on the cluster = drift only (files that differ from master for any other reason are just stale)
    [ -s /tmp/drift.$$ ] || echo "(no cluster-side edits recorded since the last sync)"
    tar -cf - -T /tmp/drift.$$ 2>/dev/null > "$HOME/.sync_pull.tar"; echo "PULLED $(wc -l < /tmp/drift.$$)"; exit 0 ;;
  status) [ -s /tmp/changed.$$ ] || [ -s /tmp/deleted.$$ ] && exit 1; exit 0 ;;
  init)
    if [ -s /tmp/changed.$$ ] || [ -s /tmp/deleted.$$ ]; then echo "cluster is NOT identical to master; cannot init (run push)"; exit 1; fi
    git tag -f master-sync HEAD >/dev/null; echo "master-sync tag set at $(git rev-parse --short HEAD)"; exit 0 ;;
  dry) exit 0 ;;
esac
if [ -s /tmp/drift.$$ ] && [ -z "${FORCE:-}" ]; then
    echo "ABORT: cluster-side edits since the last sync would be overwritten:"; sed 's/^/  /' /tmp/drift.$$ | head -20
    echo "run './sync_clusters.sh pull', fold them into master, commit, then push (or FORCE=1)"; exit 4
fi
while read -r f; do
    mkdir -p "$(dirname "$f")"; cp -p "$STAGE/$f" "$f.sync.$$" && mv -f "$f.sync.$$" "$f"
done < /tmp/changed.$$
while read -r f; do rm -f "$f"; done < /tmp/deleted.$$
git add -A
if ! git diff --cached --quiet; then
    git commit -q -m "Sync from master repo (Collaborative-VLM @ $SHA)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
    git push -q origin "$BR" || { echo "ABORT: push to origin/$BR failed (origin moved?); files are updated on disk and committed locally"; exit 5; }
fi
git tag -f master-sync HEAD >/dev/null
echo "DONE: $BR now $(git rev-parse --short HEAD), working tree == master"
EOF

run_remote() {   # $1=mode; stage first unless pull
    local mode="$1"
    if [ "$mode" != pull ]; then
        git -c core.autocrlf=false -c core.eol=lf archive --format=tar "$REF:FlexLLaVA" | $SSH "$HOST" 'rm -rf ~/.sync_stage && mkdir -p ~/.sync_stage && tar -x -C ~/.sync_stage' || { echo "staging to $HOST failed"; return 3; }
    fi
    printf '%s\n' "$REMOTE" | FORCE="${FORCE:-}" $SSH "$HOST" "FORCE='${FORCE:-}' bash -s -- '$BR' '$mode' '$(git rev-parse --short main)'" 2>&1 | grep -v "^\*\*\|^$\|Welcome\|Faculty\|Only AUTH\|All actions\|University\|Policy\|feiog\|mailto\|^ *(http"
    return "${PIPESTATUS[1]}"
}

rc_all=0
case "$CMD" in
    status|push|init|pull) ;;
    *) sed -n 2,22p "$0"; exit 2 ;;
esac
[ "$CMD" = push ] && { local_ready; git push -q origin main 2>/dev/null; }
for c in $(clusters); do
    cfg "$c"; REF="$BR"
    [ "$c" = hipster ] && { update_hipster_branch; git push -q origin hipster 2>/dev/null; }
    echo "== $c ($HOST, branch $BR, master ref $REF @ $(git rev-parse --short "$REF")) =="
    mode="$CMD"; [ "$CMD" = push ] && [ -n "$DRY" ] && mode=dry
    run_remote "$mode"; rc=$?
    if [ "$CMD" = pull ] && [ $rc -eq 0 ]; then
        rm -rf ".cluster_pull/$c"; mkdir -p ".cluster_pull/$c"
        $SSH "$HOST" 'cat ~/.sync_pull.tar' 2>/dev/null | tar -x -C ".cluster_pull/$c" 2>/dev/null
        echo "cluster-side edits copied to .cluster_pull/$c/  (diff against FlexLLaVA/, fold into master, commit)"
    fi
    [ "$CMD" = pull ] && [ "$c" = das6 ] && $SSH das6 "cat ~/FlexLLaVA/$LEDGER" > "FlexLLaVA/$LEDGER.new" 2>/dev/null \
        && mv -f "FlexLLaVA/$LEDGER.new" "FlexLLaVA/$LEDGER" && cp "FlexLLaVA/$LEDGER" "$LEDGER" \
        && echo "ledger refreshed from DAS-6 (FlexLLaVA/$LEDGER and $LEDGER): commit it"
    [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
