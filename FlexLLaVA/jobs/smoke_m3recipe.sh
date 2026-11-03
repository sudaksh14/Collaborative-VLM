#!/bin/bash
#SBATCH --job-name=smoke_m3recipe
#SBATCH -t 02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1
#SBATCH --nodelist=node206
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/skalra/FlexLLaVA/jobs/smoke_m3recipe_%j.out
# Gate for the two M3-recipe runs of docs/brainstorm.md section 18 (parcel-exact-m3recipe, m3-pool-lora):
# 8 real steps per stage (real LLaVA data, one GPU, the recipes' exact flags via RUN_INLINE), checks finite
# losses and the saved config, then a 16-sample POPE generation smoke at 256 and 16 tokens from m3-pool-lora
# (exercises project_before_pool through the eval wrapper). Submit the real runs with --dependency=afterok:<this>.
set -uo pipefail
module load cuda12.6/toolkit/12.6
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
cd /home/skalra/FlexLLaVA
export HF_HOME=/var/scratch/skalra/.cache/huggingface WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUM_GPUS=1 MAX_STEPS=8 TAG_SUFFIX=-smoke RUN_INLINE=1
CK=/var/scratch/skalra/flexllava/checkpoints
fail=0
for RECIPE in parcel-exact-m3recipe m3-pool-lora; do
    LOG=/home/skalra/FlexLLaVA/jobs/smoke_m3recipe_${RECIPE}_$SLURM_JOB_ID.log
    echo "=== $RECIPE ==="
    SLM_KEY=tinyllama bash submit_elastic_run.sh "$RECIPE" 2>&1 | tee "$LOG" | tail -5
    rc=${PIPESTATUS[0]}
    echo "$RECIPE exit=$rc"
    [ "$rc" -ne 0 ] && fail=1
    nonfinite=$(grep -ciE "\bnan\b|non-finite|\binf\b" "$LOG")
    echo "$RECIPE non-finite lines: $nonfinite"
    [ "$nonfinite" -gt 0 ] && fail=1
    python3 - "$LOG" <<'PYEOF' || fail=1
import re, sys
txt = open(sys.argv[1], errors="ignore").read()
losses = [float(m) for m in re.findall(r"'loss': ([0-9.]+)", txt)]
print(f"  logged losses: n={len(losses)} first={losses[:1]} last={losses[-1:]}")
if len(losses) < 6 or any(l != l or l > 1e3 for l in losses):
    print("  FAIL: too few or non-finite losses"); sys.exit(1)
PYEOF
    python3 - "$CK/elastic-finetune-tinyllama-${RECIPE}-smoke" "$RECIPE" <<'PYEOF' || fail=1
import json, sys
d, recipe = sys.argv[1], sys.argv[2]
m = json.load(open(f"{d}/run_manifest.json"))
c = json.load(open(f"{d}/elastic_config.json"))
print("  manifest: arch", m["resampler"]["arch"], "lr", m["learning_rate"], "kd", m["kd"]["use_prefix_kl"])
assert c["resampler_arch"] == "parcel_exact" and float(m["learning_rate"]) == 2e-5, m["learning_rate"]
assert c["single_budget_per_step"] is False and c["budget_sampling"] == "ladder"
assert m["kd"]["use_prefix_kl"] is False and m["coral"]["enabled"] is False
routing = {int(k): v for k, v in c["anchor_routing"].items()}
if recipe == "m3-pool-lora":
    assert routing == {16: 16, 64: 64, 144: 144, 256: 256} and c["project_before_pool"] is True and c["use_lora"] is True
else:
    assert routing == {16: 16, 64: 64} and not c.get("project_before_pool") and c["use_lora"] is False
print("  config OK:", routing, "project_before_pool", c.get("project_before_pool"))
PYEOF
done

echo "=== eval smoke (m3-pool-lora, 16 samples of POPE at 256 and 16 tokens) ==="
EVALROOT=/var/scratch/skalra/flexllava/eval_logs/elastic-finetune-tinyllama-m3-pool-lora-smoke
rm -rf "$EVALROOT"
for IDX in 0 3; do
    SLURM_ARRAY_TASK_ID=$IDX EVAL_LIMIT=16 TASKS=pope bash eval_lmms_level.sh \
        "$CK/elastic-finetune-tinyllama-m3-pool-lora-smoke" 2>&1 | tee "/home/skalra/FlexLLaVA/jobs/smoke_m3recipe_eval${IDX}_$SLURM_JOB_ID.log" | tail -3
    n=$(grep -c "in generating" "/home/skalra/FlexLLaVA/jobs/smoke_m3recipe_eval${IDX}_$SLURM_JOB_ID.log")
    echo "eval idx $IDX generation errors: $n"
    [ "$n" -gt 0 ] && fail=1
    ls "$EVALROOT"/*tok/*results.json >/dev/null 2>&1 || ls "$EVALROOT"/*tok/*/*results.json >/dev/null 2>&1 || { echo "FAIL: no results for idx $IDX"; fail=1; }
done

if [ "$fail" -eq 0 ]; then echo "SMOKE OK"; else echo "SMOKE FAILED"; exit 1; fi
