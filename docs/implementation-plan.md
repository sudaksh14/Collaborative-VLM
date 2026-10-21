# Implementation plan

Part of the [project overview](README.md). **[PROPOSED]** unless tagged **[FACT]** (verified in the repo this session). Nothing below has been implemented. Written 2026-10-21.

## 1. Current code, as verified

**[FACT]** The repo is the M3/LLaVA-1.5 codebase plus an elastic pipeline (details in [ELASTIC_PIPELINE.md](../FlexLLaVA/docs/ELASTIC_PIPELINE.md)):
- `llava/model/elastic/engine.py` (`ElasticEngine.reduce_tokens`, `attach_elastic_engine`), `resampler.py` (`NestedQueryResampler`, `_forward_pool_anchored`, `pool_anchors`, `n_anchors_for`), `nested_lora.py`, `config.py` (`ElasticConfig`), `losses.py`, `kd_teachers.py`, `flops.py`.
- `llava/train/train_elastic.py` (launcher), `llava/model/language_model/llava_elastic_mixin.py` (training loop over token levels).
- Eval: `lmms-eval/lmms_eval/models/llava_elastic.py` (fixed `tok_level` per run; sets `config.matryoshka_vis_token_scale`), `llava/eval/efficiency/` (analytic roofline: `analyzer.py`, `hardware.py`, `roofline.py`), `scripts/export_eval_csv.py`.
- Launchers: `submit_elastic_run.sh` (e.g. `final-parcel`), `run_matrix_hipster.sh`, SLURM job scripts, `eval_lmms_*.sh`.

**[FACT] Not present:** a Jetson measurement harness (`tegrastats`/INA3221 logging, per-stage timers), raw measurement logs, a `jobs/` directory (referenced from README and docs), any crop/ROI code, any per-query budget selection, any MCU code.

## 2. Target architecture [PROPOSED]

```
 Camera ──► [MCU: STM32H747]                         [SoC: Jetson Orin Nano]
            frame buffer (SDRAM, 1080p)               text embedder (Model2Vec, CPU)
            128² thumbnail ───┐                              │ question embedding (256-d)
            MGround (INT8) ◄──┼──────────────────────────────┘
              │ ROI centre, crop class, confidence
            Crop Engine ──► crop bytes (+ thumbnail?) ──────► controller ──► (c, r, n)
                                                                   │
                                                  vision tower @ r (CLIP + nested LoRA)
                                                       │  patch grid (variable)
                                             NTR: anchors + LDQs @ n   [option C: anchors from thumbnail]
                                                       │
                                                  MLP → small LM → answer
```

- **Controller** (new, host): inputs = question features, MGround confidence signals (crop-size logit, |centre_visual − centre_prior|), measured load (GPU util, co-runner latency, thermal), SLO/energy budget; output = (c, r, n) or "bypass crop".
- **ROI stage** is a separable module with a stable interface (question embedding in; ROI + confidence out) so a GPU-side stand-in can replace the MCU for ablations.

## 3. Work packages

### WP0: Measurement harness and reproducibility foundation (do first; also fixes existing gaps)
| Task | Details | Done when |
|---|---|---|
| WP0.1 Jetson runner | `tools/jetson/run_bench.py`: loads a checkpoint, runs N queries with CUDA-event stage timers (encoder / resampler+projector / prefill / decode), logs JSONL per query | Reproduces paper Table V TinyLlama rows within ±10 % (or discrepancy documented) |
| WP0.2 Power logger | Parse `tegrastats --interval 100` including INA3221 rails (VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0); idle-floor measurement step; clock/thermal log; energy integration net of idle | Energy per query CSV + raw log |
| WP0.3 Device state | Script to set `nvpmodel`/`jetson_clocks`, fixed fan, record JetPack/L4T/driver/torch versions | Written into every run manifest |
| WP0.4 Manifest | Extend `run_manifest.json` (exists per journal decision 38) with device state, seeds, dataset hashes, command line | Every result file links to a manifest |
| WP0.5 Restore `jobs/` | README/docs reference `jobs/*.sh`; directory missing in this clone. Recover from history/cluster or rewrite and test | `jobs/` exists, referenced scripts run |
| WP0.6 De-hardcode paths | Replace `/var/scratch/skalra/...` with env vars (`FLEX_DATA`, `FLEX_CKPT`) in scripts and CSV generators | Fresh-clone dry-run works |
| WP0.7 Fail-fast jobs | Add `set -euo pipefail` to `run_job_slm.sh` and friends; verify checkpoint exists before launching dependent eval (documented hazard in ELASTIC_PIPELINE §10) | Failed training no longer reports success |
| WP0.8 Replace analytic Jetson columns | In `results/*.csv` keep roofline numbers but rename columns `jetson_est_*`; add measured columns from WP0.1 | No estimate masquerades as measurement |
| WP0.9 Tests | CPU unit tests for: pool_anchors counts, nested-LoRA prefix identity, resampler shapes at several grids (extend existing `jobs/test_*.sh` ideas) | CI on CPU passes |

### WP1: ROI model (MGround)
- New package `llava/roi/`: `mground.py` (PyTorch; QSG gate `x*σ(W[GAP(x);e]+b)`, stem 3→32 7×7/4, stages 32→64→128, centre head 384→128→2, prior head 256→128→64→2, learnable blend, crop-class head 256→1); `labels.py` (teacher prompt → box → centre; rule-based crop class); `train_mground.py` (AdamW 1e-3, cosine, bs 64, ≤200 epochs, early stop 40, colour-jitter/flip/shift aug, as CoInfer Sec. IV-A); `eval_roi.py` (L1, IoU, evidence coverage).
- **[UNVERIFIED]** CoInfer does not give exact kernel sizes/channels beyond the text; target ≈276 K parameters and compare.
- Quantisation: PTQ then QAT; export to TFLite INT8; TFLM firmware (separate repo/folder `firmware/mground/`): Crop Engine, SDRAM buffers, USB/UART transfer, interrupt-driven protocol as in CoInfer Algorithm 1.
- Text embedder: Model2Vec 256-d on host CPU; measure CPU cost on Orin Nano.
- Label generation reuses the repo's KD-teacher plumbing where convenient (`kd_teachers.py`); teacher choice per [experimental-plan.md](experimental-plan.md) §4.

### WP2: Crop-aware, variable-resolution nested VLM
| Task | Details / risk |
|---|---|
| WP2.1 Variable grid in vision tower | `CLIPVisionTower._encode` (`llava/model/multimodal_encoder/clip_encoder.py`) currently assumes 336² / 576 patches. Add `interpolate_pos_encoding` path **[UNVERIFIED]** that the pinned `transformers` version supports it; else interpolate manually (`resampler.py:134` has `_interp_pos_embed` for the resampler's own embeddings) |
| WP2.2 Resampler generalisation | `NestedQueryResampler(num_patches=576)`, `valid_anchor_counts`, `patch_pos_embed[:, :P]`, `n_anchors_for`: make grid size a runtime argument; anchors snapped to pooling-reachable counts for grids like 16×16; `pool_anchors_adaptive` (exists) may already cover non-square counts |
| WP2.3 Crop augmentation in Stage 2 | New dataset transform producing random/teacher-box crops of area {35,65,100} % with the corresponding question; mix ratio configurable; keep full-frame majority to avoid regressing Table III |
| WP2.4 Resolution axis in nesting | Extend `ElasticConfig` with `res_levels`; decide pairing with LoRA rank (SSA currently pairs rank with budget); training loop in `llava_elastic_mixin.py` samples (n, r) pairs; `single_budget_per_step` already exists (journal) as precedent |
| WP2.5 Option C (thumbnail anchors + crop LDQs) | Two-stream forward: encode thumbnail at low res → anchors; encode crop at r → LDQs cross-attend; the self-attention over `[A; Q]` already exists in `_forward_pool_anchored`. Highest risk; gate on pilot results |
| WP2.6 Regression guard | After any change, rerun Table III benchmarks at c=100 %, r=336 and require ≤1 pp loss (and report) |
| WP2.7 Seeds | Train ≥3 Stage-2 seeds for the primary backbone; share Stage-1 if needed (state it) |

### WP3: Runtime and controller
- `llava/runtime/` (or `tools/runtime/`): inference server on the Orin Nano; accepts (question, frame/crop, config); returns answer + timing. Avoid HF `generate` overhead hiding encoder time; consider TensorRT/ONNX only if needed — **keep PyTorch FP16 for comparability with the paper unless a baseline requires otherwise**, and state the stack.
- `controller/`: (1) static table lookup; (2) learned policy (gradient-boosted trees or small MLP) on offline logs; (3) feedback controller on measured load. Decision latency must be reported and ≪ TTFT.
- Co-runner harness: YOLOv8n and DeepLabV3 loops with fixed FPS and latency logging, plus a load trace generator.

### WP4: Evaluation plumbing
- Extend `lmms-eval/lmms_eval/models/llava_elastic.py` (currently reads one `tok_level`, lines ~198–200) to accept per-sample config (c, r, n) and a ROI-provider (file of precomputed ROIs for offline eval; live MCU for system runs).
- Dataset builders: GQA-Outdoor (`scripts/build_gqa_outdoor.py`, CoInfer App. A), W5 public 1080p set builder with licence checks, question-type tagging.
- Metrics: official evaluators; additional exact-match/LLM-judge on W1/W5; paired-bootstrap utilities (`tools/stats/paired_bootstrap.py`); one results schema (long-format CSV: run_id, model, config, sample_id, correct, ttft_ms, energy_mj, …).
- Plotting scripts generating every paper figure from CSVs; figures regenerate with one command.

### WP5: Packaging and anonymity
- Anonymised mirror script: strips author metadata, rewrites remote, removes cluster paths and tokens; checks for secrets (W&B key removal is documented in journal §20 — verify nothing similar remains; never commit `.hf_token`, `_netrc`).
- `REPRODUCE.md` with exact commands per table/figure; pinned `environment.yml` and Jetson JetPack notes; checkpoint hosting plan (anonymous Zenodo or similar; the repo already has a SURFdrive upload script, which is not anonymous).

## 4. Milestones

| ID | Milestone | Depends on | Acceptance test | Target |
|---|---|---|---|---|
| M0 | Harness + pilot (H1, oracle crop) | WP0.1–0.4 | Table V reproduced ±10 % or gap explained; pilot numbers | 2026-10-27 |
| M1 | Repo hygiene (WP0.5–0.9) | — | Fresh-clone dry run; CPU tests pass | 2026-11 |
| M2 | MGround FP32 + INT8 | WP1 | L1 ≤ ~0.30 testdev (CoInfer 0.296); INT8 loss ≤ 1 pp-equivalent; TFLM latency/flash/SRAM measured | 2026-11/12 |
| M3 | Crop-aware nested model | WP2 | Table III regressions ≤1 pp; crop gain on detail questions measured | 2027-01 |
| M4 | System measurements | M0, M2, M3, WP3 | H2, H5, H6, H9 tables with CIs | 2027-02 |
| M5 | Controller | M4 | H7 evaluation | 2027-03 |
| M6 | Artifact + draft | all | `REPRODUCE.md` verified by someone who did not write it | 2027-04 |

## 5. Reproducibility checklist

- [ ] Every number in the paper traceable to a run manifest (git hash, config, seed, device state).
- [ ] Raw per-query logs (latency stages, power, correctness) archived and released.
- [ ] Device state recorded: JetPack/L4T, `nvpmodel` mode, clocks, fan, ambient temp, power supply.
- [ ] Software pins: torch, transformers (CoInfer used 4.56.1), CUDA, lmms-eval commit; `environment.yml`.
- [ ] Seeds: ≥3 for training; fixed for eval; data order fixed.
- [ ] Dataset construction scripts and hashes (GQA-Outdoor keyword list; W5 sources + licences).
- [ ] Baselines run through the same harness, same prompt, same decoding; baseline weights named and hashed.
- [ ] Analytic (roofline) numbers labelled as estimates everywhere.
- [ ] Statistical scripts committed; CIs in all tables.
- [ ] Negative/failed runs logged (the journal is a good model; keep it).
- [ ] Anonymised mirror builds from a clean checkout; no secrets, usernames, cluster paths.
- [ ] CoInfer baseline provenance stated (re-implemented; no official code found).

## 6. Deployment considerations

- **Memory:** unified 8 GB on the Orin Nano; Nest-VLM-TinyLlama peaks ≈3.05 GB (paper Table V); adding the thumbnail path, text embedder, co-runners (YOLO, DeepLabV3) and CUDA context must be re-measured, not assumed.
- **Link:** USB crop transfer took 57 ms (small) to 149 ms (full 1080p) in CoInfer's prototype (Sec. IV-C); production cameras (CSI/MIPI) may bypass an MCU entirely. Treat as an experimental variable (A10) and discuss honestly.
- **Safety:** ROI misses cause silent degradation; ship the confidence-gated fallback and report miss-conditioned accuracy.
- **Privacy/ethics:** public 1080p data only; no faces/licence plates in released sets unless the source licence permits; the CFP requires a human-subjects statement only if participants are involved.
- **Updates:** per-site MGround retraining is cheap in CoInfer's account; document the retraining recipe and its label-generation cost (teacher inference).
