# Experimental plan

Part of the [project overview](README.md). Everything here is **[PROPOSED]**; no experiment in this document has been run. Thresholds marked "target" are my guesses to make hypotheses falsifiable, not results. Facts cited come from [analysis.md](analysis.md). Written 2026-10-21.

## 1. Notation

- **c**: crop extent ∈ {35 %, 65 %, 100 %} of image area (CoInfer's two classes plus full frame).
- **r**: encoder input resolution ∈ {336², 224²} (patches 576 / 256); more values optional.
- **n**: nested visual-token budget ∈ {256, 144, 64, 16} (Nest-VLM ladder).
- **Configuration** = (c, r, n). The pipeline variants are **N** (Nest-VLM only: c=100 %, r=336), **C** (CoInfer-style crop on a non-elastic VLM), **C+N** (proposed).

## 2. Hypotheses (measurable, falsifiable)

Each states the metric, comparison, and the result that would refute it.

| ID | Hypothesis | Measure | Refuted if |
|---|---|---|---|
| H1 | **Encoder-bound floor.** On Orin Nano (15 W), for Nest-VLM-TinyLlama the vision encoder + fixed overhead accounts for ≥50 % of TTFT at n=16. | Per-stage timing (CUDA events) of encoder / resampler / projector / prefill, 100+ runs | Encoder + fixed share < 50 % at n=16 |
| H2 | **Pareto shift.** For an (accuracy, TTFT) Pareto front on detail-sensitive tasks (TextVQA, V\*-type high-res set, GQA-Outdoor at 1080p input), C+N dominates N: at matched TTFT, C+N gains ≥ +3 pp accuracy (target), or at matched accuracy it cuts TTFT ≥ 25 % (target), with 95 % CIs excluding 0. | Accuracy (official evaluator) vs p50 TTFT, paired bootstrap | No C+N point beats the N front by the target margin after including the MCU/link overhead |
| H3 | **Zoom effect.** With a question-aware crop, accuracy loss from n=256→64 is ≥ 50 % smaller than without crop on attribute/detail questions. | Δacc(256→64) with c=35 % vs c=100 %, per question type | Δ is not smaller (or CI overlaps) |
| H4 | **Transfer.** An ROI model trained with labels from one teacher and applied to Nest-VLM (unseen at ROI training time) loses ≤ 1 pp vs full-frame at the same n on GQA-Outdoor (CoInfer reports −0.4 pp for Qwen2-VL, Fig. 3; this tests it on a CLIP/TinyLlama stack). | Accuracy, paired | Loss > 1 pp with CI excluding 1 |
| H5 | **Contention.** Under co-located YOLOv8n + DeepLabV3 on the same GPU, C+N lowers (a) VLM p95 TTFT ≥ 30 % (target) and (b) co-runner p95 latency ≥ 30 % (target) vs N at equal accuracy. | p95 over ≥ 200 queries, ≥ 3 repetitions | Gains below target or not significant |
| H6 | **Sensor stage is not free on small SoCs.** CoInfer's MCU overhead (≈157 ms) is ≥ 25 % of the C+N TTFT at n ≤ 64 on Orin Nano; thus the policy must sometimes bypass the crop. | MCU-measured latency + link time vs TTFT | Overhead < 25 % of TTFT |
| H7 | **Controller.** A load- and question-aware controller attains ≥ 90 % SLO attainment (SLO = p95 TTFT bound, set per scenario) at ≤ the mean energy of the best *static* configuration with equal attainment, and is within 10 % of the oracle's accuracy. | SLO attainment, energy, accuracy | Static config matches the controller within noise |
| H8 | **Anchor handling under crop (design question).** Global anchors from the thumbnail + LDQs from the crop (option C) beat sequential crop→NTR on relational/global questions by ≥ 2 pp while within 1 pp on attribute questions. | Per-question-type accuracy | No gain, or loss on attribute questions |
| H9 | **Multi-camera scaling.** With k ∈ {1,2,4} simultaneous views, aggregate TTFT and energy of C+N grow sub-linearly relative to N (CoInfer shows host grounding grows to 842 ms at 8 views, MCU flat at 157 ms). | Aggregate latency vs k | Same slope as N |

Null results are publishable as characterisation (see [paper-plan.md](paper-plan.md) §3 fallback).

## 3. Workloads and datasets

| ID | Dataset | Role | Notes (availability/licence flagged) |
|---|---|---|---|
| W1 | **GQA-Outdoor** (rebuild from GQA via CoInfer App. A: ≥3 of 47 outdoor keywords per image; train/val/testdev = 12,505/1,418/143 images) | ROI training and in-domain eval; reproduces CoInfer's protocol | GQA licence applies; keyword list is in CoInfer App. A. Images are low-res; use for accuracy, not for latency |
| W2 | **TextVQA val** | Detail-sensitive; Nest-VLM already reports it (28.47@256) | Official evaluator (OCR-token matching) |
| W3 | **POPE, MME-P, SciQA-img, GQA testdev-balanced** | Regression check vs Nest-VLM Table III; confirm no collapse | Use the repo's `eval_lmms_*.sh` / lmms-eval |
| W4 | **High-resolution VQA** (candidates: V\* Bench, HR-Bench; **[UNVERIFIED]** availability, licence, size—check before committing) | Where crop+budget should matter most; the ≈1080p regime | Choose ≥1 public set with ≥1080p images |
| W5 | **Public 1080p CPS-style set** (to be built): short-answer questions over 1080p robot/street/industrial frames from public datasets (**[UNVERIFIED]** candidates such as BDD100K-like driving or public surveillance data; check licences), labelled with teacher (Qwen2.5-VL-7B) answers + a human-verified subset (≥300 QA) | Substitute for CoInfer's proprietary industrial set; latency/energy and contention runs use these images | Document labelling protocol; release anonymously |
| W6 | **Multi-camera replay**: k=1,2,4 views per query drawn from W5 | H9 | CoInfer's multi-image setting (Sec. IV-C3) |
| W7 | **Contention generators**: YOLOv8n @ fixed FPS, DeepLabV3, CPU stress on N−3 cores (as CoInfer Sec. IV-C2) plus a *ramped* load trace for the controller | H5, H7 | Fix FPS and model versions; log co-runner latency |

Question-type labels for H3/H8: use CoInfer's rule (object/attribute → small; relational/global → large) and GQA's structural types where available; report per type with counts.

## 4. Models

| Model | Purpose | Notes |
|---|---|---|
| Nest-VLM-TinyLlama-1.1B (repo `final-parcel`, ckpt `elastic-finetune-tinyllama-final-parcel`; paper's primary) | Main subject | Existing checkpoint; budgets 256/144/64/16 |
| Nest-VLM-SmolLM2-1.7B, MobileLLaMA-1.4B | Generality (paper has all three) | Existing checkpoints per journal; confirm they are on disk |
| **Nest-VLM+crop (new)**: same architecture, Stage-2 fine-tuned with crop augmentation and (optionally) 224² encoder resolution with interpolated positional embeddings | C+N | Training budget: warm start from Stage-2 ckpt, ≈ a fraction of 85 GPU-h (to be measured) |
| MGround (re-implemented; 276 K params; INT8) trained on W1 train + teacher labels; plus GPU-side FP32 version for ablation | ROI stage | No official code (analysis.md §2.5) |
| Teacher: Qwen2.5-VL-7B-Instruct (as in CoInfer) and/or LLaVA-1.5-7B | ROI labels | **[UNVERIFIED]** that LLaVA-7B boxes are usable; test label quality on 100 hand-checked samples |
| TinyLLaVA (CLIP/TinyLlama, 576 tokens), MobileVLM-arch (CLIP/MobileLLaMA, 144) | Non-elastic small-VLM references (paper Table V rows) | Rebuild as in the paper; report exactly which weights |
| Qwen2-VL-2B (+ MGround) | Reproduce CoInfer's headline number on W1 as a sanity check of our MGround | Fits Orin Nano? **[UNVERIFIED]**; run on a larger board if not |

## 5. Hardware

| Device | Role | Notes |
|---|---|---|
| **Jetson Orin Nano 8 GB Super**, 15 W, JetPack 6.2.1, GPU clock fixed (paper: 0.61 GHz) | Primary target; Nest-VLM's platform | Lock clocks/power mode (`nvpmodel`, `jetson_clocks`); record thermals |
| **STM32H747I-DISCO** (Cortex-M7 480 MHz, 1 MB flash, 512 kB D1 SRAM, external SDRAM 32 MB), TFLite-Micro | Sensor stage | Procurement/lead time **[UNVERIFIED]**. Link: USB as in CoInfer; log actual throughput |
| Jetson AGX Orin 32 GB | CoInfer's platform; cross-check vs published numbers | Only if available |
| Training: 2×A10 24 GB (paper) / the SLURM cluster used in the journal | Stage-2 fine-tune of C+N, MGround training | MGround fits any GPU |
| Power: on-board INA3221 via `tegrastats`; cross-check with an external meter on one run | Energy | Net of idle floor; also log absolute |

**Fallback if no MCU board:** emulate the sensor stage on an isolated CPU core of the Orin Nano (cgroup/affinity-pinned) with MGround INT8, and report MCU numbers from CoInfer's paper (157 ms) as a *labelled assumption*; state clearly that "load isolation" is then emulated, not demonstrated.

## 6. Baselines (fair-comparison rules)

Rules: identical precision (FP16 unless the baseline is defined otherwise), identical prompt template and max-new-tokens, identical decoding (greedy), identical image set and order, identical evaluator, same device clocks, warm-up discarded, interleaved run order (ABBA) to cancel thermal drift. Report parameter counts and peak memory for each. Retrain baselines with the same data/epochs where they require training.

| ID | Baseline | Why |
|---|---|---|
| B0 | Nest-VLM n=256, full frame downscaled to 336² | Reference operating point |
| B1 | Nest-VLM at n ∈ {144,64,16} (N) | The elastic-only front |
| B2 | TinyLLaVA / MobileVLM-arch, native tokens | Non-elastic small VLMs (paper's baselines) |
| B3 | Dense (non-elastic) TinyLlama VLM + CoInfer-style crop (C) | Isolates crop without elasticity |
| B4 | Nest-VLM + **centre crop** and **random crop** (35/65 %) | Controls for "any crop helps" (CoInfer's fixed-area baselines) |
| B5 | Nest-VLM + **teacher-ROI crop** (upper bound) | ROI quality ceiling |
| B6 | Training-free token pruning on the dense small VLM: FastV, SparseVLM, LLaVA-PruMerge at matched token counts | Alternative reduction (CoInfer's baselines) |
| B7 | Nest-VLM + GPU-resident cropper (CropVLM-style small model **[UNVERIFIED]** feasibility; or MGround-on-GPU) | Isolates *placement* (MCU vs host) from *method* |
| B8 | Static best configuration per scenario (grid-searched on the validation set) | Controller baseline |
| B9 | Oracle configuration per query (offline) | Controller upper bound |
| B10 | Published CoInfer numbers (Qwen2-VL-2B) | Context only; **never** in the same table as official-metric numbers without flagging the substring-match metric |

## 7. Metrics

**Accuracy:** official evaluators (TextVQA, POPE, MME, SciQA, GQA). For W1/W5 free-form answers, report *both* the CoInfer-style normalised substring match (for comparability) and exact match / LLM-judge or human-verified subset accuracy, because substring match inflates verbose answers (CoInfer's own caveat).
**ROI quality:** L1 centre distance (CoInfer), crop IoU with teacher box, and *answer-evidence coverage* (fraction of questions where GT evidence lies inside the crop).
**System:** TTFT (p50, p95, IQR) and its stage breakdown; end-to-end latency for short and long answers; decode tok/s; peak memory (Jetson unified memory: use `tegrastats` RAM plus PyTorch allocator); energy per query (mJ, net of idle) and average power; MCU latency, flash, SRAM; bytes over the link; co-runner p95 latency and throughput; SLO attainment; thermal throttle events (log GPU clock).
**Controller:** SLO attainment, regret vs oracle, switching frequency, decision latency.
**Reproducibility:** run manifest (git hash, config, seeds, device state) per run — the repo already writes a `run_manifest.json` (journal decision 38).

## 8. Procedure and phases

**Phase 0: Pilot, zero training (target: done by 2026-10-27; this is the SenSys round-2 go/no-go).**
1. Install the existing TinyLlama `final-parcel` checkpoint on the Orin Nano; build the measurement harness (§implementation-plan M0).
2. H1: stage-level timing at n ∈ {256,144,64,16} (reproduce Table V; confirm or refute the encoder-bound floor).
3. Oracle-crop pilot: for TextVQA and GQA-Outdoor testdev, crop with **teacher/GT boxes** (no ROI model yet), resize the crop to 336², run existing checkpoints at n ∈ {256,64,16}. Measure accuracy only (no latency change expected at 336²). Question: does zoom help the existing nested model at all, and does it lower budget sensitivity (H3)?
4. Gate: proceed to Phase 1 only if crop gives ≥ +2 pp on attribute/detail questions or reduces n-sensitivity, **and** H1 holds.

**Phase 1: ROI stage.** Generate teacher labels on W1; train MGround (FP32 → INT8, QAT if PTQ loses >1 pp L1 target); validate L1 ≈ 0.30 / testdev as CoInfer (Sec. IV-B); port to TFLM; measure MCU latency/flash/SRAM (compare with 123 ms / 302 kB / 256 kB).

**Phase 2: Crop-aware nested model.** Implement variable grid in the encoder/resampler; Stage-2 fine-tune with crop augmentation (and r=224 variant); evaluate W2–W4; ablate A1–A4 below.

**Phase 3: System measurements.** Build W5/W6/W7; measure C+N vs baselines on the Orin Nano (idle, loaded, multi-camera); H2, H5, H6, H9.

**Phase 4: Controller.** Collect (query features, ROI confidence, load, config → accuracy/latency/energy) tables offline from Phase 3; train/fit policy; online closed-loop test with ramped load; H7.

**Phase 5: Robustness & write-up.** Seeds, CIs, sensitivity, failure analysis, artifact packaging.

## 9. Ablations

| ID | Ablation | Tests |
|---|---|---|
| A1 | Crop only (n=256 fixed) / budget only / both | Contribution of each axis (H2) |
| A2 | Encoder res r = 336 vs 224 under crop | Where encoder savings come from |
| A3 | Crop classes {35,65} vs continuous size vs single class | CoInfer's policy under elasticity |
| A4 | Anchors: crop-anchors vs thumbnail-anchors+crop-LDQs (option C) vs no anchors | H8; Nest-VLM Table VI logic under crop |
| A5 | ROI source: MGround / centre / random / teacher / GPU cropper | ROI sensitivity |
| A6 | MGround components: QSG on/off, semantic prior on/off (CoInfer Table X) | Whether the gating transfers to our labels |
| A7 | Ladder-coupled policy (tight crop ⇒ small n) vs independent knobs | Design option B |
| A8 | SSA direction under crop (ranks 8/16/32/64 vs reversed) | Does SSA still help with crops? (low priority) |
| A9 | Question-embedding source: Model2Vec 256-d vs smaller | MCU interface cost |
| A10 | Link: USB vs emulated slower link | H6 sensitivity |
| A11 | Controller features: ROI-confidence on/off, load signal on/off | H7 |

## 10. Statistical methods

- **Accuracy comparisons are paired** (same questions): paired bootstrap, 10,000 resamples over questions (cluster by image since several questions share an image), report 95 % CI of the difference; McNemar's test as a secondary check for binary correctness. Holm–Bonferroni across the hypotheses in a table.
- **Seeds:** the repo's runs are n=1. For the primary backbone (TinyLlama) and the C+N and N models, train **≥3 seeds** of Stage-2 (cost: see M3; if too costly, ≥3 seeds only for Stage-2 and share Stage-1, and state it). Report mean ± std across seeds *and* bootstrap CI over questions; conclusions require agreement.
- **Latency/energy:** ≥100 timed runs after ≥10 warm-ups per cell; report median, p95, IQR; discard runs with thermal throttling (flag via clock log); randomise/ABBA order; repeat each cell on ≥3 separate days/boots for the headline numbers. Report 95 % bootstrap CIs for medians; for p95 use ≥200 samples. Energy: net of idle floor measured in the same session, plus absolute.
- **Effect sizes** are reported in absolute units (pp, ms, mJ) and ratios; no p-value-only claims.
- **Pre-registration (lightweight):** commit this document's hypotheses with a hash before Phase 3; label any later hypothesis "exploratory".
- **Data splits:** ROI model trained on W1 train, tuned on val, tested once on testdev; no testdev peeking. Report in-sample variants (as CoInfer does) separately and never as headline.

## 11. Risks and fallbacks

| Risk | Likelihood (my guess) | Mitigation / fallback |
|---|---|---|
| Crop at fixed 336² yields no encoder saving; 224² hurts accuracy | High | Retrain with r axis; if accuracy loss > 3 pp, use crop purely as a zoom/accuracy lever and drop the encoder-savings claim; pivot to H1/H3 characterisation |
| Nest-VLM TextVQA is too low for zoom to matter (28 % @256) | Medium | Use detail-sensitive subsets where the model is non-trivial; consider the larger SmolLM2 variant; or report on question types |
| MGround re-implementation underperforms CoInfer | Medium | Contact authors for code/weights; use teacher boxes as ROI to separate method claims from ROI quality; use a GPU cropper as stand-in |
| No MCU board in time | Medium | Emulated sensor stage on an isolated CPU core; label as emulation |
| ROI misses (≈20 % in CoInfer) hurt safety-style questions | Medium | Confidence-gated fallback to full frame (D); report miss-conditioned accuracy |
| Controller offers no gain over the best static config | Medium | Report as negative result; focus on characterisation + design |
| Measurement noise/thermal throttling | Medium | Fixed clocks, active cooling, ABBA, discard flagged runs |
| Label quality (teacher boxes with a CLIP/Llama-family teacher) | Medium | Hand-verify 100–300; try Qwen2.5-VL-7B teacher as CoInfer |
| Single-seed fragility of Nest-VLM results | Medium | ≥3 seeds (above); do not rely on ≤1 pp gaps |
| Dual submission / overlap with Nest-VLM under review | Unknown | Confirm status with co-authors; keep contributions disjoint |
| Time: SenSys round-2 deadline 2026-11-05 | Certain | Go/no-go on 2026-10-28; otherwise target a later cycle |

## 12. Timeline (assuming start 2026-10-21)

| Dates | Milestone | Output |
|---|---|---|
| 10-21 → 10-27 | Phase 0 pilot (H1 + oracle-crop zoom) | Numbers + go/no-go memo |
| 10-28 | **Gate**; if go: register abstract by 10-29 | Decision |
| 10-29 → 11-05 | *Only if go:* narrow paper on characterisation + oracle/MGround crop + N; short experiments | 12-page submission (risky) |
| otherwise 10-28 → 11-30 | Phase 1 (MGround, MCU port) in parallel with measurement harness hardening | MGround INT8 + TFLM numbers |
| Dec 2026 → Jan 2027 | Phase 2 (crop-aware retraining, W2–W4) | Trained C+N, ablations A1–A7 |
| Feb 2027 | Phase 3 (system measurements, W5–W7, H2/H5/H6/H9) | Pareto + contention figures |
| Mar 2027 | Phase 4 (controller) | H7 |
| Apr 2027 | Phase 5 (seeds, CIs, artifact, writing) | Draft |
| next SenSys cycle (**dates [UNVERIFIED]**) | Submission | — |

The Dec–Apr durations are rough; training time for Stage 2 is the main uncertainty (paper: ~85 GPU-h on 2×A10 per backbone from scratch).
