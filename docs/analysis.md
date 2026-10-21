# Analysis: CoInfer, Nest-VLM, and the FlexLLaVA repository

Part of the [project overview](README.md). Written 2026-10-21.

**Tagging convention.** Every substantive statement carries one of:
- **[FACT]**: read directly in a paper, the repo, or a fetched web page; the source is cited.
- **[INTERP]**: my inference from facts. It may be wrong; the reasoning is shown.
- **[PROPOSED]**: work I suggest doing. Nothing tagged this way has been done.
- **[UNVERIFIED]**: could not be checked in this environment.

## 0. What I inspected, and what I could not

| Material | Status |
|---|---|
| `FlexLLaVA/` repo (local clone, `main`, remote `github.com/sudaksh14/FlexLLaVA`, HEAD `28fadd0`) | Inspected: README, `docs/ELASTIC_PIPELINE.md`, `docs/ADAPTIVE_INPUT_TOKEN_BUDGET_ALLOCATION.md` (first 140 lines), `docs/EXPERIMENT_JOURNAL.md` (selected sections: §14, §16m–q, final-config, §23, decision table), `results/*.csv`, `docs/vision_vs_llm_split.csv`, grep over `llava/` and scripts. I did **not** read all 3,092 journal lines or run any code. |
| Nest-VLM paper: [`related papers/657_file_Paper.pdf`](../related%20papers/657_file_Paper.pdf) (anonymous submission, no venue stated) | Read in full (text extraction). Figures were not viewed. |
| CoInfer paper: [`related papers/093201a575.pdf`](../related%20papers/093201a575.pdf) (IEEE TCAD) | Read in full (text extraction), plus rendered pages 6–7 to read Tables II–V and Fig. 3. Other figures (Figs. 4–11) were **not** viewed; I rely on the text that describes them. |
| `AdaOff` (`093201a017.pdf`) and `Flextron` (`2406.10260v2.pdf`) | **Only the first lines were read. Not analysed.** They are in the folder but outside the requested scope. |
| CoInfer official code / MGround weights | **Not found.** See §2.5. |
| PARCEL (arXiv:2605.30126) | Only search-result summaries seen by me this session. The repo journal §14 says it was fetched in full by an earlier agent pass; I did not re-verify. |
| SenSys 2027 CFP | Fetched via the web-fetch tool, which returns a model-generated summary, not verbatim HTML. Treat details as [UNVERIFIED] until you re-read the page. |

**Names.** "Nest-VLM" is the paper's title (all caps NEST-VLM). The repo never uses that name; the corresponding code is the `pool_anchored` resampler plus nested vision LoRA, shipped under the recipe tag `final-parcel` ([INTERP], based on the paper's Table III matching the journal's v8 numbers exactly, e.g. TinyLlama@256: MME-P 1170.25, POPE 84.33, TextVQA 28.47, GQA 56.07; see journal "FINAL BEST CONFIG" and §16m). "CoInfer" is written "CoInfer" in the paper, not "Co-Infer".

---

## 1. Nest-VLM

### 1.1 Problem and claims [FACT]
- Elastic visual-token methods (M3, MQT-LLaVA) were developed on large backbones (Vicuna-7B) that do not fit an 8 GB Jetson Orin Nano; small VLMs fit but are non-elastic (Sec. I–II, Table I).
- Claim: purely learned nested query resamplers become **rank-deficient** at small LM scale (Table II: 12/256 directions without positional embeddings, 130/256 with, vs 217/256 for MQT-7B), so the token budget stops being an accuracy lever (Sec. IV-C, VI-C).
- Method (Sec. IV): frozen CLIP ViT-L/14-336 → nested **Scarcity-Scaled Adapters (SSA)**, i.e. prefix-nested LoRA whose rank *grows as the budget shrinks* (ranks {8,16,32,64} for budgets {256,144,64,16}) → **Nested Token Resampler (NTR)** = deterministic average-pooled spatial anchors (25 % of budget) + anchor-conditioned Local-Detail Query tokens (LDQs) → 2-layer MLP → small LM. Loss (Eq. 2): LM loss at a reference level (256) plus one random student level, plus 0.1 × KL to the reference level (self-distillation).
- The paper says explicitly that NTR "adapts PARCEL's division of labour" (Sec. I, III-C). The NTR is therefore **not** claimed as a new mechanism in kind; the claimed novelty is the small-LM adaptation, the rank diagnosis, SSA, and the embedded characterisation.

### 1.2 Setup [FACT] (Sec. V)
- Hardware: Jetson Orin Nano Super dev kit, 8 GB, 15 W mode, 0.61 GHz GPU clock, JetPack 6.2.1; median of 100 runs; INA3221 energy via `tegrastats`, net of idle floor.
- Backbones: TinyLlama-1.1B, SmolLM2-1.7B, MobileLLaMA-1.4B. Budgets {256,144,64,16}.
- Training: LLaVA two-stage recipe, 1 epoch each; ~10 + ~85 GPU-hours on 2×A10 (24 GB).
- Benchmarks: MME-P, POPE, SciQA, TextVQA, GQA, VQAv2. Baselines: TinyLLaVA, MobileVLM-arch, LLaVA-1.5-7B (incl. 4-bit), MQT-LLaVA-7B.

### 1.3 Key results [FACT]
| Quantity | Value | Source |
|---|---|---|
| Prefill TTFT 256→16 tokens, TinyLlama | 392.88 → 272.09 ms (1.44×) | Table V |
| Same, TinyLlama 64→16 | 271.62 → 272.09 ms (**flat**) | Table V |
| Energy/query 256→16, TinyLlama / SmolLM2 / MobileLLaMA | 6319→4845 / 9392→6653 / 7821→5887 mJ (1.30× / 1.41× / 1.33×) | Table V |
| Decode throughput | 12–13 tok/s at every budget | Table V |
| Peak memory | ≈3.04–3.05 GB TinyLlama regardless of budget | Table V |
| Accuracy 256→16, TinyLlama | TextVQA 28.47→19.71, GQA 56.07→52.61, POPE 84.33→80.49 | Table III |
| Anchor ablation | Without anchors, 256→16 spread ≤0.82 pts on %-scale metrics | Table VI |
| SSA ablation | SSA best on MME-P/POPE/TextVQA/GQA at all levels; worse on SciQA | Table VII |

### 1.4 What the repo adds [FACT, repo]
- `pool_anchored` resampler: `llava/model/elastic/resampler.py` (`pool_anchors`, `pool_anchors_adaptive`, `_forward_pool_anchored`, `n_anchors_for`); config knobs `--resampler_arch pool_anchored --anchor_mode {ratio,fixed,adaptive} --anchor_ratio 0.25` (`train_elastic.py`).
- Nested LoRA: `llava/model/elastic/nested_lora.py`; `lora_level_for_tok` in `config.py`.
- Evidence behind SSA's direction: journal §16q: reversing the ladder (v14: ranks 64/32/16/8) loses MME/POPE/TextVQA/GQA at every level (−8.23 TextVQA, −3.77 GQA at 256). Shared rank-16 adapter (v10) is *worse than no adapter* at 256 on TextVQA (19.96 vs 22.18) (journal §16m). This supports the paper's SSA claim and is **extra evidence not in the paper**.
- Fixed-resolution assumption: `NestedQueryResampler(num_patches=576)`; `valid_anchor_counts(num_patches)`; there is a `_interp_pos_embed` helper at `resampler.py:134` ([INTERP] usable as a starting point for variable grids; not verified).

### 1.5 Problems in the Nest-VLM paper that matter for any follow-up [FACT unless tagged]
1. **Table IV row mislabelled (high confidence).** Table IV's "NEST-VLM-TinyLlama (ours)" at native 576 tokens lists POPE 84.89, SciQA 57.36, TextVQA 41.00, GQA 58.30. The repo journal §23 gives **exactly these numbers for `baseline-tinyllama-576tok`**, the plain dense mlp2x_gelu baseline with *no* NTR, while the PARCEL-architecture run at 576 tokens (`final-parcel-576ctrl`) scores 81.82 / 50.72 / 21.20 / 52.57. [INTERP] The Table IV row is the dense baseline, not Nest-VLM, and the prose "NEST-VLM recovers most of the uncompressed model's accuracy" (Sec. VI-A) compares 256-token Nest-VLM to a *different, non-elastic* model. The honest framing is the repo's own: the anchor architecture carries a tax at full resolution (journal §23, TextVQA −19.8 vs dense).
2. **Latency claim vs MobileVLM-arch.** Fig. 1 caption/Sec. I say non-elastic references "sit above every NEST-VLM operating point at a matched or smaller budget". Table V: MobileVLM-arch at 144 tokens = 246.41 ms / 6046 mJ; NEST-VLM-MobileLLaMA at 144 = 333.58 ms / 6640 mJ. [INTERP] For that backbone the statement does not hold as written (the two differ in connector, so it is not a clean comparison either).
3. **Prefill gain is small and saturates.** 64→16 tokens gives no latency gain (Table V). [INTERP] The fixed part (vision encoder at 576 patches, fixed overheads) is ≤272 ms of the 393 ms at 256 tokens, i.e. up to 69 % of TTFT cannot be touched by the token budget. The repo's analytical split agrees in direction (vision = 36 % of FLOPs, 30 % of latency for TinyLlama@256; `docs/vision_vs_llm_split.csv`).
4. **Backbone size inconsistency.** Sec. V says the "2.7 B SmolLM backbone" needed A40s; Table V lists SmolLM2-1.7B at 2.09 B params.
5. **n = 1.** The journal states every result is single-run (journal "FINAL BEST CONFIG": "Every result in this journal is n=1"). The paper reports no seeds or CIs; several claimed per-level wins (e.g. SSA vs MNR on MME-P at n=16: 1123.91 vs 1118.11) are small enough that noise could matter ([INTERP]).
6. **Measured Jetson numbers are not reproducible from the repo.** `results/*.csv` columns `jetson_prefill_ms` are analytic roofline estimates (`llava/eval/efficiency/hardware.py`, `scripts/export_eval_csv.py`); e.g. TinyLlama@256 estimate 78.94 ms vs 392.88 ms measured in the paper (5× gap). A grep for `tegrastats`/`INA3221` finds only the roofline hardware table and shell mentions, no measurement harness or raw logs. The paper itself notes the FLOP model over-predicted the speedup (Sec. VI-B).
7. SciQA is the one metric where nesting hurts (−2 to −3, journal and Table VII); the paper acknowledges it.

---

## 2. CoInfer

### 2.1 Problem and goals [FACT]
Embedded CPS nodes (patrol robots, safety monitors) run VLMs on a **shared** SoC GPU/CPU alongside perception/control tasks. Visual token count (hence encoder + prefill cost) scales with image resolution; existing reduction (FastV, SparseVLM, CROP, LLaVA-PruMerge, learned croppers like CropVLM, Grounding DINO) runs *on the host* and either leaves encoder cost unchanged (post-encoder pruning) or consumes host GPU (GPU croppers). Goal: **move question-aware region selection off the host onto an on-sensor MCU**, send only the cropped region to an unmodified VLM (Sec. I, II, III-A).

### 2.2 System design [FACT] (Sec. III, Fig. 1–2)
- Runtime: host encodes the question with a CPU text embedder (Model2Vec, 256-d) → sends embedding to MCU → **MGround** predicts an ROI centre and a binary crop-size class (small 35 % / large 65 % of image area) from a 128×128 downscaled frame + embedding → **Crop Engine** crops the full-res frame in external SDRAM → sends crop to host VLM.
- MGround: 7×7 stride-4 stem then three conv stages (32→64→128), each gated by **QSG** (question-squeeze gating: `g = σ(W[GAP(x); e] + b)`, `x' = x ⊙ g`; multiplicative only, "SE-like with question", cf. FiLM); centre head 384→128→2; **semantic prior head** from the question alone, blended with a learnable weight w (init 0.5); crop-class head from the question embedding only (256→1). 276 K params, INT8, ≈300 kB flash (302 kB measured), 256 kB SRAM arena, 123 ms model latency, ~157 ms net on-sensor overhead (resize + inference + crop) on STM32H747 @480 MHz with TFLite-Micro (Sec. III-B, IV-C).
- Training: **no manual labels**. A larger VLM from the same family (Qwen2.5-VL-7B for a Qwen2-VL-2B target) is prompted for a bounding box; its centre is the label; crop-size labels come from a rule on question type (object/attribute → small; relational/global → large). L1 loss for centre, BCE for crop class. AdamW, lr 1e-3, bs 64, ≤200 epochs, early stopping (Sec. III-C, IV-A).
- Assumptions stated or implied: short-answer VQA; encoder-dominated TTFT (high-res input, dynamic-token VLM); a sensor-side MCU with frame buffering in external SDRAM; raw crop transfer over USB (CSI/in-camera integration deferred); per-site retraining is cheap; one MCU per camera; ROI misses (20 % of queries) are tolerated because crops are loose (Sec. IV-D).

### 2.3 Experimental setup [FACT] (Sec. IV-A, Tables II–III, rendered page 6)
- Hardware: STM32H747 (Cortex-M7 480 MHz) as OnSensor; **Jetson AGX Orin 32 GB** as OnDevice GPU; JetPack 6.2.1, CUDA 12.6, PyTorch 2.8.0, Transformers 4.56.1.
- VLMs: Qwen2-VL-2B-Instruct (primary, dynamic resolution), InternVL2.5-2B (448 px tiles), Gemma3-4B-it (fixed 256 tokens), all FP16; teacher Qwen2.5-VL-7B (BF16).
- Datasets: **GQA Outdoor** (rebuilt from GQA by keyword filter, Appendix A: ≥3 of 47 outdoor keywords per image): train 12,505 imgs / 60,840 samples; val 1,418 / 6,694; testdev 143 / 715. **Industrial**: proprietary 1920×1080 robot-dog set, train 4,301 imgs / 32,540 samples, val 1,066 / 8,084; **cannot be released** (Sec. IV-A).
- Metrics: ROI L1 centre distance; VQA accuracy by **case-insensitive, article-stripped substring match** (the paper itself cautions absolute values are relative only); p95 TTFT over 100 runs; per-query GPU energy on the Jetson VDD GPU SOC rail; co-located YOLOv8n / DeepLabV3 p95.
- Baselines: fixed crops (center/random/fixed-at-GDINO), FastV, SparseVLM, Grounding DINO crop, CropVLM (256 M, fine-tuned), teacher-ROI upper bound, MGround+FastV/SparseVLM, MGround on Orin CPU.
- Loaded condition: concurrent YOLOv8n + DeepLabV3 + CPU stress on 9 of 12 cores (Sec. IV-C2).

### 2.4 Reported results [FACT]
| Result | Value | Where |
|---|---|---|
| Accuracy, Qwen2-VL-2B, GQA-Outdoor testdev | 63.4 % vs 63.8 % full (−0.4 pp) at 50.8 % token reduction | Fig. 3, Sec. IV-B1 |
| Same, centre crop / FastV / SparseVLM at ≈50 % | ≈61.3 / 61.5 / 62.7 % | Fig. 3, Table V, text |
| Teacher crop upper bound; CropVLM | 64.1 %; 63.9 % at 33 % reduction | Sec. IV-B1 |
| ROI L1 (testdev) | 0.2960 vs centre baseline 0.3447; in-sample 0.1849 | Sec. IV-B1 |
| Question-type split (Table V) | Attribute questions +8.9 pp over centre crop; relational +3.6; object/category 0 | Table V |
| Cross-VLM (testdev) | InternVL −0.5 pp; Gemma3 **+1.5 pp** (no token reduction, fixed 256) | Fig. 4, Sec. IV-B3 |
| 1080p token reduction | Qwen2-VL −55.8 % (2562→1133); InternVL −41.0 %; Gemma3 0 % | Fig. 10 text |
| TTFT, 1080p industrial | −58 % idle, −48 % loaded (abstract/conclusion) | Sec. IV-C2, V |
| Energy/query (small crop) | 84.3→11.5 J forward (−86 %); MGround ≈60 mJ | Table VII text |
| Encoder share of TTFT | >70 % in baseline | Sec. IV-C2 |
| Co-located YOLO p95 | 594 ms (full) → 183 ms (35 % crop) | Fig. 9 text |
| MCU vs host-CPU grounding at 8 cameras, loaded | 157 ms flat vs 842 ms | Fig. 11 text |
| On-sensor overhead | ≈157 ms; net 65–106 ms after transfer savings | Sec. IV-C |

Stated limitations [FACT]: scene-level questions pick large crops (small saving); single device configuration; multi-turn queries over one image lose encoder-reuse; 20 % of queries miss the GT centre; confidence-gated fallback and NPU-MCU are future work; savings shrink for long generation (Sec. IV-D).

### 2.5 Is there an official codebase? 
**Not verified; no evidence of one.** The paper text contains no repository URL and no artifact statement (grep for `github`/`http`/"open-source" finds only IEEE boilerplate and the sentence that the industrial data "cannot be publicly released"). Two web searches (standard + extended) for the title, "MGround", "STM32H747", ETRI, found nothing. [UNVERIFIED] That does not prove no code exists (the paper may be too new to be indexed); ask the authors (ETRI; corresponding author Jeong-Si Kim, yun38@etri.re.kr / sikim00@etri.re.kr per the paper's header). **Plan on re-implementing MGround** from the paper (architecture is fully specified in Sec. III-B except exact channel/kernel details of stages 2–3 and Model2Vec variant).

### 2.6 What is clear vs unclear
Clear: the architecture of MGround, labelling procedure, the headline numbers above, the test hardware.
Unclear / worth flagging:
- Fig. 3–11 numeric values beyond those quoted in text (not extracted).
- "Presented at CODES+ISSS 2024" versus references dated 2025–2026 and "Received/accepted 10 August 2026" (same day for both) are internally odd [INTERP: likely template/edit artefacts; irrelevant to the technical content, but cite carefully].
- Accuracy uses substring matching that can give false positives on verbose answers (authors' own caveat), so cross-paper comparison with official-metric numbers (such as Nest-VLM's) is invalid.
- GQA images are low-res; the big wins (token reduction, latency) are shown on the **proprietary** 1080p set, whose accuracy is not reported on the public subset at 1080p [INTERP: GQA Outdoor accuracy supports "no accuracy loss", but 1080p accuracy under the deployed regime is unpublished].
- Hardware/energy numbers for one device; no Orin Nano (the 8 GB class that Nest-VLM targets).
- MCU latency (157 ms) is similar in size to the savings on small SoCs ([INTERP] on Orin Nano with TTFT ≈270–390 ms the sensor stage is a first-order cost, not negligible; see experimental plan H2).

---

## 3. How Nest-VLM could transfer to CoInfer

### 3.1 Complementarity (core interpretation)
- **[FACT]** CoInfer reduces *encoder input* (crop before encoder) and shows encoder >70 % of TTFT for a dynamic-resolution VLM at 1080p. Nest-VLM reduces only *LLM-side visual tokens*, leaves CLIP at 576 patches, and its measured TTFT floors at ≈272 ms (Table V).
- **[INTERP]** The two attack different terms of cost: `T ≈ T_enc(pixels) + T_prefill(n_visual + n_text) + T_decode`. CoInfer cuts `T_enc` (and `n_visual` for dynamic-token encoders); Nest-VLM cuts `T_prefill` and KV size, and provides a *continuous, trained-in* knob at fixed encoder cost. Neither alone gives a good Pareto front on a small SoC.
- **[INTERP]** Nest-VLM answers a CoInfer weakness: CoInfer's token reduction depends on the VLM's tokeniser (zero for Gemma3-style fixed budgets, 41 % for tiles). A nested resampler gives a *tokeniser-independent* knob.
- **[INTERP]** CoInfer answers a Nest-VLM weakness: at low budgets TextVQA collapses (28.5→19.7 at 16 tokens). A question-aware crop concentrates the pixel budget on the relevant region ("higher effective resolution", Sec. IV-B2), which is exactly what small-budget models lack. Gemma3's +1.5 pp from cropping with *no* token savings (Sec. IV-B3) is supporting evidence of a zoom effect independent of token reduction.

### 3.2 Compatibility problems (all [INTERP] from the cited code/text)
| Issue | Why | Consequence |
|---|---|---|
| CLIP-L/336 is fixed-resolution | A crop resized to 336² still costs 576 patches | **No encoder saving unless the encoder runs at lower resolution** (e.g. 224² = 256 patches ≈ 0.44× linear FLOPs, less in attention). Needs pos-embed interpolation and probably retraining; a new (resolution) axis in the nesting. |
| Nest-VLM never saw crops | Training images are full-frame, 336² | Distribution shift in zoomed crops (usually helpful for detail, risky for context) → add crop augmentation to Stage 2. |
| Anchors assume global layout | Anchors are average-pooled grids of the *input to the encoder* | On a crop, anchors encode the crop's layout, not the scene's → global context is lost; the "scene" is only what the MCU kept. |
| Low-res frame already at the sensor | MCU holds a 128² thumbnail | Could provide *global anchors for free* (see §3.3 option C). |
| Crop class (35/65 %) vs token budget | Both control "how much detail" | Redundant knobs if composed naively; should be tied or jointly chosen. |
| Short-answer assumption | CoInfer's gains are TTFT-dominated | Nest-VLM decode is 12–13 tok/s, so long answers wash out both. |
| SoC mismatch | CoInfer measured on AGX Orin 32 GB; Nest-VLM on Orin Nano 8 GB | Must re-measure everything on one platform; conclusions can flip (MCU overhead share, contention). |
| Teacher/labels | CoInfer's teacher is a 7B VLM of the target's family | For a CLIP/TinyLlama VLM, use the repo's LLaVA-7B or Qwen2.5-VL-7B as teacher; label quality is untested for this pairing. |

### 3.3 Ways to combine [PROPOSED]
- **A. Sequential composition.** MGround crop → (optionally lower-res) encoder → Nest-VLM at budget n. Simplest; first baseline. Knobs: crop {35,65,100} % × encoder res × n.
- **B. Coupled policy.** One "scarcity" rule: tighter crop ⇒ fewer tokens needed (the information is concentrated); wide crop or fallback ⇒ larger budget. Reduces the 3-axis space to a 1-D ladder per question class.
- **C. Anchors from the global thumbnail, LDQs from the crop** (foveation). Anchors pool a cheap low-res full-frame encoding; LDQs cross-attend to a high-res crop encoding. Preserves global layout (fixes the anchor issue above). Costs two encoder passes (low-res one is cheap) and new training; **highest novelty and highest risk**.
- **D. Confidence-gated escalation.** Use MGround's own signals (crop-size logit, disagreement between visual and prior centres; Sec. IV-D) to pick budget / fall back to full frame. The paper lists this as future work.
- **E. Load-aware controller.** Choose (crop, res, n) per query under a latency/energy SLO and measured contention (Nest-VLM's stated future work: "embedded runtime controller", Sec. VII).
- **F. ROI as soft prior instead of hard crop** (bias LDQs toward ROI on-host). Keeps context, loses encoder saving and sensor isolation; use only as an ablation.

---

## 4. Improvements needed in the current repo (summary; details in [implementation-plan.md](implementation-plan.md))
1. No measurement harness or raw logs for the paper's Jetson numbers; repo CSVs hold roofline estimates (see §1.5 item 6).
2. Fixed 576-patch assumption throughout the resampler/vision tower; no crop or variable-resolution path.
3. Eval wrapper selects a single fixed `tok_level` per run (`lmms-eval/lmms_eval/models/llava_elastic.py:198-200`); no per-query budget, crop, or controller.
4. `jobs/` directory is referenced by README and docs (`jobs/*.sh`, `jobs/test_lora_rank_order.sh`) but does not exist in this clone. Reproducibility gap.
5. Single-seed results; no CIs.
6. Cluster-specific absolute paths (`/var/scratch/skalra/...`) in scripts and CSVs.
7. `run_job_slm.sh` has no `set -e` (documented in `ELASTIC_PIPELINE.md` §10): a failed training still reports success.
8. Anonymity for submission: the remote is a personal GitHub account; the SenSys CFP requires anonymised artifact links (see [paper-plan.md](paper-plan.md)).
9. [FACT] The home directory of this machine is itself a git repository with untracked secrets-like files (`.ssh/`, `.hf_token`, `_netrc`, `.claude.json`) per the session's `git status`. Do not run `git add -A` from there; the FlexLLaVA repo has its own `.git`.

---

## 5. Sources
- Nest-VLM paper (local): [`657_file_Paper.pdf`](../related%20papers/657_file_Paper.pdf), Sec. I–VII, Tables I–VII, Fig. 1–2.
- CoInfer paper (local): [`093201a575.pdf`](../related%20papers/093201a575.pdf), DOI 10.1109/TCAD.2026.3724452; Sec. II–V, Tables II–VII, App. A–D.
- Repo: [`ELASTIC_PIPELINE.md`](../FlexLLaVA/docs/ELASTIC_PIPELINE.md), [`EXPERIMENT_JOURNAL.md`](../FlexLLaVA/docs/EXPERIMENT_JOURNAL.md) (§14, §16m, §16q, §23), [`vision_vs_llm_split.csv`](../FlexLLaVA/docs/vision_vs_llm_split.csv), `llava/model/elastic/*`, `llava/train/train_elastic.py`.
- PARCEL: https://www.emergentmind.com/papers/2605.30126 (summary only).
- MatryoshkaLoRA: https://arxiv.org/pdf/2605.07850.
- SenSys 2027: https://sensys.acm.org/2027/cfp.html and https://sensys.acm.org/2027/.
- Other related-work links are in [paper-plan.md](paper-plan.md) §2.
