# Brainstorm: a dual-elastic, HR-aware small VLM for Jetson-class devices

Part of the [project overview](README.md). Written 2026-10-22. Tags follow [analysis.md](analysis.md): **[FACT]** = read in a paper, the repo or a fetched page (source cited); **[INTERP]** = my inference; **[PROPOSED]** = not done. **Nothing here has been run or measured.** Paper numbers come from the papers' own hardware and backbones (mostly 7B-class models on server GPUs). None of them were measured on TinyLlama or an Orin Nano.

Sources read for this note: all nine PDFs in [`related papers/adaptive HR VLM/`](../related%20papers/adaptive%20HR%20VLM/) (body text through the conclusion, via `pdftotext`; figures not viewed; some tables were garbled in extraction and only clearly legible numbers are quoted), the four existing docs in this folder, `FlexLLaVA/docs/EXPERIMENT_JOURNAL.md` (decisions 23 and 45 only), `ADAPTIVE_INPUT_TOKEN_BUDGET_ALLOCATION.md` (grep only), `llava/model/multimodal_encoder/clip_encoder.py`, and the SenSys 2027 CFP (re-fetched today).

---

## 0. Your summary, checked

> *"There is a 2-D elasticity at the encode stage and the tokens at the prefill stage, and nothing could be saved in the decode stage."*

You are mostly right, with three corrections.

**(a) Today the pipeline has only one cost knob, not two.** [FACT] Nest-VLM varies the visual-token budget *n*. The vision encoder always runs CLIP-L/14 at 336² (576 patches). The nested LoRA ranks change with *n*, but that is an accuracy lever, not a cost lever: ranks *grow* as *n* shrinks (8→64), so the encoder does not get cheaper. Measured on the Orin Nano (Nest-VLM Table V, TinyLlama), TTFT goes from 392.88 ms at 256 tokens to 272.09 ms at 16, and is flat from 64 to 16. Up to ≈69 % of TTFT is outside the reach of *n*. **The encoder axis is the one that has to be added**, which is what the HR/ROI papers below give us.

**(b) Decode: right for visual-token elasticity, with caveats.** [FACT] Decode stays at 12–13 tok/s at every budget (Table V), and peak memory stays at ≈3.04 GB for every budget. [INTERP, corrected in §8.1 item 4: the logs show decode is about 3× slower than a weight-bandwidth floor, so "bandwidth-limited" is not established; the conclusion below still holds] Decode time does not depend on the visual tokens because the weights are read for every generated token. TinyLlama uses grouped-query attention (22 layers, 4 KV heads × 64 dims), so each visual token adds about 22 × 2 × 256 × 2 B ≈ 22 KB of KV cache. All 256 visual tokens together come to ≈6 MB, against ≈2.2 GB of FP16 weights read for every generated token. Cutting visual tokens therefore cannot make decode faster. What *can* cut decode cost:
- **Not decoding with the expensive configuration at all**: early exit or a cascade (SGL's SEE, Q-Zoom's gate). This saves whole passes, not tokens.
- **Shorter answers**: this is why the paper should be framed on short-answer VQA, as CoInfer does.
- **Weight quantisation** (HyperVL Table 7: W4A16 loses 0.1–2.2 pts). This is orthogonal to elasticity. Use it for both our model and the baselines, or for neither.
- **Self-speculative decoding with the same weights does not help here.** Every draft token still reads all the weights (the journal's item M6 discusses this).

So the honest claim for the paper is: *both elastic axes act on TTFT and energy per query, and decode is a fixed tail that the paper reports but does not claim to reduce.*

**(c) Memory does not come from elasticity.** [FACT] The model fits in the 8 GB budget because the LM is small (≈3.05 GB peak), not because of the token knob. [INTERP] Adding HR views does add *activation* memory to the encoder, and HyperVL shows how to bound it (serial tile encoding; §1, item 8).

**Restated thesis [PROPOSED]:** *pixel elasticity* (which pixels the encoder sees, and at what scale: a crop, a ROI tile, or a skipped second view) plus *token elasticity* (how many visual tokens the LM prefills, possibly split per view) together give a per-query (encoder cost × prefill cost) operating point. Decode is reduced only by exiting early.

---

## 1. What each paper shows, and what it means for our stack

Our stack: CLIP-L/14-336 (frozen, with nested LoRA), then the NTR resampler (pooled anchors + LDQs), then an MLP, then TinyLlama / SmolLM2 / MobileLLaMA, on a Jetson Orin Nano 8 GB.

| # | Paper | Core idea | Evidence in the paper [FACT] | Transfer to our stack [INTERP] |
|---|---|---|---|---|
| 1 | **ViCrop** (ICLR 2025) | Training-free. The MLLM's own answer-to-image attention (or gradients) locates the ROI. Crop it, resize to the encoder's input size, and **concatenate** the crop's tokens with the full image's tokens. | TextVQA with a GT ("human") crop, LLaVA-1.5-7B: small-object accuracy 39.38→69.95, medium 47.74→65.36 (Tab. 1). Automatic `rel-att`: TextVQA 47.80→55.17, V\* 42.41→62.30, GQA 60.48→60.97 (no loss on large objects) (Tab. 2). The attention ratio on the GT box is >1 **even when the answer is wrong** ("they know where to look", Fig. 3). Cost: 1.16 s on an A6000 GPU vs 0.17 s per answer token (Tab. 4). Fails on relational and counting questions (single region). **Its future-work section names MQT (nested tokens) as a way to cut the cost of the extra crop tokens**, which is essentially our proposal. | Very high relevance; the zero-training starting point. Our resampler is query-based, so attention has to be mapped back to patches through the resampler's cross-attention (ViCrop's A_st·A_ti product for Q-Former-style connectors, Sec. 4). Anchors are spatial pools, so they map back directly. Open questions: whether TinyLlama's attention localises as well as Vicuna-7B's, and what a second CLIP pass costs on the Orin. |
| 2 | **SGL: small VLM guides large** (2024) | Attention aggregated over **all layers** of a small VLM ranks visual tokens for pruning in a large VLM. **SEE** exits early when the small model's confidence + consistency score is high enough. | At 9 % token retention: SGP keeps 89.58 % of the InternVL2-26B score, against 46.99 % for FastV and 54.28 % for ToMe (Tab. 1). **FastV collapses below ≈35 % retention** (Fig. 1a). Aggregating all layers of the 2B model beats one layer of the 26B model (Tab. 2: 92.64 % vs 48.37 %). A 1B guide works (Tab. 4). The confidence + consistency score beats entropy and quantile scores for early exit (Fig. 4). | (i) Use **"low budget guides high budget" inside one nested model**: the n=16 pass is our "small VLM", and its all-layer attention supplies the ROI. (ii) SEE gives a training-free gate: exit at n=16 when confident. This is the only real decode saving. (iii) A strong prediction for our FastV baseline: at 16–64 of 576 tokens (2.8–11 %), FastV should collapse while Nest-VLM degrades gracefully. |
| 3 | **Q-Zoom** (2026 preprint) | A gate decides whether HR is needed. **SD-RPN** predicts a ROI heatmap from intermediate LLM layers in the *same* prefill. It is trained on attention-derived pseudo-labels (sink tokens removed, three-way label assignment). The ROI tokens are inserted before the question, and the **prefix KV cache is reused** for the second prefill. | LLaVA-1.5-7B: TextVQA 46.1→58.4, DocVQA 21.5→34.0, V\* 50.3→70.7, throughput 0.58–0.64× vs ViCrop's 0.42× (Tabs. II, III). Training-free attention ROI runs at only 0.34–0.39× throughput; GroundingDINO ROIs are weaker (Tab. V). **10K pseudo-labels already give 77.7 vs 78.9 at 185K** (Tab. VII). The gate bypasses HR for 16.5 % more queries at equal accuracy (Fig. 9). On LLaVA the gate "almost universally" triggers on HR benchmarks, because 336² is low resolution. | High. It shows the learned successor to ViCrop beats it on both accuracy and throughput, and the training is cheap (frozen backbone; only a 3-layer branch is trained). The KV-reuse layout fits a nested model directly: global tokens at n₁ form the cached prefix, and ROI tokens at n₂ are appended. Warning for us: the trigger rate on LLaVA-class 336² encoders is high, so the gate saves little on detail-heavy benchmarks. |
| 4 | **AdaptVision** (CVPR 2026) | Start from a ¼-resolution image. An RL-trained policy (DTPO) emits a bbox tool call to fetch an HR crop only when needed. | Qwen2.5-VL-7B: 33 % of tokens at 97.9 % of vanilla performance, vs a plain ¼-res downsample at 25 % tokens and 92.1 % (Tab. 1); 1.67× end-to-end speed-up (Fig. 4). Rewards need a GPT-4o judge; vanilla GRPO collapses to calling the tool every time. | Low for us. RL on a 1.1B LM, an LLM judge, and long tool-call decoding (which costs *decode* time, our fixed tail) are all a poor fit. Useful as evidence that "coarse first, crop on demand" keeps ≈98 %. |
| 5 | **S2: Scaling on Scales** (ECCV 2024) | Run the *frozen* encoder on native-size **tiles** of 2× and 3× upscaled images, then average-pool back to the base grid and concatenate channels. **The LLM token count does not change.** | LLaVA-1.5-7B at 1008² with 576 tokens: V\* attribute 43.5→51.3, spatial 56.6→61.8, TextVQA 58.2→61.0, VQAv2 78.5→80.0 (Tab. 1). Its discussion names "**scale-selective processing**" (high scale only where needed) as future work. | High as a *decoupling* tool: pixel cost rises while tokens stay controlled by the NTR. **The repo already contains an unused `CLIPVisionTowerS2`** (`clip_encoder.py:131`, with no nested-LoRA support). Full S2 multiplies encoder cost (1+4+9 tiles at 336/672/1008), which is the wrong direction on the Orin. Applied only to the ROI tile, it is our idea. |
| 6 | **Monkey** (CVPR 2024) | Tile HR images into native-size crops, use a per-tile LoRA on a shared encoder, and add a global view. | **Interpolating position embeddings to 896 instead of tiling: TextVQA 44.7 vs 67.6, DocVQA 53.9 vs 66.5** (Tab. 5, r1 vs r9). LLaVA-1.5 retrained at 224/336/448: TextVQA 56.1/59.8/60.2 (Tab. 6). | Important *negative* evidence: do **not** change CLIP's resolution by position-embedding interpolation without substantial training; feed native 336² tiles or crops instead. This weakens the "r=224 encoder" item in [experimental-plan.md](experimental-plan.md) (A2). The per-tile LoRA idea parallels our nested LoRA (a "view-specific" level is cheap to add). |
| 7 | **ViCO** (2025) | Two connectors (4× and 16× compression), **consistency (KL) training** across compression rates, and a **patch-level router** trained on loss-ratio pseudo-labels. | InternVL3.5: ≈50 % fewer tokens at 99.6 % retention (Tab. 1). Ablation (Tab. 3): patch-level router 99.6 % > image-level 98.8 % > random routing 97.6 % > compressing everything 95.1 %. **Applying the router without consistency training drops to 87.9 %.** FastV at a comparable ratio 95.9 %, SparseVLM 91.2 %. 50 % compression gives ≈1.8–2.0× LLM first-token throughput (Tab. 4). | Very high. Nest-VLM *already has* the consistency training (Eq. 2: KL to the reference level). The missing piece is **per-region budget allocation**: global view vs ROI tile at different *n*. The loss-ratio labelling recipe gives router labels without human annotation. |
| 8 | **HyperVL** (Xiaomi, 2025) | An on-device MLLM: **serial tile encoding** to cap memory; a **Visual Resolution Compressor** (MobileNet, predicts a 0.1–1.0 scale from loss-ratio labels); **dual consistency learning** between a SigLIP2-Base and a SigLIP2-Large encoder sharing one LLM. | Real phone NPU (Qualcomm 8750): **6.8× lower peak memory** and ≈12.9× ViT speed-up vs Qwen3-VL-2B at high resolution. VRC: **2 ms overhead**, −20.2 % tokens at 98.7 % retention; on Qwen3-VL-2B −29.3 % at 98.4 % (Tab. 6). It keeps full resolution on ChartQA and cuts 49–63 % on DocVQA. DCL lifts the small-encoder branch by +5.0 on AI2D and +22 on OCRBench (Tab. 5). | High, and the closest in spirit to a SenSys paper (real-device latency and memory). (i) A VRC-style predictor is our cheapest *learned* controller and could run on the Orin CPU or the sensor MCU. (ii) Serial tiles keep the extra HR view inside the 8 GB budget. (iii) DCL is a third elastic axis (encoder size): expensive, keep it for later. |
| 9 | **VISOR** (Samsung, 2026) | Keep all visual tokens. Most LLM layers are text-only; a few cross-attention and self-attention layers attend to vision; a policy chooses how many self-attention layers to run. | LLaVA-OV-0.5B: easy avg 63.6, hard avg 58.4 vs baseline 61.5 / 57.1, at 8.6× FLOP savings. Token-reduction methods collapse on hard tasks (VisionZip hard 32.9, FastV-line similar). **M3 (nested Matryoshka tokens) holds up: hard avg 56.6 at 8×** (Tab. 1). Measured 0.0738 → 0.0384 s per sample. | The architecture itself is low priority: it requires changing the LLM and retraining on millions of samples, and the LLM is not our bottleneck on the Orin. **Its Table 1 is useful evidence for us**: nested/learned token compression beats training-free pruning on fine-grained tasks, which supports Nest-VLM vs FastV. |

**Three cross-cutting lessons [INTERP]:**
1. **The ROI signal is nearly free if it comes from a pass you run anyway.** ViCrop, SGL and Q-Zoom all show that an MLLM's own attention localises the evidence (ViCrop shows this even when the model answers wrongly). In a nested model, the n=16 pass is that cheap pass.
2. **Change pixels with crops or tiles, not position-embedding interpolation** (Monkey, S2). For CLIP-L/336, the unit of encoder cost is one 336² view.
3. **Consistency training is what makes mixed or adaptive allocation work** (ViCO 87.9 % → 99.6 %; HyperVL DCL; Q-Zoom post-SFT). Nest-VLM's KL self-distillation is the right base. Extending it to a global view plus a ROI view is the training change that matters most.

---

## 2. The pipeline this points to [PROPOSED]

```
 HR frame (e.g. 1080p) ──► thumbnail 336² ──► CLIP (+LoRA@level) ──► NTR @ n₁ (16) ──► TinyLlama prefill ──┐
                                                                                                         │ answer-to-image attention,
                                                                                                         │ confidence + consistency (SGL)
                                              ┌───────────────────── gate ◄──────────────────────────────┘
                                              │  exit: answer at n₁                  (≈floor TTFT, no extra decode)
                                              │  escalate-tokens: same view, n₁→n₂   (no 2nd encoder pass; journal: ≈0.5× cost)
                                              │  escalate-pixels: ROI crop from HR frame (native res) ─► CLIP ─► NTR @ n₂
                                              │       └► append ROI tokens after cached global prefix (Q-Zoom KV reuse) ─► answer
 optional external ROI: MCU / CPU MGround (CoInfer) or MobileNet VRC (HyperVL) replaces or primes the self-ROI
```

The two elastic axes, stated per query:
- **Pixel axis P** ∈ {thumbnail only; thumbnail + 1 ROI view (crop of the *full-resolution* frame, at native 336²); crop only (CoInfer-style, one pass)}. Cost unit: number of 336² CLIP passes, encoded serially (HyperVL) so peak memory stays flat.
- **Token axis T**: (n₁ for the global view, n₂ for the ROI view) ∈ the nested ladder {16, 64, 144, 256}, allocated per view (ViCO).

Cost model to fill in with measurements (H1 in [experimental-plan.md](experimental-plan.md)):
`TTFT ≈ T_enc + T_pre(n₁) + p_esc·[T_enc + T_pre(n₂ | cached prefix)] + T_gate`.
[INTERP] Since T_enc dominates on the Orin, **the second view pays off only if the escalation rate p_esc is low, or if the accuracy gain on the escalated queries is large.** Q-Zoom's warning (the gate almost always fires for LLaVA-class models on HR benchmarks) means we should report results per benchmark and per question type, not as one average.

Why this can "maintain performance vs non-elastic small VLMs" [INTERP]. The analysis found an anchor-architecture tax at full resolution: TextVQA 21.20 for the NTR at 576 tokens vs 41.00 for the dense baseline (journal §23; analysis.md §1.5). TextVQA is exactly where a GT crop gave LLaVA-1.5 +30 pts on small objects (ViCrop Tab. 1). **The HR crop is the most plausible way to close the TextVQA gap without giving up elasticity.** That is a hypothesis to test first (I-2), not a result.

---

## 3. Ideas, each tied to evidence

**Do first.** Profile and remove fixed overhead (**I-0**). This is not an HR idea but it gates everything else. [FACT] MobileVLM-arch at 144 tokens is *faster* (246.41 ms) than Nest-VLM-TinyLlama at 16 tokens (272.09 ms) (Table V). [INTERP] Candidate causes to measure, in order:
- unmerged nested LoRA in all 24 CLIP layers (rank 64 at n=16), which costs extra kernel launches on Jetson;
- resampler self-attention;
- Python and HF `generate` overheads.

Merging the active level's LoRA into the weights on level switch could recover tens of ms. That is a hypothesis to profile, not a claim. Without this step, every pixel-axis idea starts behind the non-elastic baselines.

Each idea below lists its evidence, what it adds, and what would refute it.

- **I-1. FastV / training-free pruning baselines on the same backbones.** Run FastV (K=2, R∈{50,75,90,97 %}) on the *dense* TinyLlama-576 baseline (journal §23 checkpoint) and on TinyLLaVA / MobileVLM-arch, at token counts matched to Nest-VLM's ladder. Measure on the Orin.
  - Evidence: SGL Tab. 1 (FastV collapses below 35 %), VISOR Tab. 1 (pruning collapses on hard tasks; M3 does not), ViCO Tab. 3.
  - Adds: the "elastic beats pruning on the same backbone" claim you asked for. On our stack FastV also keeps the full encoder plus 2 LM layers at 576 tokens, so its TTFT floor is *higher* than ours.
  - Refuted if: FastV@64 matches Nest-VLM@64 on accuracy at lower TTFT.

- **I-2. Oracle-crop zoom pilot (already Phase 0 in the experimental plan).** On TextVQA (OCR boxes as GT), and on V\* if it can be obtained, crop the smallest square around the GT box (ViCrop protocol) from the *original* image. Run existing Nest-VLM checkpoints at n∈{256, 64, 16} in two modes: (a) crop only, (b) global + crop concatenated (two images; the model never saw this, so it is a training-free test).
  - Evidence: ViCrop Tab. 1 (LLaVA-1.5 small 39→70).
  - Adds: the upper bound, and whether zoom reduces budget sensitivity (H3).
  - Refuted if: no gain on small/medium boxes, in which case the HR direction is dead for this model.

- **I-3. Self-guided ROI: "low budget guides high budget."** Extract answer-to-image attention from the n=16 pass, aggregated over all layers (SGL). Map it through anchors and LDQ cross-attention to the 24×24 patch grid, normalise by a generic-prompt pass (ViCrop `rel-att`), pick a box with ViCrop's sliding-window rule, and crop the *full-resolution* frame. Compare the n=16 guide with an n=256 guide.
  - Evidence: ViCrop Tabs. 2–3 (layer averaging is robust); SGL Tabs. 2, 4 (small guides work, all-layer aggregation matters).
  - Adds: a training-free pixel axis, with no external detector and no MCU needed.
  - Refuted if: ROI hit rate (GT-box coverage) is near that of a centre crop.

- **I-4. Confidence gate / early exit (the only decode saving).** Use SGL's score S = ½(length-normalised sequence probability + consistency under pruning), or simply the first-token probability for single-word answers. Exit at n₁, escalate tokens, or escalate pixels.
  - Evidence: SGL Fig. 4; Q-Zoom Fig. 9; AdaptVision's balance reward (shows naive policies over-escalate).
  - Adds: per-query adaptivity. Easy queries pay the floor; decode with the expensive configuration is skipped for exited queries.
  - Refuted if: the score's AUROC for correctness at n=16 is <0.7 (choose the threshold on a validation set).

- **I-5. Tiny learned operating-point predictor (HyperVL VRC / CoInfer MGround style).** A MobileNet-sized model (optionally conditioned on a question embedding, as MGround does) predicts (P, n) from loss-ratio labels (HyperVL Eq. 2–3; ViCO Eq. 2) generated with our own model.
  - Evidence: HyperVL Tab. 6 (2 ms, −20–29 % tokens, ≈98.5 % retention); ViCO image-level router 98.8 %; Q-Zoom consistency-aware labels.
  - Adds: a decision made *before* the LLM. The journal (decision 23) found this is what changes the economics, because cascades only break even at ≈50 % escalation. It can also run on the CPU or an MCU (the SenSys sensing angle, without needing MGround).
  - Refuted if: the predictor does no better than the best static configuration.

- **I-6. Two-view consistency fine-tune (crop-aware Nest-VLM).** Stage-2 fine-tune with (global@n₁ + ROI@n₂) inputs, ROIs taken from teacher or attention boxes plus random jitter, extending Nest-VLM's KL to the n=256 single-view reference. Use positional tagging for the ROI (box coordinates as text, or a learned view embedding). LLaVA has no MRoPE, so Q-Zoom's alignment trick needs adapting.
  - Evidence: Q-Zoom Tab. IV (SFT fixes context loss), ViCO Tab. 3 (87.9→99.6 % with consistency training), Monkey (per-view LoRA).
  - Adds: makes I-2, I-3 and I-4 robust.
  - Refuted if: the two-view model is no better than concatenation without training.

- **I-7. Per-view nested budget router (ViCO).** Choose (n₁, n₂) jointly, e.g. a 16-token global view plus a 64-token ROI vs 64 + 16. Train the router from loss-ratio labels.
  - Evidence: ViCO Tab. 3 (region-level 99.6 % > image-level 98.8 % > random 97.6 %).
  - Adds: a true 2-D control surface rather than two independent knobs.
  - Refuted if: (16, 64) and (64, 16) give the same accuracy.

- **I-8. Selective multi-scale (S2 only on the ROI).** Encode the ROI region at 2× scale as native 336² tiles, pool to one grid, and feed the NTR. Revive `CLIPVisionTowerS2` with `l_enc` support.
  - Evidence: S2 Tab. 1 (V\* +5–8 at constant tokens), Monkey Tab. 5 (tiling ≫ position-embedding interpolation), HyperVL (serial tiles keep memory flat).
  - Adds: the HR regime (1080p, V\*) where a single 336² crop of a wide ROI is still too coarse.
  - Refuted if: +4 encoder passes cost more TTFT than the accuracy is worth on the Orin.

- **I-9. Learned ROI head on the LM (Q-Zoom SD-RPN).** 2–3 TinyLlama layers initialised from the backbone, trained on ≈10K attention pseudo-labels with three-way label assignment and sink-token filtering, sharing the first prefill.
  - Evidence: Q-Zoom Tabs. II, V, VII (beats attention ROI and GroundingDINO; 10K labels are enough).
  - Adds: removes the generic-prompt pass that I-3 needs (≈2× LM cost at n₁).
  - Refuted if: it does not beat I-3 on ROI hit rate at the same TTFT.

- **I-10. Sensor-side ROI (CoInfer MGround on an MCU).** Already planned ([implementation-plan.md](implementation-plan.md) WP1).
  - Evidence: CoInfer.
  - Adds: the strongest SenSys "sensing" story (load isolation, multi-camera).
  - Refuted if: the MCU overhead (≈157 ms) exceeds the saving on the Orin Nano (H6).

- **I-11. Elastic encoder size (HyperVL DCL).** Add a CLIP-B/16 branch that shares the LM, trained with KL to the CLIP-L branch.
  - Evidence: HyperVL Tab. 5.
  - Adds: a genuine encoder-cost knob for the thumbnail pass, without position-embedding hacks.
  - Refuted if: CLIP-B loses >3 pts after DCL.

- **I-12. Not recommended for this paper.** AdaptVision-style RL tool calling: needs 7B-class reasoning and an LLM judge, and costs decode tokens. VISOR-style LLM surgery: retrains the LM on millions of samples and targets a cost term that is not our bottleneck. Cite both, do not build them.

---

## 4. Ideas ordered by effort and time

Effort assumes one person, the existing repo and checkpoints, access to the cluster (2×A10 or A40) and one Orin Nano. "GPU-h" is a guess, scaled from the paper's ≈85 GPU-h Stage-2. All times are rough guesses; the harness (I-0) is a prerequisite for every row that has Orin numbers.

| Rank | Idea | Training? | Needs Orin? | Est. time | Main evidence | What it buys for SenSys | Fits the 2026-11-05 deadline? |
|---|---|---|---|---|---|---|---|
| 0 | **I-0** Jetson harness + stage profile + LoRA-merge check | none | yes | 2–3 days | Nest-VLM Tab. V; analysis.md §1.5 | Real measured breakdown (H1); fixes the "slower than MobileVLM" problem | **Yes (mandatory)** |
| 1 | **I-2** Oracle-crop zoom pilot (crop / global+crop × n) | none | accuracy: no; latency: yes | 2 days | ViCrop Tab. 1 | Upper bound; go/no-go for the whole HR direction | **Yes** |
| 2 | **I-1** FastV (+ dense, TinyLLaVA, MobileVLM-arch) at matched tokens | none | yes | 2–4 days (baseline env issues in `peak_memory_comparison.csv`) | SGL Tab. 1, VISOR Tab. 1, ViCO Tab. 3 | Baseline table the reviewers will ask for | **Yes** |
| 3 | **I-4** Confidence gate / early exit (SGL score) | none (threshold on val) | yes | 2–3 days | SGL Fig. 4, Q-Zoom Fig. 9 | Per-query adaptivity; the only decode saving | **Yes** |
| 4 | **I-3** Self-guided ROI from the n=16 pass | none | yes | 4–6 days (attention back-projection through NTR) | ViCrop Tabs. 2–3, SGL Tabs. 2/4 | Training-free pixel axis | **Tight** (only if I-2 is positive by ≈10-27) |
| 5 | **I-5** Tiny operating-point predictor (VRC/MGround-like) | small model only (CPU-hours) | yes (2 ms check) | 5–7 days (label generation dominates) | HyperVL Tab. 6, ViCO, Q-Zoom | Decision before the LLM; MCU/CPU deployable | No (follow-up) |
| 6 | **I-8** Selective S2 on the ROI | none (training-free first) / projector retrain | yes | 1–2 weeks | S2 Tab. 1, Monkey Tab. 5, HyperVL | HR (1080p, V\*) regime; memory-bounded tiling | No |
| 7 | **I-6** Two-view consistency fine-tune | Stage-2 warm start, ≈20–40 GPU-h guess, ×3 seeds | for eval | 2–3 weeks | Q-Zoom Tab. IV, ViCO Tab. 3, Monkey | Robust crop-aware elastic model (core method) | No |
| 8 | **I-7** Per-view budget router | router + labels; on top of I-6 | yes | +1–2 weeks after I-6 | ViCO Tab. 3 | True 2-D control surface | No |
| 9 | **I-9** Learned ROI head (SD-RPN) | 2–3 layer branch, ≈10K pseudo-labels | yes | 3–4 weeks | Q-Zoom Tabs. II/V/VII | Removes the extra ROI pass | No |
| 10 | **I-10** MCU-side ROI (MGround + firmware) | small CNN + INT8 + TFLM | yes + STM32 board | 4–6 weeks + hardware lead time | CoInfer | Sensing/load-isolation story, multi-camera | No |
| 11 | **I-11** Elastic encoder size (DCL) | full Stage-1+2 for a new branch (≈100+ GPU-h) | for eval | 5–8 weeks | HyperVL Tab. 5 | Encoder-cost knob without crops | No |
| — | AdaptVision RL / VISOR | heavy | — | 6+ weeks | — | Poor fit (§3, I-12) | No |

---

## 5. Two paper scopes

**A. 2026-11-05 round: "training-free dual elasticity" (ranks 0–4).** [PROPOSED]

*Claim, only if the data supports it:* on an Orin Nano, a nested small VLM that adds (i) a self-guided ROI crop from the full-resolution frame and (ii) a confidence gate traces an accuracy–TTFT–energy front that dominates, on detail-sensitive benchmarks, both:
- the same backbone with FastV, and
- non-elastic TinyLLaVA / MobileVLM-arch,

while matching them on POPE/GQA.

*Must-have figures:*
1. TTFT stage breakdown vs *n* (the encoder floor);
2. Pareto plots for TextVQA, V\*, POPE and GQA, per question type and object-size bin (ViCrop's S bins);
3. escalation-rate vs accuracy curve;
4. energy per query and memory;
5. a FastV collapse curve on the same backbone.

*Honest limits:* it is a training-free composition, small models are not shown to localise as well as 7B ones, decode is reported but not reduced, and there is no sensor-side component.

[INTERP] This is a narrower, characterisation-plus-composition paper, and **its risk is high with 7 days to the abstract and 14 to the paper.** The go/no-go is I-2 on about 2026-10-27: if the oracle crop gives no gain on small/medium TextVQA boxes for Nest-VLM, do not submit this round. The dual-submission question about the Nest-VLM manuscript (README, open question 7) also has to be settled before registering an abstract, because both papers would report the same checkpoints.

**B. Later cycle: "dual-elastic, HR-aware small VLM with an on-device controller" (ranks 0–8, optionally 10).**
- Method: the crop-aware nested model (I-6) with per-view budgets (I-7), selective multi-scale for HR frames (I-8), and a pre-LLM controller (I-5), optionally on the sensor MCU (I-10).
- Evaluation: the full plan in [experimental-plan.md](experimental-plan.md): contention, multi-camera, ≥3 seeds, CIs.
- This is the version that matches SenSys's "embedded AI / edge computing / sensor systems" framing best, because the HR frame comes from the sensor and the controller makes a per-query, resource-aware decision on real hardware.

**SenSys fit (CFP re-fetched 2026-10-22) [FACT].**
- Round 2: abstract 2026-10-29, paper 2026-11-05 (AoE); full papers ≤12 pages plus unlimited references and appendices.
- The topics include embedded AI and edge computing. The page states no review criteria for regular papers and does not require a real prototype. [INTERP] Reviewers at this venue will still expect measured on-device latency, energy and memory, not FLOPs.
- Submissions must be fully anonymised (desk rejection otherwise), and artifact links must be anonymous.

---

## 6. Things to watch

- **Baseline numbers.** The Nest-VLM Table IV "576-token Nest-VLM" row appears to be the dense baseline (analysis.md §1.5). Any "maintains performance vs non-elastic" claim must use the correct rows.
- **TextVQA protocol.** ViCrop evaluates without external OCR tokens. Use one protocol for every model and state which.
- **Transfer gap.** Every effect size in §1 is from 7B-class models (or 0.5–8B for VISOR/ViCO/HyperVL) on server GPUs or a phone NPU, and none is from TinyLlama on an Orin. Treat them as motivation, not predictions.
- **Second encoder pass.** On the Orin it may cost ≈ the whole TTFT floor. Design and report around the escalation rate.
- **Single-seed fragility.** The repo's results are n=1 (journal). Gaps ≤1 pt need seeds and CIs before they go in a table.

---

## 7. Crop pilot results (I-2, run 2026-10-22 on DAS-6)

**Setup [FACT].** Checkpoint `elastic-finetune-tinyllama-v8-parcel` (the `final-parcel` recipe), TextVQA val restricted to the 4,326 questions that have a ViCrop ground-truth answer box (`jrzhang/TextVQA_GT_bbox`, joined by question id; 44 zero-area boxes dropped, so the small bin is 729, not ViCrop's 773). Three input modes per token budget (256/144/64/16): the full image, the smallest square crop around the GT box (resized by the model's normal preprocessing), and both images together (ViCrop's "human-CROP"). Same `lmms-eval` wrapper, prompt, template and scorer as `eval_lmms_level.sh`. Code: `lmms-eval/lmms_eval/tasks/textvqa_croppilot/`, `eval_croppilot_level.sh`, `scripts/croppilot/` in the FlexLLaVA repo. CIs are paired bootstraps resampling images (10,000 draws).

**Control [FACT].** The "full image" mode reproduces the earlier `eval_lmms_level.sh` run of the same checkpoint to within 0.2 points at every budget (e.g. 27.37 vs 27.61 at 256 tokens; 27–32 of 4,326 answers differ), so this pipeline is the one behind the paper's numbers.

Accuracy (TextVQA metric ×100), full image / crop only / full image + crop:

| Tokens | All (4,326) | Small box (729) | Medium (2,411) | Large (1,186) |
|---|---|---|---|---|
| 256 | 27.4 / 42.4 / 46.1 | 16.5 / 44.2 / 47.8 | 25.8 / 43.3 / 47.0 | 37.2 / 39.5 / 43.4 |
| 144 | 25.3 / 41.4 / 45.2 | 15.5 / 44.2 / 47.2 | 23.9 / 41.8 / 45.8 | 34.0 / 38.7 / 42.7 |
| 64 | 22.5 / 40.3 / 42.8 | 14.6 / 43.0 / 47.1 | 20.8 / 40.9 / 43.0 | 30.8 / 37.3 / 39.9 |
| 16 | 18.5 / 33.8 / 34.5 | 12.3 / 37.7 / 37.3 | 17.0 / 33.7 / 34.5 | 25.6 / 31.6 / 32.9 |

Paired crop − full (95 % CI), all questions: +15.1 [13.6, 16.5] at 256, +16.1 [14.7, 17.5] at 144, +17.7 [16.3, 19.2] at 64, +15.3 [14.0, 16.6] at 16. Small boxes: +25 to +29 at every budget; large boxes: +2.3 [0.2, 4.4] at 256 up to +6.5 at 64. Every cell's CI excludes zero.

**What it shows [INTERP].**
- Zoom is worth more than any amount of token budget on this model. Crop-only at 16 tokens (33.8) beats the full image at 256 tokens (27.4).
- H3 (zoom makes the model less sensitive to the budget): holds from 256 to 64 tokens (full image loses 4.9 points, crop loses 2.2, i.e. 55 % less) and **fails from 256 to 16** (both lose about 8.7).
- Adding the full image to the crop helps when the budget is large (+3.7 at 256, +3.8 at 144, +2.6 at 64) and not at 16 (+0.7, inside the noise). It also doubles the visual tokens and the encoder work, so it is the expensive way to buy those points.
- The pilot passes the plan's go/no-go (≥ +2 points on detail questions, or less budget sensitivity).

**Caveats [FACT/INTERP].** The crops use ground-truth answer boxes: this is an upper bound for any ROI method, and on TextVQA the answer text is literally what sits in the box. Only TextVQA was tested (the benchmark where reading small text is the whole task), one checkpoint, one run per cell, no seeds, no latency measured, and 44 questions dropped.

---

## 8. What the Nest-VLM Jetson logs already tell us (until I-0 re-measures)

Source: `Matryoshka Learning/jetson_logs/` (final report plus raw `nestbench-results_*` JSONL; I used the latest snapshot `nestbench-results_20260919_0422`). Board: Orin Nano Super dev kit, 15 W mode, JetPack 6.2.1, stock DVFS, FP16, 8 timed iterations after 3 warm-ups, 4 queries, 8 new tokens, stage timers via forward hooks (hook overhead 1.6 %). **The elastic resampler/projector weights were untrained in these runs (`elastic_weights_pretrained: false`), which is valid for timing but not for accuracy.** The log set has no encoder-resolution, crop, or two-pass measurement, and the level-switch-cost file (`exp06_switch.jsonl`) contains only a provenance header, no data.

### 8.1 Per-stage time, TinyLlama-1.1B (ms per query)

| Visual tokens | Prefill seq | Vision tower | Resampler | Projector | LM prefill | TTFT (median) | Vision + resampler share |
|---|---|---|---|---|---|---|---|
| 256 | 307 | 184.1 | 7.3 | 0.8 | 188.6 | 392.9 | 49 % |
| 144 | 195 | 184.1 | 6.0 | 0.9 | 107.9 | 306.9 | 62 % |
| 64 | 115 | 184.8 | 5.8 | 0.5 | 75.1 | 271.6 | 70 % |
| 16 | 67 | 186.5 | 5.8 | 0.5 | 74.2 | 272.1 | 71 % |

The same table at MAXN_SUPER: vision tower 140–142 ms at every budget, LM prefill 63.5 ms at 16 tokens, TTFT 216 ms. Decode is 77.3 ms/token at 15 W (12.9 tok/s) and does not change with the budget.

**What this settles or changes [FACT unless tagged]:**
1. **H1 holds, from existing data.** At 16–64 tokens the vision tower plus resampler is ≈ 70 % of TTFT (hypothesis H1 asked for ≥ 50 %). The tower costs the same 184–187 ms whatever the budget, which is exactly why TTFT stops improving below 64 tokens.
2. **The LM prefill is flat from 67 to 115 tokens (74.2 → 75.1 ms)** and only starts to grow after that (107.9 at 195, 188.6 at 307). [INTERP] Prefill at these lengths is bound by reading the weights, not by token count. So **64 tokens cost nothing over 16 on this board**, and the 16-token level has no latency advantage over 64. In the pilot, crop-only at 64 tokens scores 6.5 points higher than at 16 (40.3 vs 33.8), at no extra modelled latency. Any second pass in our pipeline should therefore use 64 tokens.
3. **The vision tower looks about 70–80 ms slower than it should be, probably because of the unmerged nested LoRA [INTERP, to be tested by I-0].** Evidence: the MobileVLM-arch baseline (same CLIP-L tower, same MobileLLaMA-1.4B, no LoRA) takes 246.4 ms at 144 tokens, while Nest-VLM-MobileLLaMA takes 333.6 ms at the same sequence length, where the logs put its LM prefill at 131.8 ms and its tower at 181 ms. Subtracting the same LM time from the baseline leaves ≈ 105–115 ms for the baseline's tower and connector, i.e. ≈ 70–75 ms less than ours. The resampler (≈ 6 ms) cannot explain that gap. The TinyLlama pair (TinyLLaVA at 576 tokens, 440 ms, vs Nest at 256, 392.9 ms) is consistent with a tower near 105 ms and not with 184 ms, given Nest's own LM-vs-length slope. The nested LoRA is applied as extra matmuls on all six linears of all CLIP layers (`NestedLoRALinear.forward`), so this is plausible. If merging the active level's LoRA into the weights recovers even 50 ms, it saves more than the entire 256→16-token gain (121 ms). Caveats: different harnesses for baseline and Nest rows, random-weight baselines, and merging per level costs memory (≈ 0.6 GB per merged copy of CLIP-L in FP16) or a switch cost that was never measured.
4. **Correction to §0(b).** I said decode is limited by reading the weights. The logs do not support that: 77 ms/token means ≈ 28 GB/s effective for 2.2 GB of FP16 weights, against a board bandwidth of ≈ 102 GB/s (spec as I recall it, not in the logs). So decode on this stack is ≈ 3× slower than a bandwidth floor, which points at launch/Python overhead or low clocks. The conclusion that visual tokens do not change decode time stands, but decode is a real engineering target on its own (CUDA graphs, static KV cache, a leaner runtime). With a fixed 8 new tokens it is 541 of 813 ms end to end (67 %); for TextVQA-style answers of 2–4 tokens it is still ≈ 150–230 ms on top of a 272 ms TTFT.
5. **Memory** is unaffected by the budget (CUDA peak 3.05 GB TinyLlama). The swap readings in the logs are noisy (the report shows they are not a model-size signal), so only CUDA/RSS peaks should be used for "fits in 8 GB". Adding a second image costs little memory in the analytic estimate (1.98 → 2.03 GB in the pilot), but that estimate is not a Jetson measurement.

### 8.2 Latency model for planning (TinyLlama, 15 W, from the table above)

`TTFT(one pass, n tokens) ≈ 184.5 + 6 + LM(n)`, with LM(16) = 74.2, LM(64) = 75.1, LM(144) = 107.9, LM(256) = 188.6 ms. This is a **model built from the logs, not a measurement of any pipeline in this repo**:

| Configuration | Modelled TTFT | Pilot accuracy (4,326 q) |
|---|---|---|
| full image, 256 tokens | 393 ms | 27.4 |
| full image, 64 tokens | 272 ms | 22.5 |
| oracle crop only, 64 tokens (needs the box from somewhere free) | 272 ms | 40.3 |
| oracle crop only, 256 tokens | 393 ms | 42.4 |
| self-guided: 16-token guide pass, then crop at 64 (two passes) | 272 + 272 = 544 ms | to be measured (§9) |
| same with an early exit that skips the second pass for a fraction p of queries | 272 + (1 − p)·272 ms | to be measured |

[INTERP] The two-pass self-guided pipeline costs about 150 ms more than the full image at 256 tokens unless a gate skips the second pass for roughly 55 % of the queries or more, or unless the guide pass is made much cheaper. A box that arrives for free (sensor-side, as in CoInfer) makes the single-pass crop strictly better than the full-image baselines on this metric.

---

## 9. I-3: self-guided ROI (status)

**Design [PROPOSED, implemented as `scripts/croppilot/selfroi_*.py` in the FlexLLaVA repo].**
- *Stage A (GPU):* run the global image through the model at a guide budget (16 or 256 visual tokens) and record, for every LM layer, the answer-to-image attention on the 24×24 patch grid. LM attention goes to the visual tokens (head-averaged, last prompt position, i.e. the position that predicts the first answer token); the visual tokens are mapped back to patches through the resampler (anchors: uniform over their 12×12 or 3×3 pooling window; learned queries: the resampler's cross-attention weights, captured by forward hooks). The same is done for ViCrop's generic prompt ("Write a general description of the image.") so relative maps can be formed. The guide pass's own greedy answer, its TextVQA score, first-token probability and mean token probability are saved for the early-exit gate.
- *Stage B (CPU):* candidate maps are raw per layer, relative per layer, all-layer raw sum (SGL-style) and all-layer relative. Each becomes a crop window by ViCrop's rule (square windows of 336 × {1, 1.2, …, 2} px, best position per size, then the size whose window sum differs most from its neighbours'). The map variant is chosen on a seeded 500-question calibration split; every reported number uses the other 3,826 questions. Controls with the same window sizes: centre, random, and centred on the GT box (separates "where" from "how big").
- *Stage C (GPU):* the pilot's `crop` and `globalcrop` tasks answer with these windows (`SRC_TAG=g16 MODE=crop sbatch eval_croppilot_level.sh`).

Results: see §10 once the runs finish.
