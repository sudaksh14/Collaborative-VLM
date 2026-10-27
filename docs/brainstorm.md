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

**Status of this pipeline after I-2, I-3, I-4 and the Jetson-log analysis (revised 2026-10-24).** The diagram above stays as the *proposal*; the evidence so far says this about each branch:

| Branch of the pipeline | Evidence | Verdict |
|---|---|---|
| First stage at n₁ = 16 | §8: LM prefill is flat from 67 to 115 tokens (74.2 vs 75.1 ms), the tower costs 184 ms at every budget. Full image at 64 tokens scores 22.6 vs 18.6 at 16 on the pilot questions. | **Change n₁ to 64.** n₁ = 16 buys no latency on the Orin and costs 4 accuracy points. |
| Escalate tokens (same view, n₁ → n₂, no second tower pass) | Not run as a cascade. | **Untested.** Cheapest escalation by the cost model; only meaningful once n₁ > 64 or when decode time matters. |
| Gate = SGL-style confidence | §10.4: mean-token probability separates correct from wrong answers (AUROC 0.89), first-token probability does not clear 0.7, and neither predicts that a crop will help (AUROC 0.38–0.46). No consistency-under-pruning term implemented. | **Partly supported.** Usable as an "answer is probably fine" exit, not as a "zoom now" trigger. |
| Escalate pixels with a *self-guided* ROI from the n₁ pass | §10: +2.4 over the full image at 64 tokens, +3.2 over a centre window, small boxes ≈ centre crop; the best cascade (24.3–24.7 at 400–427 ms) is dominated by full-image 256 tokens (27.3 at 381 ms). | **Not supported on this model.** The second tower pass is the cost; the guide is too coarse. |
| Escalate pixels with a *free* ROI (oracle or external: MCU / CPU MGround, MobileNet VRC) | §7: oracle crop at 64 tokens = 39.9–40.3 vs 22.6 for the full image, 15 points above the self-guided crop. | **Supported as the upper bound.** This is the arm that keeps the pipeline alive; I-9/I-10 matter more than I-3. |
| KV reuse of the global prefix for the ROI pass (Q-Zoom) | Not implemented. The latency model charges a full second pass; with the tower at 184 ms of ≈ 270, reuse saves only the LM share. | **Low priority.** Revisit if the LM share grows. |
| Pixel axis P: thumbnail only / thumbnail + ROI / ROI only | §7, §10: both ROI modes run; thumbnail + ROI is +1.4 (64 tokens) over ROI only for self-ROI, +2.8 for the oracle. | **Both kept.** |
| Evidence base | TextVQA val images (not true 1080p frames), one model, accuracy from the lmms scorer, latency modelled from the Nest-VLM logs. V\* and the Orin harness (I-0) are still missing. | **Provisional.** |

[INTERP] Net: the self-guided two-pass form of the pipeline loses to simply running the full image at 256 tokens on this model; the pipeline is worth building only around a box that arrives without a second tower pass, or with a much cheaper guide.

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

---

## 10. I-3 results (run 2026-10-23 on DAS-6, TextVQA val, 3,826 evaluation questions)

Source: `croppilot/logs/selfroi_analyze_27934.out` and `selfroi_boxes_27926.out` on DAS-6. The map variant was picked on 500 calibration questions (16-token guide: raw layer 16; 256-token guide: raw layer 17); every number below uses the other 3,826. Paired differences are in points with 95 % CIs. Accuracy is the pilot's lmms TextVQA score, so absolute values are low (full image at 256 tokens = 27.3).

### 10.1 Where does the guide look? (ROI hit rate)

| Box source | GT-box coverage | Centre of the window inside the GT box |
|---|---|---|
| 16-token guide | 77.5 % | 81 % |
| 256-token guide | 85.8 % | 89 % |
| centre window (same sizes) | 70.9 % | — |
| random window (same sizes) | 67.2 % | — |
| GT-centred window (same sizes) | 98.1 % | — |

[FACT] Refutation test for I-3 ("hit rate near a centre crop") **is not met, but the margin is small**: +6.6 points over the centre window at 16 tokens, +14.9 at 256. Window IoU with the GT box is only 7.5 % (16 tok) and 9.7 % (256 tok); on small boxes (n = 637) IoU is 0.5 %, i.e. the window covers the box but is far larger than it.

### 10.2 Accuracy, crop only (windows from the 16-token guide)

| Eval budget | full image | self-ROI (16-tok guide) | centre window | random window | GT-centred window | oracle GT crop |
|---|---|---|---|---|---|---|
| 256 tokens | 27.3 | 29.6 | 27.1 | 25.4 | 33.5 | 42.3 |
| 64 tokens | 22.6 | 24.9 | 21.7 | 21.1 | 27.4 | 39.9 |
| 16 tokens | 18.6 | 20.4 | 18.0 | 17.5 | 22.1 | 33.6 |

[FACT] Paired gains of self-ROI (16-tok guide) at 64 tokens: +2.4 [+1.3, +3.4] over the full image, +3.2 [+2.1, +4.3] over the centre window, and −2.5 [−3.3, −1.6] against a window centred on the true box with the same size. The oracle crop is +15.0 [+13.5, +16.4] above it (small boxes +23.9). The 256-token guide adds +1.4 to +2.5 over the 16-token guide at the same eval budget (e.g. 26.3 vs 24.9 at 64 tokens).

[FACT] **Small boxes are where it fails.** At 64 tokens self-ROI is +3.9 [+1.7, +6.2] over the full image on small boxes, but only +2.1 [−0.3, +4.4] over the centre window. At 256 tokens, large boxes are *worse* than the full image (−2.5 [−4.8, −0.2]) because the crop discards context.

[FACT] Adding the global image to the crop (two images, no training) helps self-ROI by +1.4 [+0.7, +2.1] at 64 tokens, mostly on large boxes (+3.0), and is neutral on small ones. Oracle crop + image reaches 42.8 at 64 tokens.

### 10.3 Cascade (16-token guide pass, escalate to a 64-token self-ROI crop when not confident)

Guide pass alone: 19.3 accuracy at 267 ms (model, §8.2). Always escalating: 24.9 at 533 ms. Reference: full image at 256 tokens = 27.3 at 381 ms (model).

| Escalate | gate = first-token prob. | gate = mean token prob. | TTFT, model (ms) |
|---|---|---|---|
| 0 % | 19.3 | 19.3 | 267 |
| 40 % | 23.4 | 22.8 | 373 |
| 50 % | 24.3 | 23.5 | 400 |
| 60 % | 24.7 | 24.2 | 427 |
| 80 % | 24.3 | 25.4 | 480 |
| 100 % | 24.9 | 24.9 | 533 |

Random gate at 40 %: 21.5; oracle gate (escalate only the wrong guide answers): 28.4 at 83 % escalation.

### 10.4 What this settles

1. **[INTERP] As a latency win on this model, I-3 is negative.** The best cascade point (first-token gate, 50–60 % escalation: 24.3–24.7 at 400–427 ms) is *below* simply running the full image at 256 tokens (27.3 at 381 ms) in the §8.2 model, so it is dominated on both axes. Even the oracle gate only reaches 28.4 at 83 % escalation. The two-pass design pays the vision tower twice (§8.1), which is what §8.2 predicted.
2. **[INTERP] The training-free pixel axis is real but weak.** Self-ROI beats centre and random windows with CIs excluding zero overall (+2.5 to +4.3), so the guide carries signal, and the oracle gap (12–15 points) shows the headroom. But about two thirds of that headroom comes from *where exactly* the box is (GT-centred window with the same size is +3.9 above self-ROI at 256 tokens), and small boxes get almost nothing over a centre crop.
3. **[INTERP] The guide quality is the lever.** A better guide (256-token) gives +1.4 to +2.5; a free, accurate box (sensor-side, I-10, or a learned ROI head, I-9) is the case where the crop's +12–15 points over a 64-token full image becomes usable without the second tower pass. This strengthens I-9/I-10 relative to I-3, and makes the pre-LLM predictor (I-5) the more promising way to use the gate signal.
4. **Gate signal (I-4 check, `scripts/croppilot/gate_auroc.py`, job 27950; 3,826 evaluation questions, image-cluster bootstrap CIs).**

   | Score from the guide pass | AUROC for "guide answer correct" (16-tok / 256-tok guide) | AUROC for "crop@64 beats the guide answer" (16-tok / 256-tok guide) |
   |---|---|---|
   | first-token probability | 0.687 [0.669, 0.705] / 0.676 [0.659, 0.694] | 0.397 [0.369, 0.425] / 0.379 [0.345, 0.415] |
   | mean token probability | 0.888 [0.874, 0.901] / 0.883 [0.871, 0.895] | 0.463 [0.438, 0.492] / 0.451 [0.416, 0.487] |

   [FACT] Against I-4's refutation test (AUROC < 0.7 for correctness at n = 16): the first-token score fails it (0.687, upper CI 0.705 only just reaches it), the mean-token score passes clearly (0.888). [INTERP] The mean-token score may partly measure answer length (long, hard-to-read strings get a low mean), so it is not evidence of calibrated uncertainty. [FACT] Neither score predicts *when the crop helps*: AUROC for "crop beats the guide answer" is below 0.5 for both (positives are only 10 % of questions), so the gate separates "answer is probably right" from "probably wrong" but not "wrong and fixable by zooming". That explains why the cascade in §10.3 sits close to random escalation plus a small offset, and why the oracle gate (28.4) is far above anything the confidence scores reach. An operating point fixed on the calibration split (escalate the least-confident half there) gives 47 % / 44 % escalation and 23.9 / 23.1 accuracy on the evaluation split (first-token / mean-token), again below full-image 256 tokens (27.3).

**Caveats:** TTFT figures are the §8.2 model, not a measured pipeline; one model (TinyLlama nested), one dataset (TextVQA, OCR-heavy), absolute accuracy is low because of the lmms prompt setup; the selected layers come from a 500-question calibration split.

---

## 11. I-1: FastV and other training-free baselines (status, 2026-10-24)

**Implemented [FACT, FlexLLaVA commit d39b65e].** `llava/model/fastv.py` implements FastV as a key mask: the first K = 2 layers see all 576 visual tokens, the lowest-scoring fraction R (by last-token attention at layer K−1, mean over heads) is masked as keys in all later layers, at prefill and during decode. Logits equal those of physically removing the tokens (positions unchanged, as in FastV's reference code). It does *not* reproduce FastV's latency saving, because the dropped tokens' keys and values are still computed; latency has to come from the analytic model or the Orin harness. Self-test (job 27953, on the M3-ladder TinyLlama checkpoint): R = 0 is token-for-token identical to the unpatched model, R = 0.75 keeps 144 of 576 tokens and changes the output, and a left-padded batch of two prompts of different lengths gives the same text as running them one by one.

**Eval design.** Dense TinyLlama-576 (same backbone, vision tower and data as Nest-VLM) with FastV at kept-token counts 256 / 144 / 64 / 16 (R = 1 − n/576), five tasks (MME, POPE, ScienceQA-img, TextVQA, GQA), compared with `final-parcel` at the same token counts and with the dense 576-token numbers (MME-P 1248.3, POPE 84.9, SciQA 57.4, TextVQA 41.0, GQA 58.3). Also planned from the doc: K = 4 as a sensitivity check, and the Orin latency once I-0 exists.

**Blocker found and handled.** The dense checkpoint `baseline-tinyllama-576tok-finetune` was deleted in the September cleanup (only its eval logs survive). Stage 2 is being re-trained from the existing plain-projector Stage 1 (`baseline-tinyllama-4tok-pretrain`, trained on all 576 tokens) with the original recipe: job 27952 (A10:2, node208; the original run took 69,983 s ≈ 19.4 h). Chained behind it with `afterok`: the dense re-evaluation (27954, logs under `eval_logs/baseline-tinyllama-576tok-rerun/`) and the FastV array (27955, logs under `eval_logs/baseline-tinyllama-576tok-finetune-fastv-k2/<n>tok/`). The dense checkpoint was not on the SURFdrive share (checked 2026-10-24: only `baseline-tinyllama-4tok-pretrain` and `baseline-tinyllama-ladder-finetune` among the baselines), so an upload of the new `baseline-tinyllama-576tok-finetune` (without its `checkpoint-N/` optimizer states) is chained behind the training as job 27956. [INTERP] The re-trained model should land near the old numbers but will not be bit-identical; the re-evaluation tells us how far, and the old numbers in the table above should be replaced by the re-evaluated ones if they differ.

**Results:** §12.

---

## 12. I-1 results: FastV (K = 2) on the dense backbone vs Nest-VLM (run 2026-10-24 to 26 on DAS-6, full benchmarks)

Source: `scripts/summarize_fastv.py` (job output `croppilot/logs/summarize_fastv_*.out`). FastV rows use the *re-trained* dense TinyLlama-576 (A10:2, finished 2026-10-26); Nest-VLM rows are `final-parcel` (v8-parcel). "Kept" is the number of visual tokens visible from layer 2 on; the first two layers always see all 576. MME-P is the perception score; the other columns are percentages.

| Config | MME-P | POPE | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| dense 576, old checkpoint (deleted; from eval logs) | 1248.3 | 84.89 | 57.36 | 41.00 | 58.30 |
| dense 576, re-trained | 1250.2 | 84.58 | 56.52 | 41.19 | 58.39 |
| FastV keep 256 | 1235.7 | 84.39 | 51.91 | 39.63 | 57.52 |
| Nest-VLM @ 256 | 1170.3 | 84.33 | 48.14 | 28.47 | 56.07 |
| FastV keep 144 | 1196.4 | 82.96 | 51.71 | 37.30 | 56.36 |
| Nest-VLM @ 144 | 1142.2 | 84.23 | 48.09 | 26.34 | 55.30 |
| FastV keep 64 | 1132.7 | 77.32 | 50.57 | 30.23 | 52.39 |
| Nest-VLM @ 64 | 1155.1 | 83.03 | 47.45 | 23.73 | 54.52 |
| FastV keep 16 | 744.0 | 59.90 | 46.26 | 14.98 | 42.77 |
| Nest-VLM @ 16 | 1123.9 | 80.49 | 46.36 | 19.71 | 52.61 |

[FACT] **The re-trained dense model reproduces the deleted one** (TextVQA 41.19 vs 41.00, MME-P 1250.2 vs 1248.3, GQA 58.39 vs 58.30, POPE −0.3, SciQA −0.8), so FastV is measured on the same dense baseline the journal used. The checkpoint is now also on the SURFdrive share.

**Retention against each method's own full-budget model [computed from the table; Nest's own 256-token level is its reference, the dense model is FastV's]:**

| Kept tokens | FastV TextVQA | Nest TextVQA | FastV POPE | Nest POPE | FastV GQA | Nest GQA |
|---|---|---|---|---|---|---|
| 144 | 90.6 % | 92.5 % | 98.1 % | 99.9 % | 96.5 % | 98.6 % |
| 64 | 73.4 % | 83.3 % | 91.4 % | 98.5 % | 89.7 % | 97.2 % |
| 16 | 36.4 % | 69.2 % | 70.8 % | 95.4 % | 73.3 % | 93.8 % |

**What this settles**

1. **[FACT] "Elastic beats pruning" holds below about 64 tokens and not above.** At 16 tokens Nest-VLM wins every benchmark except SciQA (tie, 46.4 vs 46.3), by 380 MME-P points, 20.6 POPE, 4.7 TextVQA, 9.8 GQA. At 64 tokens it wins POPE (+5.7), GQA (+2.1), MME-P (+22) and loses TextVQA (−6.5) and SciQA (−3.1). At 256 and 144 tokens FastV is ahead on TextVQA by 11–13 points, SciQA by 3.6–3.8 and MME-P by 54–65.
2. **[INTERP] The comparison is not "same backbone".** Most of FastV's absolute lead at ≥ 64 tokens is the architecture tax already in the journal (§23): the NTR/anchor model sits at 28.5 TextVQA at its own best level against 41.2 for the dense projector, a 12.7-point gap that exists before any pruning. In retention terms Nest degrades much more gracefully (TextVQA 83 % at 64 tokens against FastV's 73 %; at 16 tokens 69 % against 36 %). The claim we can defend is about robustness at low budgets, not about absolute accuracy at moderate ones.
3. **[FACT] I-1's refutation test ("FastV@64 matches Nest-VLM@64 on accuracy at lower TTFT") is not met on accuracy** (mixed: FastV better on TextVQA and SciQA, Nest better on POPE, GQA and MME-P), and TTFT is not lower for FastV in the model below, so the idea is not refuted; but the unconditional claim is not supported either.
4. **[INTERP, extrapolated model, not measured] FastV's TTFT floor is a little above Nest's.** Using §8.2's per-stage numbers (tower 184.5 ms, projector ≈ 1 ms), the 22-layer LM at 2/22 of its cost on the full 627-token sequence (LM(627) ≈ 419 ms, linearly extrapolated from the 195 → 307 token slope of 0.72 ms/token, which is a guess) plus 20/22 of LM(n + 51): FastV ≈ 395 / 322 / 292 / 291 ms at 256 / 144 / 64 / 16 kept tokens against Nest's measured 393 / 307 / 272 / 272. So FastV is 2 to 21 ms slower at equal kept tokens. **Caveat that can flip this:** the same §8.1 analysis suspects Nest's 184 ms tower is ≈ 70–80 ms inflated by the unmerged nested LoRA, while a dense tower should be nearer 105–115 ms. If that is right, FastV would be about 60–80 ms *faster* than Nest at every budget until the LoRA is merged, which would reverse the latency half of the story. The merge check in I-0 is therefore decisive for this comparison.
5. **[INTERP] It reinforces the HR-crop direction.** In the crop pilot, the oracle crop on Nest-VLM at 64 tokens scored 39.9–40.3 TextVQA (§7; the pilot's question subset, so not exactly comparable), i.e. roughly the dense model's 41 at 64 tokens, while FastV at 64 tokens is at 30.2. A crop closes the tax that pruning cannot, but only when the box comes for free (§10).

**Not yet done:** K = 4 and K = 8 sensitivity (jobs 27962 and 27963; FastV at 97 % pruning with K = 2 is known to be harsh, so these decide whether the 16- and 64-token FastV rows are a fair baseline), TinyLLaVA/MobileVLM-architecture baselines, and any Orin measurement (I-0).

---

## 13. Why Nest-VLM trails the dense model: recipe and architecture audit vs M3 / MQT / PARCEL (2026-10-26)

Source: `scripts/summarize_tinyllama.py` over every TinyLlama eval on DAS-6; code read in `llava/model/elastic/resampler.py`, `llava_elastic_mixin.py`, `llava_trainer.py`, the Stage-1/2 launchers and `submit_elastic_run.sh`; profile `results/profile_overhead_elastic-finetune-tinyllama-v8-parcel_NVIDIA_A10.json`.

### 13.1 Same-backbone numbers (TinyLlama-1.1B, CLIP-L/336, LLaVA-1.5 data)

| Model @ visual tokens | MME-P | POPE | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| dense 576 (TinyLLaVA-equivalent, re-trained) | 1250.2 | 84.58 | 56.52 | 41.19 | 58.39 |
| **final-parcel** @ 256 / 144 / 64 / 16 | 1170 / 1142 / 1155 / 1124 | 84.3 / 84.2 / 83.0 / 80.5 | 48.1 / 48.1 / 47.5 / 46.4 | **28.5 / 26.3 / 23.7 / 19.7** | 56.1 / 55.3 / 54.5 / 52.6 |
| MQT ladder (own recipe, budgets 256/144/64/16) | 1194 / 1146 / 1160 / 1146 | 83.2 / 81.9 / 81.2 / 79.4 | 53.3 / 52.9 / 52.9 / 52.3 | 24.9 / 22.7 / 21.6 / 18.4 | 54.8 / 54.2 / 53.4 / 51.8 |
| parcel-faithful (single budget per step, no KD, no LoRA) | 1089 / 1102 / 1099 / 1046 | 80.2 / 80.2 / 79.9 / 79.1 | 43.9 / 44.1 / 44.6 / 44.3 | 20.5 / 20.0 / 19.6 / 17.9 | 51.3 / 51.0 / 50.8 / 49.8 |
| v11 = final-parcel without vision LoRA | 1087 / 1117 / 1101 / 987 | 83.5 / 83.5 / 82.4 / 79.7 | 50.2 / 50.4 / 49.6 / 47.5 | 22.2 / 19.3 / 16.2 / 12.1 | 54.0 / 53.6 / 52.5 / 50.3 |
| final-parcel-576ctrl (same architecture, one level, 576 tokens) | 1113 | 81.8 | 50.7 | 21.2 | 52.6 |
| M3, 4-token model | 1176 | 80.6 | 53.5 | 17.1 | 54.1 |
| MQT, 4-token model | 1106 | 78.9 | 51.5 | 14.6 | 50.9 |
| **M3 ladder** (256/144/64/16; re-evaluated 2026-10-27, job 27973) | 1252 / 1255 / 1253 / 1245 | 84.5 / 84.6 / 83.9 / 82.6 | 56.8 / 56.9 / 55.4 / 53.7 | **35.1 / 34.3 / 30.6 / 21.2** | 57.9 / 58.2 / 57.7 / 55.7 |

[FACT] **Against the elastic baseline we have on the same backbone, we are not behind.** final-parcel beats the MQT ladder on TextVQA (+1.3 to +3.6), POPE (+1.1 to +2.3) and GQA (+0.8 to +1.3) at every budget; it loses SciQA by 4.8 to 5.9 and MME-P by 4 to 24 points. It also beats our PARCEL reproduction (parcel-faithful) by 8 TextVQA points at 256. The large gap is to the **dense** model (−12.7 TextVQA at 256), and MQT shares it (−16.3).

### 13.2 The missing piece: the tokens the LLM sees are no longer CLIP patch features

[FACT, from the code] At budget 256, final-parcel emits 64 average-pooled anchors plus 192 learned queries. The anchors are not passed through unchanged: they go through a self-attention block jointly with the queries and then the resampler's output LayerNorm, before the projector. The queries come from a two-layer cross-attention stack initialised at random. M3, the dense model and FastV all hand the projector raw (pooled or selected) CLIP patch features.

[FACT] The 576ctrl run is consistent with this: the same architecture at 576 tokens, with no compression and no ladder (so no elastic training signal and no loss weighting at all), scores TextVQA 21.2 against 41.2 for the dense projector. It does not isolate the connector, though: 576ctrl also starts from a randomly initialised resampler and a fresh LM, uses the Stage 1/2 recipe of §13.3 (Stage 1 at the 256 level only, resampler at the projector's 2e-5), and may carry the vision LoRA. FastV (§12), which keeps raw patch tokens and only drops some after layer 2, holds 39.6 with 256 tokens and 37.3 with 144.

[INTERP] So the likeliest explanation is that most of the deficit is the connector, not elasticity, the loss or the LoRA (not yet isolated; change 1 in §13.3 is the test). A randomly initialised resampler trained for one epoch of 558K captions (Stage 1, at the 256 level only) and then at LR 2e-5 inside a 665K Stage 2 does not learn to carry the fine detail (OCR) that the identity path keeps for free. MQT's one-layer resampler has the same problem. PARCEL's published retention comes from a setting we have not matched (its main text does not give the LM training setup; our faithful run is 8 points below final-parcel).

**Prediction to check with job 27967:** if M3's pooled grid at 256/144 tokens lands near 35–40 TextVQA on TinyLlama, the pass-through explanation is confirmed and is the change to make.

### 13.3 What to change (ranked by expected effect)

1. **Anchor pass-through.** Feed anchors as raw pooled CLIP features through the projector (no self-attention, no output LayerNorm) and let queries add detail on top. Raise the anchor share at high budgets (256 = the 16×16 pooled grid with queries only at lower levels, or 144 anchors + 112 queries). This keeps M3's robustness and spends queries only where pooling loses information.
2. **Warm start from the dense model.** Initialise Stage 2 from `baseline-tinyllama-576tok-finetune` (LM and projector already aligned to raw CLIP tokens; the checkpoint exists again) instead of a fresh LM.
3. **Train every level in Stage 1** (today Stage 1 trains only the 256 level, rank 64), and give the resampler its own LR in Stage 2 (it currently shares the projector's 2e-5).
4. Loss weighting: second-order (13.5).

### 13.4 Vision LoRA (question 1)

[FACT] Nested, level-specialised LoRA is the largest single accuracy gain in the ablations: v8 vs v11 TextVQA +6.3 / +7.1 / +7.5 / +7.6 (256 / 144 / 64 / 16), MME-P +84 / +25 / +54 / +137, GQA +2.1 to +2.3, POPE about +0.8; it costs SciQA 1.1 to 2.3. A shared rank-16 adapter (v10) is worse than none at 256 and better at 16, and the reversed rank ladder (v14) loses at 256, so per-level nesting, not capacity, is what helps.

[FACT, A10 profile] Unmerged LoRA nearly doubles the vision tower's time: 23.2–24.0 ms with LoRA vs 12.9–13.2 ms without at 144/64/16 tokens (41 ms at 256, noisy), independent of rank, so it is kernel-launch overhead, not FLOPs. TTFT is about 44 ms at *every* budget on the A10, because the LM prefill is also flat (17.8–18.6 ms for 117 to 309 tokens).

[INTERP] Keep the LoRA and remove its latency by merging the active level into the CLIP weights (W + B_ℓA_ℓ) at level switch, which costs nothing at run time. The cost is memory for merged copies (about 0.6 GB per CLIP-L copy in FP16) or a one-off merge per switch. Side effect: with a level-specialised tower the encoder output differs per level, so §2's "escalate tokens without a second encoder pass" is not available for this model.

### 13.5 Loss weighting and MOO (question 4)

[FACT] Current per-step loss: mean CE over {256, one sampled level} + 0.1 × prefix-KL to the 256 level, i.e. effective weights of about 0.5 on 256 and about 0.17 on each other level. [INTERP] Weighting cannot explain the dense gap: 576ctrl puts all weight on a single level and is still at 21.2. In Elastoformer-MOO (SmolLM2-360M, one seed), a fixed 3× weight on the full model moved full-model perplexity by about 1.7 % and the smallest configuration by about 10 %; hypervolume weighting did not beat fixed 3×, and its weights drifted toward the smallest configuration. Recommendation: keep fixed weights now, revisit after the connector change, and use a sandwich (largest + smallest + one random) if the low levels need more signal.

### 13.6 Latency in the objective (question 3)

[INTERP] Training needs no latency term: the budgets are discrete and the token count is the cost knob. Report FLOPs and parameters as the hardware-independent cost, but note that FLOPs mis-rank configurations on real hardware here: on the A10 and on the Orin (§8.1), TTFT is flat across 16–256 tokens because the vision tower and the weight-bound LM prefill dominate, and the LoRA overhead is invisible to FLOPs. A FLOPs-only paper is defensible for the method, but any latency claim (the FastV comparison in §12, the LoRA question) needs one measured table. The A10 profile is a usable stand-in until the Orin (I-0) is available.

## 14. Results update 2026-10-27: M3 ladder and FastV K-sensitivity (DAS-6 jobs 27962, 27963, 27973; summary job 27977)

Source: `scripts/summarize_tinyllama.sh` output `croppilot/logs/summarize_tinyllama_27977.out`. Single seed, TinyLlama-1.1B, CLIP-L/336, LLaVA-1.5 data. The M3 ladder checkpoint was trained on the Hipster cluster; its recipe was not re-verified against ours here.

### 14.1 M3 ladder: the §13.2 prediction holds, and M3 beats final-parcel everywhere

[FACT] M3 ladder at 256 / 144 / 64 / 16 tokens: TextVQA 35.1 / 34.3 / 30.6 / 21.2 (§13.2 predicted 35-40 at 256/144: 256 is at the bottom of the range, 144 a hair under). Against final-parcel (TextVQA 28.5 / 26.3 / 23.7 / 19.7) M3 is better on **all five tasks at all four budgets**: TextVQA +6.6 / +7.9 / +6.9 / +1.4, SciQA +7 to +9, MME-P +82 to +121, GQA +1.8 to +3.1, POPE +0.2 to +2.1. It is also well above our MQT ladder (24.9 / 22.7 / 21.6 / 18.4).

[FACT] M3 vs the dense 576 model (TextVQA 41.2): −6.1 at 256, −6.9 at 144, −10.6 at 64, −20.0 at 16. M3 vs FastV K=2 on TextVQA: FastV is better at 256 (39.6 vs 35.1) and 144 (37.3 vs 34.3), level at 64 (30.2 vs 30.6), and M3 wins clearly at 16 (21.2 vs 15.0). On MME-P, POPE, SciQA and GQA the M3 ladder is flat across budgets (MME-P 1245-1255).

[INTERP] The connector explanation of §13.2 is supported (raw pooled CLIP features through the projector reach 35 TextVQA at 256, final-parcel's learned queries 28.5), but only as "pooling with no new parameters beats our resampler"; it still does not isolate which part of the resampler (random init, anchor self-attention + LayerNorm, queries, training budget) is responsible.

### 14.2 FastV K-sensitivity (K = 2 / 4 / 8)

| kept tokens | TextVQA K2 / K4 / K8 | SciQA K2 / K4 / K8 | MME-P K2 / K4 / K8 |
|---|---|---|---|
| 256 | 39.6 / 39.0 / 38.5 | 51.9 / 51.5 / 44.5 | 1236 / 1238 / 1222 |
| 144 | 37.3 / 36.5 / 36.7 | 51.7 / 51.7 / 44.3 | 1196 / 1211 / 1198 |
| 64 | 30.2 / 29.1 / 30.5 | 50.6 / 49.3 / 44.4 | 1133 / 1120 / 1064 |
| 16 | 15.0 / 13.5 / 13.8 | 46.3 / 46.2 / 42.6 | 744 / 742 / 742 |

[FACT] Moving the pruning layer deeper does not rescue the high-pruning rows (TextVQA at 16 tokens 13.5-15.0 for every K; K = 8 loses about 7 SciQA points). [INTERP] K = 2 is a fair FastV baseline; §12's conclusions stand, and FastV at 97 % pruning is simply bad, not mis-tuned.

### 14.3 Why our NTR trails M3 although PARCEL reports it beating M3/MQT

[FACT, journal §7-8] Our NTR **is** PARCEL's PCQR: pooled anchors + a self-attention block over [anchors; queries] + cross-attention of the pool-aware queries to the raw ViT patches. PARCEL's claimed win is on *retention relative to each method's own full-budget model*, against MQT/M3 trained in their own setting, not against a dense model in absolute scores.

[FACT, journal §8-note] final-parcel (v8) deviates from PARCEL's routing: PARCEL uses 16 anchors below B = 64 and 64 anchors up to B = 256, i.e. N_q = 0 at 16 and 64 tokens; v8 uses {256:64, 144:36, 64:16, 16:4}. **Correction (2026-10-27, replaces an earlier wrong correction):** the `parcel-faithful` run was meant to be the literal routing ({256:64, 144:64, 64:64, 16:16}) but the `pool_anchored` code always keeps at least one query (`n_anchors_for` snaps the anchor count down until it is below the budget), so it actually trained **9 anchors + 7 queries at 16 tokens and 36 + 28 at 64 tokens**, with post-self-attention anchors; it never ran a pure-anchor budget (unit-tested in `jobs/test_parcel_exact.sh`). So the literal-PARCEL connector had **not** been tested before 2026-10-27; it is now `resampler_arch=parcel_exact` (section 15).

[FACT, journal measurements] Effective rank at 256 tokens: the 192 learned queries alone have 26.8 effective dimensions (13.9 %), the 64 anchors 62.6 (97.9 %), the full set 43.8 (17.1 %); the queries are more collapsed than the plain query baseline's.

[INTERP] The gap is therefore (1) the queries add little unique information and are randomly initialised and trained for one epoch at one level, whereas M3 adds no parameters and keeps the Stage-1 projector aligned with raw CLIP features, (2) the anchors pass through self-attention and an output LayerNorm that raw-feature baselines (and, per Fig. 4, PARCEL's own anchors) skip, and the low budgets spend tokens on queries (parcel-faithful: 9+7 and 36+28), where PARCEL and M3 use pure pooling. Neither is yet tested in isolation; the parcel-exact runs of section 15 test both at once.

### 14.4 Next steps (decision-relevant, in order)

1. **PARCEL pipeline** (section 15): raw pass-through anchors, PARCEL's literal routing with pure-anchor budgets, uniform budget sampling, plain cross-entropy. `parcel-faithful` did not test this (it ran 9+7 / 36+28 with post-self-attention anchors).
2. **Anchor pass-through + dense/M3-aligned warm start + all levels in Stage 1** (§13.3 items 1-3) in one run, then ablate.
3. **M3 ladder + nested vision LoRA** as a serious alternative system: no resampler, and the LoRA gave +6 to +8 TextVQA in the Nest ablations (§13.4). If it beats the best NTR variant, the elastic-token mechanism should be pooling and the paper's contribution moves to the LoRA, the crop and the controller.
4. Hold the HR-crop work (I-2) until one of 1-3 closes the connector gap; the crop gains are measured on a Nest model that trails plain pooling.


## 15. The PARCEL pipeline (built 2026-10-27, next to Nest-VLM, not instead of it)

Source: PARCEL (arXiv 2605.30126) Sec. 3.2-3.3, Fig. 4 and Appendix E; code in `llava/model/elastic/{resampler,engine,config}.py`, `llava/model/language_model/llava_elastic_mixin.py`, recipes `parcel-exact-nolora` / `parcel-exact-lora` in `submit_elastic_run.sh`; tests `jobs/test_parcel_exact.sh` (14 checks, pass) and `jobs/smoke_parcel_exact.sh`. The Nest-VLM pipeline (`pool_anchored`, `final-parcel`) is untouched.

### 15.1 What the paper specifies and what the new pipeline does

| | PARCEL (paper) | `parcel_exact` here |
|---|---|---|
| Anchors | pooled, **concatenated to the output as-is** (Fig. 4, Sec. 3.2) | raw avg-pooled anchors, bit-identical in the output (unit test) |
| Connector blocks | one Query<->Pool self-attention + one Query->ViT cross-attention (App. E) | exactly these, depth 1 enforced; only the updated queries leave the self-attention block |
| Routing | 16 <= B < 64: 4x4 = 16 anchors; 64 <= B <= 256: 8x8 = 64 anchors; N_q = B - N_p | the same step function; B = 16 and B = 64 are **pure-anchor budgets** (== M3 pooling) |
| Budgets in training | B ~ even values in [16, 256], one per step | B ~ Uniform{16, 18, ..., 256}, one forward per step, arbitrary B (not a ladder entry) |
| Loss | PaliGemma-2 recipe; **no distillation, no teacher** is described | plain next-token cross-entropy on response tokens: KD, CORAL, decorrelation all off |
| LR / optimiser | Adam(W) defaults, LR 2e-5 x 0.5 = 1e-5 (App. E) | AdamW, Stage 2 LR 1e-5 (Stage 1 keeps the connector-only 1e-3 of the LLaVA recipe) |
| Data | PaliGemma-2 mixtures, 100M + 100M samples, global batch 8192 | unchanged LLaVA-1.5 (558K + 665K), batch 128 |
| Encoder | SigLIP-so400m/14 at 224 (256 patches), trainable | CLIP-L/336 (576 patches), frozen or nested vision LoRA |
| Choices the paper leaves open | final norm, positional embeddings, head count | LayerNorm on the queries only (flag), no extra positional embeddings, 8 heads (12 when the width is 1152) |

[FACT] The anchor counts (16 and 64) are the paper's, but on our 24 x 24 CLIP grid they are 6 x 6 and 3 x 3 pooling blocks (the paper's 16 x 16 SigLIP grid pools 4 x 4 and 2 x 2): 64 anchors are 11 % of our 576 patches against 25 % of PARCEL's 256, so our anchors are relatively finer, and our 256-token budget is a compression of the grid where PARCEL's 256 is the uncompressed grid.

### 15.2 Is there KD from a larger model in PARCEL?

[FACT, from the pages read: main text and Appendix E] No. PARCEL describes only the PaliGemma-2 pretraining recipe (nested dropout over budgets, per-benchmark transfer tuning); no teacher, no distillation term and no auxiliary loss is mentioned. The loss is the standard autoregressive cross-entropy of PaliGemma-2, which the pipeline reproduces. (Appendix pages 21-28 were not read; they are results, FLOP and benchmark details.)

### 15.3 Runs

parcel-exact-nolora (train 27983, eval 27984) and parcel-exact-lora (train 27989, eval 27990; first submitted as 27985, cancelled because it landed on the excluded node205) are queued behind the smoke gate (27982, passed), on node208 A10:2 one after the other (27989 is `afterany:27983`) with identical settings (one world size, so the only difference is the vision LoRA). Unit tests 27979 and the two older regression tests (27980, 27981) pass.

### 15.4 Would SigLIP-so400m change the picture? (decision: not queued yet)

[FACT] There is **no SigLIP accuracy result in our logs**. Two TinyLlama + so400m-384 runs (`elastic-finetune-tinyllama-v8-siglip-so400m-parcel`, fixed anchors 81/9/9/1, and the `-adaptive-parcel` variant) finished training on 2026-09-28/29 (5197/5197 steps) but were never evaluated; the other SigLIP runs were 6-step smoke tests (journal section 22). Evaluation of the first one is queued (job 27987, node207, 4 levels in sequence): it is the clean encoder swap inside the Nest-VLM pipeline (same recipe as final-parcel, CLIP replaced by so400m-384).

[INTERP, from 2 x params x tokens, not measured] Latency depends on the resolution, not only the model: so400m has 428M parameters against CLIP-L's 304M (1.41x). At 384 px it sees 729 patches (about 0.69 TFLOPs, roughly 1.8x CLIP-L/336's 0.38), so it is slower; at 224 px it sees 256 patches (about 0.23 TFLOPs, 0.6x), so it is faster, at the price of OCR resolution (TextVQA). PARCEL itself uses so400m at 224 (and a 448 variant for high resolution).

[INTERP] An encoder swap lifts every method that uses it (dense, M3, FastV, Nest, PARCEL) and does not touch the connector problem of sections 13-14, so it cannot by itself decide Nest/PARCEL against pooling; a SigLIP comparison is only interpretable with a SigLIP dense baseline and a SigLIP M3 ladder next to it, which do not exist. Queue it only if (a) the 27987 evaluation shows a clear encoder gain at equal budget, and (b) the CLIP parcel-exact runs show the connector is no longer the bottleneck. The recipes `parcel-exact-so400m224-nolora|lora` are ready (`submit_elastic_run.sh`; the so400m-224 weights must be downloaded once, the compute nodes are offline).

### 15.5 SigLIP-so400m/224 PARCEL runs on hipster, and the rank-64 band (2026-10-27)

- `lora_small_budget_max=30` (new, default 0 = off): under uniform sampling, B in {16..30} uses the level-16 (rank-64) vision-LoRA adapter. The plain smallest-level->=B rule gave rank 64 only to B = 16 (1 step in 121). Enabled for the parcel-exact `*-lora` recipes from now on; the CLIP LoRA run 27993 started before this and uses the plain rule, so CLIP-vs-SigLIP LoRA comparisons carry this extra difference.
- Queued on hipster (`performance`, 4x RTX 6000 Ada, checkpoints under `/scratch/skalra/flexllava_saves/checkpoints`): `parcel-exact-so400m224-nolora` train 381317 / eval 381318, `parcel-exact-so400m224-lora` train 381319 / eval 381320. Both run at the same time, queued separately.
