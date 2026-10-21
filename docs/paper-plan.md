# Paper plan: contributions, novelty checks, outline, SenSys 2027 fit

Part of the [project overview](README.md). Facts/interpretations/proposals are tagged as in [analysis.md](analysis.md) ([FACT], [INTERP], [PROPOSED], [UNVERIFIED]). Written 2026-10-21.

## 1. SenSys 2027 call for papers

Source: https://sensys.acm.org/2027/cfp.html and https://sensys.acm.org/2027/, fetched with a web tool that returns **summaries**, not raw HTML. **[UNVERIFIED] until you re-read the page yourself** (especially deadlines and policies).

### 1.1 What the pages say [FACT per the fetched summaries]
- Name/venue: ACM/IEEE International Conference on Embedded Artificial Intelligence and Sensing Systems; **May 17–20, 2027, Boulder, CO, USA** (home page). A third-party listing says New York→Boulder was a recent move; the CFP page itself gives no dates/location.
- Deadlines (AoE): **1st**: abstract 2026-05-29, paper 2026-06-05, notification 2026-08-31 (passed). **2nd**: abstract **2026-10-29**, paper **2026-11-05**, early notification 2026-12-17, notification 2027-01-28. Camera-ready date: TBD. No later round is listed.
- Format: full papers ≤12 pages (+ unlimited references/appendices; reviewers need not read appendices), short papers ≤6 pages (visions/experiences-type only), two-column 9-pt, `acmart` `sigconf`; submission via HotCRP (https://sensys27nov.hotcrp.com/). Surveys/tutorials out of scope.
- Categories: regular research papers ("original, high-impact research" in embedded AI, sensing systems, CPS/IoT) and visions/experiences/tools/datasets/benchmarks.
- Topics (selected, relevant): *LLMs for edge systems*, *collaborative sensing*, *in-network processing*, *coding and compression* (Algorithms); *industrial IoT*, *autonomous vehicles and drones*, *AR/VR* (Applications); *digital twins and physical AI*, *edge and fog computing*, *heterogeneous sensor networks and data fusion*, *sustainability* (Systems); *secure or privacy-sensitive IoT* (Security).
- Review criteria: only stated for short papers (originality, clarity, potential impact; extensive evaluation not required). **No explicit criteria for regular papers on the page.**
- Double-blind: full anonymisation (no names, affiliations, acknowledgments/funding), cite own work in third person, arXiv preprints allowed, **artifact links must be anonymous** (blinded GitHub/GitLab, anonymised Zenodo; no personal/lab/company sites; scrub file metadata). Violations → desk rejection.
- No formal artifact-evaluation process on the page.
- Dual submission: must be unpublished and not under review elsewhere. Papers rejected in round 1 may resubmit to round 2 only with substantive revision plus a mandatory ≤4-page "Response to Reviewers" appendix.
- Generative AI: describe AI used *in the research* (data, code, analysis, figures) in the methods; AI used only for writing needs no disclosure per the page.
- At least one author must present in person; one full registration per accepted paper.

### 1.2 Cannot be verified from the page [UNVERIFIED]
Explicit review rubric for full papers; whether a prototype on real hardware is "required" (it is not stated; [INTERP] given the venue's name and topics, reviewers will expect real-hardware evaluation); the dates of any future SenSys cycle; PC composition and preferences.

### 1.3 Deadline reality check [FACT + INTERP]
Today is 2026-10-21. The only open round is the 2nd deadline: **abstract in 8 days, full paper in 15 days**. The experimental plan needs MCU firmware, a crop-capable retrained VLM, Jetson measurements and a controller. [INTERP] That is not feasible in 15 days at the quality the venue implies. Two options:
1. **Go/no-go gate on 2026-10-28** (see [experimental-plan.md](experimental-plan.md) §8, Phase 0). Only if the zero-training pilot (oracle crop + existing Nest-VLM checkpoints, measured on the Orin Nano) shows a clear Pareto shift, register an abstract (10-29) and write a narrow 12-page paper for 11-05.
2. Otherwise target a later cycle (SenSys 2028 CFP not yet published; **[UNVERIFIED]**) or another embedded-AI/systems venue. I have not checked other venues' deadlines; do so before committing.
Also note a Nest-VLM-related manuscript may already be under review somewhere (the PDF is an anonymous submission with an access date of Sep 19 2026 in its references, i.e. after SenSys round 1). **Dual-submission rules apply**: the combined paper must be a substantively different contribution and must not overlap with whatever is under review. I do not know where it was submitted; please confirm.

## 2. Novelty claims, each checked against the literature

Search effort: ~9 web searches this session plus the repo's own earlier literature passes (`docs/EXPERIMENT_JOURNAL.md` §6, §14; `docs/ADAPTIVE_INPUT_TOKEN_BUDGET_ALLOCATION.md`). **Absence of evidence in search is weak evidence.** Before any claim goes in a draft, a systematic search (Google Scholar, arXiv, ACM DL, IEEE Xplore, last 12 months) must be redone and logged in `docs/related-work-log.md` [PROPOSED].

| # | Candidate claim | Closest prior work found | Assessment |
|---|---|---|---|
| C1 | Anchors + learned detail queries for elastic tokens (NTR) | **PARCEL**, arXiv:2605.30126 (Kuzucu et al.; pooled anchors + conditioned elastic queries) https://www.emergentmind.com/papers/2605.30126 ; Nest-VLM itself says it adapts PARCEL | **Not novel.** Cite as prior; at most "evaluation on small LMs". |
| C2 | Scarcity-Scaled Adapters (rank grows as budget shrinks) | DyLoRA (prefix-nested ranks); **MatryoshkaLoRA** https://arxiv.org/pdf/2605.07850 (hierarchical rank, nests in the opposite direction); MoNE (nested experts, arXiv:2407.19985) | **Partly novel, modest:** the *direction* of the rank-vs-budget pairing for vision-encoder adapters. Evidence is n=1 (repo journal §16m/q supports direction). Do not make it a headline claim. |
| C3 | Small-LM rank-collapse diagnosis of nested query resamplers | MQT-LLaVA (nested query bank); M3 | **Plausible novelty (diagnostic), unproven:** the rank metric (1 % singular-value threshold) and 24-image sample are thin. Needs more images, seeds, other thresholds. |
| C4 | **Composing sensor-side, question-aware ROI cropping with a trained-in nested token budget** (the proposed paper) | CROP (EMNLP 2025, lightweight localiser + pruning, host-side) https://aclanthology.org/2025.emnlp-main.492.pdf ; CropVLM arXiv:2511.19820; FOCUS arXiv:2506.21710; GapSight/loss-gap crop routing arXiv:2608.21762; mmVB "Visual Tokens on Demand" arXiv:2610.18858 (question-conditioned per-image allocation level, not nested); AVG-LLaVA arXiv:2410.02745 (router over pooling granularity, needs instruction); InputAdapt arXiv:2512.20839 | **No exact match found in my searches**, but each half exists separately and adjacent hybrids exist. Defensible only as a *systems* composition with a measured Pareto study, not as an algorithmic first. Re-search specifically: "crop + token budget", "foveated VLM tokens", "region-then-resample". |
| C5 | MCU-resident question-conditioned grounding | **CoInfer/MGround** (the paper under study; claims first, Sec. II-B). My search found no counter-example (CROP/SGL/CropVLM are GPU-scale) | Novel *to CoInfer*, not to us. We reuse it; cannot claim. |
| C6 | **Measured finding: token-budget elasticity saturates on the SoC because the vision encoder dominates; crop-before-encoder is the missing lever** | "Phase Matters: Characterizing Heterogeneous Vision-Language Inference on a Mobile SoC" arXiv:2606.27906 (TTFT roughly linear in image tokens, tokens as a latency knob; mobile SoC, different stack); CoInfer encoder-share measurement (AGX Orin, Qwen2-VL) | **Likely incremental as a characterisation**, but a clean, rigorous Orin-Nano decomposition (per-stage latency/energy) for *elastic small VLMs* is useful. Check "Phase Matters" in full before claiming. |
| C7 | Load-aware controller choosing (crop, resolution, token budget) per query under SLO/contention | EcoVLA arXiv:2608.15502 (SLO-constrained plan selection for device-edge VLA; picks partition plans, not token budgets); CoVSpec arXiv:2605.02218 (device-edge speculative decoding); Phase Matters; generic DNN-scheduling-under-contention literature **not searched** | **Most promising system claim**, but the scheduling literature (multi-DNN GPU contention, e.g. in MobiSys/SenSys/EuroSys) must be searched before claiming. |
| C8 | Co-location benefit (co-located perception task tail latency drops when VLM token load shrinks) | CoInfer Fig. 9 (YOLO p95 594→183 ms) | **Not novel for crops**; novel only if shown for *nested budgets* and for a controller that exploits it. |
| C9 | Cross-VLM reuse of a single ROI model | CoInfer Sec. IV-B3 (Qwen2-VL, InternVL, Gemma3) | Extending to elastic small VLMs is incremental. |

## 3. Recommended contribution (for SenSys)

**Position:** a *systems* paper, regular research category, framed on **embedded physical-AI nodes that run a question-answering VLM next to perception/control tasks on a shared SoC**.

**Thesis [PROPOSED]:** On small VLMs, visual-token elasticity alone is encoder-bound; question-aware spatial gating alone is tokeniser-bound; their *joint* control surface (crop extent × encoder resolution × nested token budget), driven by a load-aware controller, gives a materially better accuracy–latency–energy–contention frontier on an 8 GB Jetson-class SoC.

**Contributions to claim (only if evidence supports each):**
1. **Characterisation (C6):** per-stage latency/energy decomposition of elastic small VLMs on Jetson Orin Nano showing where the token budget stops helping, with raw logs released (anonymised).
2. **Design (C4):** a crop-aware elastic VLM, with training changes that make the nested model valid under crops and reduced encoder resolution, plus the sensor-side ROI stage; evaluate the hard-vs-global-anchor variants (§3 of analysis.md).
3. **Controller (C7):** a lightweight policy mapping (question features, ROI confidence, measured load, SLO) to a configuration, with SLO attainment, energy and co-runner impact vs static and vs the oracle.
4. **Evaluation methodology:** paired statistics, multi-seed, official metrics (not substring match), contention scenarios, multi-camera scaling.

**Explicitly not claimed:** NTR/anchors (PARCEL), MGround (CoInfer), SSA as headline.

**Why SenSys:** touches *LLMs for edge systems*, *collaborative sensing / in-network processing* (sensor MCU ↔ SoC), *physical AI*, *heterogeneous platforms*. The weak spot is "sensing": the paper must keep the MCU/camera path real (actual MCU, link, frame buffering), not simulated, or reviewers may see it as a pure model-compression paper.

**Fallback contributions** (if the joint design underperforms): (a) a measurement/characterisation paper (Tools/Benchmarks category) with a released harness; (b) negative-result-informed design guidance ("when does crop+budget help"), which the CFP's "experiences" category might accommodate (≤6 pages, lower bar; **[UNVERIFIED]** whether this fits).

## 4. Paper outline (12 pages, ACM sigconf) [PROPOSED]

1. **Introduction** (1 p): CPS node, shared SoC, VLM cost; two partial solutions; the encoder-bound gap; contributions; headline numbers (placeholders until measured).
2. **Background & motivation** (1.25 p): cost model `T_enc + T_prefill + T_decode`; measurement study on Orin Nano (the C6 result as motivating figure); contention example.
3. **Design** (2.5 p): sensor ROI stage; crop-aware nested VLM (resolution × budget nesting, training); optional foveated variant; controller.
4. **Implementation** (0.75 p): MCU firmware, link, Jetson runtime, power instrumentation.
5. **Evaluation** (4.5 p): setup; Pareto fronts per workload; contention; multi-camera; controller; ablations; sensitivity (ROI error, question type).
6. **Related work** (1 p): token reduction (pruning, resamplers), question-aware cropping, split/collaborative inference, edge VLM serving & scheduling, TinyML.
7. **Discussion/limitations** (0.75 p): short-answer assumption, ROI misses, single board, dataset privacy.
8. **Conclusion** (0.25 p).
Appendix (unread by reviewers; keep non-essential): full tables, training details, labelling protocol.

## 5. Related-work strategy [PROPOSED]
- Maintain `docs/related-work-log.md`: query string, database, date, hits, decision. Re-run all §2 searches the week before submission, including arXiv listings for the preceding 8 weeks (this field moves monthly).
- Must read in full and compare against (none of these were read in full by me): PARCEL, CROP, CropVLM, mmVB, AVG-LLaVA, Phase Matters, EcoVLA, M3, MQT-LLaVA, MatryoshkaLoRA/DyLoRA, FastV, SparseVLM, LLaVA-PruMerge, and the TinyML VQA works CoInfer cites (TinyVQA, MCUNetV2).
- Search the systems literature the repo has not covered: multi-DNN scheduling on shared embedded GPUs, anytime/elastic DNN inference controllers, adaptive offloading (AdaOff in `related papers/` is a starting point but was not analysed).
- Position every baseline fairly: reproduce under identical hardware, precision, prompt and metric; never mix published numbers from different metrics (CoInfer's substring accuracy vs official benchmark accuracy).

## 6. Anonymity and artifacts [PROPOSED, per CFP]
- Create an anonymised mirror (e.g. anonymous.4open.science or a blinded repo) of code + measurement harness; scrub `git` history, author metadata, cluster paths (`/var/scratch/skalra/...`), remote URL, W&B entities.
- No acknowledgments or funding in the submission. Cite own prior work (Nest-VLM, repo) in third person.
- Describe any AI-assisted *experiments/code* in the methods (the CFP wants that); writing-only assistance needs no disclosure per the page.
- Proprietary-data issue: CoInfer's industrial set is unreleasable. Build and release a public 1080p evaluation set (see [experimental-plan.md](experimental-plan.md) §3) so reviewers can reproduce.
