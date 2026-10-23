# Research project: composing Nest-VLM with CoInfer

Planning documents added 2026-10-21. They are separate from the existing FlexLLaVA docs in [FlexLLaVA/docs/](../FlexLLaVA/docs/) ([ELASTIC_PIPELINE.md](../FlexLLaVA/docs/ELASTIC_PIPELINE.md), [EXPERIMENT_JOURNAL.md](../FlexLLaVA/docs/EXPERIMENT_JOURNAL.md), etc.), which they cite but do not modify.

**Status: planning only.** I read two papers and parts of the repo, searched the web, and wrote these files. I did not run any code, train anything, or measure anything on hardware. Every number in these documents is either quoted from a cited source or explicitly marked as a target/guess.

## One-paragraph summary

- **Nest-VLM** ([paper](../related%20papers/657_file_Paper.pdf)) makes the visual-token budget of a *small* VLM a runtime knob (anchors + nested learned queries + budget-dependent vision LoRA). Measured on a Jetson Orin Nano, going from 256 to 16 tokens only buys 1.44× prefill and 1.30× energy (TinyLlama), and latency stops improving below 64 tokens.
- **CoInfer** ([paper](../related%20papers/093201a575.pdf)) offloads *question-aware cropping* to a sensor-side MCU (MGround, 276 K params), so the unchanged host VLM sees ~50 % fewer tokens at 1080p, with −0.4 pp accuracy on a GQA-based test and up to 58 % lower TTFT; the vision encoder is >70 % of baseline TTFT.
- **Interpretation:** the two cut different cost terms (LLM prefill vs vision encoding). A combined system could reduce both and add a controller under contention. The main technical obstacle is that Nest-VLM's CLIP encoder is fixed at 336², so a crop saves encoder time only if the encoder is also run at lower resolution, which needs retraining.
- **Recommendation:** pursue a *systems* paper (characterisation + crop-aware elastic VLM + load-aware controller), not an algorithmic-novelty paper. Do not claim the anchor resampler (PARCEL) or MGround (CoInfer) as ours.
- **SenSys 2027 timing:** the only open round has its abstract due 2026-10-29 and paper 2026-11-05 (15 days from writing). I recommend a go/no-go on 2026-10-28 from a zero-training pilot; otherwise target a later cycle.

## Research question

> Can a question-aware, load-isolated sensor-side stage and a trained-in nested token budget be composed into a single controllable pipeline that gives a better accuracy–latency–energy–contention frontier for VLM question answering on 8 GB embedded SoCs than either alone, and can a runtime controller pick the operating point per query under load?

## Documents

| File | Contents |
|---|---|
| [analysis.md](analysis.md) | What was inspected; Nest-VLM and CoInfer analysed (problem, design, setup, results); official-code status; problems found in the Nest-VLM paper; transfer/compatibility analysis; repo gaps |
| [experimental-plan.md](experimental-plan.md) | Hypotheses H1–H9, workloads, datasets, models, hardware, baselines, metrics, phases, ablations, statistics, risks/fallbacks, timeline |
| [implementation-plan.md](implementation-plan.md) | Target architecture, work packages (harness, ROI model, crop-aware nested VLM, controller, evaluation, packaging), milestones, reproducibility checklist, deployment notes |
| [brainstorm.md](brainstorm.md) | (2026-10-22) Adaptive-HR papers (ViCrop, SGL, Q-Zoom, AdaptVision, S2, Monkey, ViCO, HyperVL, VISOR) mapped onto a dual-elastic (pixel × token) Nest-VLM pipeline; ideas ranked by effort; scope for the 2026-11-05 round vs later |
| [paper-plan.md](paper-plan.md) | SenSys 2027 CFP summary, novelty claims checked against literature, recommended contribution, outline, related-work and anonymity strategy |

## Things you should know before relying on this

1. **Possible error in the Nest-VLM paper.** Table IV's "NEST-VLM-TinyLlama (ours)" at 576 tokens (POPE 84.89, SciQA 57.36, TextVQA 41.00, GQA 58.30) matches, to the digit, the *dense baseline* in the repo journal §23, not the Nest-VLM-architecture run (81.82 / 50.72 / 21.20 / 52.57). Please check this before any submission. Details: [analysis.md](analysis.md) §1.5.
2. **No official CoInfer code found**; the paper gives no repository link and its industrial dataset is proprietary. MGround would have to be re-implemented, or the authors contacted.
3. **The repo has no Jetson measurement harness.** Jetson columns in `results/*.csv` are roofline estimates; the paper's measured latencies/energies cannot be regenerated from this repo as it stands.
4. **Existing results are single-run.** Any claim resting on ≤1 pp gaps needs seeds and confidence intervals.
5. **SenSys facts are from a summarising fetch tool**; re-read https://sensys.acm.org/2027/cfp.html before acting on deadlines or policies.
6. **Not analysed:** `AdaOff` and `Flextron` PDFs in `related papers/` (only their first lines were read), most of the 3,092-line experiment journal, and Figs. 4–11 of CoInfer.
7. **Open questions for you:** where is the Nest-VLM manuscript submitted/under review (dual-submission rules)? Do you have an STM32H747 board and an Orin Nano? Can the CoInfer authors be contacted? Which 1080p public data can you use?
