# Kaggle Free GPU (2× T4) & TPU (v5e-8): What Actually Runs

Research date: 2026-11-08. Source material for the Chinese-language final report. Full source notes: `notes/key-sources.md`.

Fixed constraints (given): GPU = 2× Tesla T4, 16 GB VRAM each, NOT pooled (no NVLink; a normal pipeline sees one 16 GB card, the second card is a second worker unless software explicitly shards), 30 h/week, ~12 h/session. TPU = v5e-8, 8 chips × 16 GB HBM = 128 GB aggregate, NOT pooled (models must shard; ~125.98 GiB reported usable in one public run), 20 h/week, ~9 h/session. Sessions are ephemeral notebooks, not servers.

Two cross-cutting truths:
1. **T4 = Turing, CC 7.5 (2018): 65 TFLOPS FP16, but no BF16 tensor-core datapath (BF16 emulated/slow), no FP8 hardware, no FlashAttention-2.** Run pipelines in FP16, not BF16, on Kaggle. FP8 checkpoints save VRAM (dequantise to FP16) but give no speed-up. Sources: https://www.thundercompute.com/blog/nvidia-t4-pricing ; https://github.com/kusk24/jtrax-docs/blob/HEAD/bugs/bf16-on-a-t4-is-emulated.md (both index, 2026-11-08).
2. **Kaggle GPU sessions have only 29 GB system RAM** (official, verified live: https://www.kaggle.com/docs/notebooks). Recipes needing 32–50 GB RAM for quantisation/offload are at risk — use pre-quantised checkpoints (NF4/GGUF), not on-the-fly quantisation.

Verdicts: **fits** / **fits-with-quantization-or-offload** / **does-not-fit**, always against ONE 16 GB T4 (§3: sharded across v5e-8).

## Summary
- Image: SD 1.5 + SDXL native and fun (SDXL ≈25–60 s/img); FLUX.1 schnell/dev quantised-only, ~1–3 min/img. ComfyUI + A1111 have real public Kaggle notebooks. SDXL LoRA training fits; FLUX LoRA training NF4-only, slow.
- Video (one T4): AnimateDiff, Wan2.1-1.3B, CogVideoX-2B/5B (offload), LTX-Video 2B, HunyuanVideo-1.5 (14 GB floor w/ offload) = short 480p clips in minutes–tens of minutes. Wan-14B FP16, HunyuanVideo 13B, Mochi-1 do not fit. TPU video = effectively GPU-only in practice (research JAX ports only; no public Kaggle TPU video notebook found).
- TPU superpower = 128 GB sharded HBM for big LLMs: proven on Kaggle — Qwen 27B BF16 ~130 tok/s, Gemma 26B JAX ~111 tok/s (community, single-source, arithmetically sound). Honest BF16 ceiling ~32B (64 GB weights). **70B BF16 = 140 GB > 128 GB: does-not-fit**; 70B needs INT8/INT4 (Kaggle evidence thin). Full fine-tuning caps ~7–8B params (Adam ≈16 B/param); LoRA 2–27B is the realistic training play; Tunix demos on Kaggle are 1–2B.
- Small workloads (Whisper, TTS/RVC, QLoRA 7–14B, OCR VLMs, embeddings) all fit one T4 easily; 2nd T4 = 2 parallel jobs.
- "Deploying a service" = temporary personal demo inside a live session only (Gradio/cloudflared/ngrok, random URL, dies ≤9–12 h). Kaggle AUP (verified live 2026-11-08, https://www.kaggle.com/aup) lists "server farming" + "activity unrelated to ML data science" as resource abuse.

## 1. Image generation on T4×2

| Model/tool | VRAM | Verdict | Speed on T4 | Kaggle notebook? |
|---|---|---|---|---|
| SD 1.5 FP16 | ~4–6 GB | fits | ~10–15 s/img (512², 50 steps, official KerasCV-vs-diffusers T4 benchmark: 29–41 s per 3 imgs) | Yes, many |
| SDXL base FP16 | ~8–10 GB (8.5 GB measured loaded); +refiner/ControlNet stacks → toward/past 16 GB | fits (base; heavy stacks tight) | ~25–60 s/img 1024², 20–30 steps (sources vary by sampler) | Yes |
| FLUX.1 schnell 12B | BF16 31.4 GB load / 36.2 GB peak; NF4 12.6 load / 17.3 peak; NF4+CPU-offload ≈12.2 GB; GGUF Q4/Q5 ~6–7.5 GB + T5 offload | fits-with-quantization-or-offload (BF16 alone: does-not-fit) | ~1–3 min/img 1024² quantised (thin evidence) | ComfyUI-Kaggle notebooks claim FLUX support; no first-party benchmark |
| FLUX.1 dev 12B | Same memory, 20–28 steps vs schnell's 4 | fits-with-quantization-or-offload, slower | Plan ~2–5 min/img quantised (estimate, flag) | Same |
| ComfyUI | Low overhead | fits — recommended UI | — | Yes: public notebooks + GitHub launchers e.g. https://github.com/kayas881/comfyui-kaggle (T4, Pinggy tunnel, Wan preconfigured) |
| A1111 WebUI | Higher overhead; FLUX awkward | fits (SD1.5/SDXL) | — | Yes: https://www.kaggle.com/code/abdu5511/automatic1111-s-stable-diffusion — recorded successful run on GPU T4 x2 |
| SDXL LoRA train (Kohya) | ~10–12 GB min, 16 comfortable | fits | ~1.5–4 h typical LoRA (estimate; 1–3 h quoted on 24 GB cards) | Yes: Kohya-on-Kaggle tutorials for T4 x2 |
| FLUX LoRA train | BF16 ~30 GB+ no; INT8 ~18 GB no; NF4 ~9.1 GB yes (SimpleTuner lowest-VRAM config) | fits-with-quantization-or-offload | Slow (NF4 + offload + T4); hours-scale | Recipes exist; Kaggle-specific FLUX-LoRA evidence thin (flag) |

Key sources (§1): HF diffusers quantisation blog — FLUX BF16/NF4/8-bit load+peak numbers, H100 (index): https://github.com/huggingface/blog/blob/HEAD/diffusers-quantization.md ; HF bitsandbytes docs — FLUX-dev on <16 GB / free Colab with offload, quant+compile+offload 12.24 GB (index): https://huggingface.co/docs/diffusers/quantization/bitsandbytes ; SDXL 8 GB official minimum (index): https://www.thundercompute.com/blog/how-to-run-stable-diffusion ; SDXL-on-T4 8.5 GB / ~55 s @20 steps (index): https://github.com/dungnotnull/life-chronicle-agent/blob/HEAD/docs/HF_MODEL_BENCHMARKS.md ; SD1.5-on-T4 official benchmark (index): https://www.tensorflow.org/tutorials/generative/generate_images_with_stable_diffusion ; FLUX Q4_K_S on 2×T4: 62 s @768²/10 steps, 105.6 s @1024² tiled (index): https://cran.rstudio.com/web/packages/sd2R/readme/README.html ; SimpleTuner FLUX training VRAM + ~50 GB RAM warning for quantise-at-start (index): https://raw.githubusercontent.com/bghira/SimpleTuner/main/documentation/quickstart/FLUX.md ; LoRA VRAM synthesis (index): https://localaimaster.com/blog/image-lora-training-local-guide

Speed honesty: only the SD1.5 T4 benchmark is first-party. SDXL T4 timings scatter 25–60 s by sampler/steps/UI. FLUX-on-T4 timings are the weakest numbers in this report (one GGUF benchmark on 2×T4, one low-quality social post claiming ~8 min for offloaded schnell on Kaggle) — present the range, not a point figure. FLUX licensing: schnell = Apache-2.0; dev = non-commercial licence.

## 2. Video generation

| Model | VRAM | Verdict (1×16 GB T4) | Output / rough time on T4 |
|---|---|---|---|
| AnimateDiff (SD1.5 motion) | ~6–8 GB (SDXL version ~13 GB) | fits | 512², 16 frames (~2 s loops); ~1–3 min/clip |
| Wan2.1 T2V-1.3B | **8.19 GB (official)** | fits — best quality-per-GB | 480p, 5 s (81 frames); ~4 min on RTX 4090 official → est. ~10–20 min on T4 (flag: extrapolation) |
| CogVideoX-2B | SAT FP16 18 GB; diffusers FP16 from ~4 GB with ALL optimisations (INT8 ~3.6 GB) | fits-with-quantization-or-offload (offload = slow) | 720×480, 49f @8fps = 6 s; est. ~10–25 min on T4 |
| CogVideoX-5B / 5B-I2V | SAT BF16 26 GB; diffusers BF16 from ~5 GB optimised (INT8 ~4.4 GB) | fits-with-quantization-or-offload, marginal | 6 s 720×480; documented on free T4: sequential offload "extremely slow (1 hour+)" — plan 30–60+ min |
| LTX-Video 2B (distilled) | ~6–8 GB FP8+tiling (community); ~10.4 GB peak validated on 16 GB Ada | fits | Fastest of the group; 512×320–768×512, ~97–121f; minutes on T4 |
| LTX-Video 13B | ~14–22 GB FP16; FP8+offload borderline | fits-with-quantization-or-offload (borderline, not recommended; T4 has no FP8 HW) | Short low-res clips only, slow |
| Wan2.1/2.2 14B | FP16 ~45–65 GB; FP8+T5-offload ~12–16 GB @480p; GGUF Q4/Q5+T5-offload claimed ~5–9 GB @480p | FP16: does-not-fit. GGUF route: fits-with-quantization-or-offload on paper only | 480p 5 s only in quantised form; expect tens of minutes; evidence = community guides (WillItRunAI), no official/Kaggle benchmark (flag) |
| HunyuanVideo (original, 13B) | 45 GB @544×960; 60 GB @720p (model card) | does-not-fit | — |
| HunyuanVideo-1.5 (8.3B, Nov 2025) | **14 GB minimum with model offloading (official)**; 24 GB comfortable | fits-with-quantization-or-offload — tight, expect slow (50 steps default) | 480p/720p, 121f (5–10 s); no T4 timing found (flag) |
| Mochi-1 (10B) | Official repo: ~60 GB single-GPU; ComfyUI-optimised "<20 GB" — still >16 GB | does-not-fit (practically) | 480p only even where it runs |

Key sources (§2): Wan 1.3B 8.19 GB + 4090 timing, official HF card (index): https://huggingface.co/Wan-AI/Wan2.1-VACE-14B ; CogVideoX official VRAM table incl. "disabling optimisations ≈3× peak", LoRA FT 47/63 GB (index, HF mirror of THUDM card): https://huggingface.co/FreeAi1111/PrepGenius ; CogVideoX-5B-I2V on free T4 = sequential offload, 1 h+ (index): https://huggingface.co/spaces/svjack/CogVideoX-5B-Space/blob/main/readme.py ; HunyuanVideo-1.5 official 14 GB floor (index): https://huggingface.co/tencent/HunyuanVideo-1.5 ; Mochi ~60 GB official / <20 GB ComfyUI (index): https://hysenlabs.com/projects/genmoai-mochi ; Wan-14B quantised figures, community only (index): https://willitrunai.com/blog/wan-2-2-vram-requirements

TPU for video: **effectively GPU-only.** All official video repos are PyTorch/CUDA. The vidax paper states open video models "ship almost exclusively as PyTorch/CUDA reference implementations… leav[ing] Cloud TPU pods without a production-ready inference path"; vidax itself is a research JAX/Flax engine benchmarked on TPU **v4-8** (not Kaggle, not v5e-8) — https://arxiv.org/pdf/2609.18077.pdf (index). AoiZora (Wan2.1/Hunyuan on TPU v5e subslices) is likewise a research auto-parallelisation paper — https://arxiv.org/html/2606.17566v2 (index). No public Kaggle TPU video notebook was found in searches on 2026-11-08. PyTorch/XLA in principle runs PyTorch code on TPU, but video pipelines depend on CUDA-only kernels (FlashAttention, xformers, torchao FP8 — which itself requires H100-class per the CogVideoX card) and BF16-first code; do not promise it.

## 3. TPU v5e-8 specialities (128 GB aggregate)

Memory maths (BF16 weights = 2 bytes/param): 20B→40 GB, 27B→~55 GB, 32B→64 GB, **70B→140 GB > 128 GB (and >125.98 GiB usable)**.

| Play | Size / memory | Verdict | Evidence |
|---|---|---|---|
| Qwen 27B BF16 inference (vllm-tpu) | 55.6 GB weights, sharded TP=8 | fits — proven on Kaggle | Community project: ~130 tok/s single-stream w/ MTP (78 w/o), ~540 tok/s @8 streams, prefill ~10,300 tok/s, 262k ctx, ~22 min to live endpoint. Notebook: kaggle.com/code/rahim3/qwen3-8-27b-bf16-on-kaggle-tpu-130-tok-s-api ; README (index): https://github.com/ugvfpdcuwfnh/kaggle-tpu-lab/blob/HEAD/qwen38-27b/README.md |
| Gemma 26B (A4B) JAX inference | ~52 GB download, ~6 GB/chip | fits — proven on Kaggle | YLiu95/kaggle-tpu-inference via guide (index): https://technicalpaji.pages.dev/tpu/ — ~111 tok/s, TTFT 44–69 ms |
| Qwen3-32B / Gemma-3-27B / Gemma-4-31B inference | 54–64 GB BF16 | fits (by maths + stack support) | vllm tpu-inference tested-model list (TPU generally, recommended v5e/v6e/v7x — not Kaggle-specific) (index): https://github.com/vllm-project/tpu-inference/blob/HEAD/README.md |
| Llama-3.3-70B BF16 | 140 GB weights | does-not-fit (BF16) | Arithmetic above; note 70B appears in vLLM-TPU tested lists for larger/other TPU setups — that does NOT transfer to v5e-8 |
| 70B INT8 (~70 GB) / INT4 (~35 GB) | Fits HBM on paper, KV cache extra (INT8 tight at long ctx) | fits-with-quantization-or-offload (in principle) | No Kaggle v5e-8 70B run found; vLLM-TPU quantisation support is model-dependent (flag: thin) |
| MoE with mostly-quantised experts (repo claims GLM-5.3-Flash 320B MoE, 3-bit experts, custom JAX engine, ~64 tok/s) | — | Unverified | Single-source claim, same kaggle-tpu-lab repo README (index): https://github.com/ugvfpdcuwfnh/kaggle-tpu-lab/blob/HEAD/README.md — do not present as established |
| Full fine-tuning (MaxText/JAX, Adam) | ~16–18 B/param all-in → ceiling ~7–8B params before activations | fits only ≤~7B realistically | Rule-of-thumb consistent with §4 full-FT maths |
| LoRA / SFT / GRPO (Tunix, MaxText+Tunix) | Frozen BF16 base 2 B/param + small adapter optimiser states | fits: sweet spot ~7–27B; demonstrated on Kaggle only at 1–2B | Gemma2-2B LoRA r16 completed inside one 9 h Kaggle v5e-8 session (index): https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/tunix-robotics-reasoning-single-session ; MaxText model library incl. Llama-3.3-70B/Qwen3-32B/Gemma-4 configs (index): https://github.com/kocchop/maxtext/blob/HEAD/README.md |
| Image models in JAX (diffusers Flax) | SD1.5/SDXL replicated or sharded over 8 chips | fits | HF Flax guide explicitly names Kaggle TPUs; ~7 s/img after >1 min compile (v2-8 era) (index): https://huggingface.co/docs/diffusers/v0.23.0/using-diffusers/stable_diffusion_jax_how_to ; SDXL JAX on Cloud TPU v5e: 4×1024² imgs ~4 s on v5e-4 serving (index): https://raw.githubusercontent.com/ariG23498/blog/main/sdxl_jax.md . Caveat: Flax pipelines are legacy (SD1.5/SDXL only; no Flax FLUX/video) |

TPU caveats: XLA compile time is a real tax (minutes; the 27B recipe ships a pre-built compile cache; shape changes recompile). Only JAX/Flax, PyTorch-XLA or vLLM-TPU stacks work — CUDA software (ComfyUI, llama.cpp, Ollama, bitsandbytes) does not run on TPU at all. TPU VM host RAM is large (330 GB per Kaggle docs for TPU 1VM — that spec predates v5e-8 on the docs page, treat as indicative), which helps loading/checkpointing, not HBM limits.

## 4. Other useful workloads (brief)

| Workload | Memory | Verdict |
|---|---|---|
| Whisper large-v3 STT (faster-whisper INT8 much lighter) | ~10 GB (original impl., FP16); weights ~2.9 GB | fits (1×T4); batch-transcribe hours of audio/session; 2nd T4 = 2nd parallel stream |
| F5-TTS / CosyVoice zero-shot TTS + voice cloning | ~2–3 GB / ~3–4 GB inference | fits easily |
| RVC voice conversion (Applio) | A few GB, train + inference | fits; documented Kaggle notebook path w/ ngrok/Gradio tunnels: https://docs.aihub.gg/rvc/cloud/applio-cloud/ (index) |
| LLM QLoRA fine-tune (Unsloth/Axolotl tables) | 7–8B: 5–14 GB · 13–14B: 8.5–20 GB · 32B: 24–32 GB · 70B: 40–48 GB | fits ≤8B; 13–14B fits-with-quantization-or-offload (short seq/Unsloth, tight); 32B/70B does-not-fit one T4. LoRA on BF16 base: 7B needs 16–24 GB → QLoRA is the T4 route. Full FT 7B ~112 GB+: does-not-fit. Source (index): https://quantacloud.net/guides/lora-qlora-full-fine-tuning-vram |
| LLM inference on T4 (bonus) | 7B FP16 ~14 GB; 7–14B 4-bit GGUF/AWQ ~4–9 GB | fits / fits-with-quantization; llama.cpp/transformers can shard 32B 4-bit (~19 GB) across both T4s, slowly |
| OCR / documents | PaddleOCR-class: CPU/light GPU (<2 GB); Qwen2.5-VL-7B: 4-bit ~8–10 GB, BF16 ~17 GB | fits (dedicated OCR trivially; 7B VLM in 4-bit/FP16, BF16 just over one T4) |
| Embeddings / batch processing | bge-m3, Qwen3-Embedding ≤8B: a few GB (4-bit/FP16) | fits; ideal Kaggle batch job — 2 T4s = 2 shards; also a natural TPU batch job |

## 5. Platform reality check

Verified live 2026-11-08 from https://www.kaggle.com/docs/notebooks (official): sessions max **12 h (CPU/GPU) / 9 h (TPU)**; interactive sessions end after **20 min idle**; Save & Run All must finish within the same limits; **20 GB auto-saved disk** at /kaggle/working, everything else is scratch and vanishes; GPU session = 4 CPU cores / 29 GB RAM; free GPU/TPU availability is limited and **queued at busy times**; internet is a per-notebook Settings toggle (pip/HF downloads work when on); "abuse or misuse of GPU resources may result in suspension or termination". The docs page still mentions P100/TPU v3-8 in places — it lags the current T4 x2 / v5e-8 offering, which is confirmed by multiple Nov-2026 secondary sources (e.g. https://reconscribe.com/free-credits/ai-api/, index: 30 GPU h/wk "sometimes higher", 20 TPU h/wk, P100 retired 2026-09-15, v5e-8 replaced v3-8). Phone verification is required for GPU/TPU + internet per those secondary sources. Treat the mission's quota figures as the operative ones and the in-account Settings page as final authority.

Exposing a UI/API inside a session (all index, 2026-11-08): Gradio `share=True` → public `*.gradio.live` link; `cloudflared tunnel --url` → random `*.trycloudflare.com` (this is what the Qwen-27B TPU recipe uses); ngrok (needs auth token); Pinggy/Localtunnel (used by Kaggle ComfyUI launchers). All share the same properties: **random URL per session, no SLA, tunnel dies with the session, anyone with the URL can burn your quota** (the TPU recipes therefore generate an API key).

Policy (verified live 2026-11-08): https://www.kaggle.com/aup (eff. 2025-06-22) — users agree not "to abuse resources offered as part of the Services, including using such resources to engage in: cryptomining; DDOS attacks; **server farming**; creation of harmful or offensive content; malicious activity…; **activity unrelated to ML data science**; excessive crawling". Secondary summary of Kaggle's Terms (personal, internal, non-commercial use; not for the benefit of any third party) via https://www.maborak.com/blog/kaggle-free-gpu-ai-server (index, checked 2026-10-29 by its author) — the Terms page itself was not re-opened in this run, flag accordingly. Multiple accounts to farm quota are also a ban risk per Kaggle community guidance cited in secondary sources.

**What "deploying a service" on Kaggle can mean:** a personal, temporary ML demo/experiment endpoint that lives only while you keep a session open (≤9–12 h, inside weekly quota), for your own testing, with a throwaway URL. **What it cannot mean:** an always-on or production API, a backend for an app/website/customers, a service for third parties, or anything kept alive by keepalive scripts/multi-account rotation as a standing resource — technically impossible (idle timeout, session caps, quota, ephemeral disk, queueing) and squarely inside the AUP's "server farming" prohibition. Public TPU-endpoint recipes that hold a session "a little under nine hours" sit in this grey zone: fine to cite as a demo of what is technically possible, not as a sanctioned deployment pattern.

## Could not verify / thin evidence
- No first-party or Kaggle-measured FLUX-on-T4 generation time; range given is inference from RTX-4090/5060Ti numbers + one 2×T4 GGUF benchmark + one low-quality social post.
- No T4 timing for HunyuanVideo-1.5 or Wan-14B-GGUF; Wan-14B GGUF VRAM figures are community-guide claims only.
- Qwen-27B (~130 tok/s) and GLM-MoE TPU figures: single community project + forks, not independently reproduced; the 27B size/speed is plausible (55.6 GB BF16), the 320B-MoE claim is not corroborated.
- No public Kaggle TPU notebook for Llama-3.3-70B (any precision) or for any video model — absence-of-evidence from searches on 2026-11-08, not proof of impossibility.
- Kaggle's own docs page lags hardware names (P100/v3-8); quota numbers rest on the mission's fixed facts + consistent secondary sources, not a live in-account check (this run is signed out by design).
- Kaggle Terms of Use "personal non-commercial" wording is quoted via a secondary source; the AUP wording is verified live.

## Sources
Verified live (browser_open, 2026-11-08): https://www.kaggle.com/docs/notebooks · https://www.kaggle.com/aup
Index (browser_search, 2026-11-08) — all URLs are listed inline next to the claims they support in §§1–5 and in full in `notes/key-sources.md`.
