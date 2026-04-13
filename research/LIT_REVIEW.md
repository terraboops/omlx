# Hypercar Literature Review
_Last updated: 2026-04-12 (pass 6)_

Focused pass against the six Hypercar goals (1=context, 2=intelligence-breadth,
3=decode, 4=prefill, 5=swap<8GB, 6=M4 Pro 48GB fit). Every paper below maps to
at least one numbered goal and to a concrete file/system in the repo.

Already-cited prior art (NOT re-reviewed): TurboQuant (2504.19874), SpecPrefill
(2502.02789), STARC (2505.05772), IndexCache (2603.12201), Mamba-3 (2603.15569),
GPTQ (2210.17323).

## Papers reviewed this pass

### [MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](https://arxiv.org/abs/2407.02490) — 2407.02490
- **Authors**: Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu (Microsoft Research)
- **Published**: 2024-07 (NeurIPS 2024)
- **Hypercar goals it addresses**: Goal 4 (prefill speed, constant across context), Goal 1 (1M context validation)
- **TL;DR**: Empirically identifies three recurring sparse-attention patterns in
  long-context LLMs — A-shape (sink + local), Vertical-Slash (a few vertical
  columns plus diagonal stripes), and Block-Sparse. A cheap offline per-head
  search assigns each head one pattern; at inference the prefill kernel only
  computes the selected sparse shape. Reports up to 10x prefill speedup at 1M
  context on A100 with negligible quality loss on RULER and InfiniteBench.
- **Why it matters for Hypercar**: Goal 4 is our second-biggest gap (~60% of
  target, and *not* constant). Our existing `omlx/patches/specprefill.py` is
  a related but different sparse-prefill approach (attention-score driven
  token drop); MInference's per-head pattern dispatch is complementary and
  can likely be layered on top of it, or replace it for heads where the
  vertical-slash pattern dominates. The offline pattern search is a
  one-time calibration against Qwen3-Coder — the runtime cost is just a
  sparse matmul, which MLX already expresses well.
- **Cost of adoption**: M (1-3 days). Pattern search script + per-head
  dispatch table, then a vertical-slash kernel. The block-sparse path is
  already expressible via MLX's existing attention primitives. Risk: vertical
  indices are input-dependent, so the kernel must be dynamic — this is where
  Apple Silicon sparse kernels historically struggle.
- **Local PDF**: research/2407.02490_minference.pdf

### [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2406.10774) — 2406.10774
- **Authors**: Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han (MIT, NVIDIA, UW)
- **Published**: 2024-06 (ICML 2024)
- **Hypercar goals it addresses**: Goal 3 (decode speed, constant across context), Goal 5 (swap, indirectly)
- **TL;DR**: Pages the KV cache and, per query, computes an upper-bound score
  per page using only the element-wise min and max of keys. Only the top-K
  pages participate in full attention. No accuracy degradation on PG-19,
  LongBench, PasskeyRetrieval with K=4096 pages even at 32K+ context, and
  7.03x self-attention speedup at long context during decode.
- **Why it matters for Hypercar**: Goal 3 is our single biggest gap (20 tok/s
  vs 50 target, and it degrades with context). Quest gives constant-cost
  decode because the attention work is bounded by K regardless of total
  context — exactly the shape we need. It also composes cleanly with our
  existing 3-bit KV: the page min/max bounds can be stored in the unquantized
  headroom we already maintain per-group, so the dequant hot path doesn't
  change. Fork/rewind in `omlx/turboquant_kv.py` already has a page abstraction
  we can reuse.
- **Cost of adoption**: M (2-4 days). Add page-level min/max tracking to
  TurboQuantKVCache, a top-K page-selection primitive, and a gather-then-attend
  path in the attention forward. Biggest risk: top-K on a 1M-page KV (K=4096,
  N=64K pages) needs a fast MLX top-K — we'd want to check whether MLX's
  argpartition is competitive before committing.
- **Local PDF**: research/2406.10774_quest.pdf

### [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) — 2402.02750
- **Authors**: Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu (Rice, TAMU, Stevens, CMU)
- **Published**: 2024-02 (ICML 2024)
- **Hypercar goals it addresses**: Goal 5 (swap headroom), Goal 1 (longer context in same budget)
- **TL;DR**: Analyses KV statistics and finds Key has persistent per-channel
  outliers while Value does not. Quantizes K per-channel (preserving channel
  scale) and V per-token. At 2 bits this matches fp16 quality on LongBench
  and yields 2.35-3.47x throughput on A100. The asymmetry is the trick —
  symmetric 2-bit destroys K.
- **Why it matters for Hypercar**: Our current KV is 3-bit (native MLX
  QuantizedKVCache and TurboQuant). Goal 5 (swap < 8GB under load) is marked
  "borderline, not under load" — dropping KV from 3 to 2 bits frees ~7.5GB at
  1M context (22.5GB → ~15GB), which is exactly the headroom we need at 16K
  NIAH under Chrome+editor load. The per-channel-K / per-token-V asymmetry is
  a small patch to `omlx/turboquant_kv.py` — the codebook is already
  per-group, we'd generalise "group axis" to be channel or token depending on
  whether we're quantizing K or V.
- **Cost of adoption**: S-M (1-2 days). Refactor the quant axis in the TQ
  codec, re-run NIAH and HumanEval gates, compare against native. Risk: KIVI
  predates TurboQuant's WHT rotation — we need to check whether WHT kills the
  per-channel outliers (making per-channel unnecessary) or whether outliers
  survive the rotation. If WHT already handles them, we get KIVI-style 2-bit
  for free in the existing per-group format. That experiment alone is worth
  running.
- **Local PDF**: research/2402.02750_kivi.pdf

### [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) — 2404.06654
- **Authors**: Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg (NVIDIA)
- **Published**: 2024-04 (COLM 2024)
- **Hypercar goals it addresses**: Goal 2 (intelligence breadth, independent evals), Goal 1 (context validation)
- **TL;DR**: Extends single-needle NIAH into 13 synthetic tasks in 4 categories:
  multi-key/multi-value/multi-query retrieval, variable tracking (multi-hop),
  frequent-word aggregation, and long-context QA. Crucially it exposes a
  very large gap between claimed and *effective* context length — most models
  rated for 128K collapse well before that.
- **Why it matters for Hypercar**: Our Goal 2 status is "3 evals, GPT-4 parity
  on coding only". RULER gives us the long-context retrieval eval we need —
  and the task generators are fully synthetic, so we can run them offline
  against our stack at 4K, 16K, 64K, 256K, 1M to *prove* that 3-bit KV and
  sparse-attention changes don't silently destroy quality. It slots directly
  into `omlx/bench/hypercar_bench.py` as a new gate tier.
- **Cost of adoption**: S (half a day to a day). RULER's task generators are
  open source and pure Python. Wrap them as a new benchmark phase, pick one
  retrieval and one tracing task for `--quick`, add full 13-task run to
  `--full`. Risk: at 256K+ the eval itself becomes expensive — gate only a
  handful of lengths by default.
- **Local PDF**: research/2404.06654_ruler.pdf

### [Text-to-LoRA: Instant Transformer Adaption](https://arxiv.org/abs/2506.06105) — 2506.06105
- **Authors**: Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, Robert Tjarko Lange (Sakana AI)
- **Published**: 2025-06 (ICML 2025)
- **Hypercar goals it addresses**: Goal 2 (intelligence, agentic breadth), complements TTT
- **TL;DR**: Trains a hypernetwork that maps a natural-language task description
  to a full set of LoRA adapter weights in a single forward pass. The
  hypernetwork is trained once offline on a library of pre-computed LoRAs;
  inference is gradient-free and produces adapters competitive with per-task
  fine-tuning on unseen tasks.
- **Why it matters for Hypercar**: We already have TTT (`omlx/ttt.py`) with
  LoRA-style fast weights on MLP down_proj — but it needs gradient steps per
  task, which caps how aggressively we can specialise per-request. A
  Text-to-LoRA hypernetwork over our existing down_proj LoRA shape would let
  the server mint a task-specific adapter *before* the first decoded token
  from just the system-prompt/tool-schema description, and then optionally
  let TTT refine it online. That's a direct compounding win on agentic
  quality without touching decode/prefill.
- **Cost of adoption**: L (multi-day, possibly week+). Requires (a) assembling
  a library of LoRAs for Qwen3-Coder across representative agentic tasks, (b)
  training the hypernetwork, (c) wiring into `/v1/ttt/*`. The training step
  is the expensive part and may not fit on the M4 Pro — likely needs a
  one-off cloud run. Highest risk is whether the T2L recipe transfers to a
  30B MoE; the paper's experiments are on dense models in the 1-8B range.
- **Local PDF**: research/2506.06105_text_to_lora.pdf

## Pass 2 — 2026-04-13

Bucket coverage from the brief: (1) Apple Silicon / MLX-native attention, (2)
speculative decoding for quantized/MoE, (3) reasoning/agentic evals beyond
coding+retrieval, (4) cheap online/continual fine-tuning, (5)
constant-throughput long-context beyond Quest/MInference. The strongest
clusters were buckets 2, 3, 4, and 5 — bucket 1 is intentionally underweighted
(see "Gap not closed" at the end of this section).

### [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](https://arxiv.org/abs/2410.10819) — 2410.10819
- **Authors**: Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, Song Han (MIT, NVIDIA, Edinburgh)
- **Published**: 2024-10 (ICLR 2025)
- **Hypercar goals it addresses**: Goal 3 (decode), Goal 4 (prefill), Goal 5
  (swap headroom), Goal 1 (longer context in same budget)
- **TL;DR**: Uses a tiny synthetic-data optimisation step to identify two head
  populations: a small fraction of *retrieval heads* that genuinely need full
  KV history, and the majority *streaming heads* that only ever attend to
  attention-sink + a recent window. Retrieval heads keep a full KV cache;
  streaming heads keep a constant-length cache (sink + sliding window).
  Reports 2.55x KV memory reduction on MHA, 1.67x on GQA, with up to 2.18x
  decode and 1.73x prefill speedup, and demonstrates 3.3M-token context on a
  single A100. Crucially the head classification is *static* — once
  calibrated per model, no runtime decision is needed.
- **Why it matters for Hypercar**: This is the cleanest lever we have for
  *constant-cost decode and prefill* simultaneously. Quest (yesterday's pick)
  bounds attention work per query at decode time but still needs the full KV
  resident; DuoAttention drops the cache itself for streaming heads, which
  composes orthogonally with both Quest's selection (apply on retrieval heads
  only) and our 3-bit codec (3-bit retrieval cache + length-bounded
  streaming). On Qwen3-Coder's 48-layer GQA, if even half the heads turn out
  to be streaming, KV at 1M context drops from 22.5GB toward ~12GB — that's
  the swap headroom Goal 5 needs and a strict super-set of the KIVI 2-bit
  experiment. The retrieval-head identifier is one calibration script very
  similar in shape to the MInference offline pattern search we already added.
  Slots into `omlx/turboquant_kv.py` as a per-(layer, head) policy table.
- **Cost of adoption**: M (3-5 days). One-time calibration script (synthetic
  passkey-style data, gradient-descent on a per-head gate), per-head policy
  table shipped under `omlx/patches/duoattention_policies/`, and a forked
  KV cache that holds two storage classes per head. The composition with
  TurboQuant pages is the trickiest part — streaming heads probably should
  not be quantized at all (the cache is tiny and the quality cost of
  retrieval-head misclassification compounds at the streaming heads). Risk:
  retrieval-head fraction on MoE models is poorly characterised in the
  paper — we may discover Qwen3-Coder needs a higher fraction than the
  ~25% Llama numbers, eroding the win.
- **Local PDF**: research/2410.10819_duoattention.pdf

### [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — 2406.16858
- **Authors**: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang (Peking University, Microsoft Research, Waterloo)
- **Published**: 2024-06 (EMNLP 2024)
- **Hypercar goals it addresses**: Goal 3 (decode speed)
- **TL;DR**: Speculative decoding with a tiny EAGLE draft model that operates
  on the *target model's hidden states* (not just tokens), and a
  *context-aware dynamic draft tree* that re-shapes itself per-step using the
  draft model's confidence as an acceptance-rate estimator. EAGLE-2 reports
  3.05x-4.26x lossless speedup, 20-40% better than EAGLE-1, across diverse
  generation tasks. The output distribution is provably identical to greedy
  / vanilla sampling.
- **Why it matters for Hypercar**: Goal 3 is our largest absolute gap. Quest
  attacks the per-step attention cost; EAGLE-2 attacks the *number of model
  forwards* — the two are multiplicative, not competing. EAGLE is uniquely
  attractive for our MoE target because the draft network is shallow and
  *operates on hidden states*, which means its forward pass goes through the
  same MoE router as the target — so accepted draft tokens incur no extra
  expert-dispatch cost. Medusa-style multi-head approaches by contrast bolt
  extra heads onto the dense projection and are awkward with MoE expert
  routing. The MLX path: train a single EAGLE head once against
  Qwen3-Coder-30B-A3B's hidden-state distribution (this is days of GPU time,
  not weeks), ship the head as a small file, and add a verification loop in
  `omlx/hypercar_server.py`'s decode path. Verification reuses prefill (one
  forward, multiple positions) which we have already optimised.
- **Cost of adoption**: M-L (1-2 weeks including the draft-head training).
  The draft-head training is the only real obstacle — EAGLE's repo has the
  recipe but it assumes PyTorch + a single dense base model. We'd need an
  MLX port of the head and a one-off training run on a rented box (or a
  Qwen-team-released EAGLE head if one exists for the 30B-A3B variant).
  Risk: speculative speedup compounds with batch size, and our local-server
  use case is batch=1 — the upside is bounded compared to the paper's
  multi-batch numbers, but the 3-bit KV decode is already memory-bound, so
  cutting forwards is exactly the right axis.
- **Local PDF**: research/2406.16858_eagle2.pdf

### [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045) — 2406.12045
- **Authors**: Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan (Sierra)
- **Published**: 2024-06
- **Hypercar goals it addresses**: Goal 2 (intelligence breadth — agentic, not retrieval)
- **TL;DR**: Two synthetic customer-service domains (retail, airline) where
  the model is given a tool API + a policy document and must hold a
  multi-turn conversation with a *language-model-simulated user* whose state
  evolves. Scoring is exact: at end of conversation, the database state is
  diffed against the annotated goal state. Introduces `pass^k` (passes all
  k independent trials of the same task) as a reliability metric. GPT-4o
  scores < 50% on tasks and < 25% pass^8 in retail — i.e., the eval is hard
  and reveals reliability problems that single-shot evals miss.
- **Why it matters for Hypercar**: Goal 2 status is "3 evals, GPT-4 parity on
  coding only" — RULER (yesterday) closes the long-context retrieval gap,
  τ-bench closes the *agentic tool use* gap. Critically, our project ships a
  tool-using OpenCode integration and a TTT engine that trains on
  *trajectories* — τ-bench is the natural validation that the TTT loop is
  improving anything other than HumanEval pass@1. The harness is
  pure-Python and OpenAI-compatible, so it talks to `hypercar_server`'s
  `/v1/chat/completions` directly with no adapter work. Because the user is
  also an LLM, you can run the user-side against a small local model
  (gemma2-2b, qwen2.5-3b) to keep evaluation fully offline.
- **Cost of adoption**: S-M (1-2 days). Vendor or pip-install
  `tau-bench`, write a `omlx/eval/tau_bench/` shim that points at our
  server, and add a `phase_tau()` to `hypercar_bench.py` running 5
  airline + 5 retail tasks in `--full`. Gate at e.g. >= 30% pass^1 on
  retail to start (well below GPT-4o, which is the ceiling we're chasing).
  Risk: long-running — each task is many turns, and a full 100-task run
  is a ~30 minute eval even with our server. Use a sampled subset.
- **Local PDF**: research/2406.12045_tau_bench.pdf

### [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — 2405.14734
- **Authors**: Yu Meng, Mengzhou Xia, Danqi Chen (Princeton, UVA)
- **Published**: 2024-05 (NeurIPS 2024)
- **Hypercar goals it addresses**: Goal 2 (intelligence breadth, agentic
  quality), composes with TTT
- **TL;DR**: A DPO variant that drops the frozen reference model entirely:
  the reward is just the average log-probability of the candidate sequence,
  and the loss is a margin on (winner_logp - loser_logp). Adds a hand-tuned
  target margin γ. Empirically beats DPO by 6-7 points on AlpacaEval2 and
  Arena-Hard at the same data, on the same trained model, with roughly half
  the GPU memory because there's no reference forward pass.
- **Why it matters for Hypercar**: Our `omlx/ttt.py` engine currently does
  supervised LoRA updates on *passing* trajectories — it can't learn from
  the *contrast* between a passing and a failing trajectory of the same
  task, which is exactly the signal a tool-using agent generates by the
  bucketload (multiple attempts at the same problem, some good, some bad).
  SimPO is the cheapest known way to consume that contrast: no reference
  model means the existing TTT LoRA *is* the policy, no second copy of the
  base weights, no extra Metal memory pressure (which we are already tight
  on per Goal 5). The math is also embarrassingly small — about 30 lines
  of MLX once the trajectory pair format is defined. This is the
  "complement to TTT, cheaper than Text-to-LoRA" that the brief asked for:
  T2L needs a multi-day offline hypernetwork training; SimPO needs only
  pairs from yesterday's agent runs and a tiny gradient step at the same
  cadence the existing TTT engine already uses.
- **Cost of adoption**: S (1-2 days). Add `simpo_step()` next to the
  existing `ttt_step()` in `omlx/ttt.py`, take pairs of (winning,
  losing) trajectories from the agent log, compute per-token logprobs
  with the existing forward path, apply the margin loss, and gradient-step
  the same down_proj LoRA we already have. Add `/v1/ttt/simpo` endpoint
  parallel to existing TTT endpoints. Risk: SimPO's quality relies on a
  well-tuned γ and a length-normalised reward — small-batch online use
  has not been studied in the original paper (their experiments are
  100K+ pairs). We will likely overshoot if we apply naive learning rates.
  Mitigation: cap the per-step LoRA update norm, exactly as the existing
  TTT loop already does.
- **Local PDF**: research/2405.14734_simpo.pdf

## Pass 3 — 2026-04-13

Bucket coverage from the brief: (1) MoE-specific serving optimizations,
(2) prompt/context compression, (3) reasoning+code evals beyond
coding+retrieval+agentic, (4) paged/ring attention for single-device
unified memory, (5) non-autoregressive/parallel decoding for code. The
strongest clusters were buckets 1, 2, and 3 — buckets 4 and 5 are
intentionally left empty (see "Gap not closed" at the end of this
section) rather than padded with weak papers.

### [ProMoE: Fast MoE-based LLM Serving using Proactive Caching](https://arxiv.org/abs/2410.22134) — 2410.22134
- **Authors**: Xiaoniu Song, Zihang Zhong, Rong Chen, Haibo Chen (Shanghai Jiao Tong University, IPADS)
- **Published**: 2024-10 (preprint; v3 2025-09)
- **Hypercar goals it addresses**: Goal 6 (M4 Pro 48GB fit), Goal 3
  (decode speed, constant across context), Goal 5 (swap headroom), Goal 4
  (prefill speed)
- **TL;DR**: Proactively predicts which MoE experts the *next few*
  tokens will route to, using cheap linear probes over the current
  token's hidden state / gating logits, and prefetches cold experts
  into the fast tier *before* the router demands them. Treats the cache
  miss as a scheduling problem rather than a reactive fault. Reports
  2.20x prefill and 2.07x decode speedup (up to 3.21x / 5.02x) over
  the best reactive offloading baselines by eliminating cache misses
  on the critical path.
- **Why it matters for Hypercar**: We currently pay the full 32GB of
  8-bit expert weights resident in Metal, even though Qwen3-Coder
  activates only 8 of 128 experts per token (~6% of the weight bytes
  per step). That is the single largest un-exploited inefficiency in
  the project: we are budget-bound on Goal 6 because of weights we
  don't need per-step. Apple Silicon unified memory radically improves
  ProMoE's economics — the "prefetch" on a discrete GPU is a PCIe
  copy, on the M4 Pro it's a pointer flip / page touch with no bulk
  transfer, so the policy doesn't need to be as aggressive to win. A
  tiered layout (hot experts 8-bit resident, warm experts 4-bit
  resident, cold experts packed in "far" unified memory with on-demand
  unpack) would free 10-20GB at load and make `Goal 5: swap <8GB
  under load` comfortable instead of borderline. Slots into
  `omlx/hypercar_server.py` model loading, plus a per-layer router
  hook that records prior routing decisions and a small predictor
  head.
- **Cost of adoption**: L (1-2 weeks). The research payoff is very
  high but the project touches three things at once: (a) a custom
  expert-weight layout that lets us partially-load experts from
  mmap'd far memory without materialising all of them, (b) a tiny
  predictor calibrated on real traffic (can probably be a linear
  probe on the previous-layer router logits — the paper's own
  ablations support this), (c) a scheduler that overlaps predictor
  dispatch with attention. Risk: MLX's module system currently
  loads all weights at construction time — we'd need a lazy-load
  `Linear` wrapper for the expert MLPs. Worth de-risking with a
  one-day "can we even get Qwen3-Coder to run with half the experts
  zero-weighted and see a memory drop?" probe before committing to
  the full project.
- **Local PDF**: research/2410.22134_promoe.pdf

### [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968) — 2403.12968
- **Authors**: Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Dongmei Zhang (Tsinghua, Microsoft Research)
- **Published**: 2024-03 (ACL 2024)
- **Hypercar goals it addresses**: Goal 1 (effective 1M context),
  Goal 4 (prefill speed — fewer tokens to prefill)
- **TL;DR**: Trains a small (XLM-RoBERTa-size, ~270M params) token
  classifier that, for every token in a long prompt, predicts whether
  it is essential or droppable. Supervised by a distillation procedure
  where GPT-4 is asked to compress reference prompts and the
  classifier learns to reproduce the resulting keep/drop mask. The
  compressor itself runs in milliseconds, is bidirectional (unlike
  the LLMLingua-1 causal-LM entropy heuristic), and is task-agnostic.
  At 2x-5x compression ratios it preserves near-identical downstream
  quality on LongBench, ZeroScrolls, GSM8K, and BBH while cutting
  end-to-end latency 1.6x-2.9x.
- **Why it matters for Hypercar**: Prompt compression is an
  *orthogonal* axis to everything else in the backlog. Quest,
  DuoAttention, MInference all reduce the cost of processing a given
  N tokens; LLMLingua-2 reduces N itself, before the big model ever
  sees the prompt. At 1M context (Goal 1) a 3x compression turns a
  10-minute prefill at 500 tok/s into a 3-minute prefill — same
  kernel, same memory budget, same KV cache. For agentic workflows
  (OpenCode + tool outputs) compression is especially appealing
  because tool JSON, directory listings, and stale scratchpad
  contents are overwhelmingly droppable without semantic loss. The
  compressor is a one-file model load — we already use mlx-lm for
  the user-side in tau-bench, so the same infra ships it. Slot:
  a pre-processing middleware in `omlx/hypercar_server.py` with a
  `--compress-prompts llmlingua2:3x` flag, off by default and
  opt-in per request via a header so OpenCode sessions can turn
  it on without breaking exact-match replay.
- **Cost of adoption**: M (3-5 days). Port (or reuse) the official
  MS-released compressor checkpoint via mlx-lm, add the middleware,
  add a new RULER gate that specifically tests compressed-prompt
  retrieval quality (because the compressor *can* destroy multi-key
  retrieval at aggressive ratios — we need the gate to be honest).
  Risk: compression is lossy by construction, so we must gate *both*
  HumanEval AND RULER multi-key at the compression ratio we ship.
  A 3x compressor that drops multi-key@16K from 0.9 to 0.5 is a
  silent quality collapse we would not catch without RULER.
- **Local PDF**: research/2403.12968_llmlingua2.pdf

### [LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://arxiv.org/abs/2403.07974) — 2403.07974
- **Authors**: Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica (UC Berkeley, MIT, Cornell)
- **Published**: 2024-03 (ICLR 2025, continuously updated)
- **Hypercar goals it addresses**: Goal 2 (intelligence breadth —
  contamination-free coding eval, independent from HumanEval)
- **TL;DR**: Continuously scrapes new problems from LeetCode, AtCoder,
  and CodeForces contests each month, and hosts only problems with
  publication dates *after* each model's training cutoff. Beyond
  plain code generation, it scores four scenarios: generation,
  self-repair from a failing stack trace, code execution (predict
  the output), and test output prediction. Empirically shows that
  several models with strong HumanEval scores regress badly on
  LiveCodeBench — direct evidence that HumanEval numbers are
  contaminated in the public-model era.
- **Why it matters for Hypercar**: Our Goal 2 status depends on a
  90% HumanEval number that is almost certainly contaminated for
  Qwen3-Coder-30B — HumanEval was published in 2021, and Qwen3's
  training corpus trivially contains it. "4 independent evals" is
  an empty claim if one of the four is a memorisation test. A
  LiveCodeBench gate pinned to problems *after* Qwen3-Coder's known
  cutoff is the cheapest way to turn the HumanEval number into an
  honest one. The harness is pure Python, the problem dataset is
  hosted on HuggingFace as versioned releases, and the execution
  scoring reuses the same sandboxed runner we already have in
  `omlx/ttt.py`'s code verifier. This is a near-drop-in replacement
  for the existing HumanEval phase with a higher signal-to-noise
  ratio. Slot: new `omlx/eval/livecodebench/` module, `phase_lcb()`
  in `hypercar_bench.py`, gate at >= 30% on the post-cutoff release.
- **Cost of adoption**: S (half a day to a day). The official
  release ships an evaluation script; we wrap its problem loader,
  reuse our existing sandboxed Python executor from the TTT code
  verifier, and add a phase. The harder decision is sizing the
  quick variant so it stays under 60 seconds — LiveCodeBench has
  400+ problems, we likely pin 20 for `--quick` and a 100-problem
  subset for `--full`.
- **Local PDF**: research/2403.07974_livecodebench.pdf

### [BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions](https://arxiv.org/abs/2406.15877) — 2406.15877
- **Authors**: Terry Yue Zhuo et al. (BigCode collaboration — 30+ authors across Monash, HuggingFace, Sea AI Lab, etc.)
- **Published**: 2024-06 (ICLR 2025)
- **Hypercar goals it addresses**: Goal 2 (intelligence breadth —
  tool-using code generation distinct from agentic dialogue)
- **TL;DR**: 1,140 fine-grained Python tasks that each require calling
  across a curated set of 139 real third-party libraries spanning 7
  domains (data analysis, web, ML, scientific computing, etc.), with
  complex compositional natural-language instructions. Each task has
  an average of 5.6 unit tests and ~99% branch coverage — scoring is
  exact execution, not model-graded. On the original paper, even
  top frontier models scored 50-60% vs 97% for humans; the benchmark
  is specifically designed to *not* saturate. A companion
  `BigCodeBench-Instruct` strips the docstring down to a short
  natural-language instruction, testing whether models can infer the
  correct library usage from intent alone.
- **Why it matters for Hypercar**: We run Qwen3-*Coder*, on a project
  where the realistic workload is "generate Python that uses pandas
  / requests / pytest / BeautifulSoup" — exactly BigCodeBench's
  shape. tau-bench (pass 2) covers *conversational* tool use; what
  it doesn't cover is the single-shot code-synthesis-with-real-
  library-calls case that is OpenCode's primary workload.
  BigCodeBench fills that gap and gives us a legitimately hard fourth
  eval at the same tier as HumanEval but without contamination and
  at realistic library breadth. It also composes with TTT: any task
  where a BigCodeBench problem fails but a retry passes becomes a
  SimPO pair (Task 15), so we get both a harder eval and a fresh
  training signal from the same run. Slot: vendor or pip-install
  BigCodeBench, reuse the existing sandboxed Python executor from
  the TTT code verifier, add `phase_bcb()` with 10 sampled tasks
  in `--quick` and 100 in `--full`.
- **Cost of adoption**: S-M (1-2 days). The pip package ships a
  sandbox-based executor (Docker on Linux, but the problem
  definitions + test cases are just Python files we can run in our
  own subprocess sandbox on macOS). Biggest risk: BigCodeBench task
  dependencies are heavy — installing pandas, scipy, etc. in the
  benchmark venv is fine, but some tasks pull in Docker-only
  dependencies that we'll need to filter out to keep the macOS run
  clean. A pinned task subset documented in
  `omlx/eval/bigcodebench/macos_compatible.json` handles this.
- **Local PDF**: research/2406.15877_bigcodebench.pdf

**Gap not closed this pass**: bucket 4 (paged / ring attention for
single-device unified memory). The 2024-2026 arxiv work in this area
targets multi-GPU serving (vLLM, SGLang, Sarathi-Serve) where the
paged allocator is load-bearing because of fragmentation across many
concurrent requests. For our single-user batch-1 local serving
profile the paging payoff collapses — we do not have concurrent
requests to pack into shared pages, and the fragmentation argument
does not apply to a single long sequence. The right next step for
this bucket is not another paper but a measurement: profile how
much of our actual 1M-context KV bill is fragmentation versus live
data, and only revisit paging if fragmentation is a real cost.

**Gap not closed this pass**: bucket 5 (non-autoregressive /
parallel decoding for code). The strongest recent papers (diffusion
LLMs for code, Jacobi/lookahead decoding) target dense models and
do not compose well with MoE expert routing — every parallel draft
token has to route through the same experts or you pay for a
routing mismatch on verification. EAGLE-2 (pass 2) remains the
best parallel-decoding bet for our specific MoE target because the
draft operates on the target model's hidden states and therefore
re-uses the router decisions. No fresh paper was worth displacing
EAGLE-2 from the backlog.

## Pass 4 — 2026-04-13

Bucket coverage from uncovered territory: (1) hard-reasoning evals beyond
RULER/τ-bench (MMLU-Pro, LiveBench), (2) adaptive computation / early-exit
decoding on the model itself (LayerSkip), (3) dynamic token-level prefill
pruning distinct from MInference's head-pattern dispatch (LazyLLM), (4)
realistic software-engineering agent eval (SWE-agent). The strongest
clusters here were (2) and (1) — (2) is the first paper in the review that
optimises decode via *skipping layers*, which is mechanically orthogonal to
every prior optimisation and directly composes with EAGLE-2 as a
self-speculative alternative. Gap-not-closed note at the end of this
section.

### [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710) — 2404.16710
- **Authors**: Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed A. Aly, Beidi Chen, Carole-Jean Wu (Meta FAIR, CMU)
- **Published**: 2024-04 (ACL 2024)
- **Hypercar goals it addresses**: Goal 3 (decode speed, constant across context), Goal 4 (prefill speed — cheaper first-draft passes)
- **TL;DR**: A training recipe plus an inference recipe. Training: a
  layer-dropout schedule (later layers dropped more aggressively) and a
  shared early-exit loss (every layer's output fed through a shared LM
  head). The resulting model can generate tokens by only running the
  first N layers and then jumping to the shared head. Inference:
  self-speculative decoding — draft tokens with an early exit (first few
  layers), verify by running the full model on the tree of drafts in one
  forward. No separate draft model. Reports up to 2.16x speedup on
  summarization and 1.6x on coding with lossless greedy-equivalent
  output.
- **Why it matters for Hypercar**: Goal 3 is our biggest gap, and we
  already have EAGLE-2 on the backlog (Tasks 28, 29) as the speculative-
  decoding lever. EAGLE-2 needs a one-off offline training run on cloud
  GPUs to produce the draft head, which is the single biggest scheduling
  obstacle to landing it. LayerSkip is the *same speedup axis* without a
  separate draft model — the "draft" is the Qwen3-Coder first-K-layers
  and the "verifier" is the full 48-layer pass, both routed through the
  same MoE experts. This completely sidesteps the MoE routing-mismatch
  problem that makes Medusa awkward, AND it avoids the cloud training
  step. The catch is that Qwen3-Coder was NOT trained with LayerSkip's
  layer-dropout recipe, so the early-exit quality at small K is bad out
  of the box. But there's a middle path: use LayerSkip's inference-time
  self-speculation without the training-time recipe, and calibrate a
  per-layer exit-confidence threshold so only *easy* tokens (where the
  early layers are already confident) exit early. That is a runtime
  engineering task that touches `omlx/hypercar_server.py`'s decode loop
  and can be tried without retraining the base model.
- **Cost of adoption**: M (3-5 days for the calibration-only variant,
  multi-week if we retrain). New `omlx/patches/layerskip_decode.py`
  hooking the decode loop, an offline script to measure per-layer
  exit confidence on a calibration corpus, and a gate that verifies
  lossless output (exact-match vs greedy baseline) on HumanEval+.
  Biggest risk: on a model not trained with LayerSkip's recipe, the
  early-exit confidence may be uncalibrated — "confident wrong
  tokens" would silently destroy quality. Mitigation: the verification
  pass in self-speculative decoding catches this at the cost of the
  speedup degrading gracefully; only the *already-equivalent* tokens
  are accepted, so quality is provably identical to greedy decoding.
- **Local PDF**: research/2404.16710_layerskip.pdf

### [MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574) — 2406.01574
- **Authors**: Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, Wenhu Chen (Waterloo, Toronto, CMU, HKUST)
- **Published**: 2024-06 (NeurIPS 2024)
- **Hypercar goals it addresses**: Goal 2 (intelligence breadth — hard reasoning eval, fills the "MMLU-style reasoning" gap from CLAUDE.md's status row)
- **TL;DR**: A hard upgrade of MMLU. Three concrete changes: (a) 10
  answer options per question instead of 4 (cuts random-guess baseline
  from 25% to 10%), (b) heavy filtering to remove low-quality or
  trivially-answerable items from the original MMLU pool, and (c) adds
  ~12K new reasoning-heavy questions sourced from TheoremQA and
  STEM textbook problem sets. The resulting benchmark drops GPT-4o from
  ~86% on MMLU to ~72% on MMLU-Pro and is much more sensitive to
  chain-of-thought prompting — i.e. it actually measures reasoning, not
  recall.
- **Why it matters for Hypercar**: The CLAUDE.md "Current status" row
  for Goal 2 literally calls out "MMLU-style reasoning" as a missing
  independent eval. RULER covers long-context retrieval, τ-bench covers
  agentic tool use, LiveCodeBench/BigCodeBench cover coding — the gap
  is general world-knowledge reasoning. MMLU-Pro is specifically
  designed to resist the contamination that makes plain MMLU useless for
  a 2025-era coder model, so it is the right modern replacement. The
  harness is trivial: the dataset is a HuggingFace dataset, each
  question is a multiple-choice prompt, scoring is regex extraction of
  the answer letter. Slots into `omlx/bench/hypercar_bench.py` as a new
  phase with ~25 questions in `--quick` and 500-1000 in `--full`.
- **Cost of adoption**: S (half a day to a day). No code changes to the
  model, no new dependencies beyond `datasets`, and the answer-
  extraction regex is already used in LiveCodeBench integration (Task
  18). The main scoping question is which subject subset to run in
  `--quick` — computer science + math give the strongest signal for a
  coder model and keep the quick tier under 60 seconds. Risk:
  multiple-choice evals can reward lucky guessing if the model produces
  bad chain-of-thought — mitigate by running with the paper's own
  chain-of-thought prompt template rather than raw Q&A.
- **Local PDF**: research/2406.01574_mmlu_pro.pdf

### [LiveBench: A Challenging, Contamination-Free LLM Benchmark](https://arxiv.org/abs/2406.19314) — 2406.19314
- **Authors**: Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Benjamin Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Siddartha Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, Micah Goldblum (Abacus.AI, NYU, Nvidia, Maryland, USC)
- **Published**: 2024-06 (NeurIPS 2024)
- **Hypercar goals it addresses**: Goal 2 (intelligence breadth — contamination-free general-purpose eval, complements LiveCodeBench by covering math, reasoning, data-analysis, instruction-following)
- **TL;DR**: A monthly-updated multi-task benchmark across six
  categories (math, reasoning, coding, language, instruction-following,
  data analysis) where each category draws its questions from *recent*
  sources (arxiv papers from the last month, very recent competition
  problems, fresh news articles). Every question is checkable
  automatically with ground-truth-based scoring, no LLM-as-judge. At
  release, GPT-4o scored ~54% average, Claude-3.5-Sonnet ~58%, and the
  gap between "contaminated" and "uncontaminated" versions of the same
  task family was as large as 15 points on some model families —
  direct evidence that training data contamination inflates most other
  public evals.
- **Why it matters for Hypercar**: LiveCodeBench (pass 3, Task 18)
  closes the contamination gap on *coding* specifically; LiveBench
  closes it on everything else, and in particular covers the
  "instruction-following" and "data analysis" axes that our actual
  agentic workload (OpenCode + tool output + repo analysis) lives on.
  Together, LiveCodeBench + LiveBench + MMLU-Pro give us *three*
  independent reasoning/coding evals that are all resistant to the
  HumanEval-style contamination problem. The harness is a pip package
  (`livebench`) that talks to any OpenAI-compatible endpoint —
  `hypercar_server` is already compatible. We'd pin a dated release
  (e.g. `livebench-2025-10`) so the bench is reproducible across
  Hypercar commits. Slots into `omlx/bench/hypercar_bench.py` as a
  `phase_livebench()` alongside the existing coding phases.
- **Cost of adoption**: S-M (1-2 days). Install the pip package, pin a
  dated release, write the phase shim, and pick a default subset: math
  + reasoning + data_analysis is ~80 tasks per release and runs in
  under 5 minutes on our server. Risk: LiveBench's scoring scripts
  sometimes depend on `pandas`/`numpy` at specific versions — need to
  pin those or sandbox the scoring in a subprocess.
- **Local PDF**: research/2406.19314_livebench.pdf

### [LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference](https://arxiv.org/abs/2407.14057) — 2407.14057
- **Authors**: Qichen Fu, Minsik Cho, Thomas Merth, Sachin Mehta, Mohammad Rastegari, Mahyar Najibi (Apple)
- **Published**: 2024-07 (COLM 2024)
- **Hypercar goals it addresses**: Goal 4 (prefill speed, constant across context), Goal 1 (1M context validation — the paper's headline experiments are at 32K-128K prefill)
- **TL;DR**: During prefill, at each transformer layer, score every
  token's importance using the attention scores the layer already
  produces, and keep only the top-K tokens for the *next* layer's
  forward pass. Pruned tokens are not discarded — their KV entries are
  cached and can be revived on demand during decode if a future token
  attends strongly to them. At 128K context with Llama-2, reports up to
  2.34x prefill speedup on LongBench with < 1% accuracy loss; reports
  that 43% of tokens are pruned by the final layer on average.
  Critically: training-free, no recalibration, works on an off-the-shelf
  model.
- **Why it matters for Hypercar**: LazyLLM is distinct from every prior
  prefill paper in the backlog. MInference (pass 1) dispatches per-head
  *attention shapes* but every token goes through every layer.
  SpecPrefill (existing prior art) drops tokens upfront based on a
  surrogate model. LazyLLM drops tokens *layer-by-layer* using the
  target model's own attention, and — crucially — retrieves dropped
  tokens *on demand* at decode time. This is exactly the shape we need
  for agentic workloads where most of the context (tool output, repo
  noise) is skippable for most layers but occasionally a later question
  needs to revive a previously-dropped span. LazyLLM is also from the
  Apple ML team, so the design is informed by Apple Silicon memory
  hierarchies — it's the closest thing in the literature to a paper
  that was written with MLX in mind. Slots into
  `omlx/patches/lazyllm_prefill.py` as a per-layer token-selection hook
  alongside `omlx/patches/specprefill.py` (same registration point,
  different mechanism), gated on `--prefill-prune lazyllm` so we can
  A/B against SpecPrefill on the same workload.
- **Cost of adoption**: M (2-4 days). The per-layer attention scoring
  happens in the same kernel we already run — we just need to extract
  the top-K selection and thread an "active token" mask through the
  next layer. The revival-on-decode path is more delicate: decode step
  must check if the current token attends strongly enough to any
  pruned tokens to trigger a revive, which requires keeping the
  dropped KV addressable. Composes orthogonally with MInference's
  per-head sparse dispatch (Task 5) — LazyLLM reduces N (number of
  tokens), MInference reduces the sparsity shape per head. Both
  together bound prefill work from two directions. Risk: the revival
  path adds a branch to the decode hot path; in the worst case (heavy
  revival) the speedup turns into a slowdown. Gate the task on
  measuring decode regression vs baseline alongside the prefill
  speedup.
- **Local PDF**: research/2407.14057_lazyllm.pdf

### [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793) — 2405.15793
- **Authors**: John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, Ofir Press (Princeton)
- **Published**: 2024-05 (NeurIPS 2024)
- **Hypercar goals it addresses**: Goal 2 (intelligence breadth — realistic software engineering eval, complements τ-bench's customer-service agentic eval and LiveCodeBench's single-file coding)
- **TL;DR**: Introduces the concept of an *Agent-Computer Interface*
  (ACI) — a purpose-built tool API designed for LLM consumption (file
  viewer with line numbers, scoped editor with syntax check, search
  tool with deduplicated hits, etc.) — and shows that interface design
  dominates raw model capability on real software engineering tasks.
  On SWE-bench (2294 real GitHub issues from 12 Python repos), their
  minimal 7-tool ACI lifts GPT-4 from 1.6% resolved (naive bash
  interface) to 12.5% resolved, and Claude-3 Opus from 3.8% to 18.1%.
  The paper also releases SWE-bench Lite (300 issues filtered to the
  most tractable subset) as a cheaper eval tier.
- **Why it matters for Hypercar**: Our project actually ships a
  tool-using server (`hypercar_server.py`'s OpenCode integration) — and
  our Goal 2 current-status row explicitly lists "agentic tool use" as
  a gap. τ-bench (Task 14) covers *conversational* tool use against a
  simulated user, but a customer-service conversation is not the
  workload our users actually run: our users run code-repair-on-a-real-
  repository, which is exactly what SWE-bench measures. SWE-agent's
  ACI-driven evaluation is the realistic agentic-code gate that closes
  the Goal 2 claim, and its Lite variant (300 issues) is small enough
  to wrap as a bench phase. The harness is open source, is OpenAI-
  compatible, and runs tasks in isolated Docker containers — we'd need
  to adapt the container path for macOS (lima / colima or a subprocess
  sandbox) but the agent loop itself is pure Python. Slot: a new
  `omlx/eval/swe_agent/` module plus a `phase_swe()` in
  `hypercar_bench.py`, gating on 5 Lite issues in `--quick` and 30 in
  `--full`. The eval is slow (each task is many tool calls and a
  long-context exchange) so we'd sample heavily and cap wall clock.
- **Cost of adoption**: M (3-5 days). The agent loop is small; the
  tricky part is the containerised task runner because SWE-bench runs
  each issue's tests against the exact repo state at that commit, and
  that requires git + pytest reproducibility. On macOS we'd need lima/
  colima or a pinned Docker Desktop. Biggest risk: Qwen3-Coder-30B-A3B
  is smaller than the frontier models the paper evaluates (GPT-4,
  Claude-3), so the baseline pass-rate might be single-digit percent —
  which makes regression detection noisy. Mitigation: gate at
  `swe_lite_resolved >= 0.05` (well below frontier) purely as a
  "didn't break the tool loop" signal, not as a performance comparison.
- **Local PDF**: research/2405.15793_swe_agent.pdf

**Gap not closed this pass**: MoE-aware weight quantization beyond
ProMoE's layout-centric approach. Candidate papers in 2024-2026 (MoQE,
MC-MoE, EdgeMoE follow-ups) mostly target either (a) per-expert
calibration of a uniform quantization scheme, which is a subset of what
GPTQ and AWQ already do and does not change the decode-time memory
shape, or (b) routing-aware mixed precision, which is promising but
every recent paper couples the analysis to a specific MoE family
(Mixtral, DeepSeek-MoE) and does not straightforwardly transfer to
Qwen3's fine-grained 128-expert layout. The right next step for this
bucket is not another arxiv download but an *empirical* measurement
on Qwen3-Coder: run the existing 8-bit GPTQ weights through the ProMoE
activation-frequency profile (Task 32), sort experts by activation
count, and measure whether the bottom-50% (cold) experts could be
re-quantized to 4-bit with negligible output drift. That is a Task
32 follow-up, not a paper review.

## Pass 5 — 2026-04-13

Bucket coverage from uncovered territory: (1) rotation-invariant 4-bit
*weight* quantization distinct from KV quant (QuaRot), (2) cross-request
KV prefix reuse / semantic prefix caching (CacheBlend), (3) training-free
memory-augmented long-context extension distinct from attention sparsity
(InfLLM), (4) CPU-offloaded long-context KV for throughput-bounded decode
(ShadowKV). Prior passes covered attention sparsity, KV quant, speculative
decoding, prefill token-drop, evals, and MoE expert-weight offload — this
pass explicitly targets the *weight-quant* and *cache-reuse-across-requests*
axes which were not touched by any prior paper in this review. Gap-not-
closed note at the end of this section.

### [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) — 2404.00456
- **Authors**: Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman (ETH Zürich, EPFL, Microsoft Research, IST Austria)
- **Published**: 2024-04 (NeurIPS 2024)
- **Hypercar goals it addresses**: Goal 6 (48GB fit — weight bytes), Goal 5 (swap headroom), Goal 3 (decode — memory-bound cost)
- **TL;DR**: Applies a computational-invariant Hadamard rotation to both
  weights *and* activations end-to-end so that the rotated representation
  has no per-channel outliers. Because all outliers are smeared across the
  rotation basis, plain round-to-nearest 4-bit quantization (W4A4 KV4)
  becomes lossless-quality — no mixed precision, no activation-aware
  scaling, no calibration data beyond a few hundred samples. Reports
  <0.3 perplexity loss on LLaMA-2-70B at W4A4KV4 and end-to-end 2.16x
  prefill speedup on A100 because the whole matmul is genuine 4-bit.
- **Why it matters for Hypercar**: The entire TurboQuant codec in
  `omlx/turboquant_kv.py` is built on the *same* Walsh-Hadamard rotation
  trick that QuaRot uses — we already know the rotation works on Qwen3-
  Coder (the TQ3 KV cache ships in production). QuaRot is the paper that
  says "now apply this to the weights too." Our 32GB of 8-bit expert
  weights are Goal 6's biggest line item; a 4-bit re-pack via QuaRot's
  rotation on top of the existing `omlx/turboquant_kv.py` machinery would
  drop the model footprint from ~32GB → ~16GB and free the swap headroom
  that DuoAttention + KIVI can't reach on their own (because they only
  touch the KV, not the 32GB of weight bytes). Crucially, QuaRot's
  *online* Hadamard on activations happens inside the attention layer,
  where MLX already has `mx.fast.scaled_dot_product_attention` — the
  activation rotation is one contiguous matmul we can express in MLX
  without a custom Metal kernel. This is the cleanest path to a 4-bit
  Qwen3-Coder that keeps quality, because it reuses machinery we have
  already proven on the KV side. Note: BENCHMARKS.md lists "TQ3.5 weight
  quantization" as abandoned, but that was a *different* recipe that did
  not use a rotation — QuaRot-style weight quant on top of our existing
  WHT is fundamentally new territory, not a revival of the abandoned
  approach.
- **Cost of adoption**: M-L (1-2 weeks). New
  `omlx/turboquant_weights.py` module that holds the 4-bit rotated-weight
  packer (reuses the existing WHT rotation from
  `omlx/turboquant_kv.py`), a calibration script to pick per-layer
  scales, and an MLX `Linear` replacement that runs the online Hadamard
  before each matmul. Biggest risk: QuaRot's numbers are on dense
  LLaMA, not on fine-grained MoE. A 4-bit Qwen3-Coder expert that was
  never trained with rotation-aware quantization may regress on the
  coding evals even with Hadamard smoothing. Mitigation: gate the port
  against LiveCodeBench (Task 18) and RULER multi-key at 16K before
  shipping.
- **Local PDF**: research/2404.00456_quarot.pdf

### [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444) — 2405.16444
- **Authors**: Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, Junchen Jiang (University of Chicago, Stanford, Microsoft Research)
- **Published**: 2024-05 (EuroSys 2025)
- **Hypercar goals it addresses**: Goal 4 (prefill speed — cross-request reuse), Goal 1 (effective 1M context reuse)
- **TL;DR**: The problem: prefix caching only works when the whole prefix
  matches byte-for-byte. For RAG / agentic workloads where each request
  concatenates a *different* set of cached chunks (retrieved docs, tool
  outputs, prior turns), naive prefix caching fails because position
  embeddings and cross-chunk attention don't survive concatenation.
  CacheBlend caches each chunk's KV independently, then at request time
  *selectively recomputes* only the tokens whose attention would
  actually differ from a full prefill — typically a small fraction of
  each chunk. Reports 2.2x-3.3x TTFT reduction on RAG workloads with
  <1% quality loss vs full prefill, and most importantly the win
  *grows* with the number of distinct chunks being composed.
- **Why it matters for Hypercar**: Our `omlx/hypercar_server.py` has
  prompt caching today but it's strict-prefix only — it saves on the
  second decode of a repeat system prompt, but does nothing for the
  agentic common case where the system prompt is fixed and each tool
  call appends a *different* file or search result to a previously-
  cached context. That is 90% of the OpenCode workload and 100% of
  the reason Goal 4 (prefill constant across context) is still the
  biggest gap. CacheBlend turns every distinct tool output into a
  one-time KV compute, and every *re-use* of it into a selective
  patch — which for a session that includes the same repo files across
  many tool calls is dramatic. This composes with LLMLingua-2 (Task
  17): compress once, cache the compressed KV, reuse across requests.
  The TQ3 cache already has `save/load` primitives (the basis for
  session persistence), so storing and reloading per-chunk KV segments
  is mostly plumbing rather than new cache machinery.
- **Cost of adoption**: M (3-5 days). Requires: (a) a per-chunk KV
  store keyed on `hash(chunk_tokens)` inside
  `omlx/hypercar_server.py`'s caching layer, (b) a selective-
  recompute policy (the paper's "HKVD" — Highest Key-Value Divergence
  tokens), (c) a new prefill path that concatenates cached chunks and
  recomputes only the HKVD subset. Biggest risk: selective recompute
  correctness. Get the HKVD threshold wrong and downstream tokens
  diverge silently from a full prefill — we need a gate that compares
  CacheBlend-prefilled outputs to full-prefilled outputs on the same
  prompt and bounds the KL divergence. Mitigation: start with
  CacheBlend disabled by default, opt-in via a request header, and
  gate on HumanEval parity before flipping the default.
- **Local PDF**: research/2405.16444_cacheblend.pdf

### [InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory](https://arxiv.org/abs/2402.04617) — 2402.04617
- **Authors**: Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Song Han, Maosong Sun (Tsinghua, Renmin, MIT)
- **Published**: 2024-02 (NeurIPS 2024)
- **Hypercar goals it addresses**: Goal 1 (effective 1M context beyond trained length), Goal 3 (decode — constant attention cost), Goal 5 (swap headroom)
- **TL;DR**: Splits the context into three populations: a small fixed
  sink window at the start, a local sliding window at the end, and a
  large *memory bank* of past blocks in between. The memory bank lives
  in CPU or "far" memory; at each attention step a cheap
  representative-key score picks the top-K memory blocks most likely to
  be attended to, and only those are loaded back into the fast-tier
  cache. Because only a tiny fraction of the memory bank is touched per
  step, the effective context is essentially unbounded while the decode
  cost stays constant. Shows a 4K-trained Mistral extended to 1024K
  context with no fine-tuning and no quality loss on long-range
  benchmarks, and a 90%+ memory reduction on the hot tier vs holding
  the full KV resident.
- **Why it matters for Hypercar**: InfLLM is the cleanest known way to
  make Goal 5 (swap <8GB) comfortable at 1M context without touching
  decode quality. DuoAttention (pass 2) drops cache for *streaming
  heads* but keeps the full cache for retrieval heads; InfLLM keeps all
  heads but tiers the cache so only the hot blocks live in the Metal
  heap. On M4 Pro unified memory the "memory bank" doesn't even need to
  be on a separate device — it's the *same* memory, just a different
  allocator region, which means the "load a cold block back" cost that
  the paper measures in PCIe bandwidth is effectively free on our
  hardware. That makes InfLLM's economics *strictly better* on the M4
  Pro than on the A100 the paper evaluates. Slots into
  `omlx/turboquant_kv.py` as a cache-tier policy — hot blocks in the
  current 3-bit WHT format, cold blocks in the same format but in a
  separate mx.array region flagged for mmap eviction. This also
  composes with Quest (Task 2/3): Quest picks top-K pages within the
  hot tier, InfLLM decides which blocks *are* in the hot tier.
  Orthogonal mechanisms, multiplicative win.
- **Cost of adoption**: M (3-5 days). Per-block representative-key
  scoring (mean-of-group key, per the paper) in the cache writer path,
  a top-K block-selection pass at attention time, and a two-tier
  allocator in `omlx/turboquant_kv.py`. Biggest risk: Apple's unified-
  memory allocator may not actually free the "cold" tier back to the
  system if we keep a Python reference alive — we'd need to either
  explicitly `mx.clear_cache()` between block evictions or
  round-trip the cold tier through an `mmap`-backed file. The second
  option is robust but adds filesystem I/O to the decode path; the
  first option is cheap but depends on MLX allocator behaviour.
- **Local PDF**: research/2402.04617_infllm.pdf

### [ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference](https://arxiv.org/abs/2410.21465) — 2410.21465
- **Authors**: Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, Beidi Chen (CMU, ByteDance)
- **Published**: 2024-10 (ICLR 2025)
- **Hypercar goals it addresses**: Goal 5 (swap — explicit CPU offload), Goal 1 (effective 1M context in fixed memory), Goal 3 (decode throughput with offloaded cache)
- **TL;DR**: Observes that the K cache after RoPE is extremely
  low-rank (rank ~160 in Llama-3 at 128K context) and that the V cache
  has heavy locality — sibling tokens attend to overlapping V entries.
  ShadowKV keeps a low-rank factorization of K plus a compact landmark
  index in the fast tier, and evicts V blocks to CPU "shadow" memory.
  At decode time the landmark pass picks the top blocks and only those
  V blocks are staged back. Reports 6x larger batch and 3.04x higher
  throughput on A100 at 128K context with <1% quality loss on
  Needle-in-a-Haystack and LongBench.
- **Why it matters for Hypercar**: This is a different answer to the
  same "tier the KV" question that InfLLM addresses, but with a very
  specific mechanism (low-rank K + landmarks) that may compose better
  with our *existing* WHT-rotated 3-bit KV. The WHT rotation is
  specifically a decorrelating transform, which is the same reason K
  becomes low-rank after RoPE — so the representations are friendly to
  each other. Crucially for Hypercar's M4 Pro target, ShadowKV's
  "shadow" tier on A100 costs a PCIe round trip per block stage; on
  unified memory it is free. Goal 5 (swap <8GB) is the gap this paper
  most directly attacks: at 1M context our 22.5GB KV bill is almost
  entirely what's driving us into swap, and ShadowKV's landmark-based
  eviction is the clearest path to keeping only ~2GB of K indices +
  top-K V blocks in the hot tier. Slots into `omlx/turboquant_kv.py`
  as a per-head low-rank-K factorization plus a landmark scorer,
  layered under the existing 3-bit codec.
- **Cost of adoption**: L (1-2 weeks). The low-rank factorization of
  the RoPE'd K is an offline SVD per layer per head — that's a
  calibration script, not a runtime cost. The runtime additions are a
  landmark top-K pass and a V-block staging loop. Biggest risk: the
  SVD rank is input-dependent; the paper picks rank 160 on Llama-3 but
  we don't know the Qwen3-Coder number and the paper doesn't evaluate
  MoE. Mitigation: run the SVD-rank probe as a one-day experiment
  before committing. Overlap with InfLLM (Task 42) is significant —
  we should pick *one* of the two to implement first rather than both,
  and the choice depends on the SVD-rank measurement.
- **Local PDF**: research/2410.21465_shadowkv.pdf

**Gap not closed this pass**: semantic prefix caching *beyond* exact-
token match. CacheBlend addresses the cross-chunk composition problem,
but the deeper question — "can we reuse KV across prompts that differ
in wording but share meaning?" — has no credible 2024-2026 paper that
works without a full re-embedding pass. The candidate directions
(embedding-similarity KV retrieval, prefix-distillation caches) are
mostly papers that target batched serving systems where the reuse
economics are driven by concurrent-request sharing, which we do not
have in a single-user local setting. The honest next step for this
bucket is not another paper download but the CacheBlend integration
(new Task) — if exact-token composition turns out to be the entire
practical savings, the semantic-match question becomes academic.

**Gap not closed this pass**: rotation-invariant W4A4 quantization
*specifically validated on fine-grained MoE* (128-expert Qwen3 style).
QuaRot evaluates on dense LLaMA and Mistral; SpinQuant evaluates on
LLaMA and MoE-8x7B (Mixtral). Neither paper has data on a 128-expert
model where each expert is much smaller and the outlier distribution
is expert-conditional. The right next step for this bucket is not to
wait for a paper but to run QuaRot's recipe on Qwen3-Coder's experts
one layer at a time and measure perplexity drift — that is the Task
41 spike, not a literature question.

## Pass 6 — 2026-04-12

Bucket coverage from the brief: (1) Jacobi/Lookahead parallel decoding
(genuinely new territory — no draft model, no training), (2) grammar-
constrained / structured generation for tool-call reliability, (3) KV
cache eviction policies via observation-window importance (distinct
from the prior selection/sparsity papers), (4) paged attention without
PagedAttention's fragmentation cost — single-device serving economics
for 1M-token KV. None of these buckets has a paper anywhere in passes
1–5, including the "Already-cited prior art" list. This is the first
pass to explicitly attack decode-throughput via non-speculative parallel
decoding, and the first to attack *tool-call correctness* as an
intelligence lever rather than via preference tuning.

### [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 2402.02057
- **Authors**: Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang (UCSD, Google, UC Berkeley, MosaicML)
- **Published**: 2024-02 (ICML 2024)
- **Hypercar goals it addresses**: Goal 3 (decode speed, constant across context)
- **TL;DR**: Reframes autoregressive decoding as solving a nonlinear
  system by Jacobi iteration, then accelerates it with a two-branch
  "lookahead" that maintains an n-gram pool of historical Jacobi
  trajectories. At each step the model runs one forward pass that (a)
  advances the Jacobi window and (b) verifies n-gram guesses — both in
  the same batch — yielding 1.5–2.3x lossless decode speedup with *no
  draft model and no training*. Output distribution is identical to
  greedy decoding.
- **Why it matters for Hypercar**: EAGLE-2 (pass 2) is on the backlog
  for Goal 3, but pays a draft-head training cost we have not yet
  absorbed — and training a draft model against a 30B MoE target on
  M4 Pro is a multi-week expedition, not a spike. Lookahead is the
  "free" counterpart: zero training, zero extra parameters, composes
  with any KV cache format (so it layers on top of TQ3 without touching
  `omlx/turboquant_kv.py`), and it happens entirely inside the decode
  loop in `omlx/hypercar_server.py`. On MoE specifically the win is
  magnified because each verified n-gram token amortises one MoE
  expert-dispatch cost across multiple positions — the same reason
  EAGLE-2 benefits MoE, but without the training. The paper's own 1.5x
  floor at 13B-70B dense models is a conservative estimate for our
  workload since our decode is MoE-dispatch-bound, not FLOP-bound.
- **Cost of adoption**: M (2-3 days). A Jacobi-window rollout, an
  n-gram pool indexed by trailing-token key, a tree-mask primitive for
  the single forward pass that verifies the guesses, and a
  `--lookahead-window` flag on `hypercar_server`. The hardest part is
  getting the tree-attention mask right on MLX's scaled-dot-product
  attention path — this is the same mask shape EAGLE-2 uses and is the
  reason we would want to land Lookahead before EAGLE-2 (it de-risks
  the tree-attention primitive with a simpler consumer). Risk: n-gram
  hit rate on code generation is workload-dependent; the paper reports
  strong results on code but we'd want to validate on HumanEval-style
  generations before committing.
- **Local PDF**: research/2402.02057_lookahead_decoding.pdf

### [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469) — 2404.14469
- **Authors**: Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Wang, Fan Yang, Deming Chen, Minjia Zhang, Chao Zhang, Tri Dao (UIUC, Cohere, Princeton, Microsoft)
- **Published**: 2024-04 (NeurIPS 2024)
- **Hypercar goals it addresses**: Goal 5 (swap p90), Goal 1 (context), Goal 3 (decode indirectly)
- **TL;DR**: Observes that for each query, a small "observation window"
  at the end of the prompt already carries strong information about
  which historical KV positions matter for the rest of generation.
  SnapKV pools attention scores from that observation window across
  heads, picks the top-k historical positions per head, and *evicts*
  the rest before decoding even starts. Reports 3.6x decode memory
  reduction at 16K context with no quality loss on LongBench and
  NeedleInAHaystack across six models and up to 380K context.
- **Why it matters for Hypercar**: Every KV-reduction paper in passes
  1–5 either *selects* (Quest — still holds the full cache) or
  *tiers* (InfLLM, ShadowKV — still reads the cold tier on miss) or
  *compresses* (KIVI, QuaRot-ish). None of them *permanently drop*
  positions from the KV. SnapKV is the only paper in the review that
  reduces the *resident* KV memory by evicting before decode — which
  is exactly what moves Goal 5's swap p90 metric (currently 460 MB/s,
  target 100 MB/s). Eviction is also uniquely free at 1M context
  because the memory savings stack multiplicatively with every other
  KV optimisation we've banked: SnapKV then Quest then DuoAttention
  reduces resident KV, then bounds per-step work, then removes
  streaming-head KV entirely. For agentic workloads where the
  question is fixed at prompt time (tool calls, code review, long-
  document QA) the "observation window is the question" assumption
  holds by construction — SnapKV is nearly always correct for our
  workload. Plugs into `omlx/turboquant_kv.py` as a prefill-time
  `.compact(keep_indices)` primitive.
- **Cost of adoption**: S-M (1-2 days). The compaction primitive is a
  mx.take on the existing per-page layout; the per-head top-k is
  cheap. Biggest integration wrinkle: our TQ3 cache stores quantised
  pages, so compaction must re-pack (trivial — existing fork/rewind
  path already does this). Risk: for continuous conversations where
  later turns ask about *different* earlier passages, eviction is
  irreversible and can silently hurt multi-turn quality — we'd gate
  the feature on single-turn requests only, then extend later.
- **Local PDF**: research/2404.14469_snapkv.pdf

### [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100) — 2411.15100
- **Authors**: Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi Xu, Yilong Zhao, Tianqi Chen (CMU, SJTU, NVIDIA)
- **Published**: 2024-11
- **Hypercar goals it addresses**: Goal 2 (intelligence breadth — agentic tool-call correctness), Goal 3 (decode speed — zero runtime overhead for structured outputs)
- **TL;DR**: A context-free-grammar-constrained decoding engine that
  precomputes an "adaptive token mask cache" from the grammar so the
  per-token mask lookup at decode time is a constant-cost table read
  plus a small runtime pushdown-automaton update. Reports up to 100x
  speedup over Outlines/llguidance on JSON-mode decoding with zero
  measurable overhead in end-to-end inference on Llama-3 and Qwen.
  Crucially the engine handles context-sensitive grammars (not just
  regular languages) so it supports arbitrary tool-call JSON schemas.
- **Why it matters for Hypercar**: Our `omlx/hypercar_server.py`
  serves OpenAI-compatible tool calls to OpenCode, and "tool parse
  safety" is listed as a specific concern in the server description.
  Today tool-call correctness is entirely on the model — a single
  malformed JSON bracket breaks the entire agentic turn, which is a
  Goal 2 failure mode we don't currently measure. XGrammar makes the
  malformed-JSON failure mode *impossible by construction*, at zero
  runtime cost, which converts one of the fuzziest quality axes (tool
  use) into a hard guarantee. It composes with tau-bench (Task 14)
  multiplicatively: tau-bench measures whether the model picks the
  right tool; XGrammar guarantees that once picked, the call parses.
  Neither replaces the other. The zero-runtime-cost claim means it
  does not hurt Goal 3 the way runtime FSM masking (Outlines) does.
- **Cost of adoption**: S (1 day). Off-the-shelf PyPI package with a
  small C++ extension; the integration point is the sampler in the
  decode loop. On MLX we'd either port XGrammar's mask-application as
  an MLX op (fast path) or just materialise the mask on CPU per step
  (slow path — still likely faster than Outlines). Risk: XGrammar's
  precomputation step runs on the first request for each grammar; we
  need to cache those across requests in the server to avoid a
  per-call compile hit.
- **Local PDF**: research/2411.15100_xgrammar.pdf

### [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437) — 2405.04437
- **Authors**: Ramya Prabhu, Ajay Nayak, Jayashree Mohan, Ramachandran Ramjee, Ashish Panwar (Microsoft Research India, IISc)
- **Published**: 2024-05 (ASPLOS 2025)
- **Hypercar goals it addresses**: Goal 6 (48GB fit), Goal 5 (swap), Goal 1 (1M context without fragmentation)
- **TL;DR**: Shows that PagedAttention's block-indirection and
  scatter/gather kernels exist purely to work around CUDA allocator
  fragmentation — and that if you use the GPU's virtual memory
  primitives directly (cuMemAddressReserve, cuMemMap) you can keep
  the KV cache logically contiguous while still allocating physical
  memory on demand. The resulting kernels are the standard
  contiguous-KV kernels (no custom block-indirection code), with up
  to 1.97x throughput improvement over vLLM and strictly less memory
  fragmentation. The paper's CUDA implementation is an existence
  proof; the *idea* is platform-agnostic.
- **Why it matters for Hypercar**: Goal 6 status is "Load 32.4GB,
  peak 37.7GB — PASS" with ~10GB headroom, but at 1M context the KV
  itself is 22.5GB and must coexist with 16GB of 8-bit weights. MLX
  on Metal already has a unified-memory allocator that does not
  suffer from CUDA-style fragmentation, *but* our fork/rewind path in
  `omlx/turboquant_kv.py` currently allocates a fresh page tensor on
  every grow — a pattern that on Metal produces allocator churn and
  shows up in the swap-rate metric as spurious p90 spikes even when
  total memory is not pressured (see Task 31 in TASKS.md, "Profile
  KV allocator fragmentation at 1M context"). vAttention's answer is
  exactly what Task 31 is trying to diagnose: pre-reserve a large
  virtual address range, commit physical pages on demand. On Metal
  the equivalent primitive is MTLHeap with
  MTLHeapTypePlacement — same mechanism, different API. So
  vAttention is less "port this code" and more "the paper that gives
  Task 31 a known-correct architecture to converge on." This is the
  first paper in the review that speaks directly to the allocator-
  churn class of swap-rate bugs.
- **Cost of adoption**: M (2-4 days). An MTLHeap-backed page pool
  under `omlx/turboquant_kv.py`, plus a fork/rewind path that
  commits/releases pages instead of allocating fresh tensors. The
  numerical path is unchanged — this is purely an allocator swap.
  Risk: MTLHeap placement-mode semantics differ from CUDA virtual
  memory in one important way — heaps are not trivially growable, so
  we need to reserve an upper-bound heap (sized for 1M context KV)
  at session start. That is fine for our use case because the server
  knows context budget up front; it would be wrong for a shared
  multi-tenant server.
- **Local PDF**: research/2405.04437_vattention.pdf

**Gap not closed this pass: bucket 5 — Apple Silicon / MLX-native
attention kernels.** This remains the consistently-uncovered bucket
across every pass (called out explicitly in passes 2 and 5 as well).
The reason is structural: MLX kernel optimisation work lives in the
Apple MLX repo's commit log, WWDC talks, and a handful of engineering
blogs — not in 2024-2026 arxiv papers. There is no paper to cite
because the work is not a paper. The right next step for this bucket
is *not* another literature pass; it is Task 30 ("Survey
mx.fast.scaled_dot_product_attention source for AMX binding"), which
reads the MLX source directly. We should stop listing this bucket as
a research gap — it is an implementation gap, not a literature gap.

**Gap not closed this pass: bucket 6 — adaptive computation / early-
exit routers beyond LayerSkip.** LayerSkip (pass 4) covers the self-
speculative version of this idea. A handful of 2024 papers propose
learned routers that dynamically decide how many layers each token
needs (e.g., MoD-style mixture-of-depths). None of them land cleanly
on a 30B MoE with quantised weights because the router itself needs
training, and routers trained against dense FP16 bases rarely
transfer to quantised MoEs without re-training from scratch. The
paper-to-implementation path is not concrete for our stack — this is
the kind of gap the brief explicitly says should be documented rather
than filled with a filler task.

## Synthesis

**Highest leverage right now: Quest (2406.10774)**. Goal 3 (decode speed) is our
largest absolute gap — 20 tok/s vs 50 tok/s target, degrading with context.
Quest turns the decode cost from O(N) to O(K) via query-aware page selection,
which is exactly the "constant across context window" shape the Hypercar
contract demands. It composes with our existing 3-bit KV cache (no re-quant
work), it reuses our fork/rewind page abstraction in
`omlx/turboquant_kv.py`, and the math is simple enough to prototype in MLX in
a couple of days without a custom Metal kernel. A successful Quest integration
also makes every downstream decode optimization cheaper because the
attention term is no longer dominant.

**Second highest: RULER (2404.06654)**. It's the cheapest paper to adopt (half
a day), and it directly unblocks Goal 2 — we literally cannot claim "4
independent evals" until we have a long-context retrieval eval beyond
single-needle NIAH. Critically, RULER acts as a *safety net* for the Quest
and KIVI work: both of those changes risk silently degrading long-context
retrieval quality, and without RULER in the bench suite we'd ship regressions
invisibly. RULER should land first, then Quest.

MInference is the right next step for Goal 4 but only after Quest lands —
decode is the larger gap and will show up first in user-visible latency.
KIVI is a strong "free headroom" experiment and should be tried as a
1-day probe against the existing WHT-rotated codec. Text-to-LoRA is the
highest-ceiling bet but also the most expensive — park it until decode and
prefill are closer to target.

### Pass 2 adds (2026-04-13)

**Highest-leverage find this pass: DuoAttention (2410.10819).** It is the only
paper in either pass that improves *all of* decode, prefill, and swap headroom
with one mechanism, and it composes multiplicatively with the Quest +
MInference work already on the backlog: Quest reduces per-step attention work
on retrieval heads, MInference reduces per-step prefill work on retrieval
heads, and DuoAttention removes most of the heads from the cache entirely.
Once all three land, the attention-and-cache bill at 1M context is bounded by
(a) the streaming-window length on streaming heads (constant) and (b) the
top-K page count on the small retrieval-head subset (constant). That is the
"constant across context window" shape the Hypercar contract demands, on both
the latency and the memory axes.

Among the three eval/learning picks, **tau-bench (2406.12045)** is the
single missing piece for Goal 2 — RULER from pass 1 closes long-context
retrieval, tau-bench closes agentic tool use, and together with HumanEval and
NIAH that is finally four independent eval families. **SimPO (2405.14734)** is
a small immediate upgrade to `omlx/ttt.py` that lets the existing TTT loop
learn from contrast (winner vs loser trajectory) instead of only from passing
runs — that is the kind of cheap weekly improvement the brief asked for as a
Text-to-LoRA alternative. **EAGLE-2 (2406.16858)** is the highest-ceiling
Goal-3 lever beyond Quest, but pays a one-time draft-head training cost and
should be sequenced after Quest, MInference, and DuoAttention have all
landed so we know which workload regime we are optimising the draft for.

**Gap not closed this pass**: bucket 1 (Apple Silicon / MLX-native attention
papers). I deliberately did not download a thin paper just to fill the slot.
The Apple Silicon / Metal / AMX kernel literature on arxiv 2024-2026 is
sparse — most of the useful information lives in MLX repo issues, the
mlx-examples discussion threads, and Apple's MLX team blog posts, none of
which are arxiv-cited. The right next step for that bucket is an
*engineering* survey of the MLX matmul + scaled_dot_product_attention
implementations and how they bind to AMX, not another paper download. A
candidate followup question to track: whether MLX's
`mx.fast.scaled_dot_product_attention` already routes to AMX in the prefill
shapes we care about, and what the tile/block parameters are. That is a
read-the-source task, not a read-an-arxiv task.

### Pass 3 adds (2026-04-13)

**Highest-leverage find this pass: ProMoE (2410.22134).** Every prior paper
in this review optimises the *attention* side of the bill — Quest bounds
attention work, MInference bounds prefill attention shape, DuoAttention
shrinks the KV cache. ProMoE is the first paper we have looked at that
attacks the *weight* side, which for Qwen3-Coder-30B-A3B is ~32GB out of a
~37GB working set and is currently 100% resident even though only ~6% is
active per token. On Apple Silicon unified memory, the usual "PCIe copy is
expensive" counter-argument against expert offloading does not apply, so
the ProMoE policy space is strictly cheaper than on the discrete-GPU
baselines the paper evaluates. If we can turn even half the experts into
lazy-loaded weights, Goal 6 (48GB fit) and Goal 5 (swap <8GB under load)
both become comfortable instead of borderline. This composes *orthogonally*
with every pre-existing backlog item: Quest+DuoAttention still bound the
attention cost at decode, MInference+SpecPrefill still bound the prefill
attention cost, EAGLE-2 still bounds the number of forwards, and ProMoE
additionally bounds how many expert weights each of those forwards has to
touch. The rest of the stack gets cheaper the moment ProMoE lands because
every improvement they make is now applied to a smaller working set.

**Second highest: LiveCodeBench (2403.07974).** We currently claim "90% on
HumanEval" as one of our three evals for Goal 2, but HumanEval was
published in 2021 and Qwen3-Coder's training corpus almost certainly
contains it. That's a memorisation test, not an intelligence test. Pinning
LiveCodeBench to problems dated *after* Qwen3-Coder's known cutoff turns
our headline coding number into an honest one for free — half a day of
integration work for a strictly harder, strictly cleaner signal. It also
tightens the feedback loop on SimPO (Task 15): SimPO learns from
(winner, loser) trajectory pairs, and the cheapest source of such pairs is
the same hard problem solved twice — which is exactly what LiveCodeBench's
repair scenario gives us natively. **BigCodeBench (2406.15877)** is a
complementary library-usage eval: where LiveCodeBench tests algorithmic
reasoning, BigCodeBench tests the "generate Python that imports real
libraries and invokes them correctly" workload that OpenCode actually
submits to the server — so the two together cover both competitive and
production code generation.

**LLMLingua-2 (2403.12968)** is the pure orthogonal-axis win. Every other
performance paper in the backlog (Quest, MInference, DuoAttention) reduces
the *cost* of processing N tokens. LLMLingua-2 reduces N. At 3x compression
a 500K-token prefill drops to 170K — same kernel, same memory budget, same
KV. For the agentic workflow where most of the context is stale tool
output and repository noise, 3x is conservative. The only non-trivial
integration cost is making the quality gate honest: an aggressive
compressor can silently destroy multi-key retrieval, so the LLMLingua-2
task (Task 17) ships gated against RULER multi-key@16K, not just against
HumanEval.

Sequencing for the new tasks: LiveCodeBench (Task 18) lands first — it's
the cheapest and it protects every subsequent change by giving us an
honest coding signal. BigCodeBench (Task 19) follows in the same week for
the same reason. ProMoE lazy-load probe (Task 16) is the biggest-upside
item but also the highest-risk, and should be de-risked with a one-day
"can MLX partially load a module's weights at all" spike before the full
multi-day project is scheduled. LLMLingua-2 middleware (Task 17) is a
Goal-1 lever that can land in parallel with any of the above because it
touches the request path, not the model internals.

### Pass 4 adds (2026-04-13)

**Highest-leverage find this pass: LayerSkip (2404.16710).** Every prior
decode-speed paper in the review attacks one of two axes: Quest and
DuoAttention reduce the attention cost per layer, EAGLE-2 reduces the
number of tokens per forward via a separate draft model. LayerSkip attacks
a third axis none of them touch — reducing the number of *layers* per
forward via self-speculative decoding using the same model as its own
draft. This composes multiplicatively with Quest (attention bounded),
DuoAttention (cache bounded), and ProMoE (expert weights bounded): once
all four land, the decode cost per token is bounded on attention, cache,
weights, AND depth simultaneously, which is the only credible path to
the "constant across context window" half of Goal 3. Critically, unlike
EAGLE-2, LayerSkip needs no separate draft-model training run — the
calibration-only variant (per-layer exit-confidence threshold on a
short corpus) is a days-not-weeks integration and can run on the M4 Pro.
This makes LayerSkip a *cheaper* speculative-decoding lever than EAGLE-2
and probably the one we should try first; EAGLE-2 (Task 28/29) stays on
the backlog as a higher-ceiling follow-up.

**Second highest: MMLU-Pro (2406.01574) + LiveBench (2406.19314).**
These are a matched pair that finally close the Goal 2 "4 independent
evals" claim honestly. CLAUDE.md's status row for Goal 2 explicitly
calls out "MMLU-style reasoning" as missing; MMLU-Pro is the modern,
contamination-resistant, 10-option version of exactly that eval.
LiveBench is the complementary monthly-refreshed multi-category gate
that covers math + reasoning + data analysis — the non-coding axes that
LiveCodeBench (pass 3) doesn't touch. Together with LiveCodeBench and
τ-bench, they form a four-eval set where every single eval is either
post-training-cutoff (LiveCodeBench, LiveBench) or designed to resist
memorisation (MMLU-Pro's filtered pool, τ-bench's dynamic user
simulation). That is the first version of Goal 2 that can be defended
against a serious reviewer asking "what if the training data leaked?"

**LazyLLM (2407.14057)** is the orthogonal complement to MInference on
the prefill side: MInference dispatches per-head *shape*, LazyLLM drops
per-layer *tokens*, and both reduce prefill work without touching each
other's mechanism. For agentic workloads where stale tool output
dominates the context — the common case in OpenCode — LazyLLM's
training-free 43%-token-drop claim is the single highest-leverage
prefill optimisation that doesn't need a calibration step. The revival-
on-decode path is the non-trivial piece and the reason the task is
gated on decode regression alongside prefill speedup.

**SWE-agent (2405.15793)** is an agentic-eval addition rather than a
performance paper, but it closes a specific shape of the Goal 2 gap
that τ-bench does not: τ-bench's "agentic tool use" is simulated
customer service, while our actual users run "repair this real Python
repo." SWE-agent + SWE-bench Lite measures exactly that workload, on
issues from real GitHub commits. It stays gated at `resolved >= 0.05`
rather than a quality target because Qwen3-Coder-30B is smaller than
the frontier models the benchmark targets — but even at 5% it is a
robust *regression* gate for the tool-use path, which is the thing we
actually break when we ship changes to the server's chat-completions
endpoint.

Sequencing for Pass 4 tasks: MMLU-Pro (Task 36) and LiveBench (Task 37)
land first — both are half-day-to-day jobs and each one immediately
makes Goal 2 more honest. LayerSkip calibration (Task 38) is the
highest-impact decode lever and should land next because every
downstream optimisation benchmark gets more meaningful once we're
measuring a faster baseline. LazyLLM (Task 39) is a prefill lever that
composes with the MInference work from pass 1 (Tasks 4/5) and should
land after those to avoid double-counting prefill wins.

### Pass 5 adds (2026-04-13)

**Highest-leverage find this pass: QuaRot (2404.00456).** Every prior
pass has attacked the *KV cache* (Quest, DuoAttention, KIVI,
LLMLingua-2, LazyLLM, ShadowKV) or the *attention compute* (MInference,
EAGLE-2, LayerSkip) or the *expert dispatch* (ProMoE). None of them
touches the 32GB of 8-bit expert weights that dominates our Goal 6
budget and drives Goal 5 into swap the moment anything else needs
memory. QuaRot is the first paper in this review that credibly
4-bit-quantizes the weights themselves — and it does it with the
*exact* machinery we already have shipping in production. Our
`omlx/turboquant_kv.py` proves that a Walsh-Hadamard rotation is a
valid outlier-killer on Qwen3-Coder's activations at 3-bit KV; QuaRot
says "now extend that rotation to the weight matmul and you can pack
the weights to 4 bits too." The port cost is dominated by the
`Linear`-wrapper plumbing, not new research, because the rotation
itself is already in `omlx/turboquant_kv.py`. A successful QuaRot
integration drops the model footprint from ~32GB → ~16GB, which
turns Goal 5 (swap <8GB) from "borderline" to "comfortable" *and*
turns Goal 6 (48GB fit) from "passes on a clean boot" to "runs fine
with a full browser session alongside." This is the single largest
axis of unrealised wins left in the project, and it composes with
every existing backlog item — every optimisation ProMoE, Quest,
DuoAttention, and LayerSkip make is now applied to a weight set half
the size. The fact that it reuses the same WHT rotation we already
ship makes it the most de-risked "big win" on the entire backlog.

**Second highest: CacheBlend (2405.16444).** The project's Goal 4
(prefill constant across context) has two sub-gaps. LazyLLM and
MInference reduce the cost *inside* one prefill; CacheBlend reduces
the number of prefills you need to run at all, by reusing KV across
requests that share chunks but not whole prefixes. For OpenCode's
actual workload (repeat system prompt + tool calls that pull in
*different* repo files per turn) this is the difference between
"TTFT is 5 seconds every turn" and "TTFT is 5 seconds on the first
turn and <1 second on every subsequent turn that reuses the same
files." No prior paper in the review addresses cross-request KV
reuse — every optimisation so far has been within-request. This is a
pure orthogonal-axis win and slots into the existing prompt-cache
machinery in `omlx/hypercar_server.py`.

**InfLLM (2402.04617) and ShadowKV (2410.21465)** are a matched pair
both attacking Goal 5 via KV tiering — InfLLM with an activation-
frequency block policy, ShadowKV with low-rank K + landmark-selected
V blocks. On M4 Pro unified memory both papers' "cold tier" round-
trip cost is effectively zero, so their economics are strictly better
than on the A100 baselines they publish. We should pick exactly one
to land first, not both — the choice hinges on a one-day SVD-rank
probe (Task 44) to decide whether Qwen3-Coder's RoPE'd K cache is
low-rank enough for ShadowKV's landmark compression to beat InfLLM's
block-frequency heuristic. They are *not* additive: both tier the KV,
just with different policies. The gap-not-closed notes above explain
why we are not adding a semantic-prefix-caching task or a third
weight-quant paper — those buckets are better served by *measuring*
than by reading more papers.

Sequencing for Pass 5 tasks: QuaRot weight-quant probe (Task 41) is
the highest-impact bet and should land as a de-risking spike first
(one layer at a time, measure perplexity drift), then graduate to a
full port. CacheBlend (Task 42) is the lowest-risk and can run in
parallel — it touches the server request path, not the model, so it
cannot destabilise the decode loop. The InfLLM vs ShadowKV pick (Task
43/44) is gated on the SVD-rank probe and sequences after Quest and
DuoAttention have landed so we know which portion of the KV bill
actually survives to be tiered.

### Pass 6 adds (2026-04-12)

**Highest-leverage find this pass: SnapKV (2404.14469).** Every KV
paper on the backlog so far either *selects* positions at decode
time (Quest), *tiers* them hot/cold (InfLLM, ShadowKV), *compresses*
them per-element (KIVI, QuaRot-style), or *drops whole head classes*
(DuoAttention). None of them *permanently evicts* positions from the
resident KV cache. That is the single mechanism that directly moves
Goal 5's p90 swap-rate metric, because the bytes are simply no longer
resident. Crucially SnapKV composes multiplicatively with the entire
existing backlog: SnapKV shrinks the cache to the positions that
matter, then DuoAttention removes streaming-head KV from what's left,
then Quest bounds per-step work on the retrieval-head remainder,
then QuaRot halves the weight bill next to it. The observation-window
assumption holds by construction for our agentic workload (tool-call
prompts always end with "the question"), so we expect near-zero
quality cost on the single-turn path. This is the first paper in the
review that attacks Goal 5's sustained-swap metric directly rather
than via headroom, and it's a 1-2 day port against the existing
fork/rewind page layout.

**Second highest: Lookahead Decoding (2402.02057).** Pass 2 put
EAGLE-2 on the backlog as the big Goal 3 lever beyond Quest, but
EAGLE-2 pays a draft-model training cost that is a multi-week
expedition against a 30B MoE. Lookahead is the zero-training
counterpart: same tree-attention decode pattern, same 1.5–2x
lossless speedup floor, but no draft network exists. The practical
play is to *land Lookahead first* — it de-risks the tree-attention
primitive (Task 28) against a simpler consumer, exercises the same
mask machinery EAGLE-2 will later reuse, and banks a decode win
without blocking on a training run. Then if Lookahead's n-gram hit
rate on code generation turns out to be the gating factor, EAGLE-2
is the natural upgrade path on a pre-validated kernel. The
sequencing flip (Lookahead before EAGLE-2) is the most concrete
change to the backlog this pass produces.

**XGrammar (2411.15100)** is a pure orthogonal-axis quality win on
Goal 2. Every eval paper so far (tau-bench, LiveCodeBench,
BigCodeBench, MMLU-Pro, LiveBench, SWE-agent) *measures* how well
the model picks the right tool and formulates the right call. None
of them *guarantees* the call parses once picked. XGrammar converts
malformed-JSON failures into a structural impossibility at zero
runtime cost, which collapses one category of Goal 2 failure modes
entirely. It composes with tau-bench (Task 14) multiplicatively —
tau-bench now measures *only* the intelligent-choice axis, because
the parse axis is guaranteed. 1-day integration, smallest risk on
the whole pass.

**vAttention (2405.04437)** is the architecture paper behind Task 31
(allocator fragmentation at 1M context). It is *not* a port — CUDA
virtual memory primitives do not translate to Metal — but it is the
canonical "correct design" for the class of allocator-churn swap
spikes we see at long context. Pairs with Task 31 as the "read this
first, then write the MTLHeap-backed page pool" reference.

Sequencing for Pass 6 tasks: XGrammar (smallest, safest — land
first), SnapKV (single biggest Goal 5 move — land second), Lookahead
(reshapes Task 28 sequencing — land third, before the existing
EAGLE-2 work), vAttention-style MTLHeap allocator (upgrades Task 31
from diagnostic to designed fix — land fourth, gated on measuring
that Task 31's diagnosis matches the paper's premise).
