# KV Allocator Fragmentation Profile (Task 31)

**Date**: 2026-04-15
**Model**: mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit
**MLX version**: 0.31.1

## Results

| Context | Active (GB) | Peak (GB) | Frag (GB) | Frag % | KV est (GB) |
|--------:|------------:|----------:|----------:|-------:|------------:|
| 2K | 32.67 | 33.40 | 0.73 | 2.2% | 0.23 |
| 8K | 33.27 | 34.00 | 0.73 | 2.2% | 0.83 |
| 32K | 35.69 | 36.34 | 0.65 | 1.8% | 3.25 |

## Extrapolation to 1M Context

- Average fragmentation: **2.1%**
- At 1M context (22.5 GB KV): ~0.5 GB fragmentation

## Verdict

**VERDICT: Fragmentation is negligible (<5%).** The MLX Metal allocator is efficient at these sizes. Paging/compaction work (Tasks 43, 64) is NOT justified for memory savings alone — pursue only if the tiering architecture provides other benefits (e.g., CPU offload).
