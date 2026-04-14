# Benchmark Snapshots

Each `run*/` directory contains one benchmark run's artifacts.

## Directory Format

```
bench/snapshots/
├── run23_2026-04-13T00-23/
│   ├── results.json     # Phase results + gate details
│   ├── profile.json     # Per-sample memory/CPU/swap profiler data
│   ├── env.json         # System environment at run time
│   ├── console.txt      # Full console log
│   └── git.txt          # Git SHA + diff at run time
├── run44_duo_full_all_gates/
│   ├── results.json     # Milestone: first --full ALL GATES PASS with duo mode
│   └── profile.json
├── aggregate/
│   └── <sha>.json       # Per-SHA statistical aggregates
└── README.md
```

## Key Runs

| Run | SHA | Mode | Significance |
|-----|-----|------|-------------|
| 23 | 11fe743 | native | First run in session — baseline (decode 21 tok/s) |
| 32 | c71dddd | native | First run with profiler rebuild + headroom gate |
| 41 | 0e9a6bf | native | First --full ALL GATES PASS (native, loosened swap) |
| 42 | 1b7b2d5 | tq3 | First TQ3 ALL GATES PASS (RULER 100% including VT@16K) |
| 43 | 1bc3f4e | duo | First duo ALL GATES PASS (MMLU-Pro 64%, swap 0) |
| 44 | 7448e63 | duo --full | First duo --full ALL GATES PASS (HumanEval 95%) |
| 45 | d72bafe | duo | Steady-state duo (prefill 501@16K, decode 52 tok/s) |

## Aggregate Analysis

```bash
# Build aggregates from all snapshots
.venv/bin/python omlx/bench/aggregate.py

# Report for latest SHA
.venv/bin/python omlx/bench/aggregate.py --report HEAD

# Compare two SHAs
.venv/bin/python omlx/bench/aggregate.py --report HEAD --baseline 11fe743
```

## Run Numbering Convention

Two agents operate on this repository: the **analyst cron** (periodic benchmark runs)
and the **implementation loop** (devloop, autonomous task execution with validation runs).
To avoid namespace collisions in commit messages and `git log`:

- **Analyst runs**: Use `Run N` in commit messages (e.g., `bench: Run 47 — duo ALL GATES PASS`).
  These are the canonical benchmark records tracked in BENCHMARKS.md.
- **Devloop runs**: Use `devloop-N` or `sample-N` in commit messages (e.g.,
  `bench: devloop-3 — NIAH 64K PASS`). These are internal validation runs
  during implementation, not tracked in BENCHMARKS.md.

This avoids confusion when searching `git log --oneline | grep -i run` — analyst
runs and devloop runs are visually distinct.

## Session Progress (Run 23 → Run 45)

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| Decode tok/s | 21.3 | 52.3 | +146% |
| Prefill tok/s (4K) | 571 | 803 | +41% |
| NIAH time | 208s | 51s | -75% |
| Total runtime | 850s | 431s | -49% |
| Swap (default) | 5.3 GB | 0.0 GB | -100% |
| HumanEval | never ran | 95% | NEW |
| MMLU-Pro | never ran | 64% | NEW |
| RULER | never ran | 100% | NEW |
