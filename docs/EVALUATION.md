# Evaluation evidence

## Final protocol

- Final PPO checkpoint: `models/ppo_commitgym.zip` (actual SB3 serialization).
- Training seed: 7. Timesteps: 300,032; 54,307 training episodes; CPU elapsed 256.94 seconds.
- Training resets generate seeds below 1,000,000. Initial validation used 1,000,000–1,000,019.
- Final held-out seeds: 2,000,000–2,000,039, crossed with all three failure scenarios.
- 120 complete simulator episodes per policy; baseline and PPO share initial states, action space, horizon, judge and reward.
- No model requests were made for training, local evaluation or automated tests.
- Final measurements are in `results/evaluation.json`; actual learning curve is `results/training.json`.
- Failed initial training/validation evidence is retained in `results/initial-*.json`.

| Measure | Baseline | PPO |
|---|---:|---:|
| Safe completion | 100% (120/120) | 100% (120/120) |
| Critical invariant violations | 0 | 0 |
| Unsafe terminal states | 0 | 0 |
| Unresolved episodes | 0 | 0 |
| Average reward | 98.50 | 102.0667 |
| Average actions | 11.1 | 8.2 |
| Retries | 16 | 0 |
| Rollbacks | 40 | 120 |
| Recovery success | 100% (120 incidents) | 100% (120 incidents) |
| Failure rate | 0% | 0% |
| Simulation steps | 1,332 | 984 |
| Upgrade completion | 80/120 | 0/120 |
| Verified rollback completion | 40/120 | 120/120 |
| LLM requests / tokens | 0 / 0 | 0 / 0 |

PPO matches baseline safety but abandons more upgrades. The higher reward rewards efficient safe recovery under the current objective, not superior upgrade completion. The initial vanilla PPO result was 0/60, mean reward -58. Training curves include curriculum-start episodes until halfway; do not compare early curve averages directly with full-episode held-out returns.

## Reproduction check (2026-10-30)

The held-out evaluation was re-run from the saved checkpoint with `evaluate(seeds=range(2000000, 2000040))`. All aggregate metrics and all 240 per-episode rows matched the saved file exactly (only wall-clock `elapsed_seconds` differs). The saved results now also include `unresolved_episodes` (0 for both policies). The PPO checkpoint loads with a `Box(0,1,(22,))` observation space and `Discrete(10)` action space matching the environment.

## Scope of generalization

Seeds deterministically vary only a few fault parameters. Forty held-out seeds do not represent forty independent infrastructure types. The test measures unseen seeds within these finite scenarios; it does not establish transfer to novel failures, production systems, unmodeled delays, or unrecoverable data loss. There is no claim of statistical superiority from this sample.

## Nemotron status and fairness

No live model evaluation has been performed. Missing credentials or budget limits are not replaced with synthetic model output. The API supports explicit shared-seed comparisons including Nemotron after verification. Its five model-decision limit and three-actions-per-tool batch limit are recorded operational constraints; simulation actions still have identical costs. Invalid or unavailable model responses remain unresolved failures. Tokens, request counts, latency and local estimated cost are separate from simulator reward.

API-created comparisons retain episode IDs and can be reopened in History. CLI reports contain per-episode metrics and can be rerun with the saved policy. A repeat of the same seeds is reproducibility, not a new independent held-out experiment.
