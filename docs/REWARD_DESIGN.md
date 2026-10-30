# Reward design and independent correctness

At each operation:

`reward = step + invalid + retry + unsafe + progress + terminal`

| Component | Value |
|---|---:|
| Every step | -1 |
| Rejected precondition | -5 additional |
| Eligible retry | -3 additional |
| Each violated critical invariant on this step | -30 |
| Each first-time milestone | +2 |
| Independently verified safe terminal completion | +100 |
| Premature/unsafe completion or exhausted horizon | -60 |

One-time milestones: staged, deployed, fresh evidence, migrated while drained, verified, recovered from an incident. Milestone rewards total at most 12. Repeating health checks, verification or draining cannot farm them. Rollback cannot happen before an attempted transaction and an inspected incident; after rollback the snapshot is consumed. Merely stopping the initial healthy system fails the objective. A maximum-horizon exit is always a failure, even if it happens to end in an otherwise safe state.

Critical invariants check that active traffic matches the schema and does not route to an unhealthy candidate. Terminal checks additionally require fresh evidence, resolved uncertainty, independent verification, full restored traffic to the active release, an attempted transaction, and either v2/schema2 completion or an actual incident recovered to v1/schema1. Failed completion is final and cannot be relabeled by the policy.

Every event stores all six reward terms. Tests check reward sums, repeated-action rewards, unsafe migration penalties, premature completion, rollback reverification and baseline safety across seeds.

## Observed learning failure and response

An initial 100,352-step vanilla PPO run converged to four-action early failure, mean held-out reward -58, 0/60 safe completion. Sparse positive terminal feedback and discounted future action costs made this a local optimum during exploration. We did not change or counterfeit the test results.

We added a training-only curriculum: reset to reachable baseline trajectory prefixes, with full-start probability rising from 0.1 to 1.0 by half the training budget. No baseline actions serve as supervised targets or inference fallbacks. The scoring formula and evaluation reset stay unchanged. A fresh 300,032-step PPO run achieved 120/120 verified safe completions on the final unseen seed set.

## Important objective limitation

The current task values verified recovery as highly as completing an upgrade. PPO learned to roll back every incident, including recoverable ambiguous commits. It gets 102.07 mean reward versus baseline 98.50 because it uses fewer actions and can earn a recovery milestone. Both are valid safe terminal outcomes under this explicit objective; this is not evidence that PPO is the better deployment policy.

We report upgrade completion and rollback completion separately. The next version should distinguish recoverable observations from irrecoverable incidents and price downtime/abandoned upgrades, then retrain and evaluate on a newly reserved test distribution. That is a future objective change, not a post-hoc alteration of this run's scores.
