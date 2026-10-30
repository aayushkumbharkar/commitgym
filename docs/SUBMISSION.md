# Submission draft — Best Apps and Agents

**Title:** CommitGym

**Tagline:** Train agents to recover safely, not just execute successfully.

## What it does

CommitGym is a reinforcement-learning environment and interactive workbench for agents operating multi-step software transactions. It simulates ambiguous deployment commits, stale health responses and partially acknowledged schema migrations. An action returning success is not enough: an independent judge checks traffic compatibility, service health, fresh verification and transaction resolution.

The workbench compares a deterministic baseline, a genuinely trained PPO policy, and a budgeted NVIDIA Nemotron tool-calling integration. All policies use the same simulator and scoring rules. Users can step through decisions, inspect tool results and reward terms, reopen persisted episodes, and deterministically verify replays.

## What was built and measured

A Gymnasium environment, FastAPI/SQLite backend, React/TypeScript dashboard, real local PPO training and held-out evaluation, and automated backend/browser tests. On 120 final held-out episodes per policy, baseline and PPO each achieved 100% verified safe completion with zero critical violations. Baseline completed 80 upgrades and 40 rollbacks; PPO completed 120 rollbacks. This exposes the recovery/deployment tradeoff rather than declaring an artificial winner.

The initial PPO experiment failed; a reachable-state curriculum improved exploration. Training and evaluation evidence is retained. The scenarios remain synthetic and narrowly varied; production generalization is not claimed.

## NVIDIA and Nebius integration — current truthful status

The integration targets `nvidia/Nemotron-3_5-Lightning` through the OpenAI-compatible Nebius Token Factory API. It verifies account model availability, validates strict allowlisted tool calls, executes actual simulator transitions, provides tool evidence back to the model, and limits decisions, tokens, time and estimated spending. API keys remain server-side. PPO never calls the LLM during training.

**Live account authentication and actual NVIDIA inference have not been verified.** The owner paused Nebius setup on 2026-10-30. Until a key is configured, the app offers an explicitly labelled *offline mock tool adapter* (not an LLM, zero tokens) that runs the same tool-call validation and batch boundary. It must not be described as Nemotron in the submission. Do not submit text implying a completed live integration until a real probe and tool-using episode have succeeded. Replace this paragraph with the actual selected model ID, request count, token usage and episode outcome only after verification.

## Links and assets

- Repository: local Git repository with MIT license; no public URL yet.
- Working test build: local frontend/backend and a verified Docker image; `render.yaml` prepared for Render, not yet deployed (see `DEPLOYMENT.md`).
- Screenshots: genuine local browser captures in `docs/screenshots/`.
- Video: see `DEMO_SCRIPT.md`; local recording is supplementary, not proof of live inference.

## Token Factory feedback based on experience

The unauthenticated console opened, but its login flow presented a security-verification page with HTTP 403 in the automated browser. Redemption and inference could not be tested. No feedback about model quality, latency, pricing accuracy or credit redemption is claimed from that limited experience.
