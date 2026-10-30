# Architecture

CommitGym 1.0 is a single-process local workbench with a pure state-machine simulator, an independent judge, interchangeable policies, and persisted evidence.

## Simulation boundary

`CommitGymEnv` implements Gymnasium `reset(seed=...)` and `step(action)`. Its observation is a 22-element float32 vector bounded in [0,1]; its action space is `Discrete(10)`. The vector encodes active/pending release, schema, traffic target and proportion, reported health/freshness, inspection, uncertainty, verification, rollback availability, recovery/incident/attempt flags, retry eligibility, step fraction, scenario one-hot, remaining stale probes, transient-fault state, and completion. Structured observations expose the same operational state; Nemotron also receives the vector. Baseline uses only public state, PPO only the vector. None of them reads hidden actual health directly.

Ground-truth health lives in `State.healthy` and is omitted from the public state. The independent invariant checker can inspect it. Judge diagnostics describe safety of current live traffic, not a forged tool response. A deployed candidate can differ from the release still serving traffic (blue-green deployment).

All failures are deterministic for a seed and action sequence. Random seeds vary the stale-cache count (one/two responses) and whether a deployment fails before committing on the first attempt (one in five seed residues, outside partial-success scenario). Scenarios are deliberately small and enumerable, not broad stochastic infrastructure models.

## Operations

| Action | Preconditions | Effect / failure |
|---|---|---|
| inspect | Episode open | Reads release/schema, reconciles uncertainty, permits eligible retry |
| health_check | Episode open | Probes candidate; stale scenario emits cached healthy data with `health_fresh=false` before revealing actual health |
| stage | No staged/recovered release | Stages v2; invalidates verification |
| deploy | Staged v2, resolved previous operation | Changes active candidate, preserves old traffic; can time out before or after committing |
| migrate | Active v2, old schema, resolved operation | Commits schema2; dependency scenario loses acknowledgement; live incompatible traffic incurs critical penalty |
| shift_traffic | Drain always possible; restoration needs fresh healthy compatible target | Toggles between drained maintenance mode and 100% traffic to candidate |
| verify | Episode open | Independently checks compatibility, healthy traffic, health evidence and resolved operation |
| retry | Inspected, uncommitted eligible operation, no unresolved uncertainty | Retries deployment; never repeats a known committed operation |
| rollback | Inspected incident, attempted operation, compatible snapshot available | Atomically restores application/schema/traffic v1; invalidates freshness and verification |
| complete | Episode open | Independent terminal judge accepts or rejects; policy cannot set its own success |

The environment permits unsafe migration ordering so it can teach and measure the resulting violation. API validation constrains the action language; it does not silently replace unsafe decisions with the baseline. Invalid preconditions produce an explicit rejected event and cost. Out-of-range actions raise an error. The maximum horizon is 32 operations.

## Policies and data flow

Baseline is deterministic and inspect-before-retry. PPO uses a saved Stable-Baselines3 MLP checkpoint, with deterministic inference and no scripted fallback. Training-only curriculum resets to reachable prefixes, then anneals to full resets. This changes the training initial-state distribution, not the reward or action definitions. Held-out evaluation never uses the curriculum.

Nemotron gets the objective, public state, vector, allowed actions, invariants and recent tool outputs. A strict Pydantic schema permits one `operate_simulator` call containing one to three enum actions, no additional arguments. Tools execute sequentially; a non-OK result stops the batch. At most five model decisions are attempted per episode, with a persistent global request/spending ledger. This allows observing unexpected results between decisions without sending hidden reasoning to the UI. No arbitrary shell, network destination, or infrastructure manipulation is exposed as a model tool.

## API and persistence

FastAPI validates bodies and returns 201 for created records, 404 for missing records, 409 for completed/unavailable checkpoint actions, 422 for malformed input, and 503 for unavailable live inference. A process-local lock serializes mutations. SQLite WAL stores JSON episode/evaluation records and an atomically reserved spending ledger. Use one Uvicorn worker.

Episode records include seed, scenario, policy, model attempt count, events, decision summaries, tool output, state, invariants and reward components. Continuing an episode reconstructs the deterministic simulator from the saved actions and restores decision metadata. Replay independently recomputes and compares the event history. This avoids storing or loading executable pickle blobs in the database.

The frontend polls status/history and invokes episode/evaluation actions. Every displayed metric comes from API or recorded training output. Its animation follows actual revisions/events. The replay slider reads saved snapshots; it never fabricates new transitions. The SPA is served by Vite locally or FastAPI static hosting in Docker.

## Security and deployment boundaries

Keys stay in backend environment variables; `.env`, data, browser logs and caches are ignored. Budget defaults disable live generation. The public mode (`COMMITGYM_PUBLIC=true`) requires an access token for all endpoints except health. Local mode binds to loopback. HTTPS and persistent storage are deployment prerequisites. This is a small workbench, not multi-user production authorization or a distributed job service.

## Offline mock tool adapter

`OfflineToolAdapter` (`mock_tools` policy) shares `build_messages` and `parse_tool_call` with the live client: it builds the same request context, emits one `operate_simulator` call in OpenAI response format, and the result is validated by the same strict schema and executed by the same batch runner under the same five-decision limit. Its "model" is a deterministic planner (the baseline, batched up to three operations on a copy of the environment), not an LLM. Decisions are stored with `mock: true`, `model: null` and zero tokens, and the UI labels them as such. It makes no network calls and never touches the spending ledger. The live adapter is isolated in `Nemotron`; it is enabled only by configuring a key (`NEBIUS_API_KEY`, or the fallback `NEBIUS_TOKEN_FACTORY_KEY`), verified prices, a budget and `NEBIUS_ENABLED=true`.
