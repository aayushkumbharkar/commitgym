# Devpost form answers: CommitGym

Paste-ready answers. Fields marked **CONFIRM** need the submitter's personal confirmation. Everything below is based on verified evidence as of 2026-10-30 22:10 IST.

---

**Project name:** CommitGym

**Elevator pitch:** An RL environment where AI agents manage risky software releases and are judged on verified system state, not just tool-call success.

**Track:** Best Apps and Agents

**Submitter country:** India

**New or existing project:** New. The first commit is dated 2026-10-30 14:01 IST, after the 2026-08-26 cutoff. **CONFIRM** that no earlier work exists outside this repository.

**Individual or team:** Individual. The repository has a single author (`aayushkumbharkar`) and no documented teammates. **CONFIRM.**

**Organization:** N/A. **CONFIRM** that you are not submitting on behalf of an organization.

**Canadian province:** N/A

**IRL event city:** None. **CONFIRM:** select one only if you attended an eligible event in person.

**Tavily used:** No. The code makes no runtime Tavily API calls.

---

## About the project (story)

### Inspiration
Agents that run real operations fail in a specific way: a tool returns "success" (or a timeout) and the agent believes it. In release engineering, a deploy can time out after it actually committed, a health endpoint can return a cached "healthy", and a migration can write the schema but lose the acknowledgement. I wanted an environment where an agent is graded on what the system actually looks like afterwards, not on what its tools reported.

### What CommitGym does
CommitGym simulates a blue-green software release: a web frontend, an API service and a database, with releases v1 and v2. An agent has ten operations: `inspect`, `stage`, `deploy`, `health_check`, `shift_traffic`, `migrate`, `verify`, `retry`, `rollback` and `complete`. It must take a release to verified v2 or recover safely to v1. Seeded failure scenarios inject an ambiguous deploy timeout, stale health responses and a partially acknowledged schema migration. An independent judge decides whether the final state is safe; the agent cannot declare its own success.

The web workbench lets you run episodes step by step, watch the topology change, inspect each tool result and reward term, replay any saved episode deterministically, and compare policies on held-out seeds.

### How it works
- **Environment:** a Gymnasium environment (`CommitGymEnv`) with a 22-value observation vector and a `Discrete(10)` action space. Hidden "true" state is kept separate from what tools report.
- **Judge:** independent invariants check traffic/schema compatibility, a healthy traffic target, fresh health evidence, resolved operation outcomes and independent verification. `complete` is accepted only if they all hold.
- **Backend:** FastAPI and Pydantic, with SQLite persistence of episodes, events and evaluations. Replay recomputes an episode from its seed and actions and checks for an exact match.
- **Frontend:** React 19, TypeScript and Vite, with an SVG topology, event timeline, reward inspector and evaluation charts.

### Reinforcement learning and evaluation
I trained PPO (Stable-Baselines3) on CPU for 300,032 timesteps. My first run without a curriculum failed all 60 validation episodes. A curriculum that starts training from reachable intermediate states fixed exploration. All final evaluation starts from the normal initial state.

The held-out test used 40 unseen seeds × 3 scenarios, 120 episodes per policy. I re-ran it from the saved checkpoint and all 240 per-episode results matched exactly.

| | Baseline | PPO |
|---|---:|---:|
| Safe outcomes | 120 | 120 |
| Deployed v2 | 80 | **0** |
| Safe rollback | 40 | 120 |
| Unsafe or unresolved | 0 | 0 |
| Critical violations | 0 | 0 |
| Average actions | 11.1 | 8.2 |
| Average reward | 98.5 | 102.1 |

PPO earned the higher reward with fewer actions, but it never deployed the new release: it learned that an immediate verified rollback is the cheapest safe outcome. That is a reward-design finding, not a better deployment policy. CommitGym makes that trade-off visible instead of hiding it inside a single success rate, and the UI labels "Deployed v2 · verified" and "Safe rollback · v1 restored" differently everywhere.

### Safety and invariant verification
Every operation passes through strict validation. Rejected operations are recorded, never silently executed. Rewards are logged per component: step, invalid, retry, unsafe, progress and terminal. Model output is untrusted: an LLM can only emit one allowlisted `operate_simulator` tool call with 1–3 enum actions, and the batch stops at the first unexpected result.

### Live LLM agent: NVIDIA Nemotron via OpenRouter
NVIDIA Nemotron 3.5 Lightning (`nvidia/nemotron-3.5-lightning:free`) was integrated through OpenRouter and tested live:
- The model received the real simulator state and produced tool calls. These went through validation, executed in the simulator, and the model chose its next actions from the updated state. Up to 5 decisions per episode.
- In 4 live episodes with the final configuration, it handled real failures correctly: it inspected after a deploy timeout, and it reconciled a partially acknowledged migration. In an earlier test episode it chose a verified rollback.
- **None of those episodes reached a safe completion.** Twice it called `complete` while traffic was still drained, and the independent judge rejected it. Twice it ran out of its 5-decision budget.

That is exactly the failure mode CommitGym is built to catch. OpenRouter, not Nebius, was the inference provider.

### Technical architecture
React workspace → FastAPI → episode runner → shared `CommitGymEnv` and independent judge → SQLite. Policies are a deterministic baseline, the PPO checkpoint, live Nemotron, and a separate, clearly labelled offline mock tool adapter. The mock is a local planner that is not an LLM and uses zero tokens. One Docker image serves the API and the built frontend; a Render Blueprint is included.

### Challenges and lessons learned
- PPO with no curriculum learned nothing useful (0/60).
- Reward shaping rewarded rollback over deployment, and a single "success rate" hid this until I split the outcomes.
- With Nemotron, hidden reasoning used up the output-token cap before the tool call, so I disabled reasoning for tool calls. A tool-call response is not the same as a correct plan: the judge caught premature completions.

### Limitations and future work
- The environment is a finite synthetic state machine with narrow seed variation. These results are not evidence of production generalization.
- PPO's objective needs a deployment incentive or a downtime cost.
- The LLM agent needs better planning within its decision budget.
- Next: more fault families, a downtime penalty, and Nemotron evaluated across many seeds.

### Nebius disclosure
Nebius Token Factory and Nebius AI Cloud were not used. Standard onboarding required a payment card, and the organisers confirmed that no card-free participant access or sponsored account was available. The project therefore does not claim Nebius platform integration.

---

## Built with
Python, FastAPI, Pydantic, Uvicorn, Gymnasium, Stable-Baselines3, PyTorch, NumPy, Reinforcement Learning, PPO, SQLite, React, TypeScript, Vite, Tailwind CSS, GSAP, Playwright, Docker, Render, OpenAI Python SDK, OpenRouter, NVIDIA Nemotron

## Links
- **Repository:** https://github.com/aayushkumbharkar/commitgym (public; verified HTTP 200 without login)
- **Demo URL:** **BLOCKED.** Not deployed. Render needs your login: Dashboard → New → Blueprint → this repo. Leave this field empty rather than invent a URL.
- **Video:** **BLOCKED.** Upload `docs/demo/commitgym-local-demo.webm` (66 s, real app, captioned, no narration) to YouTube as unlisted, then paste the link. A narrated version was not produced before the deadline.

## Model and platform questionnaire

**Which models did you use and why?** NVIDIA Nemotron 3.5 Lightning (`nvidia/nemotron-3.5-lightning:free`) via OpenRouter, as a tool-calling agent. It supports forced tool calls and was available free without a card. I also used a PPO policy I trained myself; that is an RL policy, not an LLM.

**Nemotron output quality:** Tool-call formatting was reliable once reasoning was disabled: every live request in the final configuration returned a valid allowlisted tool call. With reasoning enabled, two requests were cut off before reaching the tool call. Planning quality was mixed. It handled timeouts and partial failures sensibly, but it completed prematurely with traffic drained, and 0 of 4 live episodes reached a safe completion. This is a small sample.

**Prompt engineering or out of the box?** Prompt engineering. A system prompt describes the actions and an operator runbook, and the context includes state, invariants, recent tool results and remaining calls. A strict JSON-schema tool and server-side validation sit around it.

**Comparison with other models:** Not done. No other LLMs were evaluated. The baseline-vs-PPO comparison is an RL comparison, not an LLM comparison.

**Valuable Nebius platform capabilities:** Not evaluated; I could not access the platform (see disclosure).

**Recommendation rating for running Nemotron on Nebius:** N/A, not evaluated. *If the form forces a number, note "Not evaluated: no platform access" in the explanation and tell the organisers the selection is not a rating.*

**Experience running Nemotron on Nebius:** I did not run Nemotron on Nebius. Token Factory onboarding required a payment card, and no card-free participant access was available.

**Requested Nebius improvements:**
- Card-free sandbox access for hackathon participants.
- Pre-provisioned individual API keys.
- Hard spending caps.
- Promotional credits usable without adding a personal card.
- Clear usage controls and budget alerts.

**What should the Nemotron team provide next?** A documented way to bound or disable reasoning for strict tool-call responses (hidden reasoning exhausted my output cap), and published tool-calling reliability guidance for multi-step agent loops.

## Image gallery checklist
The images are real Playwright captures in `docs/screenshots/`, all under 5 MB, with no credentials or private paths:
- `lab-1440.png` (Environment Lab: topology, invariants, timeline, inspector)
- `evaluations.png` (baseline vs PPO with Deployed / Rollback columns)
- `environment-lab.png` (deploy-timeout episode with replay)
- `lab-768.png` and `lab-375.png` (responsive layouts)

They are not cropped to 3:2.

## Legal declarations
The submitter must read and accept them personally. They have not been accepted on your behalf.
