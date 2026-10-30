# Three-minute demo script

Do not claim live Nebius inference unless the connectivity probe and a real tool-using episode succeed. The current local demo must explicitly say that live account verification is pending.

| Time | Action in the real application | Narration |
|---|---|---|
| 0:00–0:20 | Overview, then Environment Lab | A tool can report failure after changing infrastructure. CommitGym trains agents to recover safely and verifies the actual terminal state. |
| 0:20–0:55 | Baseline, Ambiguous deployment, seed42. Start, Step three times. Select deployment event. | The candidate release changed to v2, but the deployment returned a timeout. Old traffic still serves v1. The agent must reconcile state before retrying. |
| 0:55–1:15 | Step inspection; Run episode. Inspect verification and completion. | The agent drains traffic, migrates, probes, restores traffic and verifies invariants. The judge, not the agent, decides whether completion is safe. |
| 1:15–1:40 | Stale health scenario, baseline. Run, select stale/fresh probes and rollback. | A cached healthy result is not proof. The fresh probe detects the bad candidate; rollback restores a compatible application/schema snapshot and requires fresh verification. |
| 1:40–2:05 | PPO episode, then Training and Evaluations | This policy was actually trained for 300,032 steps. On 120 held-out episodes both policies completed safely. PPO always rolled back; baseline completed 80 upgrades. Higher reward does not mean better deployment completion. |
| 2:05–2:40 | If verified, run a real Nemotron episode and inspect Token Factory usage. Otherwise select **Mock tool adapter (offline, no LLM)**, keep its banner on screen, run an episode, and open one step in the inspector. | With verified access: NVIDIA Nemotron chooses strictly validated simulator tools through Nebius Token Factory. Without access: "This is not a model. It is a local planner that emits tool calls through the same validation boundary, three-operation batches and five-decision limit Nemotron uses. Live Nemotron inference is pending account access, and no model result is being substituted." |
| 2:40–2:55 | History, replay verification | Every action, state, invariant and reward is persisted. Replaying the original seed and actions reproduces the recorded outcome. |
| 2:55–3:00 | Return to Overview | CommitGym: train agents to recover safely, not just execute successfully. |

A local browser recording can demonstrate the working test build. A final hackathon video still needs narration and, to substantiate the mandatory integration, verified live NVIDIA inference footage. Never show API keys, account details or promotional codes during recording.
