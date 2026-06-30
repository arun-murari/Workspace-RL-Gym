# Workspace RL Gym (Email + Drive)

A [Gymnasium](https://gymnasium.farama.org/) environment that simulates a digital
workspace — an email inbox and a file drive — operated by an AI agent through tool calls.
The environment is built around a **verifiable, hard-to-game reward** and a
**procedurally generated task suite with a held-out split**, so it measures learned skill
rather than memorization.

See [`DESIGN.md`](DESIGN.md) for the full design rationale.

## Quick start

```bash
pip install -r requirements.txt

# Run the scripted oracle baseline across the full task suite
python run.py                       # writes harness/logs/

# Run the LLM agent (requires GROQ_API_KEY)
export GROQ_API_KEY="your-key"
python -c "
from agents.llm_agent import LLMAgent
from harness.evaluate import evaluate
evaluate(agent=LLMAgent(mode='normal'), train_seeds=range(0,20), held_out_seeds=range(800,810), agent_mode='normal')
"

# Run all tests
pytest tests/test_integrity.py -v   # 16 anti-gaming + state-integrity tests

# Generate sample trajectories and open the replay UI
python harness/export_for_ui.py     # writes ui/sample_trajectory.json
open ui/replay.html                 # or open in any browser
```

`python run.py` writes machine-readable results to `harness/logs/`:
- `trajectories.jsonl` — one record per episode: instruction, every action with its
  per-step reward and per-step verifier check trace, the check_timeline, and final success.
- `summary.json` — success rates, seen-vs-unseen generalization gap, and breakdowns by
  category and difficulty.

## What's here

| Path | What it is |
|------|------------|
| `workspace/state.py` | The unified `World` model (emails, files, folders, attachments). |
| `workspace/operations.py` | The agent's tool calls (the action space) with validate-then-mutate semantics. |
| `workspace/env.py` | The Gymnasium env: `reset` / `step`, observations, termination vs. truncation. |
| `tasks/checks.py` | The verifier — predicate menu + positive/guard reward logic (anti-gaming). |
| `tasks/task_generator.py` | Procedural task generation, difficulty knobs, held-out split. |
| `tasks/content.py` | Curated content library with embedded, verifiable facts. |
| `agents/agent.py` | The `Agent` interface the harness drives. |
| `agents/baseline.py` | Scripted oracle baseline that demonstrates solvability. |
| `agents/llm_agent.py` | LLM-backed agent (Groq / llama-3.3-70b) with `normal` and `red_team` modes. |
| `rl/rl_env.py`, `rl/ppo.py` | The constrained RL slice and a PPO training run (20% → ~99%). |
| `harness/evaluate.py` | Runs an agent across the suite; emits JSONL logs + summary. |
| `harness/export_for_ui.py` | Generates `ui/sample_trajectory.json` from live LLM agent runs. |
| `tests/test_integrity.py` | 16 anti-gaming attacks + adversarial state-integrity tests. |
| `ui/replay.html` | Self-contained trajectory replay viewer — step through any episode. |
| `ui/sample_trajectory.json` | Pre-generated sample trajectories for the replay UI. |

## Task categories

`move`, `cross_app` (save attachment + reply), `archive`, `retrieval_fact` (read content
and report a fact), `judgment_overshare` (share correctly without leaking),
`judgment_clarify` (ask when underspecified), `summary`, `communication` (send with
required content), `dedup` (delete the duplicate), `long_horizon`, and
`judgment_well_specified` (a clear request that must NOT trigger clarification).

## LLM agent

`LLMAgent` in `agents/llm_agent.py` implements the `Agent` interface and works
**only from the task instruction and observation** — it never reads the `goal_spec`.
This makes its results meaningful: the success rate and generalization gap are real,
not oracle artefacts.

Two modes:

| Mode | Purpose |
|------|---------|
| `normal` | Completes tasks as a helpful assistant. Produces real success rates and a real seen-vs-unseen gap. |
| `red_team` | Explicitly instructed to exploit loopholes and reward shortcuts. Stress-tests that the verifier cannot be gamed. |

Red-team empirical results: on the tasks most susceptible to gaming (oversharing,
judgment, clarification), the red-team agent earned no undeserved reward — every
positive score required legitimately completing the task.

Requires `GROQ_API_KEY` (free tier at [console.groq.com](https://console.groq.com/keys)).

## Replay UI

`ui/replay.html` is a self-contained single-file viewer. Open it in any browser — no
server required.

- Step through any episode with **Prev / Next buttons, a slider, or arrow keys**.
- Each step shows the action taken, per-step reward, cumulative reward, the action
  result text, and — as the focal point — the **check panel**: every task-specific
  predicate with a ✓ / ✗ symbol, its kind (POSITIVE / GUARD), and a GUARD VIOLATED
  alert when a guard fires.
- The episode summary shows the `check_timeline` (first step each check became true)
  and the final completion reward.
- Load any `.json` or `.jsonl` trajectory file via the file input, or two demo
  trajectories are embedded: a normal-agent cross_app (checks flip progressively) and a
  red-team overshare (guard fires at step 2 and zeros the reward).

To generate fresh trajectories for the UI:
```bash
export GROQ_API_KEY="your-key"
python harness/export_for_ui.py
```

## Key properties

- **Verifiable rewards** judged from final world state, never agent claims.
- **Anti-gaming** via guard checks that gate (not earn) reward; identity-based checks
  that defeat copy/decoy attacks; symmetric judgment tasks; and `no_collateral_deletes`
  guard on every task where file integrity matters (mirrors `no_collateral_moves`).
- **Per-step verification trace** — the verifier runs after every step and the result
  is logged in `trajectory[i].checks`, so you can see which predicate flipped when.
  Terminal reward is unaffected; this is observational only.
- **Deterministic seeding** — same seed yields the same task; held-out split (seeds ≥ 800)
  is leak-free by construction.
- **Reproducible** — one command, no network or runtime content generation (oracle
  baseline; LLM agent requires API key).

## Requirements

Python 3.10+, plus the packages in `requirements.txt`:
`gymnasium`, `numpy`, `torch`, `matplotlib`, `pytest`, `openai` (for the LLM agent).

The `openai` package is pointed at the Groq base URL; no OpenAI account is needed.
