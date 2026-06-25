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

python run.py                       # run the baseline across the task suite -> logs/
python run.py --rl                  # also train PPO on the constrained env
pytest tests/test_integrity.py -v   # anti-gaming + state-integrity tests
```

`python run.py` writes machine-readable results to `harness/logs/`:
`trajectories.jsonl` (one record per episode) and `summary.json` (success rates, the
seen-vs-unseen generalization gap, and breakdowns by category and difficulty).

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
| `agents/baseline.py` | A scripted oracle baseline that demonstrates solvability. |
| `rl/rl_env.py`, `rl/ppo.py` | The constrained RL slice and a PPO training run (20% → ~95%). |
| `harness/evaluate.py` | Runs an agent across the suite; emits JSONL logs + summary. |
| `tests/test_integrity.py` | Anti-gaming attacks + adversarial state-integrity tests. |

## Task categories

`move`, `cross_app` (save attachment + reply), `archive`, `retrieval_fact` (read content
and report a fact), `judgment_overshare` (share correctly without leaking),
`judgment_clarify` (ask when underspecified), and `summary`.

## Key properties

- **Verifiable rewards** judged from final world state, never agent claims.
- **Anti-gaming** via guard checks that gate (not earn) reward, identity-based checks that
  defeat copy/decoy attacks, and symmetric judgment tasks.
- **Deterministic seeding** — same seed yields the same task; held-out split (seeds ≥ 800)
  is leak-free by construction.
- **Reproducible** — one command, no network or runtime content generation.

## Requirements

Python 3.10+, plus the packages in `requirements.txt` (`gymnasium`, `numpy`, `torch`,
`matplotlib`, `pytest`).
