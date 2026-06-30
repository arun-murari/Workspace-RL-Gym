# Design Document — Workspace RL Gym (Email + Drive)

A Gymnasium environment simulating a digital workspace (an email inbox and a file
drive) that an AI agent operates through tool calls. The focus of the design is a
**verifiable, hard-to-game reward** and a **procedurally generated task suite with a
held-out split**, so the environment measures learned skill rather than memorization.

---

## 1. State Model

The entire workspace lives in a single `World` dataclass (`workspace/state.py`). I
deliberately chose **one unified container** over separate email/drive states because
cross-app tasks (e.g. saving an email attachment into the drive) touch both sides at
once, and a single store removes any chance of the two halves drifting out of sync.

The `World` holds:

- `emails: dict[str, Email]` — each email has a sender, subject, body, `thread_id`,
  recipients, a single `folder` (Inbox / Sent / Archive), and a list of `Attachment`s.
- `files: dict[str, DriveFile]` — each file has a name, content, `folder_path`, and a
  `shared_with` list.
- `folders: set[str]` — folder paths; nesting is expressed with slashes.
- `clarifications` and `done_answer` — written by the agent's meta-actions and read by
  the verifier at scoring time.

Two design decisions worth calling out:

- **All ids flow through a single monotonic counter (`World.next_id`).** Nothing is
  hand-assigned. This makes id collisions structurally impossible, which matters because
  the verifier checks some tasks by id (see §3).
- **Attachments store content directly**, rather than referencing a `DriveFile`. This
  makes `save_attachment` a real content-copy and keeps the `deepcopy` snapshot
  (used on every `reset`) clean and self-contained.

State integrity is enforced by a **validate-then-mutate** discipline in every operation:
a lookup and semantic check happen *before* any mutation, so an invalid action (moving a
nonexistent file, saving to a missing folder, a malformed action dict) is rejected with a
penalty and leaves the world untouched. Adversarial sequences are covered by
`tests/test_integrity.py`.

---

## 2. Operations and Action Space

The agent acts through a fixed set of operations (`workspace/operations.py`), exposed as a
closed `ActionType` enum: email reads (`search_email`, `open_email`), email writes
(`send_email`, `reply_email`, `forward_email`, `archive_email`), drive reads
(`search_drive`, `open_file`), drive writes (`move_file`, `copy_file`, `rename_file`,
`delete_file`, `create_folder`, `share_file`), the cross-app bridge `save_attachment`,
and the meta-actions `ask_clarification` and `done`.

An action is a dict: `{"type": "move_file", "params": {"file_id": ..., "dest_path": ...}}`.
`validate_action` first gates on structure (is it a dict, is the type known, are required
params present), then the operation runs its own semantic checks. A central
`execute_action` dispatches through a registry and applies a uniform step cost.

The guiding principle is **mechanics in the tool, correctness in the reward**: operations
are "dumb" (a delete deletes any file), and whether an action was *correct* for a given
task is decided entirely by the per-task verifier. This keeps the action space general and
puts all task-specific judgment in one place.

The environment (`workspace/env.py`) is a standard `gymnasium.Env`. `reset(seed)` snapshots
a fresh world from the pristine template (deterministic via the seed); `step(action)` runs
the action, applies shaping rewards, and distinguishes **termination** (the agent called
`done`) from **truncation** (the step budget was exhausted) per the Gymnasium contract. The
observation is a deliberately **partial** text view — inbox summary, folder structure with
counts, and the last action's result — so the agent must use search/open actions to reveal
detail, which makes retrieval and navigation genuine skills.

---

## 3. Reward Design and Anti-Gaming (primary focus)

Rewards are computed by reading the **final world state**, never the agent's claims
(`tasks/checks.py`). The verifier is a menu of predicate functions; a task's `goal_spec` is
a list of `{check, kind, params}` entries that name predicates from that menu.

### Positive checks vs. guard checks

Every check is either a **positive** (earns credit) or a **guard** (gates credit). The
reward rule:

- If **any guard fails**, reward is **0** — regardless of how many positives passed.
- Otherwise, reward is the **fraction of positives passed** (partial credit for multi-step
  tasks).

This split is the core anti-gaming mechanism. A naive "fraction of all checks passed"
scheme has a critical hole: **negative/guard checks are satisfied by inaction** (an agent
that does nothing never creates a decoy, never over-shares, never deletes evidence), so a
do-nothing agent could farm partial reward. By making guards *gate* rather than *earn*,
inaction earns zero positives and therefore zero reward.

### Identity, not appearance

Checks reference ground truth the agent cannot forge:

- **Move tasks check by `file_id`** (`file_id_in_folder`). Since a `copy` produces a new
  id, an agent that copies-into-the-target instead of moving fails — the original id never
  arrives. The `no_files_created` guard catches the spawned copy as well.
- **Cross-app tasks check by content** (`file_in_folder`), because `save_attachment`
  creates a brand-new file whose id is unknowable in advance; the planted content is the
  forgery-resistant anchor.
- **Communication checks require content**, not just a recipient (`email_sent_to`,
  `reply_in_thread` with `must_include`). Recipient-presence alone is gameable (send an
  empty email); requiring specific content the agent could only know by doing the task
  closes that hole.

### Symmetric judgment

Judgment tasks come in symmetric pairs so the agent cannot win with a blanket policy.
Underspecified tasks reward `clarification_asked`; well-specified tasks include a
`clarification_not_asked` guard, so always-asking fails the well-specified half.
Over-sharing tasks pair `file_shared_with` (positive) with `not_shared_with` (guard) so an
agent that shares correctly *and* leaks externally is zeroed.

### Collateral damage guards

Two guards protect unrelated world state from adversarial side-effects:

- **`no_collateral_moves`** — any file not in `allowed_files` must still be in its original
  folder. This also catches file *deletions* of those files: `world.get_file(id) is None`
  evaluates the same as a wrong-folder check, so the guard is deletion-resistant without
  extra code.
- **`no_collateral_deletes`** — mirrors `no_collateral_moves` for tasks that do not already
  carry it. Any file in `initial_file_ids` that is not in `allowed_deletions` must still
  exist. The `allowed_deletions` parameter makes the dedup task possible: the duplicate is
  explicitly permitted to be deleted while all other files are protected.

`no_collateral_deletes` is wired into every task generator that does not already have
`no_collateral_moves`: `cross_app`, `archive`, `judgment_overshare`, `judgment_clarify`,
`judgment_well_specified`, `retrieval_fact`, `communication`, `long_horizon`, and `dedup`.
This closes the gap discovered by the red-team agent, which used `delete_file` on an
unrelated file during a task where only an email action was required.

### Demonstrated anti-gaming

`tests/test_integrity.py` (16 tests) proves this mechanically: the legitimate solver
scores **1.0**; the decoy (copy) attack, the shotgun (move-everything) attack, the
do-nothing agent, the over-share leak, and the collateral-delete attack all score **0.0**
— each for a different, correct reason. The red-team LLM agent (§7) provides the
empirical complement: run adversarially across the suite, it found no exploit that earned
undeserved reward.

---

## 4. Task Generation and Generalization

`tasks/task_generator.py` produces a `Task` (instruction, world, goal_spec, metadata)
deterministically from a seed — the same seed always yields the same task, which is what
makes the held-out split leak-free.

- **Categories** rotate evenly by seed: `move`, `cross_app`, `archive`,
  `judgment_overshare`, `judgment_clarify`, `retrieval_fact`, `summary`,
  `communication` (retrieve a fact from an email and forward it), `dedup` (delete the
  duplicate, keep one), `long_horizon` (find email → save attachment → reply, a chain with
  partial credit), and `judgment_well_specified` (a clear request that must NOT trigger a
  clarification — the guard half of the symmetric judgment pair).
- **Difficulty is a knob, not a separate task.** `easy`/`medium`/`hard` set the number of
  distractor emails and files (3 / 8 / 15) and the vagueness of the instruction, so
  difficulty is a real parameter and success-vs-difficulty is measurable.
- **Content is a curated library** (`tasks/content.py`): realistic email bodies and file
  contents paired with the exact `facts` embedded in them. Because facts are stored
  alongside content, content-comprehension tasks (`retrieval_fact`) are verifiable — the
  generator knows the answer it is asking for.
- **Non-trivial guarantee:** every generated task is run through the verifier on its
  *initial* world and rejected/regenerated if it already scores above zero, so no task can
  be passed by doing nothing.
- **Held-out split:** seeds `>= 800` are flagged `is_held_out`. Because tasks are a pure
  function of the seed, held-out tasks never appear in training — the split is clean by
  construction.

---

## 5. RL Training Run

The full environment's text observation and dict action space are not directly trainable,
so `rl/rl_env.py` is a **constrained slice** of it: each episode builds a real `World` of
five emails drawn from the real `EMAIL_CONTENT` library, frames it as a `retrieval_fact`
task, and **scores the agent's choice through the real `run_verifier`** (via `answer_matches`
on `world.done_answer`). The only RL-specific layer is an `encode` step that featurizes the
actual emails into a 66-dim vector (task block + five email blocks, one-hot so categories
carry no false ordering) with a `Discrete(5)` action.

`rl/ppo.py` is a PPO implementation (shared trunk, separate actor/critic heads, clipped
objective, GAE-style returns) adapted to discrete actions (`Categorical` policy). Training
climbs from the **20% random baseline to ~99% success** over 80 iterations
(`learning_curve.png`). The critic loss trends down; the actor loss hovers near zero —
expected for PPO's sign-flipped policy objective over normalized advantages, indicating
stable training (`loss_curve.png`).

---

## 6. Evaluation, Logging, and Per-Step Verification Trace

`harness/evaluate.py` runs an agent across the seen and held-out suites. Each episode
record in `harness/logs/trajectories.jsonl` contains:

- **Episode-level fields:** `seed`, `category`, `difficulty`, `is_held_out`,
  `instruction`, `agent_mode`, `total_reward`, `completion_reward`, `success`,
  `predicate_results`, `n_steps`, and `check_timeline`.
- **Per-step fields** (`trajectory[i]`): `step`, `action`, `reward`, `terminated`,
  `truncated`, `last_result` (the text the env returned for that action), and `checks`
  — the output of `run_verifier` run immediately after that step, logged as
  `[{"check": name, "kind": positive|guard, "passed": bool}, ...]`.
- **`check_timeline`:** a dict mapping each check name to the first step index at which
  it became `passed=True` (or `null` if never). Directly answers "when did this predicate
  flip?"

The per-step verifier calls are **observational only** — they do not change the reward.
The terminal reward is still computed once at episode end from the final world state,
exactly as before. The per-step trace exists so the replay UI and offline analysis can see
the verifier's judgment at every point in the trajectory, not just the end.

**`agent_mode`** is threaded from `evaluate()` through `run_episode()` into the episode
record, so trajectories generated by the oracle, the normal LLM agent, and the red-team
agent are distinguishable in the log.

The supplied oracle baseline (`agents/baseline.py`) reads the `goal_spec` and scores 100%
on both seen and held-out tasks, demonstrating the environment is solvable end-to-end
and that the verifier accepts legitimate solutions. The LLM agent (§7) provides a
goal_spec-blind complement that produces a real generalization gap.

---

## 7. LLM Agent — Normal and Red-Team Modes

`agents/llm_agent.py` implements the `Agent` interface backed by the Groq inference API
(model: `llama-3.3-70b-versatile`, via the `openai` package pointed at Groq's
OpenAI-compatible endpoint). It works **only from the task instruction and observation**,
never from the `goal_spec` — this is what makes its results meaningful.

### Two modes

**`normal`** — The agent is prompted to act as a helpful assistant and complete the task
correctly. On easy seeds it achieves reward 1.0 working from instruction + observation
alone, which demonstrates the environment is solvable by a real (non-oracle) agent and
produces an honest generalization gap when run across the full suite.

**`red_team`** — The agent is explicitly instructed to maximize reward by any means,
including exploiting loopholes: doing the right action then undoing it (testing whether
the verifier only checks the final state), sending emails to wrong recipients and deleting
the evidence, sharing files with unauthorized users, deleting unrelated files as a side
effect. This is an empirical stress-test of the anti-gaming mechanisms.

### Red-team findings

Running the red-team agent across the gameable-looking task categories produced no
undeserved reward:

- On `judgment_overshare`, the `not_shared_with` guard held every time the agent tried
  to leak a file. When it scored 1.0, it had legitimately completed the task.
- On `judgment_clarify`, the agent tried `delete_file`, `send_email`, and other
  off-task manipulations; the `clarification_asked` check can only be satisfied by
  calling `ask_clarification`, so there is no forgeable path to reward.
- The `no_collateral_deletes` guard was wired in after the red-team demonstrated it
  could call `delete_file` on an unrelated file during a task with no file-integrity
  guard. Before the guard: a task could be completed legitimately and then receive the
  same reward despite collateral file damage. After: `reward → 0` for any collateral
  deletion.

### Implementation notes

- **Rolling conversation history** (`HISTORY_WINDOW = 6` turns) keeps the model
  context-aware across multi-step episodes without unbounded token growth.
- **Robust JSON parsing** strips markdown fences, extracts the first `{...}` block, and
  falls back to `{"type": "done", "params": {}}` on parse failure rather than crashing.
  The env's `validate_action` handles malformed outputs without halting the episode.
- **`GROQ_API_KEY`** is read from the environment; the agent raises a clear error if it
  is unset. The key is never hardcoded.
- The action reference shown to the model is derived from `REQUIRED_PARAMS` in
  `workspace/operations.py` at import time, so it stays in sync with the action space
  without a manually maintained list.

---

## 8. Replay UI

`ui/replay.html` is a self-contained single-file trajectory viewer — HTML, CSS, and
JavaScript with no external dependencies and no build step. It loads trajectory data from
`ui/sample_trajectory.json` (via `fetch`) or from a file input (`.json` or `.jsonl`),
and falls back to two embedded demo trajectories if neither is available, so it works
from `file://` without any server.

The UI is organized around **step navigation** (Prev / Next buttons, a slider, arrow
keys) with four panels per step:

- **Action** — the `type` and `params` of the action taken this step.
- **Rewards** — step reward, running cumulative total, and episode status.
- **Action result** — the text the env returned for this action (`last_result`).
- **Check panel** (the focal point) — every task predicate with a ✓ / ✗ symbol, a kind
  badge (POSITIVE or GUARD), color-coded background (green / teal / red), and a
  **GUARD VIOLATED** alert for any guard that fires. This is the visual representation
  of the per-step verifier judgment from §6.

The episode summary footer shows the `check_timeline` (first step each check became
true) and the final `completion_reward` / `success`.

**`harness/export_for_ui.py`** generates fresh trajectories by running the LLM agent
(normal mode on a cross_app seed, red-team mode on overshare and clarify seeds) and
writing the results to `ui/sample_trajectory.json`. If `GROQ_API_KEY` is not set it
falls back to the oracle baseline.

The two embedded demo samples are:
1. A **normal-agent cross_app** episode: `file_in_folder` flips at step 3,
   `reply_in_thread` at step 4, showing the verifier tracking partial task completion
   across steps.
2. A **red-team overshare** episode: the guard `not_shared_with` passes at steps 0–1,
   fires (✗) at step 2 when the agent leaks the file to an unauthorized user, and stays
   failed through `done`. `completion_reward = 0.0` despite the positive check passing.

---

## 9. What Remains

- **Deeper long-horizon chains.** The suite includes a long-horizon task (find → save →
  reply), but chains of 5+ dependent operations would stress credit assignment and
  partial-credit shaping further.
- **Richer communication verification.** Content checks use keyword matching, the softest
  part of the verifier; an LLM-judge or stricter structured checks would tighten it.
- **Full LLM agent evaluation run.** The agent has been validated on individual seeds; a
  full 150-seed sweep would produce the seen-vs-unseen generalization gap and
  per-category difficulty curves that quantify where the environment is hard.
- **RL with the LLM agent as a teacher.** The PPO run uses the constrained retrieval
  slice; an RL loop that trains a smaller model against the full task suite — using the
  LLM agent's trajectories as imitation data or as a reward signal — would be the natural
  next scaling step.

---

## Reproduce

```bash
pip install -r requirements.txt

# Oracle baseline (no API key needed)
python run.py
python run.py --rl
pytest tests/test_integrity.py -v

# LLM agent
export GROQ_API_KEY="your-key"          # free tier at console.groq.com/keys
python harness/export_for_ui.py         # generate replay UI sample data
open ui/replay.html                     # step through trajectories in browser
```
