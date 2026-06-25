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

### Demonstrated anti-gaming

`tests/test_integrity.py` proves this mechanically on a move task: the legitimate solver
scores **1.0**; the decoy (copy) attack, the shotgun (move-everything) attack, the
do-nothing agent, and the over-share leak all score **0.0** — each for a different,
correct reason.

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
  generator knows the answer it is asking for. Content is authored offline (static data),
  never generated at runtime, to preserve reproducibility.
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
carry no false ordering) with a `Discrete(5)` action. Episodes use distinct senders and
content entries to keep the retrieval near-linearly separable and reliably learnable. PPO
trains only on the vector; it never touches the `World` directly.

`rl/ppo.py` is a PPO implementation (shared trunk, separate actor/critic heads, clipped
objective, GAE-style returns) adapted from continuous control to discrete actions
(`Categorical` policy). Training climbs from the **20% random baseline to ~99% success**
over 80 iterations (`learning_curve.png`). The critic loss trends down as the value network
learns; the actor loss hovers near zero, which is expected for PPO (a sign-flipped policy
objective over normalized advantages, not a conventional error) and indicates stable
training (`loss_curve.png`).

---

## 6. Evaluation and Logging

`harness/evaluate.py` runs an agent across the seen and held-out suites, writes one JSONL
record per episode (`logs/trajectories.jsonl`: instruction, every action, per-step reward,
the itemized predicate results, and success), and saves a summary
(`logs/summary.json`) with overall / seen / held-out success, the generalization gap, and
breakdowns by category and difficulty.

The supplied baseline (`agents/baseline.py`) is a **scripted oracle**: it reads the
goal_spec to execute the correct operations. Its purpose is to demonstrate the environment
is solvable end-to-end and that the verifier *accepts* legitimate solutions (the complement
to the anti-gaming tests, which prove it *rejects* gaming). As an oracle it scores 100% on
both seen and held-out tasks, so the reported generalization gap is 0% — an oracle cannot
exhibit a gap because it does not learn. The harness, the held-out split, and the gap metric
are all in place and validated; a meaningful gap would emerge from a *learning* agent.

---

## 7. What I'd Improve With More Time

- **An LLM agent.** The `Agent` interface (`agents/agent.py`) is built so a real agent —
  one that works from only the instruction and observation, not the goal_spec — drops in
  without changing the harness. This is the natural next step and the thing that would
  produce a real generalization gap, real per-category difficulty curves, and a measure of
  how often agents fall into the anti-gaming traps.
- **A stricter summary task.** The current `summary` task verifies facts are present but
  does not yet require the agent to write a *new* file in `/Shared`; the rigorous version
  would check a newly-created summary file.
- **Deeper long-horizon chains.** The suite includes a long-horizon task (find → save →
  reply), but chains of 5+ dependent operations would stress credit assignment and
  partial-credit shaping further.
- **Richer communication verification.** Content checks use keyword matching, the softest
  part of the verifier; an LLM-judge or stricter structured checks would tighten it.

---

## Reproduce

```bash
pip install -r requirements.txt
python run.py            # baseline evaluation across the suite (writes logs/)
python run.py --rl       # also run the PPO training and save the learning curve
pytest tests/test_integrity.py -v   # anti-gaming + state-integrity tests
```