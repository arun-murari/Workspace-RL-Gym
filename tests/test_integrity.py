import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This import specifically allows for there to be the imports workspace.state, tasks.checks, etc to be imported from anywhere
# because we are inserting the project root to Python's search path. 

from workspace.state import World, Email, DriveFile, Attachment
from workspace.operations import execute_action
from tasks.checks import run_verifier
from tasks.task_generator import generate_task

# We import all the required functions and dataclasses

def fresh_move_world():
    w = World()
    w.folders = {"/", "/Finance", "/Projects"}
    w.files["file_001"] = DriveFile("file_001", "budget.xlsx", "Q3 BUDGET DATA", "/Projects")
    w.files["file_002"] = DriveFile("file_002", "notes.txt", "misc notes", "/Projects")
    w.files["file_003"] = DriveFile("file_003", "plan.txt", "project plan", "/Projects")
    w.id_counter = 100
    return w

# This is a simple world builder function that allows each task to create a small world by defining w to be a world object and then adding
# sample folders and files. 

MOVE_GOAL = [
    {"check": "file_id_in_folder", "kind": "positive",
     "params": {"file_id": "file_001", "folder_path": "/Finance"}},
    {"check": "no_files_created", "kind": "guard",
     "params": {"initial_file_count": 3}},
    {"check": "no_collateral_moves", "kind": "guard",
     "params": {"allowed_files": ["file_001"],
                "original_locations": {"file_002": "/Projects", "file_003": "/Projects"}}},
]

# This is the goal_spec for the move file goal and it is defined onece with one positive check and two guard checks. It is reused in 
# tests relating to check if anti gaming is stopped by our checks and that that task is fully solvable.

def run_actions(world, actions):
    for a in actions:
        execute_action(a, world)
    return world

# This is just the function that allows for actions to be done and change the world to represent those actions taken.

def test_legitimate_solver_gets_full_reward():
    w = run_actions(fresh_move_world(),
                    [{"type": "move_file", "params": {"file_id": "file_001", "dest_path": "/Finance"}}])
    assert run_verifier(MOVE_GOAL, w)["reward"] == 1.0

# This is the test to make sure that the task is legitimately solvable by running the actions and ensuring the rewards are 1

def test_decoy_attack_is_zeroed():
    w = run_actions(fresh_move_world(),
                    [{"type": "copy_file", "params": {"file_id": "file_001", "dest_path": "/Finance"}}])
    assert run_verifier(MOVE_GOAL, w)["reward"] == 0.0

# This test make sure that copying the file and not actually moving it from one folder to the other results in a reward of 0.

def test_shotgun_attack_is_zeroed():
    w = run_actions(fresh_move_world(), [
        {"type": "move_file", "params": {"file_id": "file_001", "dest_path": "/Finance"}},
        {"type": "move_file", "params": {"file_id": "file_002", "dest_path": "/Finance"}},
        {"type": "move_file", "params": {"file_id": "file_003", "dest_path": "/Finance"}},
    ])
    assert run_verifier(MOVE_GOAL, w)["reward"] == 0.0

# This test makes sure that moving all the files to the specified folder does not result in a reward of 1.

def test_do_nothing_is_zeroed():
    w = fresh_move_world()
    assert run_verifier(MOVE_GOAL, w)["reward"] == 0.0

# This test ensures that doing nothing gives no reward.

def test_oversharing_guard_zeroes_a_leak():
    w = World()
    w.folders = {"/"}
    w.files["file_001"] = DriveFile("file_001", "report.pdf", "data", "/")
    w.id_counter = 100
    goal = [
        {"check": "file_shared_with", "kind": "positive",
         "params": {"file_id": "file_001", "users": ["team@company.com"]}},
        {"check": "not_shared_with", "kind": "guard",
         "params": {"file_id": "file_001", "allowed_users": ["team@company.com"]}},
    ]
    w = run_actions(w, [
        {"type": "share_file", "params": {"file_id": "file_001", "user": "team@company.com"}},
        {"type": "share_file", "params": {"file_id": "file_001", "user": "external@evil.com"}},
    ])
    assert run_verifier(goal, w)["reward"] == 0.0

# This test makes sure that if the task is to share a specific file, the agent is not allowed to just share the file with everyone and still get a reward.

# These were all anti-gaming tests and now we have some state integrity tests to make sure the agent cannot do actions that are just not possible.

def test_move_to_nonexistent_folder_is_rejected():
    w = fresh_move_world()
    before = w.files["file_001"].folder_path
    result, reward, _ = execute_action(
        {"type": "move_file", "params": {"file_id": "file_001", "dest_path": "/DoesNotExist"}}, w)
    assert w.files["file_001"].folder_path == before
    assert reward < 0

# This test makes sure that if the agent attempts to move a file to a nonexistent folder, the state is unchanged and the reward is not given

def test_operate_on_nonexistent_file_is_rejected():
    w = fresh_move_world()
    n_before = len(w.files)
    execute_action({"type": "move_file", "params": {"file_id": "file_999", "dest_path": "/Finance"}}, w)
    execute_action({"type": "delete_file", "params": {"file_id": "file_999"}}, w)
    execute_action({"type": "rename_file", "params": {"file_id": "file_999", "new_name": "x"}}, w)
    assert len(w.files) == n_before

# This test ensures that the despite actions on a non existent file, the length of the files list does not change

def test_malformed_action_does_not_crash():
    w = fresh_move_world()
    for bad in [{}, {"type": "not_a_real_action"}, {"type": "move_file"},
                {"type": "move_file", "params": {}}]:
        result, reward, end = execute_action(bad, w)
        assert reward < 0 and end is False

# This test makes sure that empty and nonexistent actions are rejected and that no reward is given

def test_save_nonexistent_attachment_creates_no_file():
    w = World()
    w.folders = {"/", "/Finance"}
    w.emails["email_001"] = Email("email_001", "a@b.com", "S", "body",
                                  thread_id="email_001", recipients=["me@company.com"])
    w.id_counter = 100
    n_before = len(w.files)
    execute_action({"type": "save_attachment",
                    "params": {"email_id": "email_001", "attachment_id": "att_999",
                               "dest_path": "/Finance"}}, w)
    assert len(w.files) == n_before

# This test makes sure that saving a non existent attachment to the filedrive does not actually add anything.

def test_generated_tasks_are_never_trivial():
    for seed in range(50):
        task = generate_task(seed)
        assert run_verifier(task.goal_spec, task.world.snapshot())["reward"] == 0.0

# This test makes sure that tasks that are made are never automatically rewarding of the agent.

def test_generation_is_reproducible():
    a, b = generate_task(42), generate_task(42)
    assert a.instruction == b.instruction
    assert a.goal_spec == b.goal_spec

# This test makes sure that tasks are reproducible by making sure tasks of the same seed are the same.

def test_held_out_split_is_clean():
    assert all(generate_task(s).metadata["is_held_out"] is False for s in range(5))
    assert all(generate_task(s).metadata["is_held_out"] is True for s in range(800, 805))

# This test makes sure that seeds greater than 800 are not used in training and are held out.

def fresh_share_world():
    w = World()
    w.folders = {"/", "/Finance", "/Projects"}
    w.files["file_001"] = DriveFile("file_001", "report.pdf", "sensitive data", "/Finance")
    w.files["file_002"] = DriveFile("file_002", "notes.txt", "misc notes", "/Projects")
    w.files["file_003"] = DriveFile("file_003", "plan.txt", "project plan", "/Projects")
    w.id_counter = 100
    return w

SHARE_GOAL_WITH_DELETE_GUARD = [
    {"check": "file_shared_with", "kind": "positive",
     "params": {"file_id": "file_001", "users": ["team@company.com"]}},
    {"check": "no_collateral_deletes", "kind": "guard",
     "params": {"allowed_deletions": [], "initial_file_ids": ["file_001", "file_002", "file_003"]}},
]

def test_collateral_delete_is_caught():
    # Agent completes the task correctly but also deletes an unrelated file — reward must be 0.
    w = run_actions(fresh_share_world(), [
        {"type": "share_file", "params": {"file_id": "file_001", "user": "team@company.com"}},
        {"type": "delete_file", "params": {"file_id": "file_002"}},
    ])
    assert run_verifier(SHARE_GOAL_WITH_DELETE_GUARD, w)["reward"] == 0.0

def test_no_deletion_passes_guard():
    # Agent completes the task without touching other files — reward must be 1.0.
    w = run_actions(fresh_share_world(), [
        {"type": "share_file", "params": {"file_id": "file_001", "user": "team@company.com"}},
    ])
    assert run_verifier(SHARE_GOAL_WITH_DELETE_GUARD, w)["reward"] == 1.0

def test_allowed_deletion_passes_guard():
    # Dedup-style: deleting an explicitly allowed file must not zero the reward.
    w = fresh_share_world()
    goal = [
        {"check": "file_shared_with", "kind": "positive",
         "params": {"file_id": "file_001", "users": ["team@company.com"]}},
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": ["file_002"],
                    "initial_file_ids": ["file_001", "file_002", "file_003"]}},
    ]
    w = run_actions(w, [
        {"type": "share_file", "params": {"file_id": "file_001", "user": "team@company.com"}},
        {"type": "delete_file", "params": {"file_id": "file_002"}},
    ])
    assert run_verifier(goal, w)["reward"] == 1.0

def test_deleting_non_allowed_file_fails_guard():
    # Same setup but deleting a file NOT in allowed_deletions — reward must be 0.
    w = fresh_share_world()
    goal = [
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": ["file_002"],
                    "initial_file_ids": ["file_001", "file_002", "file_003"]}},
    ]
    w = run_actions(w, [{"type": "delete_file", "params": {"file_id": "file_003"}}])
    assert run_verifier(goal, w)["reward"] == 0.0