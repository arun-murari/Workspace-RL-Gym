import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This import specifically allows for there to be the imports workspace.state, tasks.checks, etc to be imported from anywhere
# because we are inserting the project root to Python's search path. 

from collections import defaultdict
from workspace.env import WorkspaceEnv
from tasks.task_generator import generate_task
from tasks.checks import run_verifier
from agents.baseline import ScriptedOracleBaseline

# Import all the required functions and the WorkSpace Env as well as the baseline agent

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# This line allows for access to a log folder that we create to store all the logs in

def run_episode(agent, task, step_budget=20, agent_mode="oracle"):
    env = WorkspaceEnv(task, step_budget=step_budget)
    obs, _ = env.reset()
    agent.reset(task)

    # This is the run_episode function which runs one task fully from end-to-end. It first creates the environment with WorkspaceEnv, then
    # we define obs to be the observation return from the env and then do agent.reset to start the environment from anew.

    trajectory = []
    total_reward = 0.0
    terminated = truncated = False

    # We define a empty list called trajectory to store what each action output, define the total reward to be 0 and then define terminated
    # and truncated to both be False because the task is not done and the step budget is not exceeded. 

    step_idx = 0
    while not (terminated or truncated):
        action = agent.act(obs, task)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        step_verifier = run_verifier(task.goal_spec, env.world)
        step_checks = [
            {"check": r["check"], "kind": r["kind"], "passed": r["passed"]}
            for r in step_verifier["predicate_results"]
        ]

        trajectory.append({
            "step": step_idx,
            "action": action,
            "reward": round(reward, 4),
            "terminated": terminated,
            "truncated": truncated,
            "checks": step_checks,
            "last_result": env.last_result,
        })
        step_idx += 1

    # While both terminated or truncated are False, we define the action to be based on whatever the agents act function is, and then we define
    # obs, reward, terminated and truncated to be the output of the env.step(action). We update the total reward and then add this actions outputs
    # to the trajectory list. Per-step checks are captured for tracing but do not affect reward.

    all_check_names = [c["check"] for c in (trajectory[0]["checks"] if trajectory else [])]
    check_timeline = {name: None for name in all_check_names}
    for step_entry in trajectory:
        for c in step_entry["checks"]:
            name = c["check"]
            if check_timeline.get(name) is None and c["passed"]:
                check_timeline[name] = step_entry["step"]

    verifier = run_verifier(task.goal_spec, env.world)
    return {
        "seed": task.metadata["seed"],
        "category": task.metadata["category"],
        "difficulty": task.metadata["difficulty"],
        "is_held_out": task.metadata["is_held_out"],
        "instruction": task.instruction,
        "agent_mode": agent_mode,
        "total_reward": round(total_reward, 4),
        "completion_reward": verifier["reward"],
        "success": verifier["reward"] == 1.0,
        "predicate_results": verifier["predicate_results"],
        "trajectory": trajectory,
        "n_steps": len(trajectory),
        "check_timeline": check_timeline,
    }

    # Then we run verifier with the tasks goal spec and then current world. Then we return one big dict containing seed, category of the task, difficulty, and whether
    # it is held out. It also holds task instruction, the total reward, the completion_reward from the verifier, success which is true if the reward is 1, predicate_results
    # that is basically a dict of checks, trajectory and then the number of steps taken which is length of trajectory list.

def evaluate(agent=None, train_seeds=range(0, 100), held_out_seeds=range(800, 850),
             step_budget=20, agent_mode="oracle"):
    if agent is None:
        agent = ScriptedOracleBaseline()
    os.makedirs(LOG_DIR, exist_ok=True)

    # This is the evaluate function which runs the whole suite. It defaults to the baseline agent if no agent is provided.

    records = []
    log_path = os.path.join(LOG_DIR, "trajectories.jsonl")
    with open(log_path, "w") as f:
        for seed in list(train_seeds) + list(held_out_seeds):
            task = generate_task(seed)
            rec = run_episode(agent, task, step_budget, agent_mode=agent_mode)
            records.append(rec)
            f.write(json.dumps(rec) + "\n")

    # Here we make a new list called records, and we create a new jsonl file known as trajectories. We run every single seed both the training ones and the
    # held_out ones and then generate a task based on that seed, run full episode for that task then add that dict to the recrods list. Then we make each line 
    # in the trajectories.jsonl file a json file itself containing that info from each seed.

    summary = summarize(records)
    with open(os.path.join(LOG_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_report(summary)
    return summary

    # We make a summary that summarizes all the info in the records list. Then we create a new json file knwon as summary and add that summary to it. Then we
    # return the summary variable as well as printing the report.

def summarize(records):
    def rate(recs):
        return round(sum(r["success"] for r in recs) / len(recs), 4) if recs else 0.0

    # This inner function returns a fraction of the records that succeeded. 

    seen = [r for r in records if not r["is_held_out"]]
    held = [r for r in records if r["is_held_out"]]

    # Define seen to be the records that are for training so the seeds before 800 and then define held to be a list that holds the records for the held out seeds

    by_category = defaultdict(list)
    by_difficulty = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)
        by_difficulty[r["difficulty"]].append(r)

    # Create two new defaultdicts that adds the records based on category and difficulty.

    return {
        "n_total": len(records),
        "overall_success": rate(records),
        "seen_success": rate(seen),
        "held_out_success": rate(held),
        "generalization_gap": round(rate(seen) - rate(held), 4),
        "by_category": {c: rate(rs) for c, rs in by_category.items()},
        "by_difficulty": {d: rate(rs) for d, rs in by_difficulty.items()},
        "n_seen": len(seen),
        "n_held_out": len(held),
    }

    # Return a final dict that includes all this information.

def print_report(s):
    print("=" * 55)
    print("EVALUATION REPORT")
    print("=" * 55)
    print(f"Total episodes:      {s['n_total']}")
    print(f"Overall success:     {s['overall_success']:.1%}")
    print()
    print(f"Seen success:        {s['seen_success']:.1%}  (n={s['n_seen']})")
    print(f"Held-out success:    {s['held_out_success']:.1%}  (n={s['n_held_out']})")
    print(f"Generalization gap:  {s['generalization_gap']:+.1%}")
    print()
    print("By category:")
    for c, r in sorted(s["by_category"].items()):
        print(f"  {c:22} {r:.1%}")
    print("By difficulty:")
    for d, r in sorted(s["by_difficulty"].items()):
        print(f"  {d:22} {r:.1%}")
    print("=" * 55)

# Print report is just a print function that displays the important information from summary in a nice format.

if __name__ == "__main__":
    evaluate()

# Allows for the whole thing to be ran.

# This file in general has the goal of building up a report and storing information about how the agent behaves when it fully runs through
# every task, trajectories is basically a detailed play-by-play whereas the summary json is some final statistics that are useful to know about how
# the agent did.