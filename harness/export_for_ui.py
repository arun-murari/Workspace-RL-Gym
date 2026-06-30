#!/usr/bin/env python3
"""
Generate sample trajectories for ui/replay.html.

Runs the LLM agent (normal + red_team) on a curated set of seeds and writes
the results to ui/sample_trajectory.json, which replay.html loads by default.

Usage:
    python harness/export_for_ui.py

Requires GROQ_API_KEY to be set. Falls back to oracle baseline if not available.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.evaluate import run_episode
from tasks.task_generator import generate_task

UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui")
OUT_PATH = os.path.join(UI_DIR, "sample_trajectory.json")

# Seeds chosen to produce interesting trajectories for the demo:
#   seed=1  cross_app   — multi-step, two checks flip progressively (good for normal mode)
#   seed=3  overshare   — guard fires if red-team leaks the file (good for red-team mode)
#   seed=4  clarify     — red-team tries file manipulation, guard catches it
NORMAL_SEEDS   = [1]
RED_TEAM_SEEDS = [3, 4]


def run_with_agent(agent, seeds, mode_label):
    records = []
    for seed in seeds:
        task = generate_task(seed)
        rec = run_episode(agent, task, agent_mode=mode_label)
        records.append(rec)
        print(f"  [{mode_label}] seed={seed:3d} cat={rec['category']:22s} "
              f"reward={rec['completion_reward']:.2f} success={rec['success']} "
              f"steps={rec['n_steps']}")
    return records


def main():
    os.makedirs(UI_DIR, exist_ok=True)
    records = []

    try:
        from agents.llm_agent import LLMAgent
        print("LLM agent available — generating live trajectories...\n")

        normal_agent   = LLMAgent(mode="normal")
        red_team_agent = LLMAgent(mode="red_team")

        print("Normal mode:")
        records += run_with_agent(normal_agent, NORMAL_SEEDS, "normal")

        print("\nRed-team mode:")
        records += run_with_agent(red_team_agent, RED_TEAM_SEEDS, "red_team")

    except EnvironmentError as e:
        print(f"LLM agent unavailable ({e})")
        print("Falling back to oracle baseline for sample data...\n")
        from agents.baseline import ScriptedOracleBaseline
        agent = ScriptedOracleBaseline()
        for seed in NORMAL_SEEDS + RED_TEAM_SEEDS:
            task = generate_task(seed)
            rec = run_episode(agent, task, agent_mode="oracle")
            records.append(rec)
            print(f"  [oracle] seed={seed:3d} cat={rec['category']:22s} "
                  f"reward={rec['completion_reward']:.2f} steps={rec['n_steps']}")

    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nWrote {len(records)} trajectories → {OUT_PATH}")
    print("Open ui/replay.html in your browser to explore them.")


if __name__ == "__main__":
    main()
