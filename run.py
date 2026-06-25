import argparse
import subprocess
import sys
import os
from harness.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Workspace RL Gym — run the baseline evaluation (and optionally RL training).")

    parser.add_argument("--train-seeds", type=int, default=100, help="number of seen task seeds to evaluate (default 100)")
    parser.add_argument("--held-out-seeds", type=int, default=50, help="number of held-out task seeds to evaluate (default 50)")
    parser.add_argument("--rl", action="store_true", help="also run the PPO training on the constrained env (slower)")
    args = parser.parse_args()

    print("Running baseline evaluation across the task suite...\n")
    evaluate(train_seeds=range(0, args.train_seeds), held_out_seeds=range(800, 800 + args.held_out_seeds))

    if args.rl:
        print("\nRunning PPO training on the constrained env...\n")
        ppo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl", "ppo.py")
        subprocess.run([sys.executable, ppo_path], check=True)

if __name__ == "__main__":
    main()