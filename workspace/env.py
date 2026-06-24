import gymnasium as gym
from gymnasium import spaces
from workspace.state import World, Task
from workspace.operations import execute_action
from tasks.checks import run_verifier

# We import gymnasium because it is what helps us build up the environment and the loop of actions. Spaces allows us 
# to define what the observation and action space to be where as the World and execute_action imports are things we already built

class WorkspaceEnv(gym.Env):
    def __init__(self, task: Task, step_budget=20):
        super().__init__()
        self.task = task
        self.step_budget = step_budget
        self.pristine_world = task.world
        self.world = None
        self.steps_taken = 0
        self.last_result = ""

        self.observation_space = spaces.Text(max_length=10000)
        self.action_space = spaces.Dict({"type": spaces.Text(max_length=50)})

    # This is the WorkSpace Env class which is how the world itself will work, contrasting to the World dataclass because that it is how
    # the world is structured. This is how it will run and how the agents execution function. When we start the environment we put in a task to run
    # and we also have a default step_budget of 20 steps. We define task to be the task given so the later functions can read it. Store the step budget
    # and save pristine world as the instance of the world before it is corrupted by the agent so we can use it in the reset. Then we declare world, steps_taken
    # and last_result to be None, 0, "" because they will change as the agent executes actions. Then finally we define the action and observation space to be 
    # a dict with a type and text respectively. 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world = self.pristine_world.snapshot()
        self.steps_taken = 0
        self.last_result = "(no actions taken yet)"
        return self._render_observation(), {}

    # This is the reset function it takes in an optional seed if you ever want to reproduce a specific world environment and it also takes an options dictionary
    # that allows for specific changes to be made to the environment like difficulty. Super().reset resets the environment according to the seed if it is given
    # and then steps taken is a counter where as last result is the text result of the last action. Then we return the initial self.render_observation and a info dict

    def step(self, action):
        result, reward, agent_called_done = execute_action(action, self.world)
        self.last_result = result
        self.steps_taken += 1

        terminated = agent_called_done
        truncated = self.steps_taken >= self.step_budget and not terminated

        if terminated or truncated:
            reward += self._compute_completion_reward()

        info = {
            "steps_taken": self.steps_taken,
            "agent_called_done": agent_called_done,
        }
        return self._render_observation(), reward, terminated, truncated, info

    # This is the step function that defines how the agent moves in the environment. We define the text result, reward and whether the agent is done from the 
    # return values of the operations and then we update self.last result to be the latest action taken as well as increase the steps taken counter. Then we
    # define terminated to be whetehr the agent did the done operation and we define truncated to be true if the steps taken is greater than the step budget. Then 
    # we change the reward to be the return of the compute completion reward based on the checks made throughout the task. Then we add some information to the 
    # info dictionary and return the important info as well as what the observation looks like using our method.

    def _render_observation(self) -> str:
        w = self.world
        inbox = [e for e in w.emails.values() if e.folder == "Inbox"]
        inbox_lines = "\n".join(
            f"  {e.email_id}: from={e.sender} | {e.subject}"
            + (f" [{len(e.attachments)} att]" if e.attachments else "")
            for e in inbox[:10]
        ) or "  (empty)"

        folder_lines = "\n".join(
            f"  {path} [{sum(1 for f in w.files.values() if f.folder_path == path)} file(s)]"
            for path in sorted(w.folders)
        ) or "  (no folders)"

        return (
            f"TASK: {self.task.instruction}\n"
            f"STEP: {self.steps_taken}/{self.step_budget}\n\n"
            f"INBOX:\n{inbox_lines}\n\n"
            f"DRIVE:\n{folder_lines}\n\n"
            f"LAST ACTION RESULT:\n  {self.last_result}"
        )

    # This is the render_observation function it is the text block of info that the agent sees when it works in this environment. We define w to be
    # the world itself and then we create a list of all the emails from the emails if it is in the inbox. Then we define inbox_lines to be the info for each
    # email and then we do the same with the folder lines to see all the files. Then we return all of this information for the agent to see.

    def _compute_completion_reward(self) -> float:
        result = run_verifier(self.task.goal_spec, self.world)
        return result["reward"]

    # This is the compute_completion reward function that updates the rewards of the task based on when we run the verification.
