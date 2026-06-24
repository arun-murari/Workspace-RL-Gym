import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ConstrainedEnv(gym.Env):
    N_EMAILS = 5
    N_SENDERS = 5
    N_KEYWORDS = 5
    EMAIL_FEATS = N_SENDERS + N_KEYWORDS + 1          
    TASK_FEATS = N_SENDERS + N_KEYWORDS + 1        
    OBS_DIM = TASK_FEATS + N_EMAILS * EMAIL_FEATS     

    # The whole environment we have currently is built around text but PPO needs numbers to operate so this ConstrainedEnv is the that translation. 
    # Converting that env into involves developing a retrieval task where the agent chooses the right email given conditions so this can be expressed
    # as a fixed length vector of numbers. There are 5 emails to choose with 3 properties, 5 possible senders, 5 possible keywords and whether it has an attachment.
    # So the observation dimension ends up being 66 because there are 11 ways to describe what the task wants 55 ways to describe each email possibility. 

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.N_EMAILS)
        self.rng = np.random.default_rng()
        self.target = None
        self.obs = None

    # The observation space is 66 floats between 0 and 1 indicating the 66 different dimensions of the obs space. Action space is just 5 discreate choices because
    # the policy is choosing which email is correct. Then self.rng is just a random number generator for building random tasks and then target and obs will be defined later
    # representing the correct email and the current observation. 

    def reset(self, seed=None, options=None):

        # The goal of this method is to built out the task, build 5 distinct emails and then encode it all into a vector for the policy to take.

        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Sets a seed for a specific task if there is none given

        task_sender = self.rng.integers(self.N_SENDERS)
        task_keyword = self.rng.integers(self.N_KEYWORDS)
        task_attach = self.rng.integers(2)

        # These are just random initilizations for the sender the keyword and whether the email has a file attached 

        emails = [{"sender": self.rng.integers(self.N_SENDERS),
                   "keyword": self.rng.integers(self.N_KEYWORDS),
                   "attach": self.rng.integers(2)} for _ in range(self.N_EMAILS)]

        # Then we build 5 emails here that also contain random values for its properties by doing self.rng.integers

        self.target = self.rng.integers(self.N_EMAILS)
        emails[self.target] = {"sender": task_sender, "keyword": task_keyword, "attach": task_attach}

        # We define self.target to be the correct choice out of the 5 emails by choosing a random one. Then we add a new key to the emails dict indicating this is the answer.

        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        obs[task_sender] = 1.0
        obs[self.N_SENDERS + task_keyword] = 1.0
        obs[self.N_SENDERS + self.N_KEYWORDS] = float(task_attach)

        # We define obs to initially be an array of zeros with size of the observation dimension. Then we set the corresponding values in the array to be 1 based on who
        # the sender is, what the keyword is, and whether it has an attachment or not by using one-hot encoding, we are defining these characteristics.

        base = self.TASK_FEATS

        for i, e in enumerate(emails):
            off = base + i * self.EMAIL_FEATS
            obs[off + e["sender"]] = 1.0
            obs[off + self.N_SENDERS + e["keyword"]] = 1.0
            obs[off + self.N_SENDERS + self.N_KEYWORDS] = float(e["attach"])

        # This portion essentially initializes the rest of the obs array to have their correct keyword, sender, attachment triplets by iterating through emails.

        self.obs = obs
        return obs, {}

        # Store this obs in self.obs and return empty info dict

    def step(self, action):
        reward = 1.0 if action == self.target else 0.0
        return self.obs, reward, True, False, {"target": self.target}   

        # The step function is merely the agent gueses which email is correct. If it guesses correctly the reward is 1 and if not it is 0. Then we return the observation
        # the reward, True for terminated because it only gets one guess. False for truncated because it can never run out of steps or anything and then we return the correct answer.

    