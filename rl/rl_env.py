import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from workspace.state import World, Email
from tasks.checks import run_verifier
from tasks.content import EMAIL_CONTENT

# We need these sys and os imports at the top because this file lives in the rl folder but it needs to reach up into the workspace and tasks folders to grab the real
# World, Email, run_verifier and EMAIL_CONTENT. The sys.path.insert line adds the project root to where python looks for imports so these cross folder imports actually work.
# The whole point of importing these is so this constrained env is not a separate made up thing, it actually builds a real World out of our real email content and scores
# through our real verifier.

CONTACT_ADDRS = ["sarah@company.com", "bob@company.com", "carol@company.com",
                 "david@company.com", "eve@company.com"]

# This is just the pool of 5 possible sender addresses we assign to the emails. We have 5 because N_SENDERS is 5 and each email gets a distinct one.

class ConstrainedEnv(gym.Env):
    N_EMAILS = 5
    N_SENDERS = 5
    N_KEYWORDS = 5
    EMAIL_FEATS = N_SENDERS + N_KEYWORDS + 1
    TASK_FEATS = N_SENDERS + N_KEYWORDS + 1
    OBS_DIM = TASK_FEATS + N_EMAILS * EMAIL_FEATS

    # Our full environment is built around text but PPO needs numbers, so this ConstrainedEnv is that translation. We take a slice of our real env, specifically the retrieval
    # task where the agent picks the right email out of 5 given a description, and express it as a fixed length vector. There are 5 emails each with 3 properties, 5 possible
    # senders, 5 possible keywords and whether it has an attachment. So the obs dimension ends up being 66 because there are 11 ways to describe what the task wants and
    # 55 ways to describe the 5 email possibilities. The difference from before is these emails are now real emails from our content library, not random numbers.

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.N_EMAILS)
        self.py_rng = random.Random()
        self.world = None
        self.task_goal = None
        self.email_ids = None
        self.target = None
        self.obs = None

    # The observation space is 66 floats between 0 and 1 indicating the 66 dimensions of the obs space. Action space is just 5 discrete choices because the policy is choosing
    # which email is correct. Then py_rng is a random generator for building the tasks and the rest, world, task_goal, email_ids, target and obs get defined later. world is the
    # real World object we build, task_goal is the goal_spec for the retrieval task and email_ids holds the real emails we made so step can look them up.

    def _build_world(self, seed):

        # The goal of this method is to build a real World populated with 5 distinct emails pulled from our actual EMAIL_CONTENT library. The reason we make them distinct in
        # both sender and content is so the retrieval problem stays linearly separable, meaning each email is clearly different from the others so the agent can actually learn to
        # tell them apart. If two emails were identical the target would be ambiguous and the policy could not converge.

        rng = random.Random(seed)
        world = World()
        world.folders = {"/"}

        # We make a fresh rng off the seed and define world to be a new World object with just the root folder.

        entries = rng.sample(EMAIL_CONTENT, self.N_EMAILS)
        senders = rng.sample(range(self.N_SENDERS), self.N_EMAILS)

        # entries is 5 distinct content templates from EMAIL_CONTENT using rng.sample so there are no duplicates, and senders is 5 distinct sender indices the same way. Sampling
        # without replacement on both is what guarantees the emails are all distinguishable.

        email_ids = []
        for i, entry in enumerate(entries):
            eid = world.next_id("email")
            world.emails[eid] = Email(
                email_id=eid,
                sender=CONTACT_ADDRS[senders[i]],
                subject=entry["subject"],
                body=entry["body"],
                thread_id=eid,
                recipients=["me@company.com"],
                folder="Inbox",
            )
            email_ids.append((eid, entry, senders[i]))
        return world, email_ids

        # Then for each of the 5 content entries we make a real Email with a fresh id, the distinct sender, and the real subject and body from the content library, and add it to
        # the world. We keep a list of tuples in email_ids holding the id, the entry, and the sender index so that later in reset and step we can encode them and look up their
        # facts. Then we return the world and that list.

    def reset(self, seed=None, options=None):

        # The goal of reset is to build the real world, pick which email is the target and which fact we are asking for, frame it as a retrieval_fact task with a real goal_spec,
        # and then encode the whole thing into the 66 dim vector for the policy.

        super().reset(seed=seed)
        if seed is not None:
            self.py_rng = random.Random(seed)
        s = self.py_rng.randint(0, 10_000_000)

        # Sets a seed for the task if one is given, then we draw a sub seed s off it that we use to build the world so each episode gets a fresh world.

        self.world, email_ids = self._build_world(s)
        self.email_ids = email_ids

        # We build the real world and store the email list.

        self.target = self.py_rng.randint(0, self.N_EMAILS - 1)
        target_eid, target_entry, target_sender = email_ids[self.target]
        self.fact_key = self.py_rng.choice(list(target_entry["facts"].keys()))
        self.target_fact = target_entry["facts"][self.fact_key]
        target_keyword = EMAIL_CONTENT.index(target_entry) % self.N_KEYWORDS

        # We pick a random target email out of the 5 and pull out its id, entry and sender. Then we choose a random fact from that email's facts dict to be the thing we ask for,
        # and target_fact is the actual value of that fact. target_keyword is a stable id for the target's content, we get it from the position of the entry in the content library
        # mod the number of keywords so it always maps the same way.

        self.task_goal = [
            {"check": "answer_matches", "kind": "positive", "params": {"expected": self.target_fact}},
        ]

        # This is the actual goal_spec, the exact same kind a retrieval_fact task from our generator would have. It is one positive answer_matches check expecting the target fact.
        # This is the part that makes the reward run through our real verifier instead of a made up comparison.

        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        obs[target_sender] = 1.0
        obs[self.N_SENDERS + target_keyword] = 1.0
        obs[self.N_SENDERS + self.N_KEYWORDS] = 0.0

        # We define obs to be an array of zeros the size of the obs dimension. Then we one hot encode the task descriptor, setting the slot for the target sender to 1, the slot
        # for the target keyword to 1, and the attachment flag to 0. This is telling the policy what email it is looking for.

        base = self.TASK_FEATS
        for i, (eid, entry, sender_idx) in enumerate(email_ids):
            kw = EMAIL_CONTENT.index(entry) % self.N_KEYWORDS
            off = base + i * self.EMAIL_FEATS
            obs[off + sender_idx] = 1.0
            obs[off + self.N_SENDERS + kw] = 1.0
            obs[off + self.N_SENDERS + self.N_KEYWORDS] = 0.0

        # This portion encodes the 5 actual emails into the rest of the obs array. For each email we get its keyword id the same stable way, figure out the offset into the array,
        # and set the one hot slots for its sender and keyword. This is the encode step, it is the one place where the real text emails get turned into numbers the policy can use.

        self.obs = obs
        return obs, {}

        # Store the obs in self.obs and return it with an empty info dict.

    def step(self, action):

        # In step the agent has guessed which email is correct. We take that choice, report that email's fact into world.done_answer the same way our done operation would, and
        # then score it through the real verifier. If the agent picked the target the reported fact matches the expected one so the verifier returns 1, otherwise it returns 0.

        chosen_eid, chosen_entry, _ = self.email_ids[action]

        # We look up the email the agent actually chose using the action as an index.

        self.world.done_answer = chosen_entry["facts"].get(self.fact_key, "")

        # The key point is that correctness is decided entirely by the verifier, not by us. We hand it the chosen email's fact and run_verifier compares it to the expected
        # target fact via answer_matches. If the agent chose the right email those match and reward is 1, otherwise they differ and reward is 0. The reward is genuinely
        # produced by our real verifier, not a hand written comparison.

        reward = run_verifier(self.task_goal, self.world)["reward"]
        return self.obs, reward, True, False, {"target": self.target}

        # The reward is computed by our real run_verifier checking the goal_spec against the world, not a hand written comparison. Then we return the obs, the reward, True for
        # terminated because the agent only gets one guess, False for truncated since it can never run out of steps, and the correct answer in the info dict.