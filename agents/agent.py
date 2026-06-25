class Agent:
    def reset(self, task):
        pass

    def act(self, observation, task):
        raise NotImplementedError

# This is the Agent interface, essentially a rulebook for how any agent that wishes to run on my environment will function. When
# we have the evaluate.py file running an agent through the environment, the agent has two functions, the reset function's job
# is called at the start of an episode and lets the agent prepare but its placeholder is just a pass that means the agent has
# nothing to prepare. Every agent that runs through the environment can implement the reset function so that it can plan
# what it will do and if not the function does nothing. This is opposite to how the act function works because if not implemented
# this function raises an error. This function is the heart of the agents loop through the environment, it is repeatedly called
# and takes in what the agent sees as well as the task it is trying to complete. 