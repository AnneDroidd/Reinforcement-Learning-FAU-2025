import numpy as np

from gridworld import GridWorldEnv
from util import value_function_policy_plot


class TDAgent:
    def __init__(self, env, discount_factor, learning_rate):
        self.env = env
        self.g = discount_factor
        self.lr = learning_rate

        self.num_actions = env.action_space.n

        # V[y, x] is value for grid position y, x, initialize to all zeros
        self.V = np.zeros(env.observation_space.high, dtype=np.float32)

        # uniform random policy[y, x, z], i.e. probability of action z when in grid position y, x is 1 / num_actions
        self.policy = np.ones((*env.observation_space.high, self.num_actions), dtype=np.float32) / self.num_actions

        # TODO 3: experiment with different (not fully random) policies
        #partially determinsitic policy:
        self.policy[1,0], self.policy[1, 0, 2]= 0.0, 1.0
        self.policy[2, 0], self.policy[2, 0, 2] = 0.0, 1.0  # Down with 100%
        self.policy[1, 1], self.policy[1, 1, 2] = 0.0, 1.0  # Down with 100%
        self.policy[2, 1], self.policy[2, 1, 2] = 0.0, 1.0  # Down with 100%

    def action(self, s):
        return np.random.choice(np.arange(self.num_actions), p=self.policy[*s])
        # TODO 2: Sample action following the policy
        #return self.env.action_space.sample()  # random action

    def learn(self, n_timesteps=50000):
        s, _ = self.env.reset()

        for i in range(n_timesteps):
            # TODO 1: Implement the agent-interaction loop
            # You will have to call self.update(...) at every step
            # Do not forget to reset the environment if you receive a 'terminated' signal

            #Select action based on policy
            a = self.action(s)

            #Observe next state + reward
            s_, r, done, _ , _ = self.env.step(a)

            #Update state
            self.update(s, r, s_)

            #Update state
            s = s_

            #Reset if state is terminated
            if done:
                s, _ = self.env.reset()

    def update(self, s, r, s_):
        # TODO 1: Implement the TD estimation update rule
        # V(s) <- V(s) + alpha * (r + gamma * V(s') - V(s))
        self.V[*s] = self.V[*s] + self.lr * (r + self.g * self.V[*s_] - self.V[*s])


if __name__ == "__main__":
    # Create Agent and environment
    td_agent = TDAgent(GridWorldEnv(), discount_factor=0.9, learning_rate=0.01)

    # Learn the state-value function for 100000 steps
    td_agent.learn(n_timesteps=100000)

    # Visualize V
    value_function_policy_plot(td_agent.V, td_agent.policy, td_agent.env.map)
