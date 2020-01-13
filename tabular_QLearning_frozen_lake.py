import gym
import collections
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


"""
强化学习在测试阶段也可以进行学习，可以大大加快训练速度，即在 play_episode 函数里面也进行Q迭代.
"""

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        """
        获取当前state下随机采取action，该action的reward，以及到达的new_state

        :return:
        """
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        """
        遍历当前state下的action，获取当前state下的最佳action

        :param state:
        :return:
        """

        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, state, action, reward, next_state):
        """
        更新当前state下采取action时的local reward :
        local_reward = reward + GAMMA * next_state_reward
        更新当前state下采取action的reward :
        reward = old_reward * (1-ALPHA) + local_reward * ALPHA

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return:
        """
        next_state_reward, _ = self.best_value_and_action(next_state)
        local_reward = reward + GAMMA * next_state_reward
        old_reward = self.values[(state, action)]
        self.values[(state, action)] = old_reward * (1 - ALPHA) + local_reward * ALPHA

    def play_episode(self, env: gym.Env):
        """
        计算一个周期下的总奖励

        :param env:
        :return:
        """
        total_reward = 0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            # 在测试阶段也进行学习，大大加快了训练速度
            self.value_update(state, action, reward, new_state)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")
    steps = 0
    best_reward = 0
    rewards = []
    while True:
        steps += 1
        old_state, action, reward, new_state = agent.sample_env()
        agent.value_update(old_state, action, reward, new_state)
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(env)
        reward = reward / TEST_EPISODES
        rewards.append(reward)
        writer.add_scalar("reward", reward, steps)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % steps)
            break
    plt.plot(range(len(rewards)), rewards)
    plt.show()
    writer.close()

