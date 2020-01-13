import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake8x8-v0"
GAMMA = 0.9
TEST_EPISODES = 20

"""
Q-learning 比value iteration更高效
"""


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        # 记录state a, action, state b的奖励
        self.rewards = collections.defaultdict(float)
        # 记录state a, action到不同其他不同state的转移概率
        self.transits = collections.defaultdict(collections.Counter)
        # 记录每个state，采取不同action的奖励
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        """
        在每一次迭代开始前重新初始化奖励table和转移概率table.

        :param count: 初始化的次数
        :return:
        """
        for i in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def select_action(self, state):
        """
        在该state 选择最佳的action

        :param state:
        :return:
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_action is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        """
        注意此处，把环境传进来是因为，main里面每次迭代环境没有reset，环境状态没有发生变化，reset会随机生成一个新的状态
        """

        """
        根据每一周期：
        更新state a, action, state b的奖励，
        更新state a, action到不同其他不同state的转移概率

        :param env:
        :return:
        """
        total_reward = 0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        """
        Q-learning 保存每个state，采取不同action的奖励

        :return:
        """
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for target_state, count in target_counts.items():
                    reward = self.rewards[(state, action, target_state)]
                    best_action = self.select_action(target_state)
                    action_value += (count / total) * (reward + GAMMA * self.values[(target_state, best_action)])
                self.values[(state, action)] = action_value


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="Q-learning")
    iters = 0
    best_reward = 0.0
    while True:
        iters += 1
        # 在每一次迭代开始前重新初始化奖励table和转移概率table.
        agent.play_n_random_steps(100)
        # 在每一次迭代开始前，首先计算每个state的奖励
        agent.value_iteration()
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iters)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iters)
            break
    writer.close()