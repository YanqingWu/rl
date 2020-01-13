import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from tensorboardX import SummaryWriter

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9


"""
对与frozen lake，该情景下，大部分周期没有奖励信息，需要将有奖励的周期保存起来，奖励应该和走的步数成反比.
此外还应该将奖励靠前的周期保存起来反复利用.
PERCENTILE不能太大，太大会导致每个batch里面包含的周期实例太少.
"""


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    """
    对于离散状态，需要进行onehot，才能送入神经网络
    """
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n,), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1
        return res


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterBatches(env, net, batch_size):
    """
    获取batch_size个周期

    :param env: gym env
    :param net: Net
    :param batch_size: num episodes
    :return:
    """
    batch = []
    episode_reward = 0
    episode_steps = []
    # 获取初始状态
    obs = env.reset()
    softmax = nn.Softmax(dim=1)

    while True:
        # 转化为torch tensor
        obs_torch = torch.FloatTensor([obs])
        # 根据当前观察值获取每个action的概率
        action_probs_torch = softmax(net(obs_torch))
        action_probs = action_probs_torch.data.numpy()[0]
        # 根据每个action的概率选择action
        action = np.random.choice(len(action_probs), p=action_probs)
        # 获取采取该action之后的观察值，奖励，周期是否结束及其他信息
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        # 将每次观察值采取的行动保存
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if is_done:
            # 如果周期结束了，保存该周期每一步的信息，以及该周期的奖励信息
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            # 重新初始化周期
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs


def filterBatch(batch, percentile):
    """
    从batch_size个周期里面选出总奖励比较高的周期，告诉agent哪些行动observation-action对具有比较高的奖励.
    考虑到对于该情景下，大部分周期没有奖励信息，需要将有奖励的周期保存起来，奖励应该和走的步数成反比.

    :param batch: batch_size个周期，每个周期保存了奖励以及每个迭代步的观察和行动
    :param percentile: 选择奖励较高的百分数阈值
    :return:
    """
    # 计算折扣后的奖励
    disc_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)
    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        """对于奖励"小于等于"奖励边界的不考虑"""
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    train_obs_torch = torch.FloatTensor(train_obs)
    train_act_torch = torch.LongTensor(train_act)
    return elite_batch, train_obs_torch, train_act_torch, reward_bound


if __name__ == "__main__":
    random.seed(12345)
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v0"))
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)
    writer = SummaryWriter(comment="-frozenlake-tweaked")

    # 重复利用奖励靠前的周期
    full_batch = []
    for step, batch in enumerate(iterBatches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_bound = filterBatch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue
        full_batch = full_batch[-500:]
        optimizer.zero_grad()

        action_scores = net(obs)
        loss = objective(action_scores, acts)
        loss.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (
            step, loss.item(), reward_mean, reward_bound, len(full_batch)))
        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("reward_mean", reward_mean, step)
        writer.add_scalar("reward_bound", reward_bound, step)
        if reward_mean > 0.8:
            print("Solved!")
            break
    writer.close()
