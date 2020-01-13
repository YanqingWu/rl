import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


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
    从batch_size个周期里面选出总奖励比较高的周期，告诉agent哪些行动observation-action对具有比较高的奖励

    :param batch: batch_size个周期，每个周期保存了奖励以及每个迭代步的观察和行动
    :param percentile: 选择奖励较高的百分数阈值
    :return:
    """
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_torch = torch.FloatTensor(train_obs)
    train_act_torch = torch.LongTensor(train_act)
    return train_obs_torch, train_act_torch, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    for step, batch in enumerate(iterBatches(env, net, BATCH_SIZE)):
        env.render()
        # 将比较好的observation-action作为监督，训练神经网络
        obs, acts, reward_bound, reward_mean = filterBatch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores = net(obs)
        # action作为真实标签，网络输出作为预测
        loss = criterion(action_scores, acts)
        loss.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (step, loss.item(), reward_mean, reward_bound))
        if reward_mean > 199:
            print("Solved!")
            env.close()
            break







