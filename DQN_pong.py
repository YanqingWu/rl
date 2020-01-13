import cv2
import gym
import time
import torch
import argparse
import gym.spaces
import numpy as np
import collections
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5
GAMMA = 0.99
BATCH_SIZE = 32
# experience buffer的最大容量
REPLAY_SIZE = 10000
# 开始游戏前的等待帧
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
# 每隔多少帧同步training model和target model
SYNC_TARGET_FRAMES = 1000
EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


"""
所有Wrapper只能重载一个step函数
"""


class FireResetEnv(gym.Wrapper):
    """
    在游戏开始的时候自动按开始健

    """

    def __init__(self, env: gym.Env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    # def step(self, action):
    #     self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    相邻的4帧采取相同的action

    """

    def __init__(self, env: gym.Env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def _reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    将图片resize为84*84的灰度图

    """

    def __init__(self, env: gym.Env):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, observation):
        return ProcessFrame84.process(observation)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."

        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        transformed_img = resized_screen[18:102, :]
        transformed_img = np.reshape(transformed_img, [84, 84, 1])
        return transformed_img.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    """
    将n_step帧作为一个observation

    """

    def __init__(self, env: gym.Env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())


class ImageToPyTorch(gym.ObservationWrapper):
    """
    hwc转换为chw

    """

    def __init__(self, env: gym.Env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaleFloatFrame(gym.ObservationWrapper):
    """
    0-255转换为 0-1

    """

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaleFloatFrame(env)
    return env


class DQN(nn.Module):
    """
    构建卷积神经网络

    """

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self._conv_out_size = self._get_conv_shape(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(self._conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_shape(self, input_shape):
        o = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x).view(-1, self._conv_out_size)
        return self.fc(x)


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    """
    从buffer中随机抽取一个batch的states， actions， rewards， dones, next_states

    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    """
    创建智能体

    """
    def __init__(self, env: gym.Env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0

    def play_step(self, net, epsilon=0, device='cpu'):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.observation_space.sample()
        else:
            state = np.array([self.state], copy=False)
            state = torch.tensor(state).to(device)
            q_vals = net(state)
            _, act = torch.max(q_vals, dim=1)
            action = int(act.item())
            new_state, reward, is_done, _ = self.env.step(action)
            self.total_reward += reward
            new_state = new_state
            exp = Experience(self.state, action, reward, is_done, new_state)
            self.exp_buffer.append(exp)
            self.state = new_state
            if is_done:
                done_reward = self.total_reward
                self._reset()
            return done_reward


def calculate_loss(batch, net, target_net, device="cpu"):
    """
    计算损失

    :param batch:
    :param net: training model，用来更新梯度
    :param target_net: target model，用来计算next_state的Q value, 每隔固定帧和training model同步
    :param device:
    :return:
    """
    states, actions, rewards, dones, next_states = batch
    states = torch.tensor(states).to(device)
    next_states = torch.tensor(next_states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool) .to(device)
    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net(next_states).max(1)[0]
    """ 对于每个episode的最后一个step不考虑next_state_value，否则不会收敛 """
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = rewards + next_state_values * GAMMA
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    print('using device: %s' % ('cuda' if args.cuda else 'cpu'))
    env = make_env(args.env)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    start_frame = 0
    start_time = time.time()
    best_mean_reward = None
    while True:
        env.render()
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - start_frame) / (time.time() - start_time)
            start_frame = frame_idx
            start_time = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "best.pth")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f > %.3f, model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                env.close()
                writer.close()
                break
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss = calculate_loss(batch, net, target_net, device=device)
        loss.backward()
        optimizer.step()

