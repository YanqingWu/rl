import os
import csv
import gym
import sys
import time
import enum
import glob
import ptan
import torch
import argparse
import numpy as np
import collections
import torch.nn as nn
from gym import wrappers
import torch.optim as optim
from gym.utils import seeding
from tensorboardX import SummaryWriter


"""
一个完整的强化学习算法:
Agent:
    action
环境(state):
    observation
    reward
"""


DEFAULT_COMMISSION_PERC = 0.1
BATCH_SIZE = 32
BARS_COUNT = 10
TARGET_NET_SYNC = 1000
DEFAULT_STOCKS = "stock/YNDX_160101_161231.csv"
DEFAULT_VAL_STOCKS = "stock/YNDX_150101_151231.csv"

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

REWARD_STEPS = 2

LEARNING_RATE = 0.0001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 1000000

CHECKPOINT_EVERY_STEP = 1000000
VALIDATION_EVERY_STEP = 100000


class DataLoader:
    Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])

    def __init__(self):
        pass
        # self.Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])

    def _read_csv(self, file_name, sep=',', filter_data=True, fix_open_price=False):
        """
        读取csv文件，并且去除在该时间段内，[Open, High, Low, Close] 没有发生变化的row.
        并且记录有多少开仓数据和上一个时间点的闭仓数据发生了变化.


        :param file_name:
        :param sep:
        :param filter_data:
        :param fix_open_price:
        :return:
        """
        print("Reading", file_name)
        with open(file_name, 'rt', encoding='utf-8') as fd:
            reader = csv.reader(fd, delimiter=sep)
            h = next(reader)
            if '<OPEN>' not in h and sep == ',':
                return self._read_csv(file_name, ';')
            indices = [h.index(s) for s in ('<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>')]
            o, h, l, c, v = [], [], [], [], []
            count_out = 0
            count_filter = 0
            count_fixed = 0
            prev_vals = None
            for row in reader:
                vals = list(map(float, [row[idx] for idx in indices]))
                # 去除在该时间段内，[Open, High, Low, Close] 没有发生变化的row.
                if filter_data and all(map(lambda v: abs(v - vals[0]) < 1e-8, vals[:-1])):
                    count_filter += 1
                    continue
                po, ph, pl, pc, pv = vals

                if fix_open_price and prev_vals is not None:
                    ppo, pph, ppl, ppc, ppv = prev_vals
                    # 记录有多少开仓数据发生了变化.
                    if abs(po - ppc) > 1e-8:
                        count_fixed += 1
                        po = ppc
                        pl = min(pl, po)
                        ph = max(ph, po)
                count_out += 1
                o.append(po)
                c.append(pc)
                h.append(ph)
                l.append(pl)
                v.append(pv)
                prev_vals = vals
        print("Read done, got %d rows, %d filtered, %d open prices adjusted" % (
            count_filter + count_out, count_filter, count_fixed))
        return self.Prices(open=np.array(o, dtype=np.float32),
                           high=np.array(h, dtype=np.float32),
                           low=np.array(l, dtype=np.float32),
                           close=np.array(c, dtype=np.float32),
                           volume=np.array(v, dtype=np.float32))

    def _prices_to_relative(self, prices):
        """
        将绝对增减转化为相对增减.

        :param prices:
        :return:
        """
        assert isinstance(prices, self.Prices)
        rh = (prices.high - prices.open) / prices.open
        rl = (prices.low - prices.open) / prices.open
        rc = (prices.close - prices.open) / prices.open
        return self.Prices(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)

    @classmethod
    def load_relative(cls, csv_file=None):
        """
        加载相对数据.

        :param csv_file:
        :return:
        """
        return cls._prices_to_relative(cls, cls._read_csv(cls, csv_file))


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self, bars_count, commission_perc, reset_on_close, reward_on_close=True, volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices, offset):
        """
        重置State

        :param prices:
        :param offset:
        :return:
        """
        assert isinstance(prices, DataLoader().Prices)
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        """
        一个State的shape

        :return:
        """
        if self.volumes:
            return 4 * self.bars_count + 1 + 1,
        else:
            return 3 * self.bars_count + 1 + 1,

    def encode(self):
        """
        将State转化为np.ndarray

        :return:
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count + 1, 1):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        return res

    def _cur_close(self):
        """
        返回当前退市的价格

        :return:
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        """
        返回当前操作的奖励

        :param action:
        :return:
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

        return reward, done


class State1D(State):
    """
    将row data转化为col data，方便进行1D卷积操作

    """
    @property
    def shape(self):
        if self.volumes:
            return 6, self.bars_count
        else:
            return 5, self.bars_count

    def encode(self):
        """
        重写encode

        :return:
        """
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        if self.volumes:
            res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / self.open_price
        return res


class StocksEnv(gym.Env):
    meta_data = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=BARS_COUNT, commission=DEFAULT_COMMISSION_PERC,
                 reset_on_close=True, state_1d=True, random_ofs_on_reset=True, reward_on_close=True,
                 volumes=False):
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                  volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                volumes=volumes)

        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        """
        重置环境

        :return:
        """
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx):
        """
        进行一个action返回对应的observation，reward，done，info

        :param action_idx:
        :return:
        """
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: DataLoader.load_relative(file) for file in glob.glob(data_dir + '/*.csv')}
        return StocksEnv(prices, **kwargs)


class SimpleFFDQN(nn.Module):
    """
    只有全连接，没有进行其他任何优化的base dqn

    """
    def __init__(self, obs_len, actions_n):
        super(SimpleFFDQN, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


class DQNConv1D(nn.Module):
    """
    base conv1d dqn: Dueling + Double + two step
    """

    def __init__(self, shape, actions_n):
        super(DQNConv1D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)


class DQNConv1DLarge(nn.Module):
    """
    large conv1d dqn: Dueling + Double + two step
    """

    def __init__(self, shape, actions_n):
        super(DQNConv1DLarge, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)


def validation_run(env, net, episodes=100, device="cpu", epsilon=0.02, comission=0.1):
    stats = {
        'episode_reward': [],
        'episode_steps': [],
        'order_profits': [],
        'order_steps': [],
    }

    for episode in range(episodes):
        obs = env.reset()

        total_reward = 0.0
        position = None
        position_steps = None
        episode_steps = 0

        while True:
            obs_v = torch.tensor([obs]).to(device)
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = Actions(action_idx)

            close_price = env._state._cur_close()

            if action == Actions.Buy and position is None:
                position = close_price
                position_steps = 0
            elif action == Actions.Close and position is not None:
                profit = close_price - position - (close_price + position) * comission / 100
                profit = 100.0 * profit / position
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)
                position = None
                position_steps = None

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                    profit = close_price - position - (close_price + position) * comission / 100
                    profit = 100.0 * profit / position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)
                break

        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    return {key: np.mean(vals) for key, vals in stats.items()}


class RewardTracker:
    def __init__(self, writer, stop_reward, group_rewards=1):
        self.writer = writer
        self.stop_reward = stop_reward
        self.reward_buf = []
        self.steps_buf = []
        self.group_rewards = group_rewards

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        self.total_steps = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward_steps, frame, epsilon=None):
        reward, steps = reward_steps
        self.reward_buf.append(reward)
        self.steps_buf.append(steps)
        if len(self.reward_buf) < self.group_rewards:
            return False
        reward = np.mean(self.reward_buf)
        steps = np.mean(self.steps_buf)
        self.reward_buf.clear()
        self.steps_buf.clear()
        self.total_rewards.append(reward)
        self.total_steps.append(steps)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        mean_steps = np.mean(self.total_steps[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards)*self.group_rewards, mean_reward, mean_steps, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        self.writer.add_scalar("steps_100", mean_steps, frame)
        self.writer.add_scalar("steps", steps, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_actions = net(next_states_v).max(1)[1]
    next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--data", default=DEFAULT_STOCKS, help="Stocks file or dir to train on, default=" + DEFAULT_STOCKS)
    parser.add_argument("--valdata", default=DEFAULT_VAL_STOCKS, help="Stocks data for validation, default=" + DEFAULT_VAL_STOCKS)
    parser.add_argument("-r", "--run", default='logs', help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join("saves", args.run)
    os.makedirs(saves_path, exist_ok=True)

    if os.path.isfile(args.data):
        stock_data = {"YNDX": DataLoader.load_relative(args.data)}
        env = StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False, volumes=False)
        env_tst = StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
    elif os.path.isdir(args.data):
        env = StocksEnv.from_dir(args.data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
        env_tst = StocksEnv.from_dir(args.data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
    else:
        raise RuntimeError("No data to train on")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    val_data = {"YNDX": DataLoader.load_relative(args.valdata)}
    env_val = StocksEnv(val_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)

    writer = SummaryWriter(comment="-simple-" + args.run)
    net = SimpleFFDQN(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # main training loop
    step_idx = 0
    eval_states = None
    best_mean_val = None

    with RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
        while True:
            step_idx += 1
            buffer.populate(1)
            selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)

            if len(buffer) < REPLAY_INITIAL:
                continue

            if eval_states is None:
                print("Initial buffer populated, start training")
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            if step_idx % EVAL_EVERY_STEP == 0:
                mean_val = calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, step_idx)
                if best_mean_val is None or best_mean_val < mean_val:
                    if best_mean_val is not None:
                        print("%d: Best mean value updated %.3f -> %.3f" % (step_idx, best_mean_val, mean_val))
                    best_mean_val = mean_val
                    torch.save(net.state_dict(), os.path.join(saves_path, "mean_val-%.3f.data" % mean_val))

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_v = calc_loss(batch, net, tgt_net.target_model, GAMMA ** REWARD_STEPS, device=device)
            loss_v.backward()
            optimizer.step()

            if step_idx % TARGET_NET_SYNC == 0:
                tgt_net.sync()

            if step_idx % CHECKPOINT_EVERY_STEP == 0:
                idx = step_idx // CHECKPOINT_EVERY_STEP
                torch.save(net.state_dict(), os.path.join(saves_path, "checkpoint-%3d.data" % idx))

            if step_idx % VALIDATION_EVERY_STEP == 0:
                res = validation_run(env_tst, net, device=device)
                for key, val in res.items():
                    writer.add_scalar(key + "_test", val, step_idx)
                res = validation_run(env_val, net, device=device)
                for key, val in res.items():
                    writer.add_scalar(key + "_val", val, step_idx)


