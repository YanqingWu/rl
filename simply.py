import random

class Environment:
    """
    环境需要给agent对环境的观察，agent可以采取的行动，agent采取行动后获得的奖励
    """
    def __init__(self):
        self.step_left = 10

    def get_observations(self):
        return [0, 0, 0]

    def get_actions(self):
        return [0, 1]

    def is_done(self):
        return self.step_left == 0

    def action(self, action):
        if self.is_done():
            raise Exception('Game Over')
        self.step_left -= 1
        return random.random()

class Agent:
    """
    agent需要记录自己获得的奖励，采取行动的策略，
    """
    def __init__(self):
        self.total_reward = 0

    def take_action(self, actions):
        return random.choice(actions)

    def step(self, env: Environment):
        current_obs = env.get_observations()
        actions = env.get_actions()
        reward = env.action(self.take_action(actions))
        self.total_reward += reward


if __name__ == '__main__':
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)
    print('total reward is : %s' % agent.total_reward)
