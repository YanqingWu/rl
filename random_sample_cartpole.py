import gym
import time

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    total_reward = 0
    total_step = 0
    obs = env.reset()

    while True:
        time.sleep(0.2)
        env.render()
        # action: x coordinate, speed, angle, angle speed
        action = env.action_space.sample()
        # obs: observation, reward, done flag, extra info
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_step += 1
        if done:
            env.close()
            break

    print('Episode done in %d step, total reward %4.f' % (total_step, total_reward))