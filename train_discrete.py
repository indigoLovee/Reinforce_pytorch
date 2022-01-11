import gym
import numpy as np
import argparse
from utils import plot_learning_curve
from Reinforce_discrete import Reinforce

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=3000)
parser.add_argument('--reward_path', type=str, default='./output_images/reward.png')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/Reinforce_discrete/')

args = parser.parse_args()


def main():
    env = gym.make('LunarLander-v2')
    agent = Reinforce(alpha=0.0005, state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.n, fc1_dim=256, fc2_dim=256,
                      ckpt_dir=args.ckpt_dir, gamma=0.99)
    total_rewards, avg_rewards = [], []

    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_reward(reward)
            total_reward += reward
            observation = observation_

        agent.learn()
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        print('EP:{} reward:{} avg_reward:{}'.format(episode + 1, total_reward, avg_reward))

        if (episode + 1) % 300 == 0:
            agent.save_models(episode + 1)

    episodes = [i for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)


if __name__ == '__main__':
    main()
