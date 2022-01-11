import gym


def main():
    env = gym.make('MountainCar-v0')
    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.high)
    print(env.action_space.low)


if __name__ == '__main__':
    main()
