import time

from gridworld import GridWorldEnv


actions = {
    0: 'Up',
    1: 'Right',
    2: 'Down',
    3: 'Left'
}

if __name__ == "__main__":
    # Create the environment
    env = GridWorldEnv()

    # Reset
    obs = env.reset()
    env.render()

    for i in range(50):
        action = env.action_space.sample()
        print(actions[action])
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs)
        env.render()

        # Uncomment this to enable slow motion mode
        time.sleep(3.0)
        if terminated:
            print('Reset environment')
            env.reset()
            env.render()
    env.close()
