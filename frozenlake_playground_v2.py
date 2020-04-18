# -----------------------------------------------------------
# Using SSPP algorithm to solve FrozenLake problem
#
# -----------------------------------------------------------


from frozenlake import FrozenLakeAgent, QLearnAgent, VIAgent
import gym

ENV_MAP = 'FrozenLake-v0'
# ENV_MAP = 'FrozenLake8x8-v0'

for param in [100, 100, 100, 100]:
    agent = VIAgent(big_map=(True if ENV_MAP == 'FrozenLake8x8-v0' else False),
                    episodes=param)
    pi = agent.get_policy()
    policy = [max(0, j) for j in pi]

    success = 0
    fail = 0
    timeout = 0
    env = gym.make(ENV_MAP)
    for _ in range(1000):
        env.reset()
        s = 0
        steps = 0
        while True:
            steps += 1
            # env.render()
            a = policy[s]
            s, r, done, info = env.step(a)  # take action
            if done:
                if r == 1:
                    success += 1
                    # print('move to the goal after {} steps'.format(steps))
                elif steps < (200 if ENV_MAP == 'FrozenLake8x8-v0' else 100):
                    fail += 1
                    # print('move to {} after {} steps'.format(s, steps))
                else:
                    timeout += 1
                break
    env.close()

    print("Learn with {} episodes -- "
          "success: {} / fail: {} / timeout: {}".format(param, success, fail, timeout))
    for i in range(8 if ENV_MAP == 'FrozenLake8x8-v0' else 4):
        for j in range(8 if ENV_MAP == 'FrozenLake8x8-v0' else 4):
            s = i * (8 if ENV_MAP == 'FrozenLake8x8-v0' else 4) + j
            if s == (63 if ENV_MAP == 'FrozenLake8x8-v0' else 15):
                print('G', end='')
                break
            print((pi[s] if pi[s] > -1 else 'X'), end='')
        print()
    print('----------------')
