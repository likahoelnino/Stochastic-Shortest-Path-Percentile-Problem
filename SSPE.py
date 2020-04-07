# -----------------------------------------------------------
# Functions / modules of Stochastic Shortest Paths Problem
#
# -----------------------------------------------------------


from my_env import *
import numpy as np
import copy


def step(state, action):
    reward = state_2_action[state][action]
    observation = np.random.choice(list(action_2_state[action].keys()),
                                   p=list(action_2_state[action].values()))
    done = (observation == end)
    return observation, reward, done


def generate_action(state, policy, epsilon):
    if np.random.rand() > epsilon:
        action = policy[state]
    else:
        action = np.random.choice(list(state_2_action[state]))
    return action


def vi(gamma: float = 1.,
       episodes: int = 50,
       ):
    """
    value iteration
    :param gamma: discount factor
    :param episodes: runs of iterations
    :return: values, policy
    """
    v = {}
    for k in state_2_action:
        v.update({k: 0})

    pi = {}
    for k in state_2_action:
        pi.update({k: np.random.choice(list(state_2_action[k].keys()))})

    count = copy.deepcopy(state_2_action)
    for k in count:
        for kk in count[k]:
            count[k][kk] = 0

    def bellman_op(values):
        v0 = {}
        pi0 = {}
        for i in state_2_action:
            pi0.update({i: np.random.choice(list(state_2_action[i].keys()))})
            v0.update({i: -np.inf})

        for state in state_2_action:
            for action in state_2_action[state]:
                v_temp = 0.
                for next_state in action_2_state[action]:
                    probability = action_2_state[action][next_state]
                    reward = state_2_action[state][action]
                    v_temp += (probability * (reward + gamma * values[next_state]))
                if v_temp > v0[state]:
                    v0[state] = v_temp
                    pi0[state] = action
        return v0, pi0

    for k in range(episodes):
        v, pi = bellman_op(v)

    return v, pi


def q_learn(alpha=0.01,
            gamma=1.,
            epsilon=1.,
            eps_min=0.1,
            eps_decade=0.9,
            episodes=50,
            verbose=1  # Integer. 0, 1, or 2. Verbosity mode
            ):
    q = {}
    for k in state_2_action:
        li = {}
        for kk in state_2_action[k]:
            li.update({kk: 0.})
        q.update({k: li})

    pi = {}
    for k in state_2_action:
        pi.update({k: np.random.choice(list(state_2_action[k].keys()))})

    count = copy.deepcopy(state_2_action)
    for k in count:
        for kk in count[k]:
            count[k][kk] = 0

    for e in range(episodes):
        state = start
        total_reward = 0
        for t in range(1000):
            action = generate_action(state, pi, epsilon)
            next_state, reward, done = step(state, action)
            count[state][action] += 1
            total_reward += reward
            q[state][action] += alpha * (reward + gamma * max(q[next_state].values()) - q[state][action])
            pi[state] = max(q[state], key=q[state].get)
            if verbose > 1:
                if (t + 1) % 10 == 0:
                    print("\n{}-{}-{},".format(state, action, next_state), end=' ')
                else:
                    print("{}-{}-{},".format(state, action, next_state), end=' ')
            state = next_state
            if done:
                if verbose > 0:
                    print("\nEpisode finished after {} timesteps with reward {}".format(t + 1, total_reward))
                break
        epsilon = max(eps_min, epsilon * eps_decade)
    return q, pi


def monte_carlo(gamma=1.,
                epsilon=1.,
                eps_min=0.1,
                eps_decade=0.9,
                episodes=50,
                verbose=1  # Integer. 0, 1, or 2. Verbosity mode
                ):
    q = {}
    for k in state_2_action:
        li = {}
        for kk in state_2_action[k]:
            li.update({kk: 0})
        q.update({k: li})

    pi = {}
    for k in state_2_action:
        pi.update({k: np.random.choice(list(state_2_action[k].keys()))})

    count = copy.deepcopy(state_2_action)
    for k in count:
        for kk in count[k]:
            count[k][kk] = 0

    for e in range(episodes):
        state = start
        total_reward = 0
        trajectory = []
        for t in range(1000):
            action = generate_action(state, pi, epsilon)
            next_state, reward, done = step(state, action)
            total_reward += reward
            trajectory.append([state, action, reward])
            count[state][action] += 1
            if verbose > 1:
                if (t + 1) % 10 == 0:
                    print("\n{}-{}-{},".format(state, action, next_state), end=' ')
                else:
                    print("{}-{}-{},".format(state, action, next_state), end=' ')
            state = next_state
            if done:
                if verbose > 0:
                    print("\nEpisode finished after {} timesteps with reward {}".format(t + 1, total_reward))
                break
        g = 0
        state_list, action_list, reward_list = list(zip(*trajectory))
        for tt in reversed(range(t + 1)):
            g = gamma * g + reward_list[tt]
            q[state_list[tt]][action_list[tt]] += ((g - q[state_list[tt]][action_list[tt]]) /
                                                   count[state_list[tt]][action_list[tt]])
            pi[state_list[tt]] = max(q[state_list[tt]], key=q[state_list[tt]].get)

        epsilon = max(eps_min, epsilon * eps_decade)
    return q, pi
