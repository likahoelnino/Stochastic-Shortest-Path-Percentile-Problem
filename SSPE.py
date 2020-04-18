# -----------------------------------------------------------
# Functions / modules of Stochastic Shortest Paths Problem
#
# -----------------------------------------------------------


import numpy as np
from mdp import MDP

start: object
end: object
mdp: MDP


def step(state, action):
    observation = np.random.choice(list(mdp.get_actions(state)[action]),
                                   p=list(mdp.get_actions(state)[action].values()))
    reward = mdp.get_weight(state, action, observation)
    done = (observation == end)
    return observation, reward, done


def generate_action(state, policy, epsilon):
    if np.random.rand() > epsilon:
        action = policy[state]
    else:
        action = np.random.choice(list(mdp.get_actions(state)))
    return action


def vi(markov: MDP,
       initial_state,
       terminal_state,
       gamma: float = 1.,
       episodes: int = 50,
       ):
    """
    value iteration
    :param markov: Markov Decision Process instance
    :param initial_state: initial point
    :param terminal_state: terminal point
    :param gamma: discount factor
    :param episodes: runs of iterations
    :return: values, policy
    """
    global start
    global end
    global mdp

    start = initial_state
    end = terminal_state
    mdp = markov

    v = {}
    for k in mdp.get_states():
        v[k] = 0

    pi = {}
    for k in mdp.get_states():
        pi[k] = np.random.choice(list(mdp.get_actions(k)))

    count = {}
    for k in mdp.get_states():
        count[k] = {}
        for kk in mdp.get_actions(k):
            count[k][kk] = 0

    def bellman_op(values):
        v0 = {}
        pi0 = {}
        for i in mdp.get_states():
            pi0[i] = np.random.choice(list(mdp.get_actions(i)))
            v0[i] = -np.inf

        for state in mdp.get_states():
            for action in mdp.get_actions(state):
                v_temp = 0.
                for next_state in mdp.get_actions(state)[action]:
                    probability = mdp.get_actions(state)[action][next_state]
                    reward = mdp.get_weight(state, action)
                    v_temp += (probability * (reward + gamma * values[next_state]))
                if v_temp > v0[state]:
                    v0[state] = v_temp
                    pi0[state] = action
        return v0, pi0

    for k in range(episodes):
        v, pi = bellman_op(v)

    return v, pi


def q_learn(markov: MDP,
            initial_state,
            terminal_state,
            alpha=0.01,
            gamma=1.,
            epsilon=1.,
            eps_min=0.1,
            eps_decade=0.9,
            episodes=50,
            verbose=1  # Integer. 0, 1, or 2. Verbosity mode
            ):
    global start
    global end
    global mdp

    start = initial_state
    end = terminal_state
    mdp = markov

    q = {}
    for k in mdp.get_states():
        q[k] = {}
        for kk in mdp.get_actions(k):
            q[k][kk] = 0

    pi = {}
    for k in mdp.get_states():
        pi[k] = np.random.choice(list(mdp.get_actions(k)))

    count = {}
    for k in mdp.get_states():
        count[k] = {}
        for kk in mdp.get_actions(k):
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


def monte_carlo(markov: MDP,
                initial_state,
                terminal_state,
                gamma=1.,
                epsilon=1.,
                eps_min=0.1,
                eps_decade=0.9,
                episodes=50,
                verbose=1  # Integer. 0, 1, or 2. Verbosity mode
                ):
    global start
    global end
    global mdp

    start = initial_state
    end = terminal_state
    mdp = markov

    q = {}
    for k in mdp.get_states():
        q[k] = {}
        for kk in mdp.get_actions(k):
            q[k][kk] = 0

    pi = {}
    for k in mdp.get_states():
        pi[k] = np.random.choice(list(mdp.get_actions(k)))

    count = {}
    for k in mdp.get_states():
        count[k] = {}
        for kk in mdp.get_actions(k):
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
