# -----------------------------------------------------------
# Playground of the functions / modules about Stochastic Shortest
# Path Percentile problem (SSPP)
#
# -----------------------------------------------------------


from mdp import MDP, UnfoldedMDP
from my_env import *
from SSPP import reach, reachability_optimal_policy, guaranteed_short_path
import random
import pprint as pp


def run(mdp: MDP,  # a Morkov Decision Process instance
        initial_state: object,  # initial state of the MDP (one state only)
        terminal_states: list,  # terminal states of the MDP (one or more states)
        policy: dict,  # the optimal policy
        episodes: int = 10,  # number of times to simulate the runs
        ):
    """ simulating the process under the giving policy """
    for _ in range(episodes):
        state = initial_state
        value = 0
        for step in range(100):
            actions_info = mdp.get_actions(state)
            action = pi[(state, value)]
            next_state = random.choice(list(actions_info[action].keys()))
            weight = mdp.get_weight(state, action, next_state)
            value += weight
            print('Step {}: [s: {}]-->(a: {}, c: {})-->[s: {}]'.format(step + 1, state, action, value, next_state))
            state = next_state
            if state in terminal_states:
                print('Done!\n\n')
                break
            elif (state, value) not in policy.keys():
                print('Fail!\n\n')
                break


m = MDP()
m.generate_from_my_format(state_2_action, action_2_state)
pi = guaranteed_short_path(m, 'home', ['work'], -60, 1, 0)
run(m, 'home', ['work'], pi, 10)
