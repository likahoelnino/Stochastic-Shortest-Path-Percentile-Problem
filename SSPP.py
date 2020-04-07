# -----------------------------------------------------------
# Functions / modules of Stochastic Shortest Paths Percentile Problems
# and Reachability Problems
#
# -----------------------------------------------------------

from mdp import MDP, UnfoldedMDP, BOT
import pulp
import pprint as pp


def reach(mdp: MDP,
          targets: list,
          verbose: int = 1  # Integer. 0, 1, or 2. Verbosity mode
          ):

    x = {}
    for k in mdp.get_states():
        if k in targets:
            x[k] = 1
            continue
        else:
            x[k] = 0
        for t in targets:
            if mdp.connectivity(k, t) > 0:
                x[k] = -1
                break

    # put all non-1 states into untreated_states
    untreated_states = list(filter(lambda s: x[s] == -1, x.keys()))

    if untreated_states:
        # formulate the LP problem
        linear_program = pulp.LpProblem("reachability", pulp.LpMinimize)
        # initialize variables
        for s in untreated_states:
            x[s] = pulp.LpVariable(s, lowBound=0, upBound=1)
        # objective function
        linear_program += sum(x[s] for s in x)
        # constraints
        for s in untreated_states:
            actions_dict = mdp.get_actions(s)
            for a in actions_dict:
                linear_program += \
                    x[s] >= sum(actions_dict[a][ns] * x[ns] for ns in actions_dict[a])

        if verbose > 1:
            print(linear_program)

        # solve the LP
        linear_program.solve()

        for s in untreated_states:
            x[s] = x[s].varValue

    if verbose > 0:
        print("LP solver of x: ")
        pp.pprint(x)

    return x


def reachability_optimal_policy(mdp: MDP,
                                targets: list,
                                verbose: int = 1  # Integer. 0, 1, or 2. Verbosity mode
                                ):
    """
    return a policy that returns the action that maximises the reachability probability to "targets"
    of each state s.
    """

    x = reach(mdp, targets, verbose)

    policy = {}
    for state in mdp.get_states():
        if (state in targets) or (state == BOT):
            actions_info = mdp.get_actions(state)
            for action in actions_info:
                policy[state] = action
        else:
            max_v = -float('inf')
            actions_info = mdp.get_actions(state)
            optimal_action = None
            for action in actions_info:
                value = 0.
                for next_state in actions_info[action]:
                    value += actions_info[action][next_state] * x[next_state]
                if value > max_v:
                    max_v = value
                    optimal_action = action
            policy[state] = optimal_action

    return policy, x


def guaranteed_short_path(mdp: MDP,
                          source: object,
                          targets: list,
                          length: int,
                          proba_threshold: float = 0.,  # Float in [0, 1]
                          verbose: int = 0,  # Integer. 0, 1, or 2. Verbosity mode
                          ):
    """
    Compute the maximum probability to reach a set of target states "targets" from a initial state "source" of
    a MDP "mdp" with a path length less than a threshold "length" and get the strategy on the unfolded mdp.
    """
    unfolded_mdp = UnfoldedMDP(mdp, source, targets, length, 0)
    policy, x = reachability_optimal_policy(unfolded_mdp, unfolded_mdp.get_target(), verbose)
    new_policy = {}
    for state in policy:
        if x[state] >= proba_threshold:
            new_policy[state] = policy[state]
    return new_policy
