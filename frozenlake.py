# -----------------------------------------------------------
# This module contains MDP structures implementations as a class
# for FrozenLake environment;
# And an agent for calculating the policy using SSPP algorithm
# -----------------------------------------------------------

from mdp import MDP
from SSPP import guaranteed_short_path

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3
DONE = -1  # at holes or at goal


class FrozenLakeMDP(MDP):
    """
    v1: only create a 4x4 map as below
    SFFF       (S: starting point, safe)
    FHFH       (F: frozen surface, safe)
    FFFH       (H: hole, fall to your doom)
    HFFG       (G: goal, where the frisbee is located)


    v2: be able to create a 8x8 map as below
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG"
    """

    def __init__(self,
                 new_map: bool = False  # v2: True: create a MDP of FrozenLake 8x8
                                        #     False: create a MDP of FrozenLake 4x4
                 ):
        super().__init__()

        if new_map:
            n_row = 8
            n_col = 8
            states = list(range(64))
            actions = list(range(4))
            holes = [2*8+3, 3*8+5, 4*8+4, 5*8+1, 5*8+2, 5*8+6, 6*8+1, 6*8+4, 6*8+6, 7*8+3]

            self.starting_point = 0
            self.goal = 63
        else:
            n_row = 4
            n_col = 4
            states = list(range(16))
            actions = list(range(4))
            holes = [5, 7, 11, 12]

            self.starting_point = 0
            self.goal = 15

        def to_s(_row, _col):
            return _row * n_col + _col

        def to_row_col(_state):
            _row = _state // n_col
            _col = _state - _row * n_col
            return _row, _col

        def inc(_row, _col, _a):
            if _a == LEFT:  # left
                _col = max(_col - 1, 0)
            elif _a == UP:  # up
                _row = min(_row + 1, n_row - 1)
            elif _a == RIGHT:  # right
                _col = min(_col + 1, n_col - 1)
            elif _a == DOWN:  # down
                _row = max(_row - 1, 0)
            return _row, _col

        def random_action_set(_a):
            if _a == LEFT:  # left -> [UP, LEFT, DOWN]
                actions_set = [UP, LEFT, DOWN]
            elif _a == UP:  # up -> [LEFT, UP, RIGHT]
                actions_set = [LEFT, UP, RIGHT]
            elif _a == RIGHT:  # right -> [UP, RIGHT, DOWN]
                actions_set = [UP, RIGHT, DOWN]
            elif _a == DOWN:  # down -> [RIGHT, DOWN, LEFT]
                actions_set = [RIGHT, DOWN, LEFT]
            return actions_set

        for state in states:
            self._g.add_node(state)
        for state in states:
            row, col = to_row_col(state)
            if state in holes:
                self._g.add_edge(state, state, key=DONE,
                                 action=DONE,
                                 weight=0,
                                 length=0,
                                 fr=state,
                                 to=state,
                                 proba=1,
                                 )
            elif state == self.goal:
                self._g.add_edge(state, state, key=DONE,
                                 action=DONE,
                                 weight=0,
                                 length=0,
                                 fr=state,
                                 to=state,
                                 proba=1,
                                 )
            else:
                for action in actions:
                    rnd_set = random_action_set(action)
                    next_states = [to_s(*inc(row, col, a)) for a in rnd_set]
                    for next_state in next_states:
                        if not self._g.has_edge(state, next_state, key=action):
                            self._g.add_edge(state, next_state, key=action,
                                             action=action,
                                             weight=-1,
                                             length=1,
                                             fr=state,
                                             to=next_state,
                                             proba=1 / 3,
                                             )
                        else:
                            self._g[state][next_state][action]['proba'] += 1 / 3


class FrozenLakeAgent:
    """
    Calculate the policy that maximum the survival probability under the condition
        that move to the goal within the target steps in the Frozen Lake using
        SSPP algorithm
    """
    def __init__(self,
                 big_map: bool = False,  # True: create a MDP of FrozenLake 8x8
                                         # False: create a MDP of FrozenLake 4x4
                 max_length: int = -50,  # steps that target to move to the goal within the number of steps
                 ):
        self.mdp = FrozenLakeMDP(new_map=big_map)
        self._max_length = max_length
        pi, x = guaranteed_short_path(self.mdp, self.mdp.starting_point,
                                      [self.mdp.goal], max_length, return_x=True)
        policy = [-1] * (63 if big_map else 15)
        for i in range(63 if big_map else 15):
            for k in range(len(pi)):
                if (i, -k) in pi:
                    policy[i] = pi[(i, -k)]
                    break
        self._policy = policy
        self._full_policy = pi
        self._x = x

    def get_policy(self):
        return self._policy
