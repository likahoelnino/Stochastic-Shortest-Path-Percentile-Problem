# -----------------------------------------------------------
# This module contains MDP structures implementations as class.
#
# -----------------------------------------------------------

import networkx as nx

BOT = 'BOT'


class MDP:
    """
    Implementation of Markov Decision Process structure
    using NetworkX Graph
    An MDP structure is stored in a MultiDiGraph
    States are stored in nodes.
    Actions are stored edges' keys
    Weights of the actions and transition probabilities are stored in the attributes of the edges
    """

    def __init__(self):
        self._g = nx.MultiDiGraph()

    def get_graph(self):
        return self._g

    def get_states(self):
        return list(self._g.nodes())

    def get_info(self, state=None):
        if state is None:
            _, _, d = list(zip(*(self._g.edges(keys=False, data=True))))
        else:
            _, _, d = list(zip(*(self._g.edges(state, keys=False, data=True))))
        return d

    def get_actions(self, state):
        actions = {}
        if state is not None:
            d = self.get_info(state)
            for item in d:
                if item['action'] in actions.keys():
                    actions[item['action']].update({item['to']: item['proba']})
                else:
                    actions[item['action']] = {item['to']: item['proba']}
        return actions

    def get_weight(self, state, action, next_state):
        if self._g.has_edge(state, next_state, action):
            return self._g[state][next_state][action]['weight']
        else:
            return float('nan')

    def generate_from_my_format(self, s2a, a2s):
        """
        A function to read the example in file "my_env.py"
        into the MDP class
        """
        for state in s2a:
            self._g.add_node(state)
        for state in s2a:
            for action in s2a[state]:
                for next_state in a2s[action]:
                    self._g.add_edge(state, next_state, key=action,
                                     action=action,
                                     weight=s2a[state][action],
                                     fr=state,
                                     to=next_state,
                                     proba=a2s[action][next_state],
                                     )

    def connectivity(self, fr, to):
        if self._g.has_node(fr) and self._g.has_node(to):
            return nx.node_connectivity(self._g, s=fr, t=to)
        else:
            return -1


class UnfoldedMDP(MDP):
    """
    Unfold an MDP following an initial state (s0), a list of target states (T) and a maximum length threshold (l)
    """
    def __init__(self, mdp, s0, target, length, init_value=0):
        super().__init__()
        self._target = []
        mdp_graph = mdp.get_graph()
        bot = BOT
        self._g.add_node(bot)
        self._g.add_edge(u_for_edge=bot, v_for_edge=bot,
                         key='loop', action='loop', proba=1,
                         fr=bot, to=bot)

        def unfold(state, value):
            if not self._g.has_node((state, value)):
                self._g.add_node((state, value))
                if state in target:
                    for action in mdp_graph[state][state]:
                        self._target.append((state, value))
                        self._g.add_edge((state, value), (state, value),
                                         key=action,
                                         action=action,
                                         fr=(state, value),
                                         to=(state, value),
                                         proba=mdp_graph[state][state][action]['proba'],
                                         )
                else:
                    for next_state in mdp_graph[state]:
                        for action in mdp_graph[state][next_state]:
                            next_value = value + mdp_graph[state][next_state][action]['weight']
                            if next_value >= length:
                                unfold(next_state, next_value)
                                self._g.add_edge((state, value), (next_state, next_value),
                                                 key=action,
                                                 action=action,
                                                 fr=(state, value),
                                                 to=(next_state, next_value),
                                                 proba=mdp_graph[state][next_state][action]['proba'],
                                                 )
                            else:
                                self._g.add_edge((state, value), bot,
                                                 key='loop',
                                                 action='loop',
                                                 fr=(state, value),
                                                 to=bot,
                                                 proba=mdp_graph[state][next_state][action]['proba'],
                                                 )

        unfold(s0, init_value)

    def get_target(self):
        return self._target
