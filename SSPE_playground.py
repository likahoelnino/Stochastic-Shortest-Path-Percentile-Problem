from SSPE import q_learn, monte_carlo, vi
from mdp import MDP
from my_env import *
import pprint as pp

m = MDP()
m.generate_from_my_format(state_2_action, action_2_state)

# q, pi = q_learn(m, start, end, alpha=0.01, episodes=200000, verbose=0)
# q, pi = monte_carlo(m, start, end, episodes=100000, verbose=0)
q, pi = vi(m, episodes=50)
pp.pprint(q)
pp.pprint(pi)

# for i in range(50):
#     q, pi = q_learn(alpha=0.01, episodes=10000, verbose=0)
#     print(i, pi)
#
# pp.pprint(q)
