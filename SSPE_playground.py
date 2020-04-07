from SSPE import q_learn, monte_carlo, vi
import pprint as pp

# q, pi = q_learn(alpha=0.01, episodes=200000, verbose=0)
# q, pi = monte_carlo(episodes=100000, verbose=0)
q, pi = vi(episodes=50)
pp.pprint(q)
pp.pprint(pi)

# for i in range(50):
#     q, pi = q_learn(alpha=0.01, episodes=10000, verbose=0)
#     print(i, pi)
#
# pp.pprint(q)
