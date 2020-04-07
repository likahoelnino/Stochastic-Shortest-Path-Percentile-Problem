# state --> action and cost
state_2_action = {'home': {'railway': -2, 'car': -1, 'bike': -45},
                  'waiting_room': {'wait': -3, 'go_back': -2},
                  'train': {'relax': -35},
                  'light_traffic': {'drive': -20},
                  'medium_traffic': {'drive': -30},
                  'heavy_traffic': {'drive': -70},
                  'work': {'done': 0}}

# action --> state and transition probability
action_2_state = {'railway': {'waiting_room': 0.1, 'train': 0.9},
                  'wait': {'waiting_room': 0.1, 'train': 0.9},
                  'go_back': {'home': 1},
                  'relax': {'work': 1},
                  'car': {'light_traffic': 0.2,
                          'medium_traffic': 0.7,
                          'heavy_traffic': 0.1},
                  'drive': {'work': 1},
                  'bike': {'work': 1},
                  'done': {'work': 1}}

# start node and target node
start = 'home'
end = 'work'




