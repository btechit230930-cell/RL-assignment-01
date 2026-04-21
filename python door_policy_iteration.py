import numpy as np

# States: (person, door)
states = [(0,0), (0,1), (1,0), (1,1)]

# Actions: 0=Open, 1=Close, 2=Wait
actions = [0,1,2]

gamma = 0.9

# Reward function
def reward(state, action):
    person, door = state

    if action == 0:  # Open
        if person == 1:
            return 10
        else:
            return -5

    elif action == 1:  # Close
        if person == 0:
            return 5
        else:
            return -10

    else:  # Wait
        return -1

# Transition (simple probabilistic person arrival)
def transition(state, action):
    person, door = state

    # Person randomly appears/disappears
    next_person = np.random.choice([0,1])

    if action == 0:
        next_door = 1
    elif action == 1:
        next_door = 0
    else:
        next_door = door

    return (next_person, next_door)

# Initialize random policy
policy = {s: np.random.choice(actions) for s in states}
V = {s: 0 for s in states}

def policy_evaluation(policy, V, theta=0.01):
    while True:
        delta = 0
        for s in states:
            v = V[s]
            a = policy[s]

            r = reward(s, a)
            next_s = transition(s, a)

            V[s] = r + gamma * V[next_s]
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break
    return V

def policy_improvement(V, policy):
    stable = True

    for s in states:
        old_action = policy[s]

        action_values = []
        for a in actions:
            r = reward(s, a)
            next_s = transition(s, a)
            action_values.append(r + gamma * V[next_s])

        best_action = actions[np.argmax(action_values)]
        policy[s] = best_action

        if old_action != best_action:
            stable = False

    return policy, stable

# Policy Iteration
while True:
    V = policy_evaluation(policy, V)
    policy, stable = policy_improvement(V, policy)

    if stable:
        break

print("Optimal Policy:")
for s in states:
    print(f"State {s} -> Action {policy[s]}")
