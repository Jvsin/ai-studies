from treasure_env import MultiTreasureHunterMDP as MTH

states = MTH.get_all_states()
print(states)

for s in states:
    actions = MTH.get_possible_actions(s)
    for a in actions:
        next_states = MTH.get_next_states(s, a)
        print("State: " + s + " action: " + a + " " + "list of possible next states: ", next_states)

policy = dict()

for s in states:
    actions = MTH.get_possible_actions(s)
    action_prob = 1 / len(actions)
    policy[s] = dict()
    for a in actions:
        policy[s][a] = action_prob

print(policy)

def value_iteration(mdp, gamma, theta):
    """
            This function calculate optimal policy for the specified MDP using Value Iteration approach:

            'mdp' - model of the environment, use following functions:
                get_all_states - return list of all states available in the environment
                get_possible_actions - return list of possible actions for the given state
                get_next_states - return list of possible next states with a probability for transition from state by taking
                                  action into next_state
                get_reward - return the reward after taking action in state and landing on next_state


            'gamma' - discount factor for MDP
            'theta' - algorithm should stop when minimal difference between previous evaluation of policy and current is
                      smaller than theta
            Function returns optimal policy and value function for the policy
       """



    V = dict()
    policy = dict()

    # init with a policy with first avail action for each state
    for current_state in mdp.get_all_states():
        V[current_state] = 0
        policy[current_state] = actions[0]

    while True:
        delta = 0
        for s in mdp.get_all_states():
            last_v = V[s]
            possible_actions = mdp.get_possible_actions(s)
            a_values = dict()

            for a in possible_actions:
                p_next_states = mdp.get_next_states(s, a)
                result = 0
                for ns in p_next_states:
                    reward = mdp.get_reward(s, a, ns)
                    result += p_next_states[ns] * (reward + gamma * V[ns])
                a_values[a] = result

            best_action = max(a_values, key=a_values.get)
            V[s] = a_values[best_action]
            policy[s] = best_action 
            delta = max(delta, abs(last_v - V[s]))

        if delta < theta:
            break

    return policy, V