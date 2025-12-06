from treasure_env import MultiTreasureHunterMDP as MTH
import random
import sys
sys.stdout.reconfigure(encoding='utf-8')

MAP = [
    "A....H...T",
    ".#.#...##.",
    ".H.#T..#H.",
    ".##...H.#.",
    "T...H....B"
]

def get_agents_states(mdp, agent='1'):
    V = dict()
    policy = dict()

    # init with a policy with first avail action for each state
    for current_state in mdp.get_all_states():
        V[current_state] = 0
        actions = mdp.get_possible_actions(current_state, agent)
        policy[current_state] = actions[0]
    
    return V, policy


def value_iteration(mdp, gamma, theta, agent='1'):
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
        actions = mdp.get_possible_actions(current_state, agent)
        policy[current_state] = actions[0]
    
    iter_counter = 0
    while True:
        delta = 0
        for s in mdp.get_all_states():
            last_v = V[s]
            possible_actions = mdp.get_possible_actions(s, agent)
            a_values = dict()

            for a in possible_actions:
                p_next_states = mdp.get_next_states(s, a)
                result = 0
                for ns in p_next_states:
                    reward = mdp.get_reward(s, a, ns, agent)
                    # result += p_next_states[ns] * (reward + gamma * V[ns])
                    result += 1 * (reward + gamma * V[ns])
                a_values[a] = result

            best_action = max(a_values, key=a_values.get)
            V[s] = a_values[best_action]
            policy[s] = best_action 
            delta = max(delta, abs(last_v - V[s]))
        
        # if iter_counter % 10:
        #     print(f"Wymieliłem {iter_counter} iteracje")
        
        if delta < theta:
            break

        iter_counter += 1

    print(f"Zakończono po {iter_counter} iteracjach.")
    return policy, V


if __name__ == "__main__":

    env = MTH(MAP)
    env.reset()
    v1, p1 = get_agents_states(env, '1')
    v2, p2 = get_agents_states(env, '2')
    # print(len(v1))
    # print(len(v2))
    # print(len(p1))
    # print(len(p2))
    gamma = 0.9
    theta = 0.001

    policy_agent1, v_agent1 = value_iteration(env, gamma, theta, '1')
    # print(policy_agent1[((9, 4), (6, 4), frozenset(), False, False)])

    policy_agent2, v_agent2 = value_iteration(env, gamma, theta, '2')

    