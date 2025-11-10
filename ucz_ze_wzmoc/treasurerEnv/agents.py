import random
import matplotlib.pyplot as plt
from collections import defaultdict



class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        actions_values = []
        for a in possible_actions:
            actions_values.append(self.get_qvalue(state, a))

        max_value = max(actions_values)

        return max_value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        curr_qvalue = self.get_qvalue(state, action)
        ns_value = self.get_value(next_state)

        res = (1 - learning_rate) * curr_qvalue + learning_rate * (reward + gamma * ns_value)
        self.set_qvalue(state, action, res)


    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        possible_actions_values = {}
        for a in possible_actions:
            possible_actions_values[a] = self.get_qvalue(state, a)

        max_qvalue = max(possible_actions_values.values())
        
        # wybierz najlepszą akcję losowo (jesli jest jedna to tą jedną)
        best_action = random.choice([a for a, v in possible_actions_values.items() if v == max_qvalue])
            
        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        p = random.uniform(0, 1)
        if 1 - epsilon < p: #randomowa akcja z prawdopodobienstwem epsilon (eksploracja)
            chosen_action = random.choice(possible_actions)
        else: # najlepsza akcja z prawdopodobienstwem 1-epsilon (eksploatacja)
            chosen_action = self.get_best_action(state)   

        return chosen_action
    
    def reset(self):
        pass

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0


class SARSAAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    def reset(self):
        pass

    # ---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        values = []
        for a in possible_actions:
            values.append(self.get_qvalue(state, a))

        return max(values)

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * Q(s', a'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha
        # wykorzystujemy wartości ze stanu, który zostanie faktycznie użyty po wykonaniu akcji
        
        curr_qvalue = self.get_qvalue(state, action)
        next_action = self.get_best_action(next_state)
        next_qvalue = self.get_qvalue(next_state, next_action)

        res = (1 - learning_rate) * curr_qvalue + learning_rate * (reward + gamma * next_qvalue)
        self.set_qvalue(state, action, res)

        # function returns selected action for next state
        return next_action

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        possible_actions_values = {}
        for a in possible_actions:
            possible_actions_values[a] = self.get_qvalue(state, a)

        max_qvalue = max(possible_actions_values.values())

        best_action = random.choice([a for a, value in possible_actions_values.items() if value == max_qvalue])

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        p = random.uniform(0, 1)
        if 1 - epsilon < p:
            chosen_action = random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)    

        return chosen_action
    

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0


import random
from collections import defaultdict


class ExpectedSARSAAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        values = []
        for a in possible_actions:
            values.append(self.get_qvalue(state, a))

        max_value = max(values)

        return max_value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * \sum_a \pi(a|s') Q(s', a))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha


        curr_qvalue = self.get_qvalue(state, action)
        possible_actions = self.get_legal_actions(next_state)
        
        #wylicza średnią ważoną (te najlepsze akcje mają większe prawdopodobieństwo)
        expected_qvalue = 0
        for a in possible_actions:
            qvalue = self.get_qvalue(next_state, a)
            p_a = self.epsilon / len(possible_actions)
            if a == self.get_best_action(next_state):
                p_a += 1 - self.epsilon
            
            expected_qvalue += p_a * qvalue
        
        result = (1 - learning_rate) * curr_qvalue + learning_rate * (reward + gamma * expected_qvalue)
        self.set_qvalue(state, action, result)



    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        possible_actions_values = {}
        for a in possible_actions:
            possible_actions_values[a] = self.get_qvalue(state, a)

        max_qvalue = max(possible_actions_values.values())

        best_action = random.choice([a for a, value in possible_actions_values.items() if value == max_qvalue])

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        p = random.uniform(0, 1)
        if 1 - epsilon < p:
            chosen_action = random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state) 

        return chosen_action

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0
    

class SARSALambdaAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions, lambda_value):
        """
        SARSA Lambda Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self._evalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.lambda_value = lambda_value

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    def reset(self):
        self._evalues = defaultdict(lambda: defaultdict(lambda: 0))

    # ---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0
        
        values = []
        for a in possible_actions:
            values.append(self.get_qvalue(state, a))


        return max(values)

    def update(self, state, action, reward, next_state):
        """
        You should do your SARSA-Lambda update here:
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        curr_qvalue = self.get_qvalue(state, action)
        next_action = self.get_best_action(next_state)
        next_qvalue = self.get_qvalue(next_state, next_action)

        sigma = reward + gamma * next_qvalue - curr_qvalue
        self._evalues[state][action] = 1
        for s in self._evalues:
            for a in self._evalues[s]:
                old_qvalue = self.get_qvalue(s, a)

                res = old_qvalue + learning_rate * sigma * self._evalues[s][a]
                self.set_qvalue(s, a, res)
                self._evalues[s][a] = gamma * self.lambda_value * self._evalues[s][a]

        return next_action

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        possible_actions_values = {}
        for a in possible_actions:
            possible_actions_values[a] = self.get_qvalue(state, a)

        max_qvalue = max(possible_actions_values.values())

        best_action = random.choice([a for a, value in possible_actions_values.items() if value == max_qvalue])

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        p = random.uniform(0, 1)
        if 1 - epsilon < p:
            chosen_action = random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)    

        return chosen_action

    def turn_off_learning(self):
        self.epsilon = 0
        self.alpha = 0

    def display_qvalues(self):
        for s in self._qvalues:
            print("State: " + str(s) + " " + str(self._qvalues[s]))


class DQLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Double Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)
        """

        self.get_legal_actions = get_legal_actions
        self._qvaluesA = defaultdict(lambda: defaultdict(lambda: 0))
        self._qvaluesB = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        
    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvaluesA[state][action] + self._qvaluesB[state][action] 


    #---------------------START OF YOUR CODE---------------------#

    def get_best_action(self, state, table='A'):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None
        
        possible_actions_values = {}
        for a in possible_actions:
            if table == 'A':
                possible_actions_values[a] = self._qvaluesA[state][a]
            else:
                possible_actions_values[a] = self._qvaluesB[state][a]
            # possible_actions_values[a] = self.get_qvalue(state, a)

        max_qvalue = max(possible_actions_values.values())
        
        # wybierz najlepszą akcję losowo (jesli jest jedna to tą jedną)
        best_action = random.choice([a for a, v in possible_actions_values.items() if v == max_qvalue])
        
        return best_action

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        # curr_qvalue = self.get_qvalue(state, action)
        # next_action = self.get_best_action(next_state,)

        p = random.uniform(0, 1)
        if p < 0.5:
            next_action = self.get_best_action(next_state, table='A')
            next_qvalue = self._qvaluesB[next_state][next_action]
            
            res = learning_rate * (reward + gamma * next_qvalue - self._qvaluesA[state][action])
            self._qvaluesA[state][action] += res
        else:
            next_action = self.get_best_action(next_state, table='B')
            next_qvalue = self._qvaluesA[next_state][next_action]
            
            res = learning_rate * (reward + gamma * next_qvalue - self._qvaluesB[state][action])
            self._qvaluesB[state][action] += res


    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        p = random.uniform(0, 1)
        if 1 - epsilon < p: #randomowa akcja z prawdopodobienstwem epsilon (eksploracja)
            chosen_action = random.choice(possible_actions)
        else: # najlepsza akcja z prawdopodobienstwem 1-epsilon (eksploatacja)
            chosen_action = self.get_best_action(state)        

        return chosen_action

    def reset(self):
        """Reset nie jest potrzebny dla Double Q-Learning, ale dodajemy dla kompatybilności"""
        pass

    def turn_off_learning(self):
        self.epsilon = 0
        self.alpha = 0
