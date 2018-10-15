import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        # AE: alpha param for Q-learning
        self.alpha = 0.15
        
        # AE: epsilon param for Q-learning
        self.epsilon = 0.0005
        
        self.episode_count = 0
        self.N = defaultdict(lambda: np.zeros(self.nA))
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        #print (self.Q, state)
        #return np.random.choice(self.nA)
        
        return self.choose_action_from_epsilon_greedy_set(self.get_epsilon_greedy_actions(self.Q, state))

    def step(self, state, action, reward, next_state, done, gamma=1.0):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #self.Q[state][action] += 1
        self.N[state][action] += 1
        ega = self.get_epsilon_greedy_actions(self.Q, next_state)
        nga = np.argmax(self.Q[next_state])
        
        # SARSA
        #self.Q[state][action] += self.alpha * (reward + gamma * self.Q[next_state][nga] - self.Q[state][action])
        
        # Expected SARSA
        self.Q[state][action] += self.alpha * (reward + gamma * np.dot(ega, self.Q[next_state]) - self.Q[state][action])
        
        # SARSAMAX
        #self.Q[state][action] += self.alpha * (reward + gamma * self.get_value_of_greedy_action(self.Q, next_state) - self.Q[state][action])
        
        #if (self.episode_count < 17099):
            # Expected SARSA
            #self.Q[state][action] += self.alpha * (reward + gamma * np.dot(ega, self.Q[next_state]) - self.Q[state][action])
        #else:
            # SARSAMAX
            #self.Q[state][action] += self.alpha * (reward + gamma * self.get_value_of_greedy_action(self.Q, next_state) - self.Q[state][action])
            
        
        #self.Q[state][action] = (1 / self.N[state][action]) * (reward + gamma * self.get_value_of_greedy_action(self.Q, next_state) + self.Q[state][action] * self.N[state][action])
        
        if done == True:
            self.episode_count += 1

    def get_value_of_greedy_action(self, Q, state):
        best_greedy_action = np.argmax(Q[state])

        return Q[state][best_greedy_action]
    
    ### AE: Chooses probabilities for actions at a given state using an epsilon-greedy approach
    def get_epsilon_greedy_actions(self, Q, state):
        possible_actions = np.zeros(self.nA, dtype=np.float64)
        best_greedy_action = np.argmax(Q[state])

        # AE: choosing epsilon to decay and protecting against division by 0
        eps_thr = 2000
        if (self.episode_count < 17099):
            epsilon = eps_thr / (eps_thr + self.episode_count)
        else:
            epsilon = self.epsilon
        #epsilon = 1.0 / (self.episode_count + 1)
        #if (self.episode_count < 19500):
        #    epsilon = 0.20
        #else:
        #    epsilon = 0;

        for action_ndx, action in enumerate(possible_actions):
            if (action_ndx == best_greedy_action):
                possible_actions[best_greedy_action] = 1 - epsilon + epsilon / self.nA
            else:
                possible_actions[action_ndx] = epsilon / self.nA

            #print (epsilon, env.nA, epsilon / env.nA, possible_actions[action_ndx])

        return possible_actions

    def choose_action_from_epsilon_greedy_set(self, possible_actions):
        #print(possible_actions)
        return np.random.choice(np.arange(len(possible_actions)), p=possible_actions)