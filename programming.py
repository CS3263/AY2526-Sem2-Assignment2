"""
Assignment 2 programming script.

* Group Member 1:
    - Name:
    - Student ID:

* Group Member 2:
    - Name:
    - Student ID:
"""


import numpy as np

from typing import Callable
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils


# Assignment 2 - Question 1: Policy Iteration & Value Iteration

class mdp_solver:
    
    def __init__(self):
        self.iteration = 0
        print("MDP initialized!")
        
    
    def get_action_value(
        self, s:int, a:int, V:np.ndarray, gamma:float, env_transition:Callable):
        """
        Code for getting action value. Compute the value of taking action a in state s
        I.e., compute Q(s, a) = sum_{s'} p(s'| s, a) * [r + gamma * V(s')]
        args:
            s: state
            a: action
            V: value function
            gamma: discount factor
            env_transition: transition function
        returns:
            value: action value
        """
        value = 0
        
        for prob, next_state, reward, done in env_transition(s, a):
            
            # ------- your code starts here ----- #
            

            # ------- your code ends here ------- #
        
        return value
    
    
    def get_max_action_value(
        self, s:int, env_nA:int, env_transition:Callable, V:np.ndarray, gamma:float):
        """
        Code for getting max action value. Takes in the current state and returns 
        the max action value and action that leads to it. I.e., compute
        a* = argmax_a sum_{s'} p(s'| s, a) * [r + gamma * V(s')]
        args:
            s: state
            env_nA: number of actions
            env_transition: transition function
            V: value function
            gamma: discount factor
        returns:
            max_value: max action value
            max_action: action that leads to max action value
        """
        max_value = -np.inf
        max_action = -1
        
        for a in range(env_nA):
            
            # ------- your code starts here ----- #
            

            # ------- your code ends here ------- #
        
        return max_value, max_action
    
    
    def get_policy(
        self, env_nS:int, env_nA:int, env_transition:Callable, gamma:float, V:np.ndarray):
        """
        Code for getting policy. Takes in an Value function and returns the optimal policy
        args:
            env_nS: number of states
            env_nA: number of actions
            env_transition: transition function
            gamma: discount factor
            V: value function
        returns:
            policy: policy
        """
        policy = np.zeros(env_nS)
        
        for s in range(env_nS):
            max_value = -np.inf
            max_action = -1
            for a in range(env_nA):
                
                # ------- your code starts here ----- #
              
                
                # ------- your code ends here ------- #
                    
            policy[s] = max_action
        
        return policy
    
    
    def policy_evaluation(
        self, env_nS:int, env_transition:Callable, V:np.ndarray, gamma:float, theta:float, policy:np.ndarray):
        """
        Code for policy evaluation. Takes in an MDP and returns the converged value function
        args:
            env_nS: number of states
            env_transition: transition function
            V: value function
            gamma: discount factor
            theta: convergence threshold
            policy: policy
        returns:
            V: value function
        """ 
        
        while True:
            delta = 0
            for s in range(env_nS):
                
                # ------- your code starts here ----- #

                
                # ------- your code ends here ------- #
                
            if delta < theta:
                break
                
        return V
    
    
    def policy_improvement(
        self, env_nS:int, env_nA:int, env_transition:Callable, policy:np.ndarray, V:np.ndarray, gamma:float):
        """
        Code for policy improvement. Takes in an MDP and returns the converged policy
        args:
            env_nS: number of states
            env_nA: number of actions
            env_transition: transition function
            policy: policy
            V: value function
            gamma: discount factor
        returns:
            policy_stable: whether policy is stable
            policy: policy
        """
        policy_stable = True
        
        for s in range(env_nS):
            
            # ------- your code starts here ----- #

            # ------- your code ends here ------- #
        
        return policy_stable, policy
    
    
    def value_iteration(
        self, gamma:float, theta:float, env_nS:int, env_nA:int, env_transition:Callable):
        """
        The code for value iteration. Takes in an MDP and returns the optimal policy
        and value function.
        args:
            gamma: discount factor
            theta: convergence threshold
            env_nS: number of states
            env_nA: number of actions
            env_transition: transition function
        returns:
            policy: optimal policy
            V: optimal value function 
        """
        V = np.zeros(env_nS)
        converged = False
        
        while not converged:
            delta = 0
            
            # ------- your code starts here ----- #


            # ------- your code ends here ------- #
        
        policy = self.get_policy(env_nS, env_nA, env_transition, gamma, V)
        return policy, V
    
    
    def policy_iteration(
        self, gamma:float, theta:float, env_nS:int, env_nA:int, env_transition:Callable):
        """
        Code for policy iteration. Takes in an MDP and returns the optimal policy
        args:
            gamma: discount factor
            theta: convergence threshold
            env_nS: number of states
            env_nA: number of actions
            env_transition: transition function
        returns:
            policy: optimal policy
            V: optimal value function
        """
        V = np.zeros(env_nS)
        policy = np.zeros(env_nS)
        converged = False
        
        while not converged:
            
            # ------- your code starts here ----- #
            
            
            
            # ------- your code ends here ------- #
        
        return policy, V



# Assignment 2 - Question 2: Q-Learning

class mdp_solver_q_learning:
    
    def __init__(self):
        print("MDP initialized for Q-learning!")


    def epsilon_greedy(self, Q, state, epsilon):
        if np.random.rand() < epsilon:
            
            # ------- your code starts here ----- #
            

            
            # ------- your code ends here ------- #
            
        else:
            
            # ------- your code starts here ----- #
            

        
            # ------- your code ends here ------- #
   

    def Q_learning(
        self, alpha:float, gamma:float, theta:float, epsilon:float, env_nS:int, env_nA:int, env_transition, env, num_episodes=1000, initial_action = None, initial_reward = 10):
        """
        Q-learning algorithm.
        Args:
            gamma: discount factor
            theta: convergence threshold
            env_nS: number of states
            env_nA: number of actions
            env_transition: transition function
            num_episodes: number of episodes
        Returns:
            Q: learned Q-value function
            rewards: rewards obtained in each episode
        """
        Q = np.zeros((env_nS, env_nA))
        rewards = []
        
        if initial_action is not None:
            for i, action in enumerate(initial_action):
                Q[i][action] += initial_reward
        
        for episode in range(num_episodes):
            env.reset()
            state = env.state_to_index(env.state)
            done = False
            episode_reward = 0
            
            while not done:
                
                # ------- your code starts here ----- #
            
                
                # ------- your code ends here ------- #
                
            rewards.append(episode_reward)
        
        return Q, rewards


# Assignment 2 - Question 3: Q-Learning with LLM

# describe your current state
def describe_state(state):
    describe = ''

    # ------- your code starts here ----- #
    
    
    
    
    
    # ------- your code ends here ------- #
    
    
    print(describe)
    return describe

class LLM_Model:
    def __init__(self, device, instruction, goal_state, initial_conditions, client, model='gpt-3.5-turbo'):
        self.device = device
        self.model = model
        self.client = client
        self.sampling_params = \
            {
                "max_completion_tokens": 32,
                "temperature": 0.5,
                "top_p": 0.9,
                "n": 1,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
            }
        
        self.prompt_begin = """Generate the most logical next move in the scene. 
You must strictly follow the format in the following examples and output exactly **one** action. 
The generated action must be strictly from **GROUNDED_ACTION_LIST**."""
        self.condition_list = ['ROOM_LIST', 'OBJECT_LIST', 'OBJECT_POSITION_LIST','CONTAINER_LIST', 'SURFACE_LIST', 'CONTAINER_POSITION_LIST', 'CONNECTED_ROOM','ACTION_DICT', 'GROUNDED_ACTION_LIST']
        self.initial_conditions = initial_conditions
        self.ROOM_LIST, self.OBJECT_LIST, self.OBJECT_POSITION_LIST, \
        self.CONTAINER_LIST, self.SURFACE_LIST, self.CONTAINER_POSITION_LIST, self.CONNECTED_ROOM, \
        self.ACTION_DICT, self.GROUNDED_ACTION_LIST = initial_conditions
        
        self.translation_lm = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.container_list_embedding = self.translation_lm.encode(self.CONTAINER_LIST, batch_size=8, 
                convert_to_tensor=True, device=self.device)  # lower batch_size if limited by GPU memory
        self.object_list_embedding = self.translation_lm.encode(self.OBJECT_LIST, batch_size=8,
                convert_to_tensor=True, device=self.device)
        if self.SURFACE_LIST:
            self.position_list =  self.CONTAINER_LIST + self.SURFACE_LIST
            self.surface_list_embedding = self.translation_lm.encode(self.SURFACE_LIST, batch_size=8,
                convert_to_tensor=True, device=self.device)
            self.position_list_embedding = torch.concat((self.container_list_embedding, self.surface_list_embedding), dim=0)
        self.room_embedding = self.translation_lm.encode(self.ROOM_LIST, batch_size=8,
                convert_to_tensor=True, device=self.device)
        self.action_list_embedding = self.translation_lm.encode(self.GROUNDED_ACTION_LIST, batch_size=8, 
                convert_to_tensor=True, device=self.device)  # lower batch_size if limited by GPU memory
    
    def find_most_similar(self, query_str, corpus_embedding):
        # helper function for finding similar sentence in a corpus given a query
        query_embedding = self.translation_lm.encode(query_str, convert_to_tensor=True, device=self.device, show_progress_bar=False,)
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        most_similar_idx = np.argmax(cos_scores)
        return most_similar_idx
    
    def query_llm(self, task, observe = None):
        prompt_content = self.prompt_begin
        task = 'Scene: ' + "\n".join(f"{a}:{b}" for a, b in zip(self.condition_list, self.initial_conditions) if a != 'ACTION_DICT') + '\nCurrent State: ' + observe + 'Task: ' + task

        
        generated_samples = []
        for _ in range(self.sampling_params['n']):
            try: 
                
                # ------- your code starts here ----- #
                
                
                
                
                # ------- your code ends here ------- #
                
            except Exception as e:
                print(f"Error: {e}")

        samples = generated_samples
        return samples
    
    def plan(self, task, observe):
        samples = self.query_llm(task, observe = observe)
        action_list, action_index = [], []
        for sample in samples:
            most_similar_idx = self.find_most_similar(sample, self.action_list_embedding)
            # find the most similar action in the action list
            translated_action = self.GROUNDED_ACTION_LIST[most_similar_idx]
            action_list.append(translated_action)
            action_index.append(most_similar_idx)
        best_action = [max(action_list, key=action_list.count), max(action_index, key=action_list.count)]
        return best_action