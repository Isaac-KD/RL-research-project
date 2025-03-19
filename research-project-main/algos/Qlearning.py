import numpy as np
import random
import pickle
import copy
import matplotlib.pyplot as plt

class QLearningAgent(): 
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.5):
        self.alpha = alpha                  # Taux d'apprentissage
        self.gamma = gamma                  # Facteur de discount
        self.epsilon = epsilon              # Probabilité d'exploration
        self.q_table = {}   # Table des valeurs Q avec valeurs par défaut à une valeur aléatoire.
        
    @staticmethod
    def valid_state(state):
        """
        Args:
            state (Any): état du jeux

        Returns:
            Tuple: un state mais en version immuable 
        """
        if isinstance(state, int):
            return state
        else:
            return tuple(state)  

    def get_q_value(self, state, action):
        """ Récupère la valeur Q(s, a) et initialise si elle n'existe pas """
        v_state = QLearningAgent.valid_state(state)
        if (v_state, action) not in self.q_table:
            self.q_table[(v_state, action)] = 0# np.random.rand()
        return self.q_table[(v_state, action)]
    
    def choose_action(self, state, possible_actions):
        """ Sélectionne une action avec epsilon-greedy """
        if  np.random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_actions)  # Exploration
        else:
            # Exploitation : choisir la meilleure action selon Q(s, a)         
            q_values = np.array([self.get_q_value(state, a) for a in possible_actions])
            return possible_actions[np.argmax(q_values)]
        
    def update_q_value(self, state, action, reward, next_state, possible_next_actions,done):
        """ Mise à jour de la table Q avec la règle de Bellman """
        
        tuple_state, tuple_next_state = QLearningAgent.valid_state(state),QLearningAgent.valid_state(next_state)    # passe a un type immuable.
        if done:
            target = reward
        else:
            max_q_next = np.max([self.get_q_value(tuple_next_state, a) for a in possible_next_actions])   
            target = reward + self.gamma * max_q_next
        
        current_q = self.get_q_value(tuple_state, action)
        self.q_table[(tuple_state,action)] = current_q + self.alpha * (target - current_q)
    
    def save(self, filename="q_agent.pkl"):
        """ Sauvegarde l'agent dans un fichier """
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "q_table": self.q_table
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(filename="q_agent.pkl"):
        """ Charge un agent depuis un fichier en ne restaurant que les attributs essentiels """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        agent = QLearningAgent(alpha=data["alpha"], gamma=data["gamma"], epsilon=data["epsilon"])
        agent.q_table = data["q_table"]
        return agent
    
    def set_epsilon(self, eps):
        self.epsilon = eps
    
class SARSAAgent(QLearningAgent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.5):
        super().__init__(alpha, gamma, epsilon) 
    
    def update_q_value(self, state, action, reward, next_state, next_action, done):
        """ Mise à jour de la table Q avec SARSA """
        tuple_state, tuple_next_state = SARSAAgent.valid_state(state), SARSAAgent.valid_state(next_state)

        current_q = self.get_q_value(tuple_state, action)
        if done:
            target = reward
        else:
            next_q = self.get_q_value(tuple_next_state, next_action)  # On utilise Q(s', a')
            target = reward + self.gamma * next_q
        
        self.q_table[(tuple_state, action)] = current_q + self.alpha * (target - current_q)
        
def trainSARSA(env, agent, num_episodes, save=True, namefile="new_SARSAagent.pkl"):
    reward_list = []
    reward_mean = []
    cpt = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = 0 
        
        possible_actions = env.get_legal_actions(state)
        random.shuffle(possible_actions)
        action = agent.choose_action(state, possible_actions)  # Choisir la première action

        while not done:
            next_state, reward, done = env.step(action)
            total_rewards += reward
            cpt += 1
            next_possible_actions = env.get_legal_actions(next_state)

            if not done:
                next_action = agent.choose_action(next_state, next_possible_actions)
            else:
                next_action = None  # Pas d'action si l'épisode est terminé

            # Mettre à jour la Q-table avec SARSA
            agent.update_q_value(state, action, reward, next_state, next_action, done)

            state, action = next_state, next_action  # Mettre à jour l'état et l'action pour le prochain tour
        
        reward_mean.append(total_rewards / max(cpt, 1))
        reward_list.append(total_rewards)
        cpt = 0

        if episode % 100 == 0:
            print("Épisode :", episode)
    
    if save:
        agent.save(namefile)
    
    return reward_list, reward_mean
      
def train(env, agent, num_episodes, save=False, namefile="new_Qagent.pkl"):
    reward_list=[]
    reward_mean=[]
    cpt=0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards=0 
        
        #if np.random.rand() < 0.5:
        #   state = env.first_step()

        while not done:
            possible_actions = env.get_legal_actions(state)
            random.shuffle(possible_actions)
            action = agent.choose_action(state, possible_actions)
            next_state, reward, done = env.step(action)
            total_rewards += reward
            cpt+=1
            next_possible_actions = env.get_legal_actions(next_state)

            # Mettre à jour la Q-table
            agent.update_q_value(state, action, reward, next_state, next_possible_actions,done)
            state = next_state
        
        reward_mean.append(total_rewards/cpt)
        reward_list.append(total_rewards)
        cpt=0
        if episode%100 == 0: print(" episode :",episode)
    if save:
        agent.save(namefile)
    return reward_list,reward_mean

def plot_reward(reward_list):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list, label="Total Reward per Episode", color='b', alpha=0.7)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Reward Progression over Episodes")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()