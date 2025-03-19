from Simulateurs.TicTacToe import TicTacToe
from algos.Qlearning import QLearningAgent

import random

class JQlearning():
    def __init__(self,fichier=None,agent=False):
        self.env = TicTacToe()
        self.agent = agent if agent else QLearningAgent.load(fichier) 
    
    def joue(self,validAction,oldpos):
        if oldpos != (-1,-1): 
            self.env.take_action(oldpos,2)
            
        action = self.agent.choose_action(self.env.get_state(),validAction)
        self.env.take_action(action,1)
        return action
    
class JRandom():
    def __init__(self):
        pass
        
    def joue(self,validActionCount=None,oldpos=None):
        return random.choice(validActionCount) 
    
class JMan():
    def __init__(self):
        pass
        
    def joue(self,validActionCount=None,oldpos=None):
        return tuple(map(int, input(">").split()))