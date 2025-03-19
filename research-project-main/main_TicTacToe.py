from algos.Qlearning import *
from Simulateurs.TicTacToe import *
from Simulateurs.AbstractSimulator import *

def play(j1,j2,display=True):
    game =  TicTacToe()
    p= random.choice([True,False])
    oldpos=(-1,-1)
    while(not (game.victory(1) or game.victory(2) or game.is_full())):
        if p == True:
            oldpos = j1.joue(game.get_legal_actions(),oldpos)
            if game.get_cell(oldpos):  print(" coup deja jouer ");return
            game.take_action(oldpos,1)
        else:
            oldpos = j2.joue(game.get_legal_actions(),oldpos)
            if game.get_cell(oldpos):  print(" coup deja jouer ");return
            game.take_action(oldpos,2)
        
        p = not p
        if display: game.display()
  
    if game.victory(1):  
        if display: print(" joueur 1 (X) a gagner !!") 
        return 1
    elif game.victory(2):  
          if display: print(" joueur 2 (O) a gagner !!")
          return 2
    else:
          if display: print(" !! egalité !!")
          return 0

agent = JQlearning()
env = TicTacToe()
num_episodes = 10000

train(env, agent, num_episodes, True, "t.pkl")

rewards = agent.play(env, num_episodes=1)
print("Récompenses de l'épisode joué:", rewards)







