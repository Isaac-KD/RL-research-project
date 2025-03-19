from algos.Players import JQlearning,JMan,JRandom
from algos.Qlearning import QLearningAgent,train
from Simulateurs.TicTacToe import TicTacToe,play
import matplotlib.pyplot as plt
import copy

def visualisation_TTT_moy(agent):
    l = [ [[] for _ in range(3)] for _ in range(3)]
    for (_,(x,y)),value in agent.q_table.items():
        l[x][y].append(value)
    
    for x in range(3):
        for y in range(3):
            l[x][y] = sum(l[x][y])/len(l[x][y])
    
    for i in range(len(l)):
        print(l[i])
        
def visualisation_TTT_som(agent):
    l = [ [0 for _ in range(3)] for _ in range(3)]
    for (_,(x,y)),value in agent.q_table.items():
        l[x][y]+=value
    
    for i in range(len(l)):
        print(l[i])
        
def accuracy(play_fonction,agent,witness,nb_iteration=1000):
    l=[0,0,0]
    affiche=False
    for _ in range(nb_iteration):
        i = play_fonction(agent,witness,affiche)
        if l[1]>700 or l[2]>700: affiche = True
        agent.env.reset()
        try:
            witness.env.reset()
        except: pass
        l[i]+=1
        
    return 1-l[2]/nb_iteration,l[1]/nb_iteration,l

def graphique_training():
    agent = QLearningAgent()
    env = TicTacToe()
    l_accurency_notdef = []
    l_accurency_vict = []
    l_random_notdef , l_random_vict = [],[]
    nb_train = [0]  # Initialisation avec 0 pour éviter l'erreur d'indexation
    j3 = JRandom()
    num_episodes = 100  # Défini un nombre d'épisodes d'entraînement

    for i in range(100, int(1e4) + 1, 500):
        nb_train.append(nb_train[-1]+i)  # Ajoute la somme cumulative
        train(env, agent, nb_train[-1], save=True)
        j1 = JQlearning(agent=agent)
        
        acc_notdef, acc_vict, _ = accuracy(play, j1, j3)  # Correction du nom de la fonction
        random_notdef , random_vict ,_ = accuracy(play, j3, j3) 
        
        l_accurency_notdef.append(acc_notdef)
        l_accurency_vict.append(acc_vict)
        l_random_notdef.append(random_notdef)
        l_random_vict.append(random_vict)

    # Création du graphique
    plt.figure(figsize=(10, 5))
    plt.plot(nb_train[1:], l_accurency_notdef, label="1-taux d'echec du modèle", color='blue')
    plt.plot(nb_train[1:],  l_random_notdef, label="1-taux d'echec random", color='green')
    plt.plot(nb_train[1:],  l_random_vict, label="accurency random", color='purple')
    plt.plot(nb_train[1:], l_accurency_vict, label="accuracy modèle", color='pink')

    # Ajout du titre et des labels
    plt.title(f"Train (-0.1 , 0.1) unkonw alpha = {agent.alpha}, gamma = {agent.gamma}, epsilon = {agent.epsilon}")
    plt.xlabel("Nombre d'épisodes d'entraînement")
    plt.ylabel("Taux de réussite")

    # Affichage de la légende
    plt.legend()
    plt.show()
   
agent = QLearningAgent()
env = TicTacToe()
reward_list=train(env,agent,100000,save=True) 
#agent.load("new_agent.pkl")
agent.epsilon=0

   
print()
#visualisation_TTT_som(agent)
#agent = QLearningAgent.load("new_agent.pkl")    
for (etat, (x, y)), value in sorted(agent.q_table.items(), key=lambda item: item[1], reverse=True):
    if etat == 0:
        print((x, y), " = ", value)      
agent.epsilon=0
j1 = JQlearning(agent = agent)
j2 = JMan()
j3 = JRandom()
print(accuracy(play,j1,j3))
visualisation_TTT_moy(agent)   
print(accuracy(play,j1,copy.deepcopy(j1))) 
play(j1,j2)
print()
