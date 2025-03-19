from algos.Qlearning import *
from Simulateurs.MPR import *
from Simulateurs.AbstractSimulator import *
import numpy as np
import copy

def vitese_mean(agent,nb_episode=100):
    l=[]
    pod = PodSim(0, 0, 0, 0, 0, loop=3, boost=0)
    game = PodSimulator(pod)
    for _ in range(nb_episode):
        game.reset()  
        l.append(plays(game,agent,display=False)[0])
    return np.mean(l),l

def accuracy(agent,bot,nb_episode=500):
    l=0
    pod = PodSim(0, 0, 0, 0, 0, loop=3, boost=0)
    game = PodSimulator(pod)
    game2 = copy.deepcopy(game)
    for _ in range(nb_episode):
        game = PodSimulator(pod)
        game2 = copy.deepcopy(game)
        if plays(game,agent,display=False)[1] < plays(game2,bot,display=False)[1]:
            l+=1
    return l/nb_episode

def plot_apprentisage(env,agent,nb_episode):
    l=[]
    bot = BotMpr(100)
    for _ in range((nb_episode//100)+1):
         train(env, agent, 100)
         l.append(accuracy(agent,bot,50))
    plot_reward(l)
    
    


    
    
def train_with_evolution(env, agent, num_episodes, save=False, namefile="new_Qagent.pkl"):
    reward_list=[]
    reward_mean=[]
    game = PodSimulator(PodSim(0, 0, 0, 0, 0, loop=3, boost=0)) # simulateur qu'on va utiliser à chaque épisode pour voir l'évolution
    numbers_steps = []
    cpt=0
    for episode in range(num_episodes):
        
        
        game.checkpoints = [[2000, 2000], [10000, 10000]]
        game.N_checkpoints = 2
        game.current_checkpoint = 1
        game.pod.x = 0
        game.pod.y = 0
        game.pod.loop=0
        game.tour=0
        
        
        state = env.reset()
        done = False
        total_rewards=0 
        

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
        

        
        if episode % 100 == 0:
            agent.set_epsilon(0)

                
            _, step_count = plays(game, agent, display=False, max_steps=3000)
            numbers_steps.append(step_count)
            agent.set_epsilon(0.5)
        
        
        reward_mean.append(total_rewards/cpt)
        reward_list.append(total_rewards)
        cpt=0
        if episode%100 == 0: print(" episode :",episode)
    if save:
        agent.save(namefile)
    return reward_list,reward_mean, numbers_steps
             
agent = QLearningAgent(0.1,0.9)
#agent = SARSAAgent(0.1,0.9)
pod = PodSim(0, 0, 0, 0, 0, loop=3, boost=0)
env = PodSimulator(pod)



test=1

if test : 
    #plot_apprentisage(env, agent, 3000)
    #trainSARSA(env,agent,1000)
    train(env,agent,2000)
    agent.save()
    
    with open("qtable.txt", 'w') as f:
        for key, value in agent.q_table.items():
            f.write(f'{key} {value}\n')
        
    
    agent.epsilon=0
    env.reset()  

    print(plays(env,agent,display=True))
    

    env.reset()  
    print(plays(env,agent,display=True,data=True))
    print(vitese_mean(agent)[0])
    print(vitese_mean(BotMpr(100))[0])
    print(accuracy(agent,BotMpr(90)))
    
    #print(agent.q_table)
   
else:
    agent =  QLearningAgent.load("MPR2.pkl")
    agent.q_table = { key:round(val,5) for key,val in agent.q_table.items()}
    
    with open("qtable.txt", 'w') as f:
            f.write(f'{agent.q_table}\n')
            
    agent.epsilon=0
    print(plays(env,agent,display=True))
    print(vitese_mean(agent))
    print(vitese_mean(BotMpr(90)))
    print(accuracy(agent,BotMpr(90)))