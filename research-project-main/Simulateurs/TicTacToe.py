from Simulateurs.AbstractSimulator import AbstractSimulator
import random

class TicTacToe(AbstractSimulator):
    def __init__(self,agent=None):
        self.game_state = 0                         # 9 bits pour représenter l'état des cases (vide ou occupée)
        self.player_1_mask = 0                      # 9 bits, joueur 1 (X)
        self.player_2_mask = 0                      # 9 bits, joueur 2 (O)
        self.agent = agent
        self.win_masks = [
            0b111000000, 0b000111000, 0b000000111,  # Lignes
            0b100100100, 0b010010010, 0b001001001,  # Colonnes
            0b100010001, 0b001010100                # Diagonales
        ]
    
    def get_legal_actions(self,state):
        """
        state doit etre de 18 bit !
        """
        return [ (i,j) for i in range(3) for j in range(3) if not (state >> (2 * (i * 3 + j))) & 3]
    
    def get_state(self):
        """
        renvoie un entier dont les 18 prmeiers bits reprensente l'etat du jeux.
        Returns:
            int : l'etat actuelle du jeux 
        """
        state = 0
        for i in range(3):
            for j in range(3):
                index = i * 3 + j
                case = ((self.player_2_mask >> index) & 1) << 1 | ((self.player_1_mask >> index) & 1)
                state |= (case << (index * 2))  # On utilise |= pour bien placer les bits
        return state

    def get_cell(self, action):
        """Retourne quel joueur a joué dans la case donnée"""
        index = action[0] * 3 + action[1]       #action[0] = row , action[1] = col
        if (self.player_1_mask >> index) & 1:
            return 1                            # Joueur 1 (X)
        elif (self.player_2_mask >> index) & 1:
            return 2                            # Joueur 2 (O)
        return 0                                # Case vide

    def take_action(self, action , player):
        """
        Marque la case comme occupée par le joueur
        player : int , egale a 1 ou 2
        action : Tuple(int) , (row,col) la position de l'action a jouer 
        """
        index = action[0] * 3 + action[1]           # action[0] = row , action[1] = col
        if player == 1:
            self.player_1_mask |= (1 << index)      # Marquer la case pour joueur 1
        elif player == 2:
            self.player_2_mask |= (1 << index)      # Marquer la case pour joueur 2
        self.game_state |= (1 << index)             # Marquer la case comme occupée
            
    def victory(self,player):
        """ 
        Vérifie si un joueur a gagné
        ici action n'est pas utiliser
        """
        if player == 1:
            p_mask = self.player_1_mask
        else:
            p_mask = self.player_2_mask
                
        return any((p_mask & mask) == mask for mask in self.win_masks)
    
    def is_full(self):
        """ Vérifie si la grille est pleine """
        return self.game_state==511

    def display(self):
        """ Affiche la grille """
        symbols = {0: ".", 1: "X", 2: "O"}
        for i in range(3):
            print(" ".join(symbols[self.get_cell((i, j))] for j in range(3)))
        print()
    
    def reset(self):
        """
        Reset le jeux.
        """
        self.game_state = 0                         # 9 bits pour représenter l'état des cases (vide ou occupée)
        self.player_1_mask = 0                      # 9 bits, joueur 1 (X)
        self.player_2_mask = 0                      # 9 bits, joueur 2 (O)
        return 0
    
    def step(self,action):
        self.take_action(action,1)                  # l'action de l'IA
        if self.victory(1): return self.get_state(),1,1            # victoire du joueur 1 ( IA)
        if self.is_full():                          # egalité
            return self.get_state(),0,1                            
        else:
            st = self.get_state()
            if self.agent==None: action_bot = random.choice(self.get_legal_actions(st))
            else: action_bot = self.agent.choose_action(st,self.get_legal_actions(st))
            self.take_action(action_bot,2)
            if self.victory(2): return self.get_state(),-1,1       # victoire du bot 
            if self.is_full(): return self.get_state(),0,1   
        # etat non terminal  
            #if action ==(1,1): return self.get_state(),0.1,0                
            return self.get_state(),-0.1,0        
    
    def first_step(self):
        if self.agent==None: action_bot = random.choice(self.get_legal_actions(self.get_state()))
        else: action_bot = self.agent.choose_action(self.get_state(),self.get_legal_actions(self.get_state()))  
        self.take_action(action_bot,2)
        return self.get_state()
    
def play(j1,j2,display=True):
    game =  TicTacToe()
    p= random.choice([True,False])
    oldpos=(-1,-1)
    while(not (game.victory(1) or game.victory(2) or game.is_full())):
        if p == True:
            oldpos = j1.joue(game.get_legal_actions(game.get_state()),oldpos)
            if game.get_cell(oldpos):  print(" coup deja jouer ");return
            game.take_action(oldpos,1)
        else:
            oldpos = j2.joue(game.get_legal_actions(game.get_state()),oldpos)
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
    
if "__main__" == __name__:
    # Exemple d'utilisation
    for _ in range(2):
        game = TicTacToe()
        game.take_action((0, 0), 1)
        game.display()
        game.take_action((1, 1), 2)
        game.display()
        game.take_action((0, 1), 1)
        game.display()
        game.take_action((2, 2), 2)
        game.display()
        game.take_action((0, 2), 1)  # X gagne ici
        game.display()

        print(bin(game.get_state()))
        if game.victory(1):
            print("Le joueur X a gagné !")
        elif game.victory(2):
            print("Le joueur O a gagné !")
        elif game.is_full():
            print("Match nul !")
        game.reset()
    class JRandom():
        def __init__(self):
            pass
        
        def joue(self,validActionCount=None,oldpos=None):
            return random.choice(validActionCount)
   
                
    j1,j2   = JRandom(),JRandom()
    l=[0,0,0]
    for _ in range(300): l[play(j1,j2,False)]+=1
    print(l)