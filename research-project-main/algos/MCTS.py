import random
import math
import copy
from TicTacToe.SimulateurMCTS_TTT import Grille
# Classe pour représenter un nœud du MCTS
class Node:
    def __init__(self, state, parent=None, move=None,exploration_weight=2.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.exploration_weight = exploration_weight
        
    def is_fully_expanded(self):
        return not self.state.get_legal_actions() 

    def best_child(self, exploration_weight=None):
        if not self.children:
            raise Exception("Aucun enfant n'a été généré pour ce nœud.")
        log_parent_visits = math.log(self.visits + 1)  # Pré-calcul
        if exploration_weight == None: exploration_weight = self.exploration_weight
        return max(
            self.children,
            key=lambda child: (child.wins / child.visits ) + 
                              exploration_weight * math.sqrt(log_parent_visits / child.visits )
        )

# Classe pour gérer l'algorithme MCTS
class MCTS:
    def __init__(self, game, simulations=1000,exploration_weigh=2.0):
        self.game = game
        self.simulations = simulations
        self.exploration_weigh = exploration_weigh
        
    def search(self):
        root = Node(state=self.game,exploration_weight=self.exploration_weigh)

        for _ in range(self.simulations):
            node = root

            # 1. Sélection
            if node.is_fully_expanded() and node.children:
                node = node.best_child()

            # 2. Expansion
            if not node.is_fully_expanded():
                action = node.state.get_legal_actions()[0] # Choisir une action non encore explorée
                new_state = node.state.take_action(action)
                child_node = Node(state=new_state, parent=node, move=action)
                node.children.append(child_node)
                node = child_node  # On simule immédiatement cet enfant

            # 3. Simulation
            result = self.simulate(node.state)

            # 4. Rétropropagation
            self.backpropagate(node, result)

        if root.children:
            return root.best_child(exploration_weight=0).move
        else:
            raise Exception("Aucun enfant n'a été généré. Le MCTS a échoué.")

    def simulate(self, state):
        """Simule un jeu complet aléatoire."""
        current_state = copy.deepcopy(state)
        
        player = 'o'
        action = random.choice(current_state.get_legal_actions())
        current_state = current_state.take_action(action, player)
        player = 'x'
        
        while not current_state.victoire(action,player) and not current_state.egal:
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.take_action(action, player)
            player = 'x' if player == 'o' else 'o'  # Alternance des joueurs

        return current_state.winner  # Retourne 'x', 'o' ou None en cas d'égalité

    def backpropagate(self, node, result):
        """Met à jour les statistiques des nœuds en fonction du résultat."""
        node.visits += 1
        if result == 'x':  # Victoire de 'x'
            node.wins += 1
        elif result == 'o':  # Victoire de 'o'
            node.wins -= 1 # On compte négativement pour aider 'x'
        node = node.parent

if __name__ == "__main__":
    game = Grille()
    mcts = MCTS(game, simulations=10000)
    mcts.game.play("0 0",'x')
    mcts.game.play("1 0",'o')
    mcts.game.affiche()
    # Lancer l'algorithme de recherche
    best_move = mcts.search()

    print(f"Meilleur coup trouvé : {best_move}")
    mcts.game.affiche()